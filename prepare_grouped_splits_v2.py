"""Create a reproducible 70/15/15 PlantVillage Color split by physical leaf.

Uses only the existing teammate CSV and leaf-map.json; downloads NO images
or metadata. Does not overwrite the original CSV or move any images.

Run from any directory:
    python prepare_grouped_splits_v2.py

Requirements: pandas, numpy
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "data" / "splits"
INPUT_CSV = SPLIT_DIR / "plantvillage_color_split.csv"
LEAF_MAP_FILE = SPLIT_DIR / "leaf-map.json"
OUTPUT_CSV = SPLIT_DIR / "plantvillage_color_split_grouped.csv"
CLASS_REPORT = SPLIT_DIR / "split_per_class.csv"
AUDIT_FILE = SPLIT_DIR / "split_audit.txt"

SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLITS = tuple(SPLIT_RATIOS)


def normalise_image_identifier(filename):
    """Mirror PlantVillage's filename-to-leaf-map convention."""
    identifier = Path(str(filename)).stem
    identifier = identifier.replace("_final_masked", "")
    identifier = identifier.split("___")[-1]
    identifier = identifier.lower().split("copy")[0].strip()
    return identifier


def lookup_group(class_name, filename, leaf_map):
    identifier = normalise_image_identifier(filename)
    candidates = leaf_map.get(identifier, [])
    if not isinstance(candidates, list):
        raise ValueError(f"Unexpected leaf-map entry for {identifier}")

    # Prefer precisely one metadata ID belonging to this particular class.
    prefix = f"{class_name}:::"
    matches = sorted({str(x) for x in candidates if str(x).startswith(prefix)})
    if len(matches) == 1:
        return f"mapped|{matches[0]}", "mapped"

    # The same normalized name still stays in one split. It is not proof
    # of physical-leaf uniqueness, so record the uncertainty explicitly.
    status = "ambiguous" if len(matches) > 1 else "unmatched"
    return f"fallback|{class_name}|{identifier}", status


def assign_class_groups(class_rows, rng):
    """Greedy image-count balancing without splitting any leaf group."""
    size_by_group = class_rows.groupby("group_key").size()
    if len(size_by_group) < 3:
        raise ValueError(
            f"Class {class_rows['class_name'].iloc[0]!r} has only "
            f"{len(size_by_group)} independent groups; cannot make three splits"
        )

    # Randomise ties reproducibly; place larger groups first to reduce skew.
    groups = list(size_by_group.items())
    rng.shuffle(groups)
    groups.sort(key=lambda group: group[1], reverse=True)

    targets = {s: len(class_rows) * SPLIT_RATIOS[s] for s in SPLITS}
    counts = {s: 0 for s in SPLITS}
    allocations = {}

    for index, (group_key, size) in enumerate(groups):
        remaining_groups = len(groups) - index
        empty = [s for s in SPLITS if counts[s] == 0]
        # Ensure every class gets at least one physical group in each split.
        choices = empty if len(empty) >= remaining_groups else SPLITS

        def imbalance(candidate):
            trial = counts.copy()
            trial[candidate] += size
            return sum(abs(trial[s] - targets[s]) for s in SPLITS)

        chosen = min(choices, key=lambda s: (imbalance(s), SPLITS.index(s)))
        allocations[group_key] = chosen
        counts[chosen] += size

    if not all(counts.values()):
        raise AssertionError("A class received an empty split")
    return allocations


def main():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Missing teammate CSV: {INPUT_CSV}")
    if not LEAF_MAP_FILE.is_file():
        raise FileNotFoundError(
            f"Missing {LEAF_MAP_FILE}. Reuse the file from leakage_check.py; "
            "no dataset download is needed."
        )

    df = pd.read_csv(INPUT_CSV)
    required = {"folder_class", "file_name", "class_name", "label_id", "split"}
    if missing := required - set(df.columns):
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    if df.duplicated(["folder_class", "file_name"]).any():
        raise ValueError("CSV contains duplicate image paths")
    if (df.groupby("class_name")["label_id"].nunique() != 1).any():
        raise ValueError("Some class names have inconsistent label IDs")
    if (df.groupby("label_id")["class_name"].nunique() != 1).any():
        raise ValueError("Some label IDs refer to multiple class names")

    with LEAF_MAP_FILE.open(encoding="utf-8") as f:
        leaf_map = json.load(f)
    if not isinstance(leaf_map, dict):
        raise ValueError("leaf-map.json must contain a JSON object")

    df["original_split"] = df["split"]
    group_matches = [
        lookup_group(row.class_name, row.file_name, leaf_map)
        for row in df.itertuples(index=False)
    ]
    df["group_key"] = [g for g, status in group_matches]
    df["group_status"] = [status for g, status in group_matches]

    # Measure leakage from the teammate's original train/test split.
    original_overlaps = int((df.groupby("group_key")["original_split"].nunique() > 1).sum())

    # Custom splitting across the FULL dataset: the published official
    # train/test boundary is not imposed, because it conflicted with groups.
    rng = np.random.default_rng(SEED)
    assignment = {}
    for class_name, class_rows in df.groupby("class_name", sort=True):
        assignment.update(assign_class_groups(class_rows, rng))
    df["split"] = df["group_key"].map(assignment)

    # Checks to enforce leakage-aware dataset integrity.
    assert df["split"].notna().all()
    overlaps = int((df.groupby("group_key")["split"].nunique() > 1).sum())
    if overlaps:
        raise AssertionError(f"{overlaps} leaf groups appear in several splits")

    by_class = pd.crosstab(df["class_name"], df["split"])
    by_class = by_class.reindex(columns=SPLITS, fill_value=0)
    if (by_class == 0).any().any():
        raise ValueError("At least one class is missing from train, val or test")

    # If the teammate's CSV has MD5 hashes, also prevent exact duplicates
    # from being scattered across splits (a separate leakage check).
    if "md5" in df.columns:
        actual_hashes = df[df["md5"].notna()].copy()
        duplicate_hashes = (
            actual_hashes.groupby("md5")["split"].nunique() > 1
        )
        if duplicate_hashes.any():
            raise ValueError(
                f"{int(duplicate_hashes.sum())} identical-image MD5 groups "
                "would cross splits; no CSV saved. Inspect duplicates first."
            )

    # Output only after all checks pass; original data remains untouched.
    totals = df["split"].value_counts().reindex(SPLITS, fill_value=0)
    by_class.to_csv(CLASS_REPORT)
    df.to_csv(OUTPUT_CSV, index=False)

    lines = [
        "PlantVillage Color — custom leaf-group-aware split audit",
        f"Random seed: {SEED}",
        "Target: 70% train / 15% validation / 15% test (approximate)",
        "Source: original teammate CSV + PlantVillage leaf-map.json",
        "Original teammate CSV: unchanged",
        "Images: unchanged; no files moved or downloaded",
        "Published train/test lists: NOT used to constrain custom splitting",
        "",
        f"Total images: {len(df):,}",
        f"Classes: {df['class_name'].nunique()}",
        f"Unique groups: {df['group_key'].nunique():,}",
        "",
        "Final counts:",
    ]
    for s in SPLITS:
        lines.append(f"  {s}: {totals[s]:,} ({100 * totals[s]/len(df):.2f}%)")
    lines += [
        "",
        "Leaf metadata coverage:",
    ]
    for status in ("mapped", "unmatched", "ambiguous"):
        count = int((df["group_status"] == status).sum())
        lines.append(f"  {status}: {count:,} images")
    lines += [
        "",
        f"Cross-split group overlaps in ORIGINAL teammate CSV: {original_overlaps:,}",
        f"Cross-split group overlaps in NEW CSV: {overlaps}",
        "All 38 classes represented in each new split: yes",
        "",
        "LIMITATION: Groups for unmatched/ambiguous images rely on normalized",
        "filenames; they cannot prove photos of the same physical leaf are",
        "always grouped. The metadata itself may contain inconsistencies.",
        "The test split must be held out from model selection and tuning.",
        "",
        f"New CSV: {OUTPUT_CSV}",
        f"Per-class table: {CLASS_REPORT}",
    ]
    AUDIT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
