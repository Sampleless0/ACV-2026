"""Prepare auditable PlantVillage Color train/validation/test assignments.

- Keep the published PlantVillage test set untouched.
- Select a custom grouped validation set within the published training set.
- Preserve the teammate's original CSV and all actual image files.
- Fail loudly on unmatched paths or detected group leakage.

Run from the ACV-2026 project folder:
    python prepare_grouped_splits.py

Requirements: pandas, numpy (already installed for dataset inspection).
"""

from pathlib import Path
from urllib.request import urlretrieve
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "data" / "splits"
SOURCE_CSV = SPLITS / "plantvillage_color_split.csv"
OUTPUT_CSV = SPLITS / "plantvillage_color_split_grouped.csv"
REPORT = SPLITS / "split_audit.txt"
SEED = 42
# 12.5% of the official ~80% training partition -> ~10% of all images.
VALIDATION_FRACTION_OF_OFFICIAL_TRAIN = 0.125

SOURCES = {
    "color_train.txt": "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/splits/color_train.txt",
    "color_test.txt": "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/splits/color_test.txt",
    "leaf-map.json": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/leaf-map.json",
}


def ensure_sources() -> None:
    SPLITS.mkdir(parents=True, exist_ok=True)
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(f"Teammate CSV not found: {SOURCE_CSV}")
    for filename, url in SOURCES.items():
        dest = SPLITS / filename
        if not dest.is_file():
            print(f"Downloading {filename}...")
            try:
                urlretrieve(url, dest)
            except Exception:
                dest.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Could not download {filename}. Check connection or download it "
                    f"manually to {SPLITS} from:\n{url}"
                ) from None


def read_manifest(name: str) -> set[str]:
    """Official entries may be raw/color/CLASS/FILE; use CLASS/FILE."""
    keys = set()
    with (SPLITS / name).open(encoding="utf-8") as file:
        for line in file:
            path = line.strip().replace("\\", "/")
            if not path:
                continue
            parts = path.split("/")
            if len(parts) < 2:
                raise ValueError(f"Unexpected line in {name}: {path}")
            key = "/".join(parts[-2:])
            if key in keys:
                raise ValueError(f"Duplicate entry in {name}: {key}")
            keys.add(key)
    if not keys:
        raise ValueError(f"Empty official split: {name}")
    return keys


def normalise_image_identifier(filename: str) -> str:
    """Match the PlantVillage loader's map-key convention."""
    identifier = Path(filename).stem
    identifier = identifier.replace("_final_masked", "")
    identifier = identifier.split("___")[-1]
    identifier = identifier.lower().split("copy")[0].strip()
    return identifier


def get_group(row, leaf_map):
    """Return group ID and match status. Never invent a mapped leaf ID."""
    key = normalise_image_identifier(row.file_name)
    candidates = leaf_map.get(key, [])
    class_prefix = row.class_name + ":::"
    matching = sorted(set(str(x) for x in candidates if str(x).startswith(class_prefix)))
    if len(matching) == 1:
        return "mapped|" + matching[0], "mapped"
    # Ambiguous or missing metadata: group identical normalized identifiers.
    status = "ambiguous" if len(matching) > 1 else "unmatched"
    return f"fallback|{row.class_name}|{key}", status


def choose_validation_groups(training: pd.DataFrame) -> set[str]:
    """Approximate 12.5% validation images per class, without splitting leaf groups."""
    rng = np.random.default_rng(SEED)
    chosen = set()
    for class_name, part in training.groupby("class_name", sort=True):
        sizes = part.groupby("group_key").size()
        if len(sizes) < 2:
            raise ValueError(f"Not enough leaf groups for validation: {class_name}")

        shuffled = rng.permutation(len(sizes))
        group_ids = sizes.index.to_numpy()[shuffled]
        counts = sizes.to_numpy()[shuffled]
        target = len(part) * VALIDATION_FRACTION_OF_OFFICIAL_TRAIN
        cumulative = np.cumsum(counts)

        # Select the prefix with the closest IMAGE count to the target,
        # keeping at least one group in training and one in validation.
        candidate_cuts = np.arange(1, len(group_ids))
        cut = int(candidate_cuts[np.argmin(np.abs(cumulative[:-1] - target))])
        chosen.update(group_ids[:cut].tolist())
    return chosen


def main():
    ensure_sources()
    train_keys = read_manifest("color_train.txt")
    test_keys = read_manifest("color_test.txt")
    if train_keys & test_keys:
        raise ValueError("Published manifests overlap; cannot proceed.")

    df = pd.read_csv(SOURCE_CSV)
    required = {"folder_class", "file_name", "class_name", "label_id", "split"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV missing columns: {sorted(missing_columns)}")
    if df.duplicated(["folder_class", "file_name"]).any():
        raise ValueError("CSV contains duplicate class/filename paths")
    if (df.groupby("class_name")["label_id"].nunique() > 1).any():
        raise ValueError("Inconsistent numeric class labels")

    # Use published lists only as the starting train/test boundary.
    keys = df["folder_class"].astype(str) + "/" + df["file_name"].astype(str)
    is_train = keys.isin(train_keys)
    is_test = keys.isin(test_keys)
    missing = ~(is_train | is_test)
    if missing.any():
        print("Unmatched examples:")
        print(df.loc[missing, ["folder_class", "file_name"]].head(10).to_string(index=False))
        raise ValueError(f"{int(missing.sum())} rows do not appear in the published manifests; no CSV saved.")

    df["original_split"] = df["split"]
    df["split"] = np.where(is_test, "test", "train")
    with (SPLITS / "leaf-map.json").open(encoding="utf-8") as file:
        leaf_map = json.load(file)
    if not isinstance(leaf_map, dict):
        raise ValueError("Unexpected leaf-map.json format")

    group_info = [get_group(row, leaf_map) for row in df.itertuples(index=False)]
    df["group_key"] = [item[0] for item in group_info]
    df["group_status"] = [item[1] for item in group_info]

    # A known physical leaf must never occur on both sides of official test boundary.
    before = df.groupby("group_key")["split"].nunique()
    invalid_groups = set(before[before > 1].index)
    if invalid_groups:
        conflicts = df[df["group_key"].isin(invalid_groups)]
        audit_path = SPLITS / "official_group_conflicts.csv"
        conflicts.to_csv(audit_path, index=False)
        raise ValueError(
            f"{len(invalid_groups)} groups cross published train/test. "
            f"Inspect {audit_path}; no training CSV saved."
        )

    validation_groups = choose_validation_groups(df.loc[is_train])
    df.loc[is_train & df["group_key"].isin(validation_groups), "split"] = "val"

    # Final assertions. Validation/test must never be altered during training.
    counts = df["split"].value_counts()
    if set(counts.index) != {"train", "val", "test"}:
        raise ValueError("Missing an expected split")
    if (df.groupby("group_key")["split"].nunique() > 1).any():
        raise ValueError("Same group crosses train/val/test. No CSV saved.")
    coverage = pd.crosstab(df["class_name"], df["split"])
    if (coverage[["train", "val", "test"]] == 0).any().any():
        raise ValueError("At least one class is missing from a split")
    assert (df.loc[is_test, "split"] == "test").all(), "Published test boundary changed"

    # Save without modifying the teammate's CSV.
    df.to_csv(OUTPUT_CSV, index=False)
    message = [
        f"Random seed: {SEED}",
        "Test set: published PlantVillage Color test manifest (held out)",
        "Validation: custom, per-class leaf-group-aware split of published training set",
        "Original teammate CSV preserved; images not moved",
        "",
        "Counts and fractions:",
    ]
    for split in ("train", "val", "test"):
        message.append(f"  {split}: {counts[split]:,} ({100 * counts[split]/len(df):.2f}%)")
    message.extend([
        "",
        "Leaf-map matching:",
        *(f"  {status}: {n:,}" for status, n in df["group_status"].value_counts().items()),
        "",
        "Detected cross-split group count: 0 (based on available metadata/fallback identifiers)",
        "IMPORTANT: Unmatched/ambiguous leaf IDs are grouped by filename identifier,",
        "not independently confirmed to represent unique physical leaves.",
        "Avoid claiming perfect absence of leakage for those rows.",
        "",
        "Source: https://huggingface.co/datasets/mohanty/PlantVillage",
        "Source metadata: https://github.com/spMohanty/PlantVillage-Dataset",
    ])
    REPORT.write_text("\n".join(message) + "\n", encoding="utf-8")
    print("\n".join(message))
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Audit: {REPORT}")


if __name__ == "__main__":
    main()
