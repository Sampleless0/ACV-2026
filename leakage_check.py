
import json
import pandas as pd
from pathlib import Path
from urllib.request import urlretrieve

BASE_DIR = Path(__file__).resolve().parent
SPLIT_DIR = BASE_DIR / "data" / "splits"

csv_path = SPLIT_DIR / "plantvillage_color_split.csv"
map_path = SPLIT_DIR / "leaf-map.json"

# Download the official leaf metadata once
if not map_path.exists():
    url = (
        "https://raw.githubusercontent.com/spMohanty/"
        "PlantVillage-Dataset/master/leaf-map.json"
    )
    print("Downloading leaf metadata...")
    urlretrieve(url, map_path)

df = pd.read_csv(csv_path)

with open(map_path, encoding="utf-8") as f:
    leaf_map = json.load(f)

# Match filenames to known physical leaves
def find_leaf_id(row):
    name = Path(row["file_name"]).stem.lower()
    name = name.replace("_final_masked", "")
    name = name.split("___")[-1]
    name = name.split("copy")[0].strip()

    matches = leaf_map.get(name, [])

    if len(matches) == 1:
        return matches[0]

    for match in matches:
        if row["class_name"] in match:
            return match

    return None

df["leaf_id"] = df.apply(find_leaf_id, axis=1)

# Audit images with known leaf IDs
known = df.dropna(subset=["leaf_id"]).copy()

known["leaf_group"] = (
    known["class_name"] + "||" + known["leaf_id"]
)

group_splits = known.groupby("leaf_group")["split"].nunique()
overlapping_groups = set(group_splits[group_splits > 1].index)

affected = known[
    known["leaf_group"].isin(overlapping_groups)
]

print("\nTotal images:", len(df))
print("Images with matched leaf IDs:", len(known))
print("Images without matched IDs:", len(df) - len(known))
print("Leaf groups crossing splits:", len(overlapping_groups))
print("Images in overlapping groups:", len(affected))

if overlapping_groups:
    print("\nWARNING: Same-leaf leakage detected!")
    print(affected[
        ["class_name", "file_name", "leaf_id", "split"]
    ].head(10).to_string(index=False))
else:
    print("\nNo overlap found among matched leaf groups.")
    print("Unmatched images still need consideration.")
