
import pandas as pd
from pathlib import Path

# Your existing project directories
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "raw_images" / "PlantVillage" / "color"
CSV_PATH = BASE_DIR / "data" / "splits" / "plantvillage_color_split.csv"

# 1. Read the CSV
df = pd.read_csv(CSV_PATH)

print("Total images:", len(df))
print("\nSplit distribution:")
print(df["split"].value_counts())

# 2. Check whether the actual files exist
df["full_path"] = df.apply(
    lambda row: DATASET_PATH / row["folder_class"] / row["file_name"],
    axis=1
)

df["exists"] = df["full_path"].apply(lambda path: path.is_file())

print("\nImages found:", df["exists"].sum())
print("Missing images:", (~df["exists"]).sum())

# 3. Verify class information
print("\nNumber of classes:", df["class_name"].nunique())

print("\nImages per split:")
print(pd.crosstab(df["class_name"], df["split"]))

# 4. Check duplicate paths
print("\nDuplicate image paths:", df["full_path"].duplicated().sum())

# 5. Verify class-label mapping
print(
    "Classes with inconsistent IDs:",
    (df.groupby("class_name")["label_id"].nunique() > 1).sum()
)

# 6. Display a sample record
print("\nSample image:")
print(df[["file_name", "class_name", "label_id", "split"]].head(1))
