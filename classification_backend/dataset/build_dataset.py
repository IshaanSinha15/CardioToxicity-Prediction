import os
import pandas as pd

# ==========================================================
# Paths
# ==========================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

DATA_DIR = os.path.join(
    ROOT_DIR,
    "data",
    "datasets",
)

HERG_DATASET = os.path.join(
    DATA_DIR,
    "hERG_final_training_unique.csv",
)

NAV_DATASET = os.path.join(
    DATA_DIR,
    "Nav1.5_final_training.csv",
)

CAV_DATASET = os.path.join(
    DATA_DIR,
    "Cav1.2_final_training.csv",
)

OUTPUT_DIR = os.path.join(
    ROOT_DIR,
    "classification_backend",
    "dataset",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

OUTPUT_DATASET = os.path.join(
    OUTPUT_DIR,
    "complete_ic50_dataset.csv",
)

# ==========================================================
# Load Datasets
# ==========================================================

print("=" * 60)
print("Loading IC50 Datasets")
print("=" * 60)

herg_df = pd.read_csv(HERG_DATASET)
nav_df = pd.read_csv(NAV_DATASET)
cav_df = pd.read_csv(CAV_DATASET)

print(f"hERG Dataset   : {herg_df.shape}")
print(f"Nav1.5 Dataset : {nav_df.shape}")
print(f"Cav1.2 Dataset : {cav_df.shape}")

# ==========================================================
# Rename Columns
# ==========================================================

herg_df = herg_df.rename(
    columns={
        "IC50_nM": "IC50_IKr"
    }
)

nav_df = nav_df.rename(
    columns={
        "IC50_nM": "IC50_INa"
    }
)

cav_df = cav_df.rename(
    columns={
        "IC50_nM": "IC50_ICaL"
    }
)

# ==========================================================
# Remove Duplicate SMILES
# ==========================================================

herg_df = herg_df.drop_duplicates(
    subset="smiles"
)

nav_df = nav_df.drop_duplicates(
    subset="smiles"
)

cav_df = cav_df.drop_duplicates(
    subset="smiles"
)

print("\nDuplicates removed.")

# ==========================================================
# Merge Datasets
# ==========================================================

print("\nMerging datasets...")

merged_df = herg_df.merge(
    nav_df,
    on="smiles",
    how="outer",
)

merged_df = merged_df.merge(
    cav_df,
    on="smiles",
    how="outer",
)

# ==========================================================
# Source Information
# ==========================================================

def get_source(row):

    channels = []

    if pd.notna(row["IC50_IKr"]):
        channels.append("IKr")

    if pd.notna(row["IC50_INa"]):
        channels.append("INa")

    if pd.notna(row["IC50_ICaL"]):
        channels.append("ICaL")

    return ",".join(channels)

merged_df["ic50_source"] = merged_df.apply(
    get_source,
    axis=1,
)

# ==========================================================
# Sort
# ==========================================================

merged_df = merged_df.sort_values(
    by="smiles"
).reset_index(drop=True)

# ==========================================================
# Statistics
# ==========================================================

print("\n")
print("=" * 60)
print("Merged Dataset Summary")
print("=" * 60)

print(f"Total Molecules : {len(merged_df)}")

print("\nColumns")

print(merged_df.columns.tolist())

print("\nMissing Values")

print(merged_df.isnull().sum())

print("\nAvailable Channels")

print(merged_df["ic50_source"].value_counts())

print("\nIC50 Summary")

print(
    merged_df[
        [
            "IC50_IKr",
            "IC50_INa",
            "IC50_ICaL",
        ]
    ].describe()
)

# ==========================================================
# Save
# ==========================================================

merged_df.to_csv(
    OUTPUT_DATASET,
    index=False,
)

print("\n")
print("=" * 60)
print("Dataset Saved Successfully")
print("=" * 60)

print(OUTPUT_DATASET)