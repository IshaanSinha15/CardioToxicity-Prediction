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

INPUT_CSV = os.path.join(
    ROOT_DIR,
    "classification_backend",
    "dataset",
    "classifier_dataset_clustered.csv",
)

OUTPUT_CSV = os.path.join(
    ROOT_DIR,
    "classification_backend",
    "dataset",
    "classifier_dataset_labeled.csv",
)

OUTPUT_DIR = os.path.join(
    ROOT_DIR,
    "classification_backend",
    "outputs",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Clustered Dataset")
print("=" * 60)

df = pd.read_csv(INPUT_CSV)

print(f"Dataset Shape : {df.shape}")

print("\nColumns")
print(df.columns.tolist())

print("\nMissing Values")
print(df.isnull().sum())


# ==========================================================
# Compute Cluster Statistics
# ==========================================================

print("\n" + "=" * 60)
print("Computing Cluster Statistics")
print("=" * 60)

cluster_summary = (
    df
    .groupby("cluster")
    .agg(
        Count=("cluster", "size"),

        Mean_Block_IKr=("Block_IKr", "mean"),
        Mean_Block_INa=("Block_INa", "mean"),
        Mean_Block_ICaL=("Block_ICaL", "mean"),

        Std_Block_IKr=("Block_IKr", "std"),
        Std_Block_INa=("Block_INa", "std"),
        Std_Block_ICaL=("Block_ICaL", "std"),
    )
)

# ==========================================================
# Compute Risk Score
#
# hERG is weighted higher because it contributes more strongly
# to drug-induced QT prolongation.
# ==========================================================

cluster_summary["RiskScore"] = (
      0.60 * cluster_summary["Mean_Block_IKr"]
    + 0.20 * cluster_summary["Mean_Block_INa"]
    + 0.20 * cluster_summary["Mean_Block_ICaL"]
)

cluster_summary = cluster_summary.sort_values("RiskScore")

print(cluster_summary)


# ==========================================================
# Automatic Label Assignment
# ==========================================================

print("\n" + "=" * 60)
print("Assigning Risk Labels")
print("=" * 60)

ordered_clusters = cluster_summary.index.tolist()

label_map = {
    ordered_clusters[0]: "Low",
    ordered_clusters[1]: "Intermediate",
    ordered_clusters[2]: "High",
}

print("\nCluster Mapping")

for cluster, label in label_map.items():
    print(f"Cluster {cluster} -> {label}")


# ==========================================================
# Apply Labels
# ==========================================================

df["RiskClass"] = df["cluster"].map(label_map)


# ==========================================================
# Verify Distribution
# ==========================================================

print("\nRisk Distribution")

print(df["RiskClass"].value_counts())

print("\nCluster vs Risk")

print(
    pd.crosstab(
        df["cluster"],
        df["RiskClass"]
    )
)


# ==========================================================
# Save Cluster Summary
# ==========================================================

cluster_summary["AssignedLabel"] = cluster_summary.index.map(label_map)

cluster_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cluster_risk_summary.csv",
    )
)


# ==========================================================
# Save Labeled Dataset
# ==========================================================

df.to_csv(
    OUTPUT_CSV,
    index=False,
)


# ==========================================================
# Final Summary
# ==========================================================

print("\n" + "=" * 60)
print("Labeling Complete")
print("=" * 60)

print(f"Output Dataset : {OUTPUT_CSV}")

print("\nSaved Files")

print(
    os.path.join(
        OUTPUT_DIR,
        "cluster_risk_summary.csv",
    )
)

print("\nFinal Columns")

print(df.columns.tolist())

print("\nRisk Counts")

print(df["RiskClass"].value_counts())

print("\nDone.")