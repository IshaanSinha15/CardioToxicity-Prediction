import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


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
    "classifier_dataset.csv",
)

OUTPUT_CSV = os.path.join(
    ROOT_DIR,
    "classification_backend",
    "dataset",
    "classifier_dataset_clustered.csv",
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
print("Loading Classifier Dataset")
print("=" * 60)

df = pd.read_csv(INPUT_CSV)

print(f"Dataset Shape : {df.shape}")

print("\nColumns")
print(df.columns.tolist())

print("\nMissing Values")
print(df.isnull().sum())


# ==========================================================
# Clustering Features
# ==========================================================

FEATURE_COLUMNS = [
    "Block_IKr",
    "Block_INa",
    "Block_ICaL",
]

X = df[FEATURE_COLUMNS]

print("\nUsing Features")

for feature in FEATURE_COLUMNS:
    print(feature)


# ==========================================================
# Standardization
# ==========================================================

print("\nStandardizing Features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Done.")


# ==========================================================
# Elbow Method
# ==========================================================

print("\n" + "=" * 60)
print("Computing Elbow Curve")
print("=" * 60)

K = range(2, 9)

inertia = []

for k in K:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
    )

    model.fit(X_scaled)

    inertia.append(model.inertia_)


plt.figure(figsize=(6, 4))

plt.plot(K, inertia, marker="o")

plt.xlabel("Number of Clusters")

plt.ylabel("Inertia")

plt.title("Elbow Method")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "elbow_curve.png",
    ),
    dpi=300,
)

plt.close()


# ==========================================================
# Silhouette Scores
# ==========================================================

print("\n" + "=" * 60)
print("Silhouette Scores")
print("=" * 60)

scores = []

for k in K:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels,
    )

    scores.append(score)

    print(f"k = {k} : {score:.4f}")


plt.figure(figsize=(6, 4))

plt.plot(K, scores, marker="o")

plt.xlabel("Number of Clusters")

plt.ylabel("Silhouette Score")

plt.title("Silhouette Analysis")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "silhouette_scores.png",
    ),
    dpi=300,
)

plt.close()


# ==========================================================
# Final Clustering
# ==========================================================

print("\n" + "=" * 60)
print("Running KMeans (k = 3)")
print("=" * 60)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20,
)

clusters = kmeans.fit_predict(X_scaled)

df["cluster"] = clusters


# ==========================================================
# Cluster Summary
# ==========================================================

print("\nCluster Distribution")

print(df["cluster"].value_counts().sort_index())


summary = (
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

summary = summary.sort_index()

print("\nCluster Summary")

print(summary)


summary.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "cluster_summary.csv",
    )

)


# ==========================================================
# Cluster Centers
# ==========================================================

centers = pd.DataFrame(

    scaler.inverse_transform(
        kmeans.cluster_centers_
    ),

    columns=FEATURE_COLUMNS,

)

centers.index.name = "Cluster"

print("\nCluster Centers")

print(centers)

centers.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "cluster_centers.csv",
    )

)


# ==========================================================
# Save Dataset
# ==========================================================

df.to_csv(
    OUTPUT_CSV,
    index=False,
)


# ==========================================================
# Finished
# ==========================================================

print("\n" + "=" * 60)
print("Clustering Complete")
print("=" * 60)

print(f"Output Dataset : {OUTPUT_CSV}")

print("\nOutputs Saved")

print(os.path.join(OUTPUT_DIR, "elbow_curve.png"))
print(os.path.join(OUTPUT_DIR, "silhouette_scores.png"))
print(os.path.join(OUTPUT_DIR, "cluster_summary.csv"))
print(os.path.join(OUTPUT_DIR, "cluster_centers.csv"))

print("\nDone.")