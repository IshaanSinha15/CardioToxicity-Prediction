import pandas as pd
import numpy as np

herg = pd.read_csv("data/datasets/herg_final_training_unique.csv")
nav = pd.read_csv("data/datasets/nav1.5_final_training.csv")
cav = pd.read_csv("data/datasets/cav1.2_final_training.csv")

herg = herg.rename(columns={"IC50_nM": "herg"})
nav = nav.rename(columns={"IC50_nM": "nav"})
cav = cav.rename(columns={"IC50_nM": "cav"})

def to_pIC50(x):
    return 9 - np.log10(x)

herg["herg"] = herg["herg"].apply(lambda x: to_pIC50(x) if x > 0 else np.nan)
nav["nav"] = nav["nav"].apply(lambda x: to_pIC50(x) if x > 0 else np.nan)
cav["cav"] = cav["cav"].apply(lambda x: to_pIC50(x) if x > 0 else np.nan)

herg = herg[["smiles", "herg"]]
nav = nav[["smiles", "nav"]]
cav = cav[["smiles", "cav"]]

df = herg.merge(nav, on="smiles", how="outer")
df = df.merge(cav, on="smiles", how="outer")

# NORMALIZATION (VERY IMPORTANT)
targets = df[["herg", "nav", "cav"]]

mean = targets.mean()
std = targets.std()

df[["herg", "nav", "cav"]] = (targets - mean) / std

# Save stats
mean.to_csv("data/target_mean.csv")
std.to_csv("data/target_std.csv")

df.to_csv("data/datasets/final_combined.csv", index=False)

print("✅ Dataset ready:", df.shape)