from features.rdkit_features import generate_features

# hERG
generate_features(
    "data/datasets/hERG_final_training_unique.csv",
    target_name="IC50_nM",
    save_prefix="herg"
)

# Nav
generate_features(
    "data/datasets/Nav1.5_final_training.csv",
    target_name="IC50_nM",
    save_prefix="nav"
)

# Cav
generate_features(
    "data/datasets/Cav1.2_final_training.csv",
    target_name="IC50_nM",
    save_prefix="cav"
)