import joblib
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]

model_path = (
    repo_root
    / "saved_models"
    / "random_forest_classifier.pkl"
)

bundle = joblib.load(model_path)

print("Bundle keys:")
print(bundle.keys())

print("\nNumber of features:")
print(bundle["n_features"])

print("\nFeature names:")
for feature in bundle["feature_names"]:
    print(feature)