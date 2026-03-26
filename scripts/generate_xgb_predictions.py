import numpy as np
import pandas as pd
import xgboost as xgb

from features.rdkit_features import featurize_smiles

DATASET = "data/datasets/final_combined.csv"

TASKS = ["herg", "nav", "cav"]


def generate():

    df = pd.read_csv(DATASET)
    df.columns = [c.lower() for c in df.columns]

    smiles = df["smiles"].astype(str).tolist()

    features = []
    feature_size = None

    for s in smiles:

        try:
            f = featurize_smiles(s)

            f = np.array(f)

            if feature_size is None:
                feature_size = len(f)

            if len(f) != feature_size:
                f = np.zeros(feature_size)

        except Exception:
            f = np.zeros(feature_size if feature_size else 1028)

        features.append(f)

    X = np.vstack(features)

    for task in TASKS:

        model = xgb.Booster()
        model.load_model(f"models/saved_models/xgb_{task}.json")

        dmatrix = xgb.DMatrix(X)

        preds = model.predict(dmatrix)

        np.save(f"embeddings/xgb_{task}_pred.npy", preds)

        print(f"XGBoost predictions saved for {task}: {preds.shape}")


if __name__ == "__main__":
    generate()