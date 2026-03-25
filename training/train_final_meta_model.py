import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

DATASET = "data/datasets/final_combined.csv"

TASKS = ["herg", "nav", "cav"]


def train():

    df = pd.read_csv(DATASET)
    df.columns = [c.lower() for c in df.columns]

    for task in TASKS:

        # load predictions
        fusion = np.load(f"embeddings/fusion_{task}_pred.npy")
        xgb = np.load(f"embeddings/xgb_{task}_pred.npy")

        # combine model predictions
        X = np.column_stack([fusion, xgb])

        # load labels
        y = df[task].values

        # remove rows where label is missing
        mask = ~np.isnan(y)

        X = X[mask]
        y = y[mask]

        # train meta model
        model = LinearRegression()
        model.fit(X, y)

        # save model
        joblib.dump(model, f"models/saved_models/meta_{task}.pkl")

        print(f"Final meta model trained for {task}")
        print(f"Samples used: {len(y)}")
        print(f"Model saved: models/saved_models/meta_{task}.pkl\n")


if __name__ == "__main__":
    train()