import sys
import os

# IMPORTANT: add project root to path BEFORE other imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch

from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from prediction_backend.molecular_processing.graph_builder import build_graph
from prediction_backend.models.gnn_model import GNNModel

from deepchem.splits import ScaffoldSplitter
from deepchem.data import NumpyDataset


DATASET_PATHS = {
    "herg": "data/datasets/herg_final_training_unique.csv",
    "nav": "data/datasets/nav1.5_final_training.csv",
    "cav": "data/datasets/cav1.2_final_training.csv"
}

MODEL_PATHS = {
    "herg": "models/saved_models/gnn_herg.pt",
    "nav": "models/saved_models/gnn_nav.pt",
    "cav": "models/saved_models/gnn_cav.pt"
}


# -------------------------
# Dataset Loader
# -------------------------

def load_dataset(path):

    df = pd.read_csv(path)

    df.columns = [c.lower() for c in df.columns]

    df = df[["smiles", "ic50_nm"]]

    df = df.dropna()

    df["smiles"] = df["smiles"].astype(str)

    smiles = df["smiles"].tolist()

    # IC50 (nM) → pIC50
    labels = 9 - np.log10(df["ic50_nm"])

    # normalize targets
    labels = (labels - labels.mean()) / labels.std()

    return smiles, labels


# -------------------------
# Scaffold Split
# -------------------------

def scaffold_split(smiles, labels, dataset_type):

    dataset = NumpyDataset(
        X=np.zeros(len(smiles)),
        y=labels,
        ids=smiles
    )

    splitter = ScaffoldSplitter()

    if dataset_type == "herg":

        train, val, test = splitter.train_valid_test_split(
            dataset,
            frac_train=0.7,
            frac_valid=0.15,
            frac_test=0.15
        )

        return train, val, test

    else:

        train, test = splitter.train_test_split(
            dataset,
            frac_train=0.8
        )

        return train, None, test

# -------------------------
# Weight Computation
# -------------------------

def compute_weights(labels):

    labels = np.array(labels)

    hist, bins = np.histogram(labels, bins=10)

    weights = []

    for val in labels:

        idx = np.digitize(val, bins) - 1
        idx = min(idx, len(hist)-1)

        freq = hist[idx]

        w = 1.0 / (freq + 1e-6)

        weights.append(w)

    weights = np.array(weights)

    weights = weights / np.mean(weights)

    # clip extreme weights
    weights = np.clip(weights, 0.5, 2.0)

    return weights


# -------------------------
# Weighted Loss
# -------------------------

def weighted_mse(pred, target, weight):

    loss = (pred - target) ** 2
    loss = loss * weight

    return torch.mean(loss)


# -------------------------
# Convert dataset → graphs
# -------------------------

def graphs_from_dataset(dataset):

    graphs = []

    smiles_list = dataset.ids
    labels = dataset.y

    weights = compute_weights(labels)

    for sm, y, w in zip(smiles_list, labels, weights):

        try:

            g = build_graph(sm)

            if g is None:
                continue

            g.y = torch.tensor([y], dtype=torch.float)
            g.weight = torch.tensor([w], dtype=torch.float)

            graphs.append(g)

        except:
            continue

    print("Valid molecules:", len(graphs))

    return graphs


# -------------------------
# Evaluation
# -------------------------

def evaluate(model, loader, device):

    model.eval()

    preds = []
    actual = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            out = model(batch)

            preds.extend(out.cpu().numpy())
            actual.extend(batch.y.cpu().numpy())

    preds = np.array(preds).flatten()
    actual = np.array(actual).flatten()

    rmse = np.sqrt(mean_squared_error(actual, preds))
    mae = mean_absolute_error(actual, preds)
    r2 = r2_score(actual, preds)

    return rmse, mae, r2


# -------------------------
# Training
# -------------------------

def train(dataset_name):

    smiles, labels = load_dataset(DATASET_PATHS[dataset_name])

    train_ds, val_ds, test_ds = scaffold_split(smiles, labels, dataset_name)

    print("Train size:", len(train_ds.ids))
    print("Validation size:", len(val_ds.ids) if val_ds else 0)
    print("Test size:", len(test_ds.ids))

    train_graphs = graphs_from_dataset(train_ds)
    test_graphs = graphs_from_dataset(test_ds)

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32)

    if val_ds is not None:

        val_graphs = graphs_from_dataset(val_ds)
        val_loader = DataLoader(val_graphs, batch_size=32)

    else:

        val_loader = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GNNModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    epochs = 100
    patience = 10

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for batch in train_loader:

            batch = batch.to(device)

            optimizer.zero_grad()

            out = model(batch)

            loss = weighted_mse(
                out.view(-1),
                batch.y.view(-1),
                batch.weight.view(-1)
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

        # ---------- VALIDATION (hERG only) ----------
        if val_loader is not None:

            rmse, mae, r2 = evaluate(model, val_loader, device)

            print("Validation RMSE:", rmse)

            scheduler.step(rmse)

            if rmse < best_loss:

                best_loss = rmse
                patience_counter = 0

                os.makedirs("models/saved_models", exist_ok=True)

                torch.save(model.state_dict(), MODEL_PATHS[dataset_name])

            else:

                patience_counter += 1

            if patience_counter >= patience:

                print("Early stopping triggered")
                break

    # ---------- SAVE MODEL FOR NAV / CAV ----------
    if val_loader is None:

        os.makedirs("models/saved_models", exist_ok=True)

        torch.save(model.state_dict(), MODEL_PATHS[dataset_name])

    # ---------- TEST EVALUATION ----------
    rmse, mae, r2 = evaluate(model, test_loader, device)

    print("Test RMSE:", rmse)
    print("Test MAE:", mae)
    print("Test R2:", r2)

# -------------------------
# Main
# -------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)

    args = parser.parse_args()

    train(args.dataset)