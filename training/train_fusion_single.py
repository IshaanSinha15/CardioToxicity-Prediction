from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader as GeoLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from prediction_backend.models.gnn_encoder import GNNEncoder
from prediction_backend.models.fusion_single_task import FusionSingleTask
from prediction_backend.molecular_processing.graph_builder import build_graph


# 🔥 CHANGE PER RUN

TARGET = "cav"
GNN_PATH = "prediction_backend/models/saved_models/gnn_cav.pt"


# =========================
# Dataset
# =========================
class FusionDataset(Dataset):

    def __init__(self, df, embeddings):
        self.smiles = df["smiles"].tolist()
        self.labels = df[TARGET].values.astype(np.float32)
        self.embeddings = embeddings

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx], self.embeddings[idx]


# =========================
# Collate
# =========================
def collate_fn(batch):

    graphs, labels, emb = [], [], []

    for sm, label, e in batch:
        try:
            g = build_graph(sm)
            graphs.append(g)
            labels.append(label)
            emb.append(e)
        except:
            continue

    if len(graphs) == 0:
        return None

    graph_batch = next(iter(GeoLoader(graphs, batch_size=len(graphs))))

    return (
        torch.tensor(np.array(emb), dtype=torch.float),
        graph_batch,
        torch.tensor(np.array(labels), dtype=torch.float).view(-1, 1)
    )


# =========================
# Evaluation
# =========================
def evaluate(model, loader, gnn, device):

    model.eval()
    preds_all, labels_all = [], []

    with torch.no_grad():
        for batch in loader:

            if batch is None:
                continue

            chem_emb, graph_batch, labels = batch

            chem_emb = chem_emb.to(device)
            labels = labels.to(device)

            gnn_emb = gnn.encode(graph_batch)

            preds = model(chem_emb, gnn_emb)

            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    preds = np.vstack(preds_all)
    labels = np.vstack(labels_all)

    return (
        np.sqrt(mean_squared_error(labels, preds)),
        mean_absolute_error(labels, preds),
        r2_score(labels, preds)
    )


# =========================
# Training
# =========================
def train():

    df = pd.read_csv("data/datasets/final_combined.csv")
    embeddings = np.load("data/chemberta_embeddings.npy")

    # 🔥 FIX alignment
    valid_idx = df[~df[TARGET].isna()].index
    df = df.loc[valid_idx].reset_index(drop=True)
    embeddings = embeddings[valid_idx]

    print("Dataset:", df.shape)
    print("Embeddings:", embeddings.shape)

    train_df, val_df, train_emb, val_emb = train_test_split(
        df, embeddings, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(
        FusionDataset(train_df, train_emb),
        batch_size=256,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        FusionDataset(val_df, val_emb),
        batch_size=256,
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gnn = GNNEncoder(GNN_PATH, device=device)
    model = FusionSingleTask().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-5,
        weight_decay=5e-4
    )

    loss_fn = torch.nn.SmoothL1Loss()

    best_r2 = -1
    patience = 5
    counter = 0

    for epoch in range(50):

        model.train()

        for batch in train_loader:

            if batch is None:
                continue

            chem_emb, graph_batch, labels = batch

            chem_emb = chem_emb.to(device)
            labels = labels.to(device)

            gnn_emb = gnn.encode(graph_batch)

            preds = model(chem_emb, gnn_emb)

            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()

        train_rmse, _, train_r2 = evaluate(model, train_loader, gnn, device)
        val_rmse, val_mae, val_r2 = evaluate(model, val_loader, gnn, device)

        gap = train_r2 - val_r2

        print(f"\nEpoch {epoch+1}")
        print(f"Train R2: {train_r2:.4f}")
        print(f"Val   R2: {val_r2:.4f}")
        print(f"Gap: {gap:.4f}")

        # 🔥 HARD STOP if overfitting
        if gap > 0.20:
            print("Overfitting detected → stopping")
            break

        if val_r2 > best_r2:
            best_r2 = val_r2
            counter = 0

            torch.save(
                model.state_dict(),
                f"prediction_backend/models/saved_models/fusion_{TARGET}.pt"
            )

            print("Best model saved!")

        else:
            counter += 1

        if counter >= patience:
            print("Early stopping")
            break

    print(f"\n Best R²: {best_r2:.4f}")


if __name__ == "__main__":
    train()