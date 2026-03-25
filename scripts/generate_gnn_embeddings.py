import torch
import numpy as np
import pandas as pd

from models.gnn_model import GNNModel
from molecular_processing.graph_builder import build_graph

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET = "data/datasets/final_combined.csv"

TASKS = ["herg", "nav", "cav"]


def generate(task, smiles_list):

    model = GNNModel()

    model.load_state_dict(
        torch.load(
            f"models/saved_models/gnn_{task}.pt",
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)
    model.eval()

    outputs = []

    with torch.no_grad():

        for smi in smiles_list:

            try:
                data = build_graph(smi)

                if data is None:
                    outputs.append(np.zeros(128))
                    continue

                data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

                data = data.to(DEVICE)

                emb = model(data, return_embedding=True)

                outputs.append(emb.cpu().numpy())

            except Exception:
                outputs.append(np.zeros(128))

    outputs = np.vstack(outputs)

    np.save(f"embeddings/gnn_embeddings_{task}.npy", outputs)

    print(f" GNN embeddings saved for {task}: {outputs.shape}")


if __name__ == "__main__":

    df = pd.read_csv(DATASET)
    df.columns = [c.lower() for c in df.columns]

    smiles_list = df["smiles"].astype(str).tolist()

    for task in TASKS:
        generate(task, smiles_list)