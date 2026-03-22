import torch
import numpy as np

from embeddings.chemberta_embedding import ChemBERTaEncoder
from models.gnn_encoder import GNNEncoder
from models.fusion_single_task import FusionSingleTask
from molecular_processing.graph_builder import build_graph


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chemberta = ChemBERTaEncoder(device=device)

# Load models
models = {
    "herg": (
        FusionSingleTask().to(device),
        GNNEncoder("models/saved_models/gnn_herg.pt", device=device),
        "models/saved_models/fusion_herg.pt"
    ),
    "nav": (
        FusionSingleTask().to(device),
        GNNEncoder("models/saved_models/gnn_nav.pt", device=device),
        "models/saved_models/fusion_nav.pt"
    ),
    "cav": (
        FusionSingleTask().to(device),
        GNNEncoder("models/saved_models/gnn_cav.pt", device=device),
        "models/saved_models/fusion_cav.pt"
    )
}

for key in models:
    models[key][0].load_state_dict(torch.load(models[key][2]))
    models[key][0].eval()


def predict(smiles):

    chem_emb = chemberta.encode([smiles]).to(device)
    graph = build_graph(smiles)

    outputs = {}

    for key, (model, gnn, _) in models.items():

        gnn_emb = gnn.encode(graph)

        pred = model(chem_emb, gnn_emb)

        outputs[key] = float(pred.item())

    return outputs


# Example
print(predict("CCO"))