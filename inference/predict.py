import torch
import numpy as np
import joblib
import xgboost as xgb

from embeddings.chemberta_embedding import ChemBERTaEncoder
from models.gnn_encoder import GNNEncoder
from models.fusion_single_task import FusionSingleTask
from molecular_processing.graph_builder import build_graph
from features.rdkit_features import featurize_smiles

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Convert pIC50 → IC50 (nM)
# Inverse of:
# pIC50 = 9 - log10(IC50)
# -----------------------------
def pic50_to_nm(pic50):
    ic50_nm = 10 ** (9 - pic50)
    return ic50_nm


# -----------------------------
# Load ChemBERTa encoder
# -----------------------------
chemberta = ChemBERTaEncoder(device=device)


# -----------------------------
# Load models
# -----------------------------
models = {}

for task in ["herg", "nav", "cav"]:

    fusion = FusionSingleTask().to(device)

    fusion.load_state_dict(
        torch.load(
            f"models/saved_models/fusion_{task}.pt",
            map_location=device
        )
    )

    fusion.eval()

    gnn = GNNEncoder(
        f"models/saved_models/gnn_{task}.pt",
        device=device
    )

    xgb_model = xgb.Booster()
    xgb_model.load_model(
        f"models/saved_models/xgb_{task}.json"
    )

    meta_model = joblib.load(
        f"models/saved_models/meta_{task}.pkl"
    )

    models[task] = {
        "fusion": fusion,
        "gnn": gnn,
        "xgb": xgb_model,
        "meta": meta_model
    }


# -----------------------------
# Prediction function
# -----------------------------
def predict(smiles):

    with torch.no_grad():

        chem_emb = chemberta.encode([smiles]).to(device)

        graph = build_graph(smiles)

        if graph is None:
            raise ValueError("Invalid SMILES")

        graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
        graph = graph.to(device)

        rdkit_feat = np.array(featurize_smiles(smiles)).reshape(1, -1)

        results = {}

        for task, model_pack in models.items():

            fusion_model = model_pack["fusion"]
            gnn_model = model_pack["gnn"]
            xgb_model = model_pack["xgb"]
            meta_model = model_pack["meta"]

            # GNN embedding
            gnn_emb = gnn_model.encode(graph)

            # Fusion prediction
            fusion_pred = fusion_model(chem_emb, gnn_emb)
            fusion_pred = fusion_pred.cpu().numpy().reshape(-1)

            # XGBoost prediction
            dmatrix = xgb.DMatrix(rdkit_feat)
            xgb_pred = xgb_model.predict(dmatrix)

            # Meta model
            X_meta = np.column_stack([fusion_pred, xgb_pred])

            final_pic50 = float(meta_model.predict(X_meta)[0])

            # convert pIC50 → IC50 nM
            final_nm = pic50_to_nm(final_pic50)

            results[task] = {
                "pIC50": final_pic50,
                "IC50_nM": final_nm
            }

    return results


# -----------------------------
# Example test
# -----------------------------
if __name__ == "__main__":

    smiles = "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1"

    preds = predict(smiles)

    print("\nPredictions\n")

    for task, values in preds.items():

        print(f"{task.upper()}")
        print(f"pIC50   : {values['pIC50']:.4f}")
        print(f"IC50 nM : {values['IC50_nM']:.2e}\n")