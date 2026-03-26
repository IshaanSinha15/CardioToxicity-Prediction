import torch
import numpy as np

from prediction_backend.models.fusion_single_task import FusionSingleTask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TASKS = ["herg", "nav", "cav"]


def generate(task):

    chemberta = np.load("data/chemberta_embeddings.npy")
    gnn = np.load(f"embeddings/gnn_embeddings_{task}.npy")

    chemberta = torch.tensor(chemberta, dtype=torch.float32).to(DEVICE)
    gnn = torch.tensor(gnn, dtype=torch.float32).to(DEVICE)

    model = FusionSingleTask()

    model.load_state_dict(
        torch.load(
            f"models/saved_models/fusion_{task}.pt",
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)
    model.eval()

    with torch.no_grad():

        preds = model(chemberta, gnn)

    preds = preds.cpu().numpy()

    np.save(f"embeddings/fusion_{task}_pred.npy", preds)

    print(f"Fusion predictions saved for {task}: {preds.shape}")


if __name__ == "__main__":

    for task in TASKS:
        generate(task)