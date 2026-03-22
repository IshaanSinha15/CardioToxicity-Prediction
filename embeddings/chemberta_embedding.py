from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"


class ChemBERTaEncoder:

    def __init__(self, device=None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)

        self.model.eval()

    def encode(self, smiles_list):

        tokens = self.tokenizer(
            smiles_list,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**tokens)

            # CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings  # [batch_size, 768]


# =========================
# ✅ REQUIRED FOR TESTS (IMPORTANT)
# =========================

_encoder = None

def compute_embedding(smiles: str):
    """
    Backward-compatible function for tests.
    Returns embedding for a single SMILES.
    """

    global _encoder

    if _encoder is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _encoder = ChemBERTaEncoder(device=device)

    emb = _encoder.encode([smiles])

    return emb[0].cpu().numpy()