from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()


def compute_embedding(smiles: str):

    tokens = tokenizer(
        smiles,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():

        outputs = model(**tokens)
        embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding.numpy()[0]