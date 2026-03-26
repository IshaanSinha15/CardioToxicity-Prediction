import pandas as pd
import numpy as np
from tqdm import tqdm

from prediction_backend.embeddings.chemberta_embedding import ChemBERTaEncoder

# Load dataset
df = pd.read_csv("/final_combined.csv")

encoder = ChemBERTaEncoder(device="cpu")  # CPU is fine

embeddings = []

print("Generating ChemBERTa embeddings...")

for sm in tqdm(df["smiles"]):
    emb = encoder.encode([sm])[0]  # shape (768,)
    embeddings.append(emb)

embeddings = np.array(embeddings)

# Save
np.save("data/chemberta_embeddings.npy", embeddings)

print("Saved embeddings:", embeddings.shape)