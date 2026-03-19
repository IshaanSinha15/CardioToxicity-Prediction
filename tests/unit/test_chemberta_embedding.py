import numpy as np
from embeddings.chemberta_embedding import compute_embedding


def test_embedding_generation():

    smiles = "CCO"

    emb = compute_embedding(smiles)

    # embedding should not be None
    assert emb is not None

    # embedding should be numpy array
    assert isinstance(emb, np.ndarray)

    # embedding should be 1‑D vector
    assert emb.ndim == 1

    # embedding size should be greater than 0
    assert emb.shape[0] > 0