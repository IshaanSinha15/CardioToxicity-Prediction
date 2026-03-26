import torch
import pytest

from prediction_backend.embeddings.chemberta_embedding import ChemBERTaEncoder


# -------------------------
# Test 1: Basic functionality
# -------------------------
def test_compute_embeddings_shape():

    encoder = ChemBERTaEncoder(device="cpu")

    smiles = ["CCO", "CCN"]

    embeddings = encoder.encode(smiles)

    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 768


# -------------------------
# Test 2: Single SMILES
# -------------------------
def test_single_smiles():

    encoder = ChemBERTaEncoder(device="cpu")

    smiles = ["CCO"]

    embeddings = encoder.encode(smiles)

    assert embeddings.shape == (1, 768)


# -------------------------
# Test 3: Consistency (same input → same output)
# -------------------------
def test_deterministic_output():

    encoder = ChemBERTaEncoder(device="cpu")

    smiles = ["CCO"]

    emb1 = encoder.encode(smiles)
    emb2 = encoder.encode(smiles)

    assert torch.allclose(emb1, emb2, atol=1e-6)


# -------------------------
# Test 4: Invalid SMILES handling
# -------------------------
def test_invalid_smiles():

    encoder = ChemBERTaEncoder(device="cpu")

    smiles = ["INVALID"]

    # Transformer should still return embedding (not crash)
    embeddings = encoder.encode(smiles)

    assert embeddings.shape == (1, 768)


# -------------------------
# Test 5: Batch size variation
# -------------------------
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_batch_sizes(batch_size):

    encoder = ChemBERTaEncoder(device="cpu")

    smiles = ["CCO"] * batch_size

    embeddings = encoder.encode(smiles)

    assert embeddings.shape == (batch_size, 768)