from embeddings.embedding_cache import get_embedding

def test_embedding_generation():
    emb = get_embedding("CCO")
    assert emb.shape[0] > 0