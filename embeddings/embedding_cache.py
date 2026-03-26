import os
import pickle
from embeddings.chemberta_embedding import compute_embedding

CACHE_PATH = "cache/chemberta_cache.pkl"


def load_cache():

    if os.path.exists(CACHE_PATH):

        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    return {}


def save_cache(cache):

    os.makedirs("cache", exist_ok=True)

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


cache = load_cache()


def get_embedding(smiles):

    if smiles in cache:
        return cache[smiles]

    emb = compute_embedding(smiles)
    cache[smiles] = emb
    save_cache(cache)
    return emb