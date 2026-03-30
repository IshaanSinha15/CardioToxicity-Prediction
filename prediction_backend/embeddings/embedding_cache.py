import pickle
from pathlib import Path
from prediction_backend.embeddings.chemberta_embedding import compute_embedding

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "cache"
CACHE_PATH = CACHE_DIR / "chemberta_cache.pkl"


def load_cache():

    if CACHE_PATH.exists():

        with CACHE_PATH.open("rb") as f:
            return pickle.load(f)

    return {}


def save_cache(cache):

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with CACHE_PATH.open("wb") as f:
        pickle.dump(cache, f)


cache = load_cache()


def get_embedding(smiles):

    if smiles in cache:
        return cache[smiles]

    emb = compute_embedding(smiles)
    cache[smiles] = emb
    save_cache(cache)
    return emb