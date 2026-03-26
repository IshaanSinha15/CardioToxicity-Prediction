import numpy as np

from molecular_processing.fingerprint_generator import generate_fingerprint
from embeddings.embedding_cache import get_embedding


def build_features(smiles_list):

    if not isinstance(smiles_list, (list, tuple)):
        raise ValueError("Input must be a list of SMILES")

    features = []

    for s in smiles_list:

        try:
            fp = generate_fingerprint(s)        # (2048,)
            emb = get_embedding(s)              # (~768,)

            # Ensure both are 1D
            fp = np.asarray(fp).flatten()
            emb = np.asarray(emb).flatten()

            # Combine features
            feat = np.concatenate([fp, emb]).astype(np.float32)

            features.append(feat)

        except Exception as e:
            raise ValueError(f"Error processing SMILES '{s}': {e}")

    features = np.array(features, dtype=np.float32)

    return features