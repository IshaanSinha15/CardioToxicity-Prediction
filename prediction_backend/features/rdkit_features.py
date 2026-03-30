import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import RDLogger

# 🔥 Disable warnings
RDLogger.DisableLog('rdApp.*')

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = REPO_ROOT / "data" / "features"


def featurize_smiles(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Basic descriptors
    desc = [
        Descriptors.MolWt(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
    ]

    # Morgan fingerprint (fixed version)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp = np.array(list(fp))

    return np.concatenate([desc, fp])


def generate_features(csv_path, target_name, save_prefix):

    df = pd.read_csv(csv_path)

    X = []
    y = []

    for _, row in df.iterrows():

        sm = row["smiles"]
        label = row[target_name]

        features = featurize_smiles(sm)

        if features is not None:

            # 🔥 LOG TRANSFORM (CRITICAL FIX)
            label = np.log10(label + 1)

            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Save
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES_DIR / f"{save_prefix}_X.npy", X)
    np.save(FEATURES_DIR / f"{save_prefix}_y.npy", y)

    print(f"✅ Saved {save_prefix}: {X.shape}")