from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np


def generate_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Generate a Morgan fingerprint from a SMILES string.

    Args:
        smiles (str): Input SMILES string
        radius (int): Radius for Morgan fingerprint
        n_bits (int): Length of fingerprint vector

    Returns:
        np.ndarray: Binary fingerprint array

    Raises:
        ValueError: If SMILES is invalid
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits
    )

    arr = np.zeros((n_bits,), dtype=np.int8)

    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr