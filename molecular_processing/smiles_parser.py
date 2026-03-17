from rdkit import Chem


class SmilesParserError(Exception):
    pass


def validate_smiles(smiles: str):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise SmilesParserError(f"Invalid SMILES: {smiles}")
    return mol


def canonicalize_smiles(smiles: str):
    mol = validate_smiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True)