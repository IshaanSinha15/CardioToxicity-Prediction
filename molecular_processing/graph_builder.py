import torch
from rdkit import Chem
from torch_geometric.data import Data


def build_graph(smiles: str) -> Data:
    """
    Convert SMILES string into a graph for GNN models.

    Args:
        smiles (str): SMILES representation

    Returns:
        Data: PyTorch Geometric graph object
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Node features (atomic number)
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append([atom.GetAtomicNum()])

    x = torch.tensor(node_features, dtype=torch.float)

    # Edge list
    edge_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_list.append([i, j])
        edge_list.append([j, i])  # bidirectional

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)