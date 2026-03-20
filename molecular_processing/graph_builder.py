from rdkit import Chem
import torch
from torch_geometric.data import Data


def atom_features(atom):

    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
        atom.GetImplicitValence()
    ]


def bond_features(bond):

    return [
        int(bond.GetBondTypeAsDouble()),
        int(bond.GetIsAromatic()),
        int(bond.GetIsConjugated())
    ]


def build_graph(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES")

    # ----- Node features -----
    atoms = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atoms, dtype=torch.float)

    # ----- Edge features -----
    edges = []
    edge_attrs = []

    for bond in mol.GetBonds():

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        feat = bond_features(bond)

        edges.append([i, j])
        edges.append([j, i])

        edge_attrs.append(feat)
        edge_attrs.append(feat)

    if len(edges) == 0:

        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)

    else:

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    return data