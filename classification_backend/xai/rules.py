"""
rules.py

Contains SMARTS patterns for toxic chemical
substructures associated with cardiotoxicity.
"""

from rdkit import Chem

SUBSTRUCTURES = {

    "Aromatic Ring":
        Chem.MolFromSmarts("c1ccccc1"),

    "Phenol":
        Chem.MolFromSmarts("c1ccc(cc1)O"),

    "Tertiary Amine":
        Chem.MolFromSmarts("[NX3]([#6])([#6])[#6]"),

    "Piperidine":
        Chem.MolFromSmarts("N1CCCCC1"),

    "Piperazine":
        Chem.MolFromSmarts("N1CCNCC1"),

    "Imidazole":
        Chem.MolFromSmarts("c1ncc[nH]1"),

    "Quinoline":
        Chem.MolFromSmarts("c1ccc2ncccc2c1"),
}