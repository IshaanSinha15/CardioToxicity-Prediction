"""
chemical_xai.py

Chemical Explainability (XAI)

Pipeline:

SMILES
   ↓
RDKit Molecule
   ↓
Detect Toxic Substructures
   ↓
Assign Importance
   ↓
Highlight Molecule
   ↓
Generate Explanation
"""

from rdkit import Chem

from .rules import SUBSTRUCTURES
from .importance import ImportanceCalculator
from .draw import generate_svg
from .explanation import generate_explanation



def load_molecule(smiles: str):
    """
    Convert SMILES into an RDKit molecule.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES")

    return mol


def detect_substructures(mol):
    """
    Detect toxic SMARTS patterns.
    """

    detected = []

    for name, pattern in SUBSTRUCTURES.items():

        matches = mol.GetSubstructMatches(pattern)

        if matches:

            detected.append(
                {
                    "name": name,
                    "atoms": matches,
                }
            )

    return detected


def chemical_xai(smiles):
    """
    Run Chemical XAI.

    Parameters
    ----------
    smiles : str

    Returns
    -------
    dict
    """

    # Convert SMILES
    mol = load_molecule(smiles)

    # Detect toxic fragments
    detected = detect_substructures(mol)

    # Calculate importance
    calculator = ImportanceCalculator()

    importance = calculator.compute_importance(
        detected_substructures=detected,
        top_features=None,      # SHAP will be added later
    )

    merged = []

    highlight_atoms = []

    for item in importance:

        merged.append(
            {
                "name": item["name"],
                "importance": item["importance"],
                "atoms": item["atoms"],
            }
        )

        for match in item["atoms"]:
            highlight_atoms.extend(match)

    # Remove duplicates
    highlight_atoms = sorted(set(highlight_atoms))

    # Draw molecule
    svg = generate_svg(
        mol,
        highlight_atoms,
    )

    # Generate explanation
    explanation = generate_explanation(
        merged
    )
    

    return {
        "molecule_svg": svg,
        "substructures": merged,
        "explanation": explanation,
    }