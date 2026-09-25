"""
test_draw.py

Tests the molecule drawing functionality by generating
an SVG image with highlighted atoms.
"""

from pathlib import Path
from rdkit import Chem

from .draw import generate_svg


def main():
    # Example molecule (Aspirin)
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES!")

    # Example atom indices to highlight
    highlight_atoms = [1, 2, 3]

    # Generate SVG
    svg = generate_svg(mol, highlight_atoms)

    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Output file
    output_file = results_dir / "test_molecule.svg"

    # Save SVG
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(svg)

    print("\nSVG generated successfully!")
    print(f"Saved at:\n{output_file.resolve()}")


if __name__ == "__main__":
    main()