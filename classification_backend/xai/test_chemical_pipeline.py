"""
test_chemical_pipeline.py

Tests the complete Chemical XAI pipeline.
"""

import json
from pathlib import Path

from .chemical_pipeline import ChemicalXAIPipeline


def main():

    # Example molecule (Aspirin)
    sample_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

    pipeline = ChemicalXAIPipeline()

    result = pipeline.explain(sample_smiles)

    print("\n========== CHEMICAL XAI ==========\n")

    print(json.dumps(result, indent=4, default=str))

    # Create results folder
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Save JSON
    json_path = results_dir / "chemical_xai.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, default=str)

    print("\nJSON saved at:")
    print(json_path.resolve())

    print("\nSVG saved at:")
    print(result["svg_path"])


if __name__ == "__main__":
    main()