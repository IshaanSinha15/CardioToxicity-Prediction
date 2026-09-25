"""
chemical_pipeline.py

Entry point for the Chemical XAI pipeline.

Workflow:
1. Receive a SMILES string
2. Run Chemical XAI
3. Save the generated SVG
4. Return the complete explanation
"""

from pathlib import Path

from .chemical_xai import chemical_xai


class ChemicalXAIPipeline:
    """
    Entry point for Chemical XAI.
    """

    def __init__(self, output_dir=None):
        self.results_dir = Path(output_dir) if output_dir else Path(__file__).parent / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def explain(self, smiles: str):
        """
        Run the complete Chemical XAI pipeline.

        Parameters
        ----------
        smiles : str

        Returns
        -------
        dict
        """

        result = chemical_xai(smiles)

        # Save SVG
        svg_path = self.results_dir / "chemical_xai.svg"

        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(result["molecule_svg"])

        # Add path to returned dictionary
        result["svg_path"] = str(svg_path.resolve())

        return result