"""
final_xai.py

Combines Chemical XAI and Classification XAI
into one final explainability pipeline.
"""

import json
from pathlib import Path

from .chemical_pipeline import ChemicalXAIPipeline
from .run_Xai import XAIPipeline


class FinalXAIPipeline:

    def __init__(self):

        self.chemical = ChemicalXAIPipeline()

        self.classification = XAIPipeline()

    def explain(
        self,
        smiles,
        feature_vector,
    ):

        # ---------------- Chemical XAI ----------------

        chemical_result = self.chemical.explain(
            smiles
        )

        # ---------------- Classification XAI ----------------

        classification_result = self.classification.explain(
            feature_vector
        )

        # ---------------- Merge ----------------

        final_result = {

            "chemical_xai": chemical_result,

            "classification_xai": classification_result,

        }

        # ---------------- Save JSON ----------------

        results_dir = Path(__file__).parent / "results"

        results_dir.mkdir(exist_ok=True)

        json_path = results_dir / "final_xai.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                final_result,
                f,
                indent=4,
                default=str,
            )

        print("\nFinal XAI JSON saved at:")

        print(json_path.resolve())

        return final_result