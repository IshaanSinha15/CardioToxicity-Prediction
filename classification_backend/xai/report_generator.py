"""
report_generator.py

Creates JSON reports containing prediction results
and SHAP explanations.
"""

from pathlib import Path
import json


class ReportGenerator:

    def __init__(self, output_dir=None):

        if output_dir is None:
            output_dir = Path(__file__).parent / "results"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, xai_result):
        """Return a JSON-safe report for a chemical-XAI result."""
        return {
            "summary": {
                "num_substructures": len(xai_result.get("substructures", [])),
            },
            "substructures": xai_result.get("substructures", []),
            "explanation": xai_result.get("explanation", []),
            "molecule_svg": xai_result.get("molecule_svg", ""),
        }

    def generate_report(
        self,
        prediction,
        confidence,
        probabilities,
        feature_names,
        feature_values,
        shap_values,
    ):

        report = {
            "prediction": int(prediction),
            "confidence": float(confidence),
            "probabilities": {
                f"Class_{i+1}": float(probabilities[i])
                for i in range(len(probabilities))
            },
            "features": {
                feature_names[i]: float(feature_values[i])
                for i in range(len(feature_names))
            },
            "shap_values": {
                feature_names[i]: float(shap_values[i])
                for i in range(len(feature_names))
            },
        }

        output_file = self.output_dir / "xai_report.json"

        with open(output_file, "w") as f:
            json.dump(report, f, indent=4)

        return output_file