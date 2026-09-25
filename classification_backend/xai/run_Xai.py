"""
run_xai.py

Main entry point for the XAI module.

Workflow
--------
1. Load trained model
2. Predict cardiotoxicity class
3. Compute SHAP values
4. Extract top SHAP features
5. Generate SHAP visualizations
6. Generate Classification XAI explanation
7. Generate report
8. Save Classification XAI JSON
"""

import json
from pathlib import Path

from classification_backend.xai.classification_xai import ClassificationXAI
from classification_backend.xai.model_loader import ModelLoader
from classification_backend.xai.predictor import Predictor
from classification_backend.xai.shap_explainer import ShapExplainer
from classification_backend.xai.visualization import ShapVisualizer
from classification_backend.xai.report_generator import ReportGenerator


FEATURE_NAMES = [
    "RMP",
    "Peak",
    "APD50",
    "APD90",
    "Triangulation",
    "APA",
    "Block_IKr",
    "Block_INa",
    "Block_INaL",
    "Block_ICaL",
    "Block_IKs",
    "Block_IK1",
    "Block_Ito",
    "IC50_IKr",
    "IC50_INa",
    "IC50_ICaL",
]


class XAIPipeline:

    def __init__(self, output_dir=None):

        self.results_dir = Path(output_dir) if output_dir else Path(__file__).parent / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.model = ModelLoader().load_model()

        self.predictor = Predictor(
            self.model,
            FEATURE_NAMES,
        )

        self.explainer = ShapExplainer(self.model)

        self.visualizer = ShapVisualizer(self.results_dir)

        self.report = ReportGenerator(self.results_dir)

        self.classification_xai = ClassificationXAI()

    def explain(self, feature_vector):

        # ---------------- Prediction ----------------

        prediction = self.predictor.predict(
            feature_vector
        )

        # ---------------- SHAP ----------------

        explanation = self.explainer.explain(
            prediction["input_dataframe"]
        )

        class_explanation = self.explainer.get_class_explanation(
            explanation,
            prediction["prediction"],
        )

        # ---------------- Top Features ----------------

        top_features = self.explainer.get_top_features(
            class_explanation,
            top_n=10,
        )

        # ---------------- Plots ----------------

        bar_plot = self.visualizer.bar_plot(
            class_explanation
        )

        waterfall_plot = self.visualizer.waterfall_plot(
            class_explanation
        )

        # ---------------- Report ----------------

        report_path = self.report.generate_report(
            prediction=prediction["prediction"],
            confidence=prediction["confidence"],
            probabilities=prediction["probabilities"],
            feature_names=FEATURE_NAMES,
            feature_values=feature_vector,
            shap_values=class_explanation.values,
        )

        # ---------------- Classification XAI ----------------

        classification_result = self.classification_xai.explain(
            prediction=prediction["prediction"],
            confidence=prediction["confidence"],
            top_features=top_features,
        )
        classification_result["top_features"] = top_features.to_dict(
            orient="records"
        )

        # ---------------- Final Result ----------------

        result = {

            "prediction": int(prediction["prediction"]),

            "confidence": float(prediction["confidence"]),

            "probabilities": (
                prediction["probabilities"].tolist()
                if hasattr(prediction["probabilities"], "tolist")
                else prediction["probabilities"]
            ),

            "top_features": top_features.to_dict(
                orient="records"
            ),

            "classification_xai": classification_result,

            "bar_plot": str(bar_plot),

            "waterfall_plot": str(waterfall_plot),

            "report": str(report_path),

            "shap_values": (
                class_explanation.values.tolist()
                if hasattr(class_explanation.values, "tolist")
                else class_explanation.values
            ),
        }

        # ---------------- Save JSON ----------------

        json_path = self.results_dir / "classification_xai.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                result,
                f,
                indent=4,
                default=str,
            )

        print("\nClassification XAI JSON saved at:")
        print(json_path.resolve())

        return result


if __name__ == "__main__":

    sample = [
        -90.74,
        31.99,
        231.90,
        278.40,
        46.50,
        122.70,
        26.70,
        2.60,
        0.00,
        2.10,
        0.00,
        0.00,
        4.50,
        1000.0,
        5000.0,
        8000.0,
    ]

    pipeline = XAIPipeline()

    result = pipeline.explain(sample)

    print("\n========== XAI RESULT ==========\n")

    print("Prediction :", result["prediction"])
    print("Confidence :", result["confidence"])

    print("\nTop Features")
    print(result["top_features"])

    print("\nClassification XAI")
    print(result["classification_xai"])

    print("\nBar Plot :", result["bar_plot"])
    print("Waterfall :", result["waterfall_plot"])
    print("Report :", result["report"])