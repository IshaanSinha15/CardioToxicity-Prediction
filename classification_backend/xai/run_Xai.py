"""
run_xai.py

Main entry point for the XAI module.

This module orchestrates the complete workflow:
1. Load model
2. Predict class
3. Compute SHAP values
4. Generate plots
5. Generate JSON report
"""

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

    def __init__(self):

        self.model = ModelLoader().load_model()

        self.predictor = Predictor(
            self.model,
            FEATURE_NAMES,
        )

        self.explainer = ShapExplainer(self.model)

        self.visualizer = ShapVisualizer()

        self.report = ReportGenerator()

    def explain(self, feature_vector):

        prediction = self.predictor.predict(
            feature_vector
        )

        explanation = self.explainer.explain(
            prediction["input_dataframe"]
        )

        class_explanation = self.explainer.get_class_explanation(
            explanation,
            prediction["prediction"],
        )

        bar_plot = self.visualizer.bar_plot(
            class_explanation
        )

        waterfall_plot = self.visualizer.waterfall_plot(
            class_explanation
        )

        report_path = self.report.generate_report(
            prediction=prediction["prediction"],
            confidence=prediction["confidence"],
            probabilities=prediction["probabilities"],
            feature_names=FEATURE_NAMES,
            feature_values=feature_vector,
            shap_values=class_explanation.values,
        )

        print(type(prediction["prediction"]))
        print(type(prediction["confidence"]))
        print(type(prediction["probabilities"]))
        print(type(class_explanation.values))

        return {

            "prediction": int(prediction["prediction"]),

            "confidence": float(prediction["confidence"]),

            "probabilities": (
                prediction["probabilities"].tolist()
                if hasattr(prediction["probabilities"], "tolist")
                else prediction["probabilities"]
            ),

            "bar_plot": str(bar_plot),

            "waterfall_plot": str(waterfall_plot),

            "report": str(report_path),

            "shap_values": (
                class_explanation.values.tolist()
                if hasattr(class_explanation.values, "tolist")
                else class_explanation.values
            ),
        }


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
    ]

    pipeline = XAIPipeline()

    result = pipeline.explain(sample)

    print("\n========== XAI RESULT ==========\n")

    print("Prediction :", result["prediction"])

    print("Confidence :", result["confidence"])

    print("Bar Plot :", result["bar_plot"])

    print("Waterfall :", result["waterfall_plot"])

    print("Report :", result["report"])