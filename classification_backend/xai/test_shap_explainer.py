"""
test_shap_explainer.py

Tests the SHAP explainer and prints the
top important features.
"""

from classification_backend.xai.model_loader import ModelLoader
from classification_backend.xai.predictor import Predictor
from classification_backend.xai.shap_explainer import ShapExplainer


# Updated feature list (same as run_xai.py)
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


# Make sure the sample has the same number of values
SAMPLE = [
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
    1000.0,   # Example IC50_IKr
    5000.0,   # Example IC50_INa
    8000.0,   # Example IC50_ICaL
]


def main():

    loader = ModelLoader()
    model = loader.load_model()

    predictor = Predictor(model, FEATURE_NAMES)

    prediction = predictor.predict(SAMPLE)

    explainer = ShapExplainer(model)

    explanation = explainer.explain(
        prediction["input_dataframe"]
    )

    class_explanation = explainer.get_class_explanation(
        explanation,
        prediction["prediction"],
    )

    top_features = explainer.get_top_features(
        class_explanation,
        top_n=10,
    )

    print("\nPrediction:", prediction["prediction"])
    print("Confidence:", prediction["confidence"])

    print("\n========== TOP SHAP FEATURES ==========\n")
    print(top_features)


if __name__ == "__main__":
    main()