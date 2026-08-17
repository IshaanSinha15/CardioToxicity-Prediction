from classification_backend.xai.model_loader import ModelLoader
from classification_backend.xai.predictor import Predictor
from classification_backend.xai.shap_explainer import ShapExplainer
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
]

SAMPLE = [
    -90.74,
    31.99,
    231.9,
    278.4,
    46.5,
    122.7,
    26.7,
    2.6,
    0.0,
    2.1,
    0.0,
    0.0,
    4.5,
]


def main():

    model = ModelLoader().load_model()

    predictor = Predictor(model, FEATURE_NAMES)

    prediction = predictor.predict(SAMPLE)

    explainer = ShapExplainer(model)

    explanation = explainer.explain(prediction["input_dataframe"])

    class_explanation = explainer.get_class_explanation(
        explanation,
        prediction["prediction"],
    )

    report = ReportGenerator()

    report_path = report.generate_report(
        prediction=prediction["prediction"],
        confidence=prediction["confidence"],
        probabilities=prediction["probabilities"],
        feature_names=FEATURE_NAMES,
        feature_values=SAMPLE,
        shap_values=class_explanation.values,
    )

    print("\nReport Generated:")
    print(report_path)


if __name__ == "__main__":
    main()