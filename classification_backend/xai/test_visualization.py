from classification_backend.xai.model_loader import ModelLoader
from classification_backend.xai.predictor import Predictor
from classification_backend.xai.shap_explainer import ShapExplainer
from classification_backend.xai.visualization import ShapVisualizer


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

    explanation = explainer.explain(
        prediction["input_dataframe"]
    )

    class_explanation = explainer.get_class_explanation(
        explanation,
        prediction["prediction"],
    )

    visualizer = ShapVisualizer()

    bar = visualizer.bar_plot(class_explanation)

    waterfall = visualizer.waterfall_plot(class_explanation)

    print("\nGenerated Files")

    print(bar)

    print(waterfall)


if __name__ == "__main__":
    main()