from classification_backend.xai.model_loader import ModelLoader
from classification_backend.xai.predictor import Predictor


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

    loader = ModelLoader()

    model = loader.load_model()

    predictor = Predictor(
        model,
        FEATURE_NAMES,
    )

    result = predictor.predict(SAMPLE)

    print("\nPrediction")
    print(result["prediction"])

    print("\nConfidence")
    print(result["confidence"])

    print("\nProbabilities")
    print(result["probabilities"])


if __name__ == "__main__":
    main()