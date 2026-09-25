"""
classification_xai.py

Generates explanations for the final
cardiotoxicity classification.
"""

class ClassificationXAI:

    def __init__(self):
        pass

    def explain(
        self,
        prediction,
        confidence,
        top_features,
    ):

        return {

            "prediction": prediction,

            "confidence": confidence,

            "top_features": top_features,

            "message":
                self.generate_message(
                    prediction,
                    confidence,
                    top_features,
                ),
        }

    def generate_message(
        self,
        prediction,
        confidence,
        top_features,
    ):

        if prediction == 0:
            risk = "Low"

        elif prediction == 1:
            risk = "Medium"

        else:
            risk = "High"

        features = []

        for _, row in top_features.iterrows():

            features.append(row["feature"])

        return (
            f"The compound was classified as "
            f"{risk} risk with "
            f"{confidence:.2f} confidence. "
            f"The most influential features were "
            + ", ".join(features[:5])
            + "."
        )