"""
predictor.py

Runs inference using the trained Random Forest classifier.
"""

import pandas as pd


class Predictor:

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def predict(self, feature_vector):
        """
        Predict cardiotoxicity class.

        Parameters
        ----------
        feature_vector : list

        Returns
        -------
        dict
        """

        X = pd.DataFrame(
            [feature_vector],
            columns=self.feature_names,
        )

        prediction = int(self.model.predict(X)[0])

        probabilities = self.model.predict_proba(X)[0]

        confidence = float(probabilities.max())

        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "confidence": confidence,
            "input_dataframe": X,
        }