from typing import Dict, Any
import warnings

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from .models import ClassifierModel


class ClassifierService:
    """
    Wrapper around the trained Random Forest classifier.

    Responsibilities:
    -----------------
    1. Load trained model
    2. Perform prediction
    3. Return probabilities
    4. Return XAI-ready feature information
    """

    def __init__(
        self,
        model_path: str = "saved_models/random_forest_classifier.pkl",
    ) -> None:

        self.model = ClassifierModel(model_path)
        self.classes = self.model.classes_()

        # -----------------------------------------------------
        # Optional probability calibration
        # -----------------------------------------------------
        try:
            base = self.model.model

            df = pd.read_csv("data/datasets/classifier_dataset.csv")

            X_cal = df.drop(columns=["Medication", "Class"])
            y_cal = df["Class"]

            calibrator = CalibratedClassifierCV(
                estimator=base,
                method="sigmoid",
                cv="prefit",
            )

            calibrator.fit(X_cal, y_cal)

            self._predictor = calibrator

        except Exception as exc:

            warnings.warn(
                f"Calibration unavailable, proceeding without it: {exc}"
            )

            self._predictor = self.model.model

    # ==========================================================
    # Prediction
    # ==========================================================

    def predict(
        self,
        features_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Predict cardiotoxicity class.

        Parameters
        ----------
        features_df : pd.DataFrame
            Single-row dataframe containing the exact feature
            vector used during model training.

        Returns
        -------
        Dict[str, Any]
        """

        # -----------------------------
        # Prediction
        # -----------------------------

        pred = self._predictor.predict(features_df)

        probs = self._predictor.predict_proba(features_df)

        predicted_class = int(pred[0])

        probability_list = probs[0].tolist()

        probability_dict = {
            str(cls): float(prob)
            for cls, prob in zip(
                self.classes,
                probability_list,
            )
        }

        # -----------------------------
        # Preserve feature information
        # -----------------------------

        feature_vector = features_df.iloc[0].to_dict()
        # -----------------------------
        # Final Output
        # -----------------------------

        return {

            "predicted_class": predicted_class,

            "probabilities": probability_dict,

            "raw_prediction": predicted_class,

            "raw_probabilities": [
                float(x)
                for x in probability_list
            ],

            "feature_vector": feature_vector,

        }