from typing import Dict, Any
import pandas as pd

from .models import ClassifierModel
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import warnings


class ClassifierService:
    def __init__(self, model_path: str = "saved_models/random_forest_classifier.pkl") -> None:
        self.model = ClassifierModel(model_path)
        self.classes = self.model.classes_()
        # Attempt to wrap with a calibrated classifier using local dataset if available
        try:
            base = self.model.model
            # Load calibration data from the training CSV if present
            df = pd.read_csv("data/datasets/classifier_dataset.csv")
            X_cal = df.drop(columns=["Medication", "Class"])
            y_cal = df["Class"]
            # Use cv='prefit' to calibrate the already-trained estimator
            calibrator = CalibratedClassifierCV(estimator=base, method="sigmoid", cv="prefit")
            calibrator.fit(X_cal, y_cal)
            self._predictor = calibrator
        except Exception as exc:
            warnings.warn(f"Calibration unavailable, proceeding without it: {exc}")
            self._predictor = self.model.model

    def predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        # Use calibrated predictor if available
        pred = self._predictor.predict(features_df)
        probs = self._predictor.predict_proba(features_df)
        # pred is array-like, probs is array-like shape (n_samples, n_classes)
        predicted_class = int(pred[0])
        prob_list = probs[0].tolist()
        prob_dict = {str(c): float(p) for c, p in zip(self.classes, prob_list)}

        # feature vector as dict (preserve column names)
        feature_vector = features_df.iloc[0].to_dict()

        return {
            "predicted_class": predicted_class,
            "probabilities": prob_dict,
            "raw_prediction": int(pred[0]),
            "raw_probabilities": [float(x) for x in prob_list],
            "feature_vector": feature_vector,
        }
