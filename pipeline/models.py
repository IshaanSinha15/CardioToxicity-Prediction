from typing import Any
import joblib


class ClassifierModel:
    def __init__(self, model_path: str = "saved_models/random_forest_classifier.pkl") -> None:
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self) -> Any:
        return joblib.load(self.model_path)

    def predict(self, X):
        pred = self.model.predict(X)
        probs = self.model.predict_proba(X)
        return pred, probs

    def classes_(self):
        return list(self.model.classes_)
