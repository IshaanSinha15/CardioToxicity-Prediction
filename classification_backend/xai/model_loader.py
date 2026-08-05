"""
model_loader.py

Loads the trained Random Forest classifier.

This module ONLY loads the model.
"""

from pathlib import Path
import joblib


class ModelLoader:
    def __init__(self):
        repo_root = Path(__file__).resolve().parents[2]

        self.model_path = (
            repo_root
            / "saved_models"
            / "random_forest_classifier.pkl"
        )

    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found:\n{self.model_path}"
            )

        model = joblib.load(self.model_path)

        print("\nModel loaded successfully.")
        print(f"Path: {self.model_path}")

        return model