"""SHAP explanations for the trained classification model."""

import pandas as pd
import shap


class ShapExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def explain(self, input_dataframe, feature_names=None):
        """Compute SHAP explanations in the requested feature order."""
        if feature_names is not None:
            missing = [name for name in feature_names if name not in input_dataframe.columns]
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")
            input_dataframe = input_dataframe.loc[:, feature_names]
        return self.explainer(input_dataframe, check_additivity=False)

    def get_class_explanation(self, explanation, predicted_class):
        """Extract SHAP values for the predicted class."""
        class_index = predicted_class - 1
        return shap.Explanation(
            values=explanation.values[0, :, class_index],
            base_values=explanation.base_values[0, class_index],
            data=explanation.data[0],
            feature_names=explanation.feature_names,
        )

    def get_top_features(self, class_explanation, top_n=10):
        """Return the top features ranked by absolute SHAP value."""
        frame = pd.DataFrame(
            {
                "feature": class_explanation.feature_names,
                "importance": class_explanation.values,
            }
        )
        frame["abs_importance"] = frame["importance"].abs()
        return frame.sort_values("abs_importance", ascending=False).head(top_n)
