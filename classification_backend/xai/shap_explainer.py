"""
shap_explainer.py

Computes SHAP explanations for the trained
Random Forest classifier.
"""

import shap


class ShapExplainer:

    def __init__(self, model):
        """
        Parameters
        ----------
        model : RandomForestClassifier
        """
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def explain(self, input_dataframe):
        """
        Compute SHAP values.

        Parameters
        ----------
        input_dataframe : pandas.DataFrame

        Returns
        -------
        shap.Explanation
        """

        explanation = self.explainer(input_dataframe)

        return explanation

    def get_class_explanation(
        self,
        explanation,
        predicted_class,
    ):
        """
        Extract SHAP values for the predicted class.

        Parameters
        ----------
        explanation : shap.Explanation

        predicted_class : int
            Predicted class label (1-4)

        Returns
        -------
        shap.Explanation
        """

        class_index = predicted_class - 1

        single_class = shap.Explanation(
            values=explanation.values[0, :, class_index],
            base_values=explanation.base_values[0, class_index],
            data=explanation.data[0],
            feature_names=explanation.feature_names,
        )

        return single_class