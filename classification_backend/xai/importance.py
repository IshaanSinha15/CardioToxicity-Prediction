"""
importance.py

Assigns importance scores to detected chemical
substructures.

Current implementation:
- Uses SHAP values when available.
- Otherwise assigns descending placeholder scores.
"""

import numpy as np


class ImportanceCalculator:

    def __init__(self):
        pass

    def compute_importance(
        self,
        detected_substructures,
        top_features=None,
    ):
        """
        Parameters
        ----------
        detected_substructures : list
            Output from detect_substructures()

        top_features : pandas.DataFrame, optional
            Output from SHAP explainer.

        Returns
        -------
        list
        """

        results = []

        # ---------- SHAP available ----------
        if top_features is not None and len(top_features) > 0:

            base_score = float(
                np.mean(
                    np.abs(top_features["importance"])
                )
            )

            for item in detected_substructures:

                results.append(
                    {
                        "name": item["name"],
                        "atoms": item["atoms"],
                        "importance": round(base_score, 3),
                    }
                )

            return results

        # ---------- Placeholder ranking ----------
        score = 0.90

        for item in detected_substructures:

            results.append(
                {
                    "name": item["name"],
                    "atoms": item["atoms"],
                    "importance": round(score, 2),
                }
            )

            # Decrease score for the next detected substructure
            score = max(score - 0.10, 0.50)

        return results