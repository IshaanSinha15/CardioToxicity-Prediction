"""
visualization.py

Generates SHAP visualizations.

This module ONLY creates plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import shap


class ShapVisualizer:

    def __init__(self, output_dir=None):

        if output_dir is None:

            output_dir = (
                Path(__file__).parent
                / "results"
            )

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def bar_plot(self, explanation):

        plt.figure(figsize=(8,6))

        shap.plots.bar(
            explanation,
            show=False,
        )

        path = self.output_dir / "bar_plot.png"

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        return path

    def waterfall_plot(self, explanation):

        plt.figure(figsize=(8,6))

        shap.plots.waterfall(
            explanation,
            show=False,
        )

        path = self.output_dir / "waterfall_plot.png"

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        return path