"""Dose-response curve generation and plotting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import logging

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .hill_equation import HillEquation, hill_block_percentage

LOGGER = logging.getLogger(__name__)
CHANNEL_COLORS = {"herg": "#d1495b", "nav": "#3a86ff", "cav": "#2a9d8f"}


class DoseResponseCurveError(ValueError):
    """Raised when curve inputs are invalid."""


@dataclass(frozen=True)
class DoseResponseCurve:
    channel_name: str
    ic50_nm: float
    hill_coefficient: float = 1.0
    emax: float = 100.0

    def equation(self) -> HillEquation:
        return HillEquation(self.ic50_nm, self.hill_coefficient, self.emax, channel=self.channel_name)

    def calculate(self, concentrations_nm: Sequence[float] | np.ndarray) -> np.ndarray:
        concentrations = np.asarray(concentrations_nm, dtype=float).reshape(-1)
        return np.asarray(self.equation().block_series(concentrations), dtype=float)


def generate_curve(
    ic50_nm: float,
    hill_coefficient: float = 1.0,
    concentration_range_nm: tuple[float, float] | None = None,
    num_points: int = 200,
    emax: float = 100.0,
) -> pd.DataFrame:
    if num_points < 2:
        raise DoseResponseCurveError("num_points must be at least 2")
    if concentration_range_nm is None:
        low = max(ic50_nm / 100.0, 1e-12)
        high = ic50_nm * 100.0
    else:
        low, high = concentration_range_nm
    if low <= 0 or high <= 0 or high <= low:
        raise DoseResponseCurveError("Invalid concentration range")

    concentrations = np.logspace(np.log10(low), np.log10(high), num_points)
    blocks = hill_block_percentage(concentrations, ic50_nm, hill_coefficient, emax)
    return pd.DataFrame({"concentration_nm": concentrations, "block_pct": blocks})


def _build_curve_map(curves: Mapping[str, DoseResponseCurve] | Mapping[str, Mapping[str, float]]) -> dict[str, DoseResponseCurve]:
    normalized: dict[str, DoseResponseCurve] = {}
    for key, value in curves.items():
        if isinstance(value, DoseResponseCurve):
            normalized[key.lower()] = value
        else:
            normalized[key.lower()] = DoseResponseCurve(
                channel_name=key,
                ic50_nm=float(value["ic50_nm"]),
                hill_coefficient=float(value.get("hill_coefficient", 1.0)),
                emax=float(value.get("emax", 100.0)),
            )
    return normalized


def _apply_axis_style(ax: plt.Axes, reference_concentration_nm: float | None = None) -> None:
    ax.set_xscale("log")
    ax.set_xlabel("Concentration (nM)")
    ax.set_ylabel("Channel Block (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, which="both", alpha=0.25)
    if reference_concentration_nm is not None:
        ax.axvline(reference_concentration_nm, color="#555555", linestyle="--", linewidth=1.2, label="Reference concentration")


def plot_dose_response_curves(
    curves: Mapping[str, DoseResponseCurve] | Mapping[str, Mapping[str, float]],
    reference_concentration_nm: float | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    curve_map = _build_curve_map(curves)
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)

    for key, curve in curve_map.items():
        frame = generate_curve(curve.ic50_nm, curve.hill_coefficient, emax=curve.emax)
        ax.plot(frame["concentration_nm"], frame["block_pct"], label=curve.channel_name, color=CHANNEL_COLORS.get(key, "#444444"), linewidth=2.0)

    _apply_axis_style(ax, reference_concentration_nm)
    ax.legend(frameon=False)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), bbox_inches="tight")
    return fig, ax


def plot_channel_comparison(
    block_table: pd.DataFrame,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    required = {"concentration", "herg_block", "nav_block", "cav_block"}
    missing = required.difference(block_table.columns)
    if missing:
        raise DoseResponseCurveError(f"block_table is missing required columns: {', '.join(sorted(missing))}")

    frame = block_table.sort_values("concentration").copy()
    x = np.arange(len(frame))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    ax.bar(x - width, frame["herg_block"], width=width, label="hERG", color=CHANNEL_COLORS["herg"])
    ax.bar(x, frame["nav_block"], width=width, label="Nav1.5", color=CHANNEL_COLORS["nav"])
    ax.bar(x + width, frame["cav_block"], width=width, label="Cav1.2", color=CHANNEL_COLORS["cav"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:.3g}" for value in frame["concentration"]], rotation=30, ha="right")
    ax.set_xlabel("Concentration (nM)")
    ax.set_ylabel("Channel Block (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), bbox_inches="tight")
    return fig, ax


def plot_therapeutic_vs_toxic_dose(
    reference_concentration_nm: float,
    toxic_multiple: float = 10.0,
) -> tuple[plt.Figure, plt.Axes]:
    if toxic_multiple <= 1:
        raise DoseResponseCurveError("toxic_multiple must be greater than 1")
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
    ax.axvspan(reference_concentration_nm * 0.5, reference_concentration_nm * 1.5, color="#d9f0ff", alpha=0.7, label="Therapeutic window")
    ax.axvspan(reference_concentration_nm * 5.0, reference_concentration_nm * toxic_multiple, color="#ffe0d6", alpha=0.5, label="Toxic window")
    ax.set_xscale("log")
    ax.set_xlabel("Concentration (nM)")
    ax.set_ylabel("Context")
    ax.set_yticks([])
    ax.set_title("Therapeutic vs Toxic Dose Context")
    ax.grid(True, which="both", axis="x", alpha=0.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_safety_margin_bars(
    safety_margin_frame: pd.DataFrame,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    required = {"channel", "margin", "risk_class"}
    missing = required.difference(safety_margin_frame.columns)
    if missing:
        raise DoseResponseCurveError(f"safety_margin_frame is missing required columns: {', '.join(sorted(missing))}")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    colors = ["#2a9d8f" if risk == "Safe" else "#f4a261" if risk == "Moderate" else "#e76f51" for risk in safety_margin_frame["risk_class"]]
    ax.bar(safety_margin_frame["channel"], safety_margin_frame["margin"], color=colors)
    ax.axhline(30.0, color="#2a9d8f", linestyle="--", linewidth=1, label="Safe threshold")
    ax.axhline(10.0, color="#e76f51", linestyle="--", linewidth=1, label="Moderate threshold")
    ax.set_ylabel("Safety margin (IC50 / concentration)")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), bbox_inches="tight")
    return fig, ax
