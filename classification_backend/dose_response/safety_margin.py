"""Safety-margin calculations for IC50-to-concentration comparisons."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping

import numpy as np
import pandas as pd


class SafetyMarginError(ValueError):
    """Raised when safety-margin inputs are invalid."""


def calculate_safety_margin(ic50_nm: float, concentration_nm: float) -> float:
    if not np.isfinite(ic50_nm) or ic50_nm <= 0:
        raise SafetyMarginError("ic50_nm must be positive")
    if not np.isfinite(concentration_nm) or concentration_nm <= 0:
        raise SafetyMarginError("concentration_nm must be positive")
    return float(ic50_nm / concentration_nm)


def classify_safety_margin(margin: float, safe_threshold: float = 30.0, moderate_threshold: float = 10.0) -> str:
    if not np.isfinite(margin) or margin <= 0:
        raise SafetyMarginError("margin must be positive")
    if margin >= safe_threshold:
        return "Safe"
    if margin >= moderate_threshold:
        return "Moderate"
    return "High"


@dataclass(frozen=True)
class SafetyMarginResult:
    channel: str
    ic50_nm: float
    concentration_nm: float
    margin: float
    risk_class: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SafetyMarginAnalyzer:
    def __init__(self, safe_threshold: float = 30.0, moderate_threshold: float = 10.0):
        if moderate_threshold <= 0 or safe_threshold <= moderate_threshold:
            raise SafetyMarginError("Thresholds must satisfy safe_threshold > moderate_threshold > 0")
        self.safe_threshold = float(safe_threshold)
        self.moderate_threshold = float(moderate_threshold)

    def analyze_channel(self, channel: str, ic50_nm: float, concentration_nm: float) -> SafetyMarginResult:
        margin = calculate_safety_margin(ic50_nm, concentration_nm)
        risk_class = classify_safety_margin(margin, self.safe_threshold, self.moderate_threshold)
        return SafetyMarginResult(channel=channel, ic50_nm=float(ic50_nm), concentration_nm=float(concentration_nm), margin=margin, risk_class=risk_class)

    def build_table(self, ic50_values_nm: Mapping[str, float], concentration_nm: float) -> pd.DataFrame:
        rows = [self.analyze_channel(channel, ic50_nm, concentration_nm).to_dict() for channel, ic50_nm in ic50_values_nm.items()]
        return pd.DataFrame(rows)

    def summarize_markdown(self, drug_name: str, ic50_values_nm: Mapping[str, float], concentration_nm: float) -> str:
        table = self.build_table(ic50_values_nm, concentration_nm)
        lines = [
            f"## Safety Margin Summary: {drug_name}",
            "",
            f"Reference concentration: {concentration_nm:.6g} nM",
            "",
            "| Channel | IC50 (nM) | Concentration (nM) | Margin | Risk |",
            "|---|---:|---:|---:|---|",
        ]
        for _, row in table.iterrows():
            lines.append(f"| {row['channel']} | {row['ic50_nm']:.6g} | {row['concentration_nm']:.6g} | {row['margin']:.6g} | {row['risk_class']} |")
        return "\n".join(lines)
