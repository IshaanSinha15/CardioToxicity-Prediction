"""High-level orchestration for IC50-to-block conversion."""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .concentration_profiles import ConcentrationProfile, DEFAULT_MULTIPLES
from .hill_equation import HillEquation
from .validation import normalize_channel_inputs, validate_channel_inputs, validate_ord_payload

LOGGER = logging.getLogger(__name__)


class ChannelBlockGeneratorError(ValueError):
    """Raised when generator inputs are invalid."""


@dataclass(frozen=True)
class ChannelIC50Inputs:
    """IC50 inputs for the three core cardiac ion channels."""

    herg_ic50_nm: float
    nav_ic50_nm: float
    cav_ic50_nm: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float]) -> "ChannelIC50Inputs":
        normalized = normalize_channel_inputs(mapping)
        return cls(**normalized)


@dataclass(frozen=True)
class ChannelBlockResult:
    """ORd-ready single-point block result."""

    concentration: float
    herg_block: float
    nav_block: float
    cav_block: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _normalize_hill_coefficients(hill_coefficients: Mapping[str, float] | None) -> dict[str, float]:
    default = {"herg": 1.0, "nav": 1.0, "cav": 1.0}
    if hill_coefficients is None:
        return default
    normalized = default.copy()
    for key, value in hill_coefficients.items():
        normalized[key.lower()] = float(value)
    return normalized


class ChannelBlockGenerator:
    """Convert IC50 predictions into single-point and multi-point block outputs."""

    def __init__(
        self,
        ic50_inputs: ChannelIC50Inputs | Mapping[str, float],
        hill_coefficients: Mapping[str, float] | None = None,
        channel_labels: Mapping[str, str] | None = None,
    ):
        if isinstance(ic50_inputs, ChannelIC50Inputs):
            normalized_inputs = ic50_inputs.to_dict()
        else:
            normalized_inputs = normalize_channel_inputs(dict(ic50_inputs))

        validation = validate_channel_inputs(normalized_inputs)
        if not validation.is_valid:
            raise ChannelBlockGeneratorError(validation.to_markdown())

        self.ic50_inputs = ChannelIC50Inputs(**normalized_inputs)
        self.hill_coefficients = _normalize_hill_coefficients(hill_coefficients)
        self.channel_labels = {"herg": "hERG", "nav": "Nav1.5", "cav": "Cav1.2"}
        if channel_labels:
            self.channel_labels.update({key.lower(): value for key, value in channel_labels.items()})

    def _equation_for(self, channel: str) -> HillEquation:
        key = channel.lower()
        ic50_key = f"{key}_ic50_nm"
        return HillEquation(
            ic50_nm=getattr(self.ic50_inputs, ic50_key),
            hill_coefficient=self.hill_coefficients[key],
            channel=self.channel_labels.get(key, channel),
        )

    def block_at_concentration(self, concentration_nm: float) -> ChannelBlockResult:
        """Return the exact ORd-ready payload shape for a single concentration."""

        if not np.isfinite(concentration_nm) or concentration_nm < 0:
            raise ChannelBlockGeneratorError("concentration_nm must be a non-negative finite number")

        herg = self._equation_for("herg").block(concentration_nm)
        nav = self._equation_for("nav").block(concentration_nm)
        cav = self._equation_for("cav").block(concentration_nm)
        result = ChannelBlockResult(
            concentration=float(concentration_nm),
            herg_block=float(herg),
            nav_block=float(nav),
            cav_block=float(cav),
        )
        ok, issues = validate_ord_payload(result.to_dict())
        if not ok:
            raise ChannelBlockGeneratorError("; ".join(issues))
        return result

    def to_ord_payload(self, concentration_nm: float) -> dict[str, float]:
        return self.block_at_concentration(concentration_nm).to_dict()

    def block_profile(
        self,
        reference_concentration_nm: float,
        multiples: Sequence[float] = DEFAULT_MULTIPLES,
    ) -> pd.DataFrame:
        """Return a multi-point block table at standardized concentration multiples."""

        multiples_tuple = tuple(float(v) for v in multiples)
        profile = ConcentrationProfile(reference_concentration_nm=float(reference_concentration_nm), multiples=multiples_tuple)
        concentrations = profile.concentrations_nm
        rows = [self.block_at_concentration(concentration).to_dict() for concentration in concentrations]
        frame = pd.DataFrame(rows)
        frame["multiple"] = list(multiples_tuple)
        return frame[["multiple", "concentration", "herg_block", "nav_block", "cav_block"]]

    def build_summary_frame(
        self,
        reference_concentration_nm: float,
        multiples: Sequence[float] = DEFAULT_MULTIPLES,
    ) -> pd.DataFrame:
        """Return a long-form table with channel-wise blocks and concentrations."""

        frame = self.block_profile(reference_concentration_nm, multiples)
        melted = frame.melt(id_vars=["multiple", "concentration"], var_name="channel", value_name="block_pct")
        return melted.sort_values(["concentration", "channel"]).reset_index(drop=True)

    def build_safety_margins(self, reference_concentration_nm: float) -> pd.DataFrame:
        from .safety_margin import SafetyMarginAnalyzer

        analyzer = SafetyMarginAnalyzer()
        return analyzer.build_table(self.ic50_inputs.to_dict(), float(reference_concentration_nm))
