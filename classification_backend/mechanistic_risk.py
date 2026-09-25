from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from .dose_response.concentration_profiles import ConcentrationType, TypedExposure
from .dose_response.safety_margin import SafetyMarginAnalyzer


CHANNEL_LABELS = {"herg": "hERG", "nav": "Nav1.5", "cav": "Cav1.2"}


@dataclass(frozen=True)
class ChannelClassification:
    block_pct: float
    band: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MechanisticRiskResult:
    mechanistic_classification: dict[str, object]
    level: str
    status: str
    reasons: tuple[str, ...]
    dominant_channel: str | None
    concentration_domain: str
    training_range: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        payload["warnings"] = list(self.warnings)
        return payload


class MechanisticRiskAssessor:
    def __init__(
        self,
        safe_margin_threshold: float = 30.0,
        moderate_margin_threshold: float = 10.0,
        supported_min_nm: float = 0.3,
        supported_max_nm: float = 3000.0,
    ):
        self.margin_analyzer = SafetyMarginAnalyzer(
            safe_threshold=safe_margin_threshold,
            moderate_threshold=moderate_margin_threshold,
        )
        if supported_min_nm <= 0 or supported_max_nm < supported_min_nm:
            raise ValueError("supported concentration range is invalid")
        self.supported_min_nm = float(supported_min_nm)
        self.supported_max_nm = float(supported_max_nm)

    def assess(
        self,
        ic50_values_nm: Mapping[str, float],
        block_values: Mapping[str, float],
        exposure: TypedExposure,
    ) -> MechanisticRiskResult:
        normalized_ic50 = self._normalize_values(ic50_values_nm, "IC50")
        normalized_blocks = self._normalize_values(block_values, "block")
        domain, training_range, domain_warnings = self._domain(exposure)
        if normalized_ic50 is None or normalized_blocks is None:
            return self._unknown("required IC50 or block values are missing or invalid", domain, training_range, domain_warnings)

        margins = {
            channel: self.margin_analyzer.analyze_channel(
                channel, normalized_ic50[channel], exposure.concentration_nm
            )
            for channel in CHANNEL_LABELS
        }
        classifications = {
            CHANNEL_LABELS[channel]: ChannelClassification(
                block_pct=normalized_blocks[channel],
                band=_block_band(normalized_blocks[channel]),
            ).to_dict()
            for channel in CHANNEL_LABELS
        }
        dominant = max(CHANNEL_LABELS, key=lambda channel: normalized_blocks[channel])
        high_channels = [channel for channel, result in margins.items() if result.margin < self.margin_analyzer.moderate_threshold]
        high_block_channels = [channel for channel, value in normalized_blocks.items() if value >= 50]
        moderate_block_channels = [channel for channel, value in normalized_blocks.items() if value >= 20]
        reasons: list[str] = []
        if high_channels:
            reasons.append(f"{', '.join(CHANNEL_LABELS[channel] for channel in high_channels)} safety margin is below 10")
        if normalized_blocks["herg"] >= 50:
            reasons.append("hERG block is at least 50%")
        if high_block_channels and not normalized_blocks["herg"] >= 50:
            reasons.append(f"{', '.join(CHANNEL_LABELS[channel] for channel in high_block_channels)} block is at least 50%")

        if reasons:
            level = "high"
        elif any(result.margin < self.margin_analyzer.safe_threshold for result in margins.values()) or normalized_blocks["herg"] >= 20 or high_block_channels or len(moderate_block_channels) >= 2:
            level = "moderate"
            if not reasons:
                reasons.append("mechanistic evidence meets the moderate concern criteria")
        else:
            level = "low"
            reasons.append("all channel blocks are below 20% and safety margins meet the safe threshold")

        warnings = list(domain_warnings)
        return MechanisticRiskResult(
            mechanistic_classification={
                "channels": classifications,
                "dominant_channel": CHANNEL_LABELS[dominant],
                "overall_level": level,
                "status": "model_derived",
            },
            level=level,
            status="model_derived",
            reasons=tuple(reasons),
            dominant_channel=CHANNEL_LABELS[dominant],
            concentration_domain=domain,
            training_range=training_range,
            warnings=tuple(warnings),
        )

    def _domain(self, exposure: TypedExposure) -> tuple[str, str, list[str]]:
        if exposure.concentration_type is ConcentrationType.UNSPECIFIED:
            return "unknown", "unknown", ["concentration type is unspecified"]
        if not self.supported_min_nm <= exposure.concentration_nm <= self.supported_max_nm:
            return "extrapolated", "outside", ["concentration is outside the training panel; calculation was still performed"]
        return "supported", "inside", []

    @staticmethod
    def _normalize_values(values: Mapping[str, float], value_name: str) -> dict[str, float] | None:
        normalized: dict[str, float] = {}
        for channel in CHANNEL_LABELS:
            value = values.get(channel)
            if value is None:
                value = values.get(f"{channel}_block") if value_name == "block" else values.get(f"{channel}_ic50_nm")
            if value is None:
                return None
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(value) or (value <= 0 and value_name != "block") or (value < 0 and value_name == "block") or (value_name == "block" and value > 100):
                return None
            normalized[channel] = value
        return normalized

    @staticmethod
    def _unknown(reason: str, domain: str, training_range: str, warnings: list[str]) -> MechanisticRiskResult:
        return MechanisticRiskResult(
            mechanistic_classification={"channels": {}, "dominant_channel": None, "overall_level": "unknown", "status": "dose_response_invalid"},
            level="unknown",
            status="dose_response_invalid",
            reasons=(reason,),
            dominant_channel=None,
            concentration_domain=domain,
            training_range=training_range,
            warnings=tuple(warnings),
        )


def _block_band(block_pct: float) -> str:
    if block_pct < 5:
        return "Minimal"
    if block_pct < 20:
        return "Low"
    if block_pct < 50:
        return "Moderate"
    if block_pct < 90:
        return "High"
    return "Near-maximal"