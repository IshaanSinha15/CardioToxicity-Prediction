"""Validation helpers for dose-response outputs and ORd-ready payloads."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable, Mapping

import numpy as np


class ValidationError(ValueError):
    """Raised when validation inputs are malformed."""


@dataclass(frozen=True)
class ValidationIssue:
    field_name: str
    message: str
    severity: str = "warning"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def add_issue(self, field_name: str, message: str, severity: str = "warning") -> None:
        self.issues.append(ValidationIssue(field_name=field_name, message=message, severity=severity))

    def extend(self, issues: Iterable[ValidationIssue]) -> None:
        self.issues.extend(list(issues))

    def to_dict(self) -> dict[str, object]:
        return {"is_valid": self.is_valid, "issues": [issue.to_dict() for issue in self.issues]}

    def to_markdown(self) -> str:
        lines = ["## Validation Report", "", f"Status: {'PASS' if self.is_valid else 'FAIL'}", "", "| Field | Severity | Message |", "|---|---|---|"]
        if not self.issues:
            lines.append("| - | - | No issues detected |")
            return "\n".join(lines)
        for issue in self.issues:
            lines.append(f"| {issue.field_name} | {issue.severity} | {issue.message} |")
        return "\n".join(lines)


IC50_RANGES_NM = {
    "herg": (1.0, 100000.0),
    "nav": (1000.0, 100000.0),
    "cav": (100.0, 100000.0),
}


def normalize_channel_inputs(inputs: Mapping[str, object]) -> dict[str, float]:
    aliases = {
        "herg_ic50": "herg_ic50_nm",
        "nav_ic50": "nav_ic50_nm",
        "cav_ic50": "cav_ic50_nm",
        "herg_ic50_nm": "herg_ic50_nm",
        "nav_ic50_nm": "nav_ic50_nm",
        "cav_ic50_nm": "cav_ic50_nm",
    }

    normalized: dict[str, float] = {}
    for key, target in aliases.items():
        if key in inputs:
            normalized[target] = float(inputs[key])

    missing = {"herg_ic50_nm", "nav_ic50_nm", "cav_ic50_nm"}.difference(normalized.keys())
    if missing:
        raise ValidationError(f"Missing required IC50 values: {sorted(missing)}")
    return normalized


def validate_channel_inputs(inputs: Mapping[str, object]) -> ValidationReport:
    report = ValidationReport()
    try:
        normalized = normalize_channel_inputs(inputs)
    except ValidationError as exc:
        report.add_issue("inputs", str(exc), severity="error")
        return report

    for channel_key, value in normalized.items():
        if not isinstance(value, (int, float, np.floating)) or not np.isfinite(float(value)):
            report.add_issue(channel_key, "Must be a finite numeric value", severity="error")
        elif float(value) <= 0:
            report.add_issue(channel_key, "Must be positive", severity="error")
    return report


def validate_ic50_range(ic50_nm: float, channel: str, strict: bool = False) -> tuple[bool, list[str]]:
    if not np.isfinite(ic50_nm) or ic50_nm <= 0:
        raise ValidationError("ic50_nm must be positive")
    key = channel.lower()
    if key not in IC50_RANGES_NM:
        return True, [f"No default IC50 range registered for channel '{channel}'"]

    low, high = IC50_RANGES_NM[key]
    if strict:
        low, high = low * 2.0, high / 2.0

    warnings: list[str] = []
    if ic50_nm < low or ic50_nm > high:
        warnings.append(f"IC50 {ic50_nm:.6g} nM is outside the expected range [{low:.6g}, {high:.6g}] for {channel}")
    return len(warnings) == 0, warnings


def validate_hill_coefficient(hill_coefficient: float, channel: str | None = None, strict: bool = False) -> tuple[bool, list[str]]:
    if not np.isfinite(hill_coefficient) or hill_coefficient <= 0:
        raise ValidationError("hill_coefficient must be positive")
    low, high = (0.7, 1.5) if strict else (0.5, 2.0)
    warnings: list[str] = []
    if hill_coefficient < low or hill_coefficient > high:
        warnings.append(f"Hill coefficient {hill_coefficient:.6g} is outside the expected range for {channel or 'this channel'}")
    return len(warnings) == 0, warnings


def validate_block_curve(concentrations_nm: Iterable[float], block_percentages: Iterable[float]) -> tuple[bool, list[str]]:
    concentrations = np.asarray(list(concentrations_nm), dtype=float).reshape(-1)
    blocks = np.asarray(list(block_percentages), dtype=float).reshape(-1)
    issues: list[str] = []
    if concentrations.size == 0 or blocks.size == 0:
        raise ValidationError("concentrations_nm and block_percentages cannot be empty")
    if concentrations.size != blocks.size:
        raise ValidationError("concentrations_nm and block_percentages must have the same length")
    if not np.all(np.isfinite(concentrations)) or not np.all(np.isfinite(blocks)):
        issues.append("Curve contains non-finite values")
    if np.any(concentrations < 0):
        issues.append("Concentrations must be non-negative")
    if np.any(blocks < -1e-9) or np.any(blocks > 100.0 + 1e-9):
        issues.append("Block percentages must lie within [0, 100]")
    if np.any(np.diff(blocks) < -1e-6):
        issues.append("Block percentage should be monotonic non-decreasing with concentration")
    return len(issues) == 0, issues


def validate_ord_payload(payload: Mapping[str, object]) -> tuple[bool, list[str]]:
    required = {"concentration", "herg_block", "nav_block", "cav_block"}
    missing = sorted(required.difference(payload.keys()))
    if missing:
        return False, [f"Missing required keys: {', '.join(missing)}"]

    issues: list[str] = []
    for key in required:
        value = payload[key]
        if not isinstance(value, (int, float, np.floating)) or not np.isfinite(float(value)):
            issues.append(f"{key} must be a finite numeric value")
    if issues:
        return False, issues

    if float(payload["concentration"]) < 0:
        issues.append("concentration must be non-negative")
    for key in ("herg_block", "nav_block", "cav_block"):
        value = float(payload[key])
        if value < 0 or value > 100:
            issues.append(f"{key} must be between 0 and 100")
    return len(issues) == 0, issues
