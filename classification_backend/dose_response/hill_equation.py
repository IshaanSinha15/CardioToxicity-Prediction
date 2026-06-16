"""Hill equation utilities for channel block modeling."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


class HillEquationError(ValueError):
    """Raised when Hill-equation inputs are invalid."""


def _as_1d_array(values: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float).reshape(-1)
    if array.size == 0:
        raise HillEquationError(f"{name} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise HillEquationError(f"{name} must contain finite values")
    return array


@dataclass(frozen=True)
class HillParameters:
    ic50_nm: float
    hill_coefficient: float = 1.0
    emax: float = 100.0
    channel: str | None = None

    def validate(self) -> None:
        if not np.isfinite(self.ic50_nm) or self.ic50_nm <= 0:
            raise HillEquationError("ic50_nm must be a positive finite number")
        if not np.isfinite(self.hill_coefficient) or self.hill_coefficient <= 0:
            raise HillEquationError("hill_coefficient must be a positive finite number")
        if not np.isfinite(self.emax) or self.emax <= 0:
            raise HillEquationError("emax must be a positive finite number")

    def to_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


@dataclass(frozen=True)
class HillFitResult:
    ic50_nm: float
    hill_coefficient: float
    emax: float
    r_squared: float
    n_points: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def hill_block_percentage(
    concentration_nm: float | np.ndarray,
    ic50_nm: float,
    hill_coefficient: float = 1.0,
    emax: float = 100.0,
) -> float | np.ndarray:
    params = HillParameters(ic50_nm=float(ic50_nm), hill_coefficient=float(hill_coefficient), emax=float(emax))
    params.validate()

    concentration = np.asarray(concentration_nm, dtype=float)
    if np.any(concentration < 0):
        raise HillEquationError("concentration_nm must be non-negative")

    numerator = np.power(concentration, params.hill_coefficient)
    denominator = np.power(params.ic50_nm, params.hill_coefficient) + numerator
    block = np.divide(params.emax * numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)

    if np.isscalar(concentration_nm):
        return float(block)
    return block


class HillEquation:
    def __init__(self, ic50_nm: float, hill_coefficient: float = 1.0, emax: float = 100.0, channel: str | None = None):
        self.params = HillParameters(
            ic50_nm=float(ic50_nm),
            hill_coefficient=float(hill_coefficient),
            emax=float(emax),
            channel=channel,
        )
        self.params.validate()

    @property
    def ic50_nm(self) -> float:
        return self.params.ic50_nm

    @property
    def hill_coefficient(self) -> float:
        return self.params.hill_coefficient

    @property
    def emax(self) -> float:
        return self.params.emax

    def block(self, concentration_nm: float | np.ndarray) -> float | np.ndarray:
        return hill_block_percentage(
            concentration_nm=concentration_nm,
            ic50_nm=self.ic50_nm,
            hill_coefficient=self.hill_coefficient,
            emax=self.emax,
        )

    def block_series(self, concentrations_nm: Iterable[float] | np.ndarray) -> np.ndarray:
        concentrations = _as_1d_array(concentrations_nm, "concentrations_nm")
        return np.asarray(self.block(concentrations), dtype=float)

    def conductance_scaling(self, concentration_nm: float) -> float:
        return 1.0 - float(self.block(concentration_nm)) / 100.0

    def to_dict(self) -> dict[str, float | str | None]:
        return self.params.to_dict()


def fit_hill_parameters(
    concentrations_nm: Iterable[float] | np.ndarray,
    responses_pct: Iterable[float] | np.ndarray,
) -> HillFitResult:
    concentrations = _as_1d_array(concentrations_nm, "concentrations_nm")
    responses = _as_1d_array(responses_pct, "responses_pct")

    if concentrations.size != responses.size:
        raise HillEquationError("concentrations_nm and responses_pct must have the same length")
    if concentrations.size < 3:
        raise HillEquationError("At least 3 points are required to fit Hill parameters")
    if np.any(concentrations <= 0):
        raise HillEquationError("concentrations_nm must be strictly positive for fitting")

    clipped = np.clip(responses, 1e-6, 100.0 - 1e-6)
    logits = np.log(clipped / (100.0 - clipped))
    log_concentrations = np.log(concentrations)
    slope, intercept = np.polyfit(log_concentrations, logits, 1)
    if not np.isfinite(slope) or slope <= 0:
        raise HillEquationError("Fitted Hill coefficient must be positive")

    predicted_logits = slope * log_concentrations + intercept
    predicted_responses = 100.0 / (1.0 + np.exp(-predicted_logits))
    ss_res = float(np.sum((clipped - predicted_responses) ** 2))
    ss_tot = float(np.sum((clipped - np.mean(clipped)) ** 2))
    r_squared = 1.0 if ss_tot == 0 else max(0.0, 1.0 - ss_res / ss_tot)

    ic50_nm = float(np.exp(-intercept / slope))
    hill_coefficient = float(slope)
    LOGGER.info("Fitted Hill parameters: ic50_nm=%s hill=%s r2=%s", ic50_nm, hill_coefficient, r_squared)
    return HillFitResult(ic50_nm=ic50_nm, hill_coefficient=hill_coefficient, emax=100.0, r_squared=r_squared, n_points=int(concentrations.size))


def calculate_block_with_uncertainty(
    concentration_nm: float,
    ic50_nm: float,
    hill_coefficient: float,
    ic50_std_nm: float,
    hill_std: float,
    n_samples: int = 1000,
    random_state: int | None = None,
) -> dict[str, float]:
    if n_samples <= 0:
        raise HillEquationError("n_samples must be positive")
    if ic50_std_nm < 0 or hill_std < 0:
        raise HillEquationError("Standard deviations must be non-negative")

    rng = np.random.default_rng(random_state)
    sampled_ic50 = np.clip(rng.normal(ic50_nm, ic50_std_nm, size=n_samples), 1e-12, None)
    sampled_hill = np.clip(rng.normal(hill_coefficient, hill_std, size=n_samples), 1e-12, None)
    blocks = np.asarray(hill_block_percentage(concentration_nm, sampled_ic50, sampled_hill), dtype=float)

    return {
        "mean": float(np.mean(blocks)),
        "std": float(np.std(blocks, ddof=1) if blocks.size > 1 else 0.0),
        "ci_lower": float(np.percentile(blocks, 2.5)),
        "ci_upper": float(np.percentile(blocks, 97.5)),
    }


def validate_hill_parameters(
    ic50_nm: float,
    hill_coefficient: float,
    channel: str | None = None,
    strict: bool = False,
) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    if not np.isfinite(ic50_nm) or ic50_nm <= 0:
        raise HillEquationError("ic50_nm must be a positive finite number")
    if not np.isfinite(hill_coefficient) or hill_coefficient <= 0:
        raise HillEquationError("hill_coefficient must be a positive finite number")

    low, high = (0.7, 1.5) if strict else (0.5, 2.0)
    if hill_coefficient < low or hill_coefficient > high:
        warnings.append(f"Hill coefficient {hill_coefficient:.3g} is outside the {'strict' if strict else 'expected'} range for {channel or 'this channel'}")
    if ic50_nm > 1e6:
        warnings.append(f"IC50 {ic50_nm:.3g} nM is unusually high and should be checked")
    if ic50_nm < 1e-3:
        warnings.append(f"IC50 {ic50_nm:.3g} nM is unusually low and should be checked")
    return True, warnings
