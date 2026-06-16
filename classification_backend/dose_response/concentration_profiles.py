"""Concentration selection and conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_MULTIPLES = (0.01, 0.1, 1.0, 10.0, 100.0)


class ConcentrationProfileError(ValueError):
    """Raised when concentration inputs are invalid."""


def calculate_free_concentration(total_concentration_nm: float, protein_binding_pct: float) -> float:
    if not np.isfinite(total_concentration_nm) or total_concentration_nm < 0:
        raise ConcentrationProfileError("total_concentration_nm must be non-negative")
    if not np.isfinite(protein_binding_pct) or not (0 <= protein_binding_pct < 100):
        raise ConcentrationProfileError("protein_binding_pct must be in [0, 100)")
    return float(total_concentration_nm * (1.0 - protein_binding_pct / 100.0))


def calculate_total_concentration(free_concentration_nm: float, protein_binding_pct: float) -> float:
    if not np.isfinite(free_concentration_nm) or free_concentration_nm < 0:
        raise ConcentrationProfileError("free_concentration_nm must be non-negative")
    if not np.isfinite(protein_binding_pct) or not (0 <= protein_binding_pct < 100):
        raise ConcentrationProfileError("protein_binding_pct must be in [0, 100)")
    free_fraction = 1.0 - protein_binding_pct / 100.0
    return float(free_concentration_nm / free_fraction)


def generate_concentration_series(reference_concentration_nm: float, multiples: Iterable[float] = DEFAULT_MULTIPLES) -> np.ndarray:
    if not np.isfinite(reference_concentration_nm) or reference_concentration_nm <= 0:
        raise ConcentrationProfileError("reference_concentration_nm must be positive")

    multiples_array = np.asarray(list(multiples), dtype=float).reshape(-1)
    if multiples_array.size == 0 or not np.all(np.isfinite(multiples_array)):
        raise ConcentrationProfileError("multiples must contain finite values")
    if np.any(multiples_array <= 0):
        raise ConcentrationProfileError("multiples must be positive")

    return (reference_concentration_nm * multiples_array).astype(float)


@dataclass(frozen=True)
class ConcentrationProfile:
    reference_concentration_nm: float
    multiples: tuple[float, ...] = DEFAULT_MULTIPLES
    label: str | None = None
    protein_binding_pct: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.reference_concentration_nm) or self.reference_concentration_nm <= 0:
            raise ConcentrationProfileError("reference_concentration_nm must be positive")
        if np.any(np.asarray(self.multiples, dtype=float) <= 0):
            raise ConcentrationProfileError("multiples must be positive")
        if self.protein_binding_pct is not None and not (0 <= self.protein_binding_pct < 100):
            raise ConcentrationProfileError("protein_binding_pct must be in [0, 100)")

    @property
    def concentrations_nm(self) -> np.ndarray:
        return generate_concentration_series(self.reference_concentration_nm, self.multiples)

    @property
    def free_fraction(self) -> float | None:
        if self.protein_binding_pct is None:
            return None
        return float(1.0 - self.protein_binding_pct / 100.0)

    @property
    def free_reference_concentration_nm(self) -> float:
        if self.protein_binding_pct is None:
            return float(self.reference_concentration_nm)
        return calculate_free_concentration(self.reference_concentration_nm, self.protein_binding_pct)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "multiple": np.asarray(self.multiples, dtype=float),
                "concentration_nm": self.concentrations_nm,
                "label": self.label,
            }
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["concentrations_nm"] = self.concentrations_nm.tolist()
        payload["free_reference_concentration_nm"] = self.free_reference_concentration_nm
        return payload


def categorize_exposure(concentration_nm: float, reference_concentration_nm: float) -> str:
    if reference_concentration_nm <= 0:
        raise ConcentrationProfileError("reference_concentration_nm must be positive")
    ratio = concentration_nm / reference_concentration_nm
    if ratio < 0.5:
        return "sub-therapeutic"
    if ratio <= 1.5:
        return "therapeutic"
    if ratio <= 5.0:
        return "supratherapeutic"
    return "toxic"
