"""Concentration selection and conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

DEFAULT_MULTIPLES = (0.01, 0.1, 1.0, 10.0, 100.0)


class ConcentrationProfileError(ValueError):
    """Raised when concentration inputs are invalid."""


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ConcentrationProfileError(f"{name} must be positive and finite")
    return value


def oral_cmax_nm(
    dose_mg: float,
    molecular_weight_g_mol: float,
    volume_distribution_l: float,
    bioavailability: float,
    absorption_rate_per_h: float,
    elimination_rate_per_h: float,
) -> float:
    """Estimate oral peak concentration with a one-compartment PK model.

    This converts dose to concentration; it does not replace measured exposure.
    """
    dose_mg = _positive(dose_mg, "dose_mg")
    molecular_weight_g_mol = _positive(molecular_weight_g_mol, "molecular_weight_g_mol")
    volume_distribution_l = _positive(volume_distribution_l, "volume_distribution_l")
    absorption_rate_per_h = _positive(absorption_rate_per_h, "absorption_rate_per_h")
    elimination_rate_per_h = _positive(elimination_rate_per_h, "elimination_rate_per_h")
    bioavailability = float(bioavailability)
    if not np.isfinite(bioavailability) or not 0 < bioavailability <= 1:
        raise ConcentrationProfileError("bioavailability must be in (0, 1]")
    if np.isclose(absorption_rate_per_h, elimination_rate_per_h):
        raise ConcentrationProfileError("absorption and elimination rates must differ")

    t_max = np.log(absorption_rate_per_h / elimination_rate_per_h) / (
        absorption_rate_per_h - elimination_rate_per_h
    )
    dose_umol = dose_mg / molecular_weight_g_mol * 1000.0
    concentration_umol_l = (
        bioavailability
        * dose_umol
        * absorption_rate_per_h
        / (volume_distribution_l * (absorption_rate_per_h - elimination_rate_per_h))
        * (np.exp(-elimination_rate_per_h * t_max) - np.exp(-absorption_rate_per_h * t_max))
    )
    return _positive(concentration_umol_l * 1000.0, "calculated concentration_nm")


class ConcentrationType(str, Enum):
    NANOMOLAR = "nM"
    FREE_PLASMA = "free_plasma"
    TOTAL_PLASMA = "total_plasma"
    MEASURED_ASSAY = "measured_assay"
    NOMINAL_ASSAY = "nominal_assay"
    UNSPECIFIED = "unspecified"


class ExposureContext(str, Enum):
    THERAPEUTIC = "therapeutic"
    SUPRATHERAPEUTIC = "supratherapeutic"
    TOXIC = "toxic"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TypedExposure:
    concentration_nm: float
    concentration_type: ConcentrationType
    exposure_context: ExposureContext = ExposureContext.UNKNOWN
    source_field: str = "concentration_nm"

    def __post_init__(self) -> None:
        if not np.isfinite(self.concentration_nm) or self.concentration_nm <= 0:
            raise ConcentrationProfileError("concentration_nm must be positive and finite")

    @property
    def is_domain_known(self) -> bool:
        return self.concentration_type is not ConcentrationType.UNSPECIFIED

    def to_dict(self) -> dict[str, object]:
        return {
            "concentration_nm": self.concentration_nm,
            "concentration_type": self.concentration_type.value,
            "exposure_context": self.exposure_context.value,
            "source_field": self.source_field,
        }


def parse_typed_exposure(inputs: Mapping[str, object]) -> TypedExposure:
    if not isinstance(inputs, Mapping):
        raise ConcentrationProfileError("exposure input must be a mapping")

    has_concentration = "concentration_nm" in inputs
    has_legacy_dose = "dose_nm" in inputs
    has_dose_mg = "dose_mg" in inputs
    if has_dose_mg and (has_concentration or has_legacy_dose):
        raise ConcentrationProfileError("provide dose_mg or an nM concentration, not both")
    if has_dose_mg:
        required = (
            "molecular_weight_g_mol",
            "volume_distribution_l",
            "bioavailability",
            "absorption_rate_per_h",
            "elimination_rate_per_h",
        )
        missing = [name for name in required if name not in inputs]
        if missing:
            raise ConcentrationProfileError(f"dose_mg requires PK values: {', '.join(missing)}")
        concentration_nm = oral_cmax_nm(
            inputs["dose_mg"],
            inputs["molecular_weight_g_mol"],
            inputs["volume_distribution_l"],
            inputs["bioavailability"],
            inputs["absorption_rate_per_h"],
            inputs["elimination_rate_per_h"],
        )
        return TypedExposure(
            concentration_nm=concentration_nm,
            concentration_type=ConcentrationType.UNSPECIFIED,
            exposure_context=ExposureContext.UNKNOWN,
            source_field="dose_mg_pk_cmax",
        )
    if has_concentration and has_legacy_dose:
        raise ConcentrationProfileError("provide concentration_nm or dose_nm, not both")
    if not has_concentration and not has_legacy_dose:
        raise ConcentrationProfileError("missing concentration_nm")

    source_field = "concentration_nm"
    if has_legacy_dose:
        source_field = "dose_nm"

    try:
        concentration_nm = float(inputs[source_field])
    except (TypeError, ValueError) as exc:
        raise ConcentrationProfileError("concentration_nm must be numeric") from exc

    raw_type = inputs.get("concentration_type", ConcentrationType.NANOMOLAR)
    try:
        concentration_type = ConcentrationType(raw_type)
    except (TypeError, ValueError) as exc:
        raise ConcentrationProfileError(
            f"unsupported concentration_type: {raw_type!r}"
        ) from exc

    raw_context = inputs.get("exposure_context", ExposureContext.UNKNOWN)
    try:
        exposure_context = ExposureContext(raw_context)
    except (TypeError, ValueError) as exc:
        raise ConcentrationProfileError(f"unsupported exposure_context: {raw_context!r}") from exc

    return TypedExposure(
        concentration_nm=concentration_nm,
        concentration_type=concentration_type,
        exposure_context=exposure_context,
        source_field=source_field,
    )


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
