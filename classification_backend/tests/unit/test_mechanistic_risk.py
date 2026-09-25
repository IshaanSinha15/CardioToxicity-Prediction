import pytest

from classification_backend.dose_response import ConcentrationType, ExposureContext, TypedExposure
from classification_backend.mechanistic_risk import MechanisticRiskAssessor


def exposure(concentration=100.0, kind=ConcentrationType.FREE_PLASMA):
    return TypedExposure(concentration, kind, ExposureContext.THERAPEUTIC)


def assess(blocks, ic50=(10000.0, 10000.0, 10000.0), current=exposure()):
    return MechanisticRiskAssessor().assess(
        {"herg": ic50[0], "nav": ic50[1], "cav": ic50[2]},
        {"herg_block": blocks[0], "nav_block": blocks[1], "cav_block": blocks[2]},
        current,
    )


@pytest.mark.parametrize(
    ("block", "band"),
    [(4.99, "Minimal"), (5.0, "Low"), (19.99, "Low"), (20.0, "Moderate"), (49.99, "Moderate"), (50.0, "High"), (90.0, "Near-maximal")],
)
def test_blockage_boundaries(block, band):
    result = assess((block, 1.0, 1.0))
    assert result.mechanistic_classification["channels"]["hERG"]["band"] == band


def test_high_risk_precedence_uses_margin_and_herg_block():
    assert assess((49.0, 1.0, 1.0), ic50=(900.0, 10000.0, 10000.0)).level == "high"
    assert assess((50.0, 1.0, 1.0)).level == "high"


def test_unknown_and_extrapolated_domain_are_explicit():
    invalid = assess((1.0, 1.0, 1.0), ic50=(None, 1.0, 1.0))
    assert invalid.status == "dose_response_invalid"
    extrapolated = assess((1.0, 1.0, 1.0), current=exposure(5000.0))
    assert extrapolated.concentration_domain == "extrapolated"
    assert extrapolated.training_range == "outside"
    assert extrapolated.warnings


def test_unspecified_concentration_domain_is_unknown():
    result = assess((1.0, 1.0, 1.0), current=exposure(kind=ConcentrationType.UNSPECIFIED))
    assert result.concentration_domain == "unknown"
    assert result.training_range == "unknown"