import pytest

from classification_backend.dose_response import (
    ConcentrationType,
    ExposureContext,
    parse_typed_exposure,
)
from classification_backend.dose_response.concentration_profiles import ConcentrationProfileError
from classification_backend.dose_response.concentration_profiles import oral_cmax_nm


def test_parses_explicit_free_plasma_exposure():
    exposure = parse_typed_exposure(
        {
            "concentration_nm": 100.0,
            "concentration_type": "free_plasma",
            "exposure_context": "therapeutic",
        }
    )

    assert exposure.concentration_nm == 100.0
    assert exposure.concentration_type is ConcentrationType.FREE_PLASMA
    assert exposure.exposure_context is ExposureContext.THERAPEUTIC
    assert exposure.is_domain_known


def test_legacy_dose_defaults_to_nanomolar():
    exposure = parse_typed_exposure({"dose_nm": 100.0})

    assert exposure.concentration_type is ConcentrationType.NANOMOLAR
    assert exposure.source_field == "dose_nm"
    assert exposure.is_domain_known


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_rejects_non_positive_or_non_finite_concentration(value):
    with pytest.raises(ConcentrationProfileError, match="positive and finite"):
        parse_typed_exposure({"concentration_nm": value, "concentration_type": "nominal_assay"})


def test_rejects_ambiguous_unit_input():
    with pytest.raises(ConcentrationProfileError, match="unsupported concentration_type"):
        parse_typed_exposure({"concentration_nm": 1.0, "concentration_type": "uM"})


def test_rejects_conflicting_concentration_fields():
    with pytest.raises(ConcentrationProfileError, match="not both"):
        parse_typed_exposure({"concentration_nm": 1.0, "dose_nm": 1.0})


@pytest.mark.parametrize("dose_nm", [60.0, 6000.0, 60000.0])
def test_accepts_simple_nanomolar_test_values(dose_nm):
    exposure = parse_typed_exposure({"dose_nm": dose_nm})
    assert exposure.concentration_nm == dose_nm


def test_dose_mg_requires_pk_parameters():
    with pytest.raises(ConcentrationProfileError, match="requires PK values"):
        parse_typed_exposure({"dose_mg": 10.0})


def test_oral_pk_conversion_returns_nanomolar_peak():
    concentration_nm = oral_cmax_nm(500.0, 151.16, 50.0, 0.8, 2.0, 0.2)
    assert concentration_nm > 0
    assert concentration_nm < 100000