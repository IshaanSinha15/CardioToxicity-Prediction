from classification_backend.dose_response.validation import (
    ValidationReport,
    validate_block_curve,
    validate_channel_inputs,
    validate_hill_coefficient,
    validate_ic50_range,
    validate_ord_payload,
)


def test_validate_channel_inputs_accepts_required_keys():
    report = validate_channel_inputs({"herg_ic50_nm": 1.0, "nav_ic50_nm": 2.0, "cav_ic50_nm": 3.0})
    assert report.is_valid is True


def test_validate_ord_payload_accepts_expected_schema():
    is_valid, warnings = validate_ord_payload({"concentration": 10.0, "herg_block": 1.0, "nav_block": 2.0, "cav_block": 3.0})
    assert is_valid is True
    assert warnings == []


def test_validate_block_curve_passes_for_monotonic_table():
    concentrations = [1.0, 10.0, 100.0]
    blocks = [1.0, 10.0, 50.0]
    is_valid, warnings = validate_block_curve(concentrations, blocks)
    assert is_valid is True
    assert warnings == []


def test_validate_ic50_and_hill_ranges_return_boolean_status():
    ic50_valid, ic50_warnings = validate_ic50_range(100.0, "herg")
    hill_valid, hill_warnings = validate_hill_coefficient(1.0, "herg")
    assert ic50_valid is True
    assert hill_valid is True
    assert ic50_warnings == []
    assert hill_warnings == []


def test_validation_report_markdown_contains_status():
    report = ValidationReport()
    markdown = report.to_markdown()
    assert "Status: PASS" in markdown