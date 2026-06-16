from classification_backend.dose_response.safety_margin import (
    SafetyMarginAnalyzer,
    calculate_safety_margin,
    classify_safety_margin,
)


def test_safety_margin_classification():
    margin = calculate_safety_margin(1000.0, 10.0)
    assert margin == 100.0
    assert classify_safety_margin(margin) == "Safe"


def test_build_safety_assessment_contains_expected_columns():
    analyzer = SafetyMarginAnalyzer()
    frame = analyzer.build_table({"herg": 100.0, "nav": 200.0, "cav": 300.0}, 10.0)

    assert set(["channel", "ic50_nm", "concentration_nm", "margin", "risk_class"]).issubset(frame.columns)
