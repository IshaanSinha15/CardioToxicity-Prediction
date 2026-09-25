from classification_backend.evidence_interpreter import EvidenceInterpreter


def test_low_mechanistic_result_does_not_claim_safety():
    result = EvidenceInterpreter().interpret({"level": "low"}, {"matches": []})
    assert result.level == "low_mechanistic_concern"
    assert result.status == "model_derived_no_curated_analogue_support"


def test_high_risk_analogue_conflicts_with_low_mechanism():
    result = EvidenceInterpreter().interpret(
        {"level": "low"},
        {"matches": [{"risk_label": "High"}]},
    )
    assert result.status == "conflicting_evidence"
    assert result.requires_review


def test_conflicting_analogue_labels_require_review():
    result = EvidenceInterpreter().interpret(
        {"level": "unknown"},
        {"label_summary": {"conflicting_labels": True}, "matches": []},
    )
    assert result.status == "conflicting_evidence"
    assert result.requires_review