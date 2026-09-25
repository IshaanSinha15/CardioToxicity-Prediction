import pandas as pd

from classification_backend.classification_service import ClassificationEvidenceService
from classification_backend.assessment.similarity.reference_database import ReferenceDatabase
from classification_backend.assessment.similarity.retriever import SimilarityRetriever


def test_service_preserves_mechanistic_values_and_similarity_evidence():
    database = ReferenceDatabase.from_frame(
        pd.DataFrame([{"reference_id": "ethanol", "smiles": "CCO", "risk_label": "Low"}])
    )
    result = ClassificationEvidenceService(SimilarityRetriever(database)).evaluate(
        "C(C)O",
        {"concentration_nm": 100.0, "concentration_type": "free_plasma"},
        {"herg_ic50_nm": 1000.0, "nav_ic50_nm": 5000.0, "cav_ic50_nm": 2000.0},
        top_k=1,
        minimum_similarity=0.4,
    )

    assert result["input"]["canonical_smiles"] == "CCO"
    assert result["dose_response"]["herg_block"] == 9.090909090909092
    assert result["safety_margins"][0]["margin"] == 10.0
    assert result["similarity"]["matches"][0]["reference_id"] == "ethanol"
    assert result["interpretation"]["status"] == "model_derived_no_curated_analogue_support"


def test_service_reports_missing_similarity_database_without_false_label():
    result = ClassificationEvidenceService().evaluate(
        "CCO",
        {"dose_nm": 100.0},
        {"herg_ic50_nm": 1000.0, "nav_ic50_nm": 5000.0, "cav_ic50_nm": 2000.0},
    )

    assert result["similarity"]["status"] == "reference_database_unavailable"
    assert result["warnings"]
    assert result["interpretation"]["level"] == "moderate_concern"