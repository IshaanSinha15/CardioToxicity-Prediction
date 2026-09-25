import pandas as pd

from classification_backend.assessment.similarity.reference_database import ReferenceDatabase
from classification_backend.assessment.similarity.retriever import SimilarityRetriever
from pipeline.prediction_pipeline import PredictionPipeline


def test_full_pipeline_excludes_xai_and_returns_new_evidence_contract(monkeypatch):
    monkeypatch.setattr(
        "pipeline.prediction_pipeline.predict_ic50",
        lambda smiles: {
            "herg": {"pIC50": 6.0, "IC50_nM": 1000.0},
            "nav": {"pIC50": 5.3, "IC50_nM": 5000.0},
            "cav": {"pIC50": 5.7, "IC50_nM": 2000.0},
        },
    )
    retriever = SimilarityRetriever(
        ReferenceDatabase.from_frame(
            pd.DataFrame([{"reference_id": "ethanol", "smiles": "CCO", "risk_label": "Low"}])
        )
    )

    result = PredictionPipeline(similarity_retriever=retriever, run_simulation=False).run(
        {
            "smiles": "C(C)O",
            "concentration_nm": 100.0,
            "concentration_type": "free_plasma",
            "exposure_context": "therapeutic",
            "skip_simulation": True,
        }
    )

    assert result["input"]["canonical_smiles"] == "CCO"
    assert result["dose_response"]["herg_block"] == 9.090909090909092
    assert result["similarity"]["matches"][0]["reference_id"] == "ethanol"
    assert result["simulation"]["status"] == "skipped"
    assert "interpretation" in result
    assert "classification" in result
    assert "xai" in result
    assert set(result["xai"]) == {"chemical", "classification"}