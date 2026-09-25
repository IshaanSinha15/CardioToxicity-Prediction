import pandas as pd
import pytest

from classification_backend.assessment.similarity.reference_database import ReferenceDatabase
from classification_backend.assessment.similarity.retriever import SimilarityRetriever


@pytest.fixture
def retriever():
    frame = pd.DataFrame(
        [
            {"reference_id": "z-alcohol", "smiles": "CCO", "risk_label": "Low"},
            {"reference_id": "a-methanol", "smiles": "CO", "risk_label": "High"},
            {"reference_id": "alkane", "smiles": "CCCC"},
        ]
    )
    return SimilarityRetriever(ReferenceDatabase.from_frame(frame))


def test_search_is_ranked_and_returns_provenance(retriever):
    result = retriever.search("CCO", top_k=2, minimum_similarity=0.0)

    assert result.status == "ok"
    assert result.matches[0].reference_id == "z-alcohol"
    assert result.matches[0].similarity == 1.0
    assert result.matches[0].risk_label == "Low"
    assert result.label_summary["known_labels"] == 1


def test_search_excludes_exact_reference(retriever):
    result = retriever.search("CCO", top_k=5, minimum_similarity=0.0, exclude_reference_id="z-alcohol")

    assert all(match.reference_id != "z-alcohol" for match in result.matches)


def test_search_reports_no_match(retriever):
    result = retriever.search("Cl[Cl]", minimum_similarity=1.0)

    assert result.status == "no_reference_matches"
    assert result.matches == ()


def test_search_validates_parameters(retriever):
    with pytest.raises(ValueError):
        retriever.search("CCO", top_k=0)
    with pytest.raises(ValueError):
        retriever.search("CCO", minimum_similarity=1.1)