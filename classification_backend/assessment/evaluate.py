from __future__ import annotations

from pathlib import Path

from classification_backend.dose_response import ChannelBlockGenerator
from classification_backend.mechanistic_risk import MechanisticRiskAssessor
from classification_backend.assessment.similarity.reference_database import ReferenceDatabase
from classification_backend.assessment.similarity.retriever import SimilarityRetriever
from classification_backend.dose_response.concentration_profiles import TypedExposure, ConcentrationType


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "classification_backend" / "evaluation" / "assessment_evaluation_report.md"
REFERENCE_PATH = ROOT / "classification_backend" / "dataset" / "similarity_reference.csv"


def run_evaluation() -> str:
    database = ReferenceDatabase.from_csv(REFERENCE_PATH)
    retriever = SimilarityRetriever(database)
    similarity = retriever.search("CCO", top_k=5, minimum_similarity=0.4)

    fixed_cases = [(100.0, 100.0), (1000.0, 100.0), (10.0, 1000.0)]
    dose_rows = []
    for ic50_nm, concentration_nm in fixed_cases:
        block = ChannelBlockGenerator(
            {
                "herg_ic50_nm": ic50_nm,
                "nav_ic50_nm": ic50_nm,
                "cav_ic50_nm": ic50_nm,
            }
        ).to_ord_payload(concentration_nm)["herg_block"]
        dose_rows.append((ic50_nm, concentration_nm, block))

    exposure = TypedExposure(100.0, ConcentrationType.FREE_PLASMA)
    risk = MechanisticRiskAssessor().assess(
        {"herg": 1000.0, "nav": 5000.0, "cav": 2000.0},
        {"herg": 9.090909090909092, "nav": 1.9607843137254901, "cav": 4.761904761904762},
        exposure,
    )
    lines = [
        "# Assessment Evaluation",
        "",
        "This is the single active evaluation entry point for the unified assessment module.",
        "The legacy synthetic classifier is not evaluated as authoritative evidence.",
        "",
        "## Similarity",
        "",
        f"- Reference records: {len(database.records)}",
        f"- Query status: `{similarity.status}`",
        f"- Query matches: {len(similarity.matches)}",
        "",
        "## Dose Response",
        "",
        "| IC50 (nM) | Concentration (nM) | hERG block (%) |",
        "|---:|---:|---:|",
    ]
    lines.extend(f"| {ic50:.1f} | {concentration:.1f} | {block:.10f} |" for ic50, concentration, block in dose_rows)
    lines.extend(
        [
            "",
            "## Mechanistic Risk",
            "",
            f"- Level: `{risk.level}`",
            f"- Status: `{risk.status}`",
            f"- Concentration domain: `{risk.concentration_domain}`",
            f"- Dominant channel: `{risk.dominant_channel}`",
            "",
        ]
    )
    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return report


if __name__ == "__main__":
    print(run_evaluation())
    print(f"Saved report to: {REPORT_PATH}")
