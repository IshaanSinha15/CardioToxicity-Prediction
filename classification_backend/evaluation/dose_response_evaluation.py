from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from classification_backend.dose_response import ChannelBlockGenerator, ChannelIC50Inputs, plot_dose_response_curves


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = REPO_ROOT / "classification_backend" / "evaluation"
PLOT_DIR = EVALUATION_DIR / "plots"

DEFAULT_IC50_INPUTS = ChannelIC50Inputs(herg_ic50_nm=800.0, nav_ic50_nm=5000.0, cav_ic50_nm=2000.0)
FIXED_SERIES_CASES = [
    {"ic50_nm": 100.0, "concentration_nm": 100.0, "expected_block_pct": 50.0},
    {"ic50_nm": 1000.0, "concentration_nm": 100.0, "expected_block_pct": 9.090909090909092},
    {"ic50_nm": 10.0, "concentration_nm": 1000.0, "expected_block_pct": 99.00990099009901},
]


@dataclass(frozen=True)
class DoseResponseEvaluationRow:
    ic50_nm: float
    concentration_nm: float
    expected_block_pct: float
    actual_block_pct: float
    reasonable: bool


def _ensure_output_dirs() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _evaluate_fixed_series() -> list[DoseResponseEvaluationRow]:
    rows: list[DoseResponseEvaluationRow] = []
    for case in FIXED_SERIES_CASES:
        generator = ChannelBlockGenerator(
            {
                "herg_ic50_nm": case["ic50_nm"],
                "nav_ic50_nm": case["ic50_nm"],
                "cav_ic50_nm": case["ic50_nm"],
            }
        )
        payload = generator.to_ord_payload(case["concentration_nm"])
        actual_block = float(payload["herg_block"])
        rows.append(
            DoseResponseEvaluationRow(
                ic50_nm=float(case["ic50_nm"]),
                concentration_nm=float(case["concentration_nm"]),
                expected_block_pct=float(case["expected_block_pct"]),
                actual_block_pct=actual_block,
                reasonable=abs(actual_block - float(case["expected_block_pct"])) < 1e-9,
            )
        )
    return rows


def _write_csv(rows: list[DoseResponseEvaluationRow]) -> Path:
    csv_path = EVALUATION_DIR / "dose_response_evaluation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ic50_nm", "concentration_nm", "expected_block_pct", "actual_block_pct", "reasonable"])
        for row in rows:
            writer.writerow([row.ic50_nm, row.concentration_nm, row.expected_block_pct, row.actual_block_pct, row.reasonable])
    return csv_path


def _write_markdown(rows: list[DoseResponseEvaluationRow]) -> Path:
    md_path = EVALUATION_DIR / "dose_response_evaluation_report.md"
    lines = [
        "# Dose Response Evaluation",
        "",
        "| IC50 (nM) | Concentration (nM) | Expected block (%) | Actual block (%) | Reasonable |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.ic50_nm:.4f} | {row.concentration_nm:.4f} | {row.expected_block_pct:.4f} | {row.actual_block_pct:.4f} | {'Yes' if row.reasonable else 'No'} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def _write_curve_plot(predicted_ic50_nm: dict[str, float], reference_concentration_nm: float) -> Path:
    plot_path = PLOT_DIR / "dose_response_curve.png"
    curves = {
        "hERG": {"ic50_nm": predicted_ic50_nm["herg"], "hill_coefficient": 1.0},
        "Nav1.5": {"ic50_nm": predicted_ic50_nm["nav"], "hill_coefficient": 1.0},
        "Cav1.2": {"ic50_nm": predicted_ic50_nm["cav"], "hill_coefficient": 1.0},
    }
    fig, _ = plot_dose_response_curves(curves, reference_concentration_nm=reference_concentration_nm, save_path=plot_path)
    fig.close() if hasattr(fig, "close") else None
    return plot_path


def run_evaluation() -> None:
    _ensure_output_dirs()

    print("\n===== Dose Response Evaluation =====\n")
    print("Using default IC50 inputs:")
    print(f"hERG  : {DEFAULT_IC50_INPUTS.herg_ic50_nm:.4f} nM")
    print(f"Nav1.5: {DEFAULT_IC50_INPUTS.nav_ic50_nm:.4f} nM")
    print(f"Cav1.2: {DEFAULT_IC50_INPUTS.cav_ic50_nm:.4f} nM\n")

    rows = _evaluate_fixed_series()

    print("Fixed concentration comparison")
    print("IC50 (nM) | Concentration (nM) | Expected block (%) | Actual block (%) | Reasonable")
    print("---|---:|---:|---:|---")
    for row in rows:
        print(
            f"{row.ic50_nm:.1f} | {row.concentration_nm:.1f} | {row.expected_block_pct:.4f} | {row.actual_block_pct:.4f} | {'Yes' if row.reasonable else 'No'}"
        )

    csv_path = _write_csv(rows)
    md_path = _write_markdown(rows)

    default_curve_path = _write_curve_plot(
        {"herg": DEFAULT_IC50_INPUTS.herg_ic50_nm, "nav": DEFAULT_IC50_INPUTS.nav_ic50_nm, "cav": DEFAULT_IC50_INPUTS.cav_ic50_nm},
        reference_concentration_nm=FIXED_SERIES_CASES[1]["concentration_nm"],
    )

    print(f"\nSaved evaluation CSV to: {csv_path}")
    print(f"Saved evaluation report to: {md_path}")
    print(f"Saved curve plot to: {default_curve_path}")
    print(f"Evaluation folder: {EVALUATION_DIR}\n")


if __name__ == "__main__":
    run_evaluation()