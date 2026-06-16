from pathlib import Path

from classification_backend.dose_response import ChannelBlockGenerator, ChannelIC50Inputs
from prediction_backend.inference.predict import predict


REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_DIR = REPO_ROOT / "classification_backend" / "evaluation"
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_IC50_INPUTS = ChannelIC50Inputs(herg_ic50_nm=800.0, nav_ic50_nm=5000.0, cav_ic50_nm=2000.0)
FIXED_SERIES_CASES = [
    {"ic50_nm": 100.0, "concentration_nm": 100.0, "expected_block_pct": 50.0},
    {"ic50_nm": 1000.0, "concentration_nm": 100.0, "expected_block_pct": 9.090909090909092},
    {"ic50_nm": 10.0, "concentration_nm": 1000.0, "expected_block_pct": 99.00990099009901},
]


def _get_ic50_nm(channel_result: dict[str, float]) -> float:
    if "IC50_nM" in channel_result:
        return float(channel_result["IC50_nM"])
    if "IC50_NM" in channel_result:
        return float(channel_result["IC50_NM"])
    raise KeyError("IC50_nM")


def _print_fixed_series_table() -> None:
    print("Fixed concentration comparison")
    print("IC50 (nM) | Concentration (nM) | Expected block (%) | Actual block (%) | Reasonable")
    print("---|---:|---:|---:|---")

    for case in FIXED_SERIES_CASES:
        temp_generator = ChannelBlockGenerator(
            {
                "herg_ic50_nm": case["ic50_nm"],
                "nav_ic50_nm": case["ic50_nm"],
                "cav_ic50_nm": case["ic50_nm"],
            }
        )
        payload = temp_generator.to_ord_payload(case["concentration_nm"])
        actual_block = payload["herg_block"]
        reasonable = abs(actual_block - case["expected_block_pct"]) < 1e-9
        print(
            f"{case['ic50_nm']:.1f} | {case['concentration_nm']:.1f} | {case['expected_block_pct']:.4f} | {actual_block:.4f} | {'Yes' if reasonable else 'No'}"
        )
    print()


def _save_run_report(smiles: str, concentration_nm: float, preds: dict[str, dict[str, float]], payload: dict[str, float]) -> None:
    report_path = EVALUATION_DIR / "dose_response_predict_report.md"
    lines = [
        "# Dose Response Prediction Report",
        "",
        f"- SMILES: {smiles}",
        f"- Concentration (nM): {concentration_nm:.6g}",
        "",
        "## Predicted IC50",
        "",
        "| Channel | pIC50 | IC50 (nM) |",
        "|---|---:|---:|",
    ]
    for channel in ("herg", "nav", "cav"):
        lines.append(f"| {channel.upper()} | {preds[channel]['pIC50']:.4f} | {_get_ic50_nm(preds[channel]):.6g} |")
    lines.extend(
        [
            "",
            "## Dose Response Output",
            "",
            "| concentration | herg_block | nav_block | cav_block |",
            "|---:|---:|---:|---:|",
            f"| {payload['concentration']:.6g} | {payload['herg_block']:.6f} | {payload['nav_block']:.6f} | {payload['cav_block']:.6f} |",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _print_prediction_block(preds: dict[str, dict[str, float]]) -> None:
    print("\nPredicted IC50 Values\n")
    for task, values in preds.items():
        print(task.upper())
        print(f"pIC50   : {values['pIC50']:.4f}")
        print(f"IC50 nM : {_get_ic50_nm(values):.2e}\n")


def _print_dose_response_output(payload: dict[str, float]) -> None:
    print("Dose Response Output\n")
    print(f"Concentration : {payload['concentration']:.4f} nM")
    print(f"hERG block    : {payload['herg_block']:.4f}%")
    print(f"Nav1.5 block  : {payload['nav_block']:.4f}%")
    print(f"Cav1.2 block  : {payload['cav_block']:.4f}%")
    print("\n-----------------------------\n")


def run_test():
    print("\n===== Dose Response Test =====\n")
    print("This flow accepts a SMILES string, predicts IC50 values, then asks for a concentration in nM.")
    print("The final output is the ORd-ready dose-response payload.\n")

    _print_fixed_series_table()

    while True:
        smiles = input("Enter SMILES (or type 'exit'): ").strip()

        if smiles.lower() == "exit":
            break

        try:
            preds = predict(smiles)
        except Exception as exc:
            print("Prediction failed:", exc)
            continue

        try:
            concentration_value = input("Enter concentration in nM: ").strip()
            concentration_nm = float(concentration_value)
        except ValueError:
            print("Please enter a valid numeric concentration.\n")
            continue

        ic50_inputs = ChannelIC50Inputs(
            herg_ic50_nm=_get_ic50_nm(preds["herg"]),
            nav_ic50_nm=_get_ic50_nm(preds["nav"]),
            cav_ic50_nm=_get_ic50_nm(preds["cav"]),
        )
        generator = ChannelBlockGenerator(ic50_inputs)

        try:
            payload = generator.to_ord_payload(concentration_nm)
        except Exception as exc:
            print("Dose-response calculation failed:", exc)
            continue

        print("\nPredicted Values\n")
        _print_prediction_block(preds)
        _print_dose_response_output(payload)
        _save_run_report(smiles, concentration_nm, preds, payload)


def test_dose_response_prediction_schema():
    generator = ChannelBlockGenerator(DEFAULT_IC50_INPUTS)
    payload = generator.to_ord_payload(DEFAULT_IC50_INPUTS.herg_ic50_nm)

    assert set(payload.keys()) == {"concentration", "herg_block", "nav_block", "cav_block"}
    assert payload["concentration"] == DEFAULT_IC50_INPUTS.herg_ic50_nm
    assert payload["herg_block"] == 50.0


def test_fixed_series_examples_match_hill_equation():
    for case in FIXED_SERIES_CASES:
        generator = ChannelBlockGenerator(
            {
                "herg_ic50_nm": case["ic50_nm"],
                "nav_ic50_nm": case["ic50_nm"],
                "cav_ic50_nm": case["ic50_nm"],
            }
        )
        payload = generator.to_ord_payload(case["concentration_nm"])
        assert abs(payload["herg_block"] - case["expected_block_pct"]) < 1e-9


if __name__ == "__main__":
    run_test()