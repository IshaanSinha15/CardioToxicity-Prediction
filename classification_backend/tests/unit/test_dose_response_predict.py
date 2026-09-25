from pathlib import Path

from classification_backend.dose_response import (
    ChannelBlockGenerator,
    ChannelIC50Inputs,
    plot_dose_response_curves,
)
from prediction_backend.inference.predict import predict


# ==========================================================
# Paths
# ==========================================================

REPO_ROOT = Path(__file__).resolve().parents[3]

EVALUATION_DIR = (
    REPO_ROOT
    / "classification_backend"
    / "evaluation"
)

PLOT_DIR = EVALUATION_DIR / "plots"

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# Fixed Hill Equation Test Cases
# ==========================================================

FIXED_SERIES_CASES = [
    {
        "ic50_nm": 100.0,
        "concentration_nm": 100.0,
        "expected_block_pct": 50.0,
    },
    {
        "ic50_nm": 1000.0,
        "concentration_nm": 100.0,
        "expected_block_pct": 9.090909090909092,
    },
    {
        "ic50_nm": 10.0,
        "concentration_nm": 1000.0,
        "expected_block_pct": 99.00990099009901,
    },
]


# ==========================================================
# Helpers
# ==========================================================

def _get_ic50_nm(
    channel_result: dict[str, float],
) -> float:

    if "IC50_nM" in channel_result:
        return float(
            channel_result["IC50_nM"]
        )

    if "IC50_NM" in channel_result:
        return float(
            channel_result["IC50_NM"]
        )

    raise KeyError("IC50_nM")


# ==========================================================
# Fixed Hill Equation Validation
# ==========================================================

def _print_fixed_series_table() -> None:

    print("Fixed concentration comparison")

    print(
        "IC50 (nM) | Concentration (nM) | "
        "Expected block (%) | Actual block (%) | Reasonable"
    )

    print(
        "---|---:|---:|---:|---"
    )

    for case in FIXED_SERIES_CASES:

        temp_generator = ChannelBlockGenerator(
            {
                "herg_ic50_nm": case["ic50_nm"],
                "nav_ic50_nm": case["ic50_nm"],
                "cav_ic50_nm": case["ic50_nm"],
            }
        )

        payload = temp_generator.to_ord_payload(
            case["concentration_nm"]
        )

        actual_block = float(
            payload["herg_block"]
        )

        reasonable = (
            abs(
                actual_block
                - case["expected_block_pct"]
            )
            < 1e-9
        )

        print(
            f"{case['ic50_nm']:.1f} | "
            f"{case['concentration_nm']:.1f} | "
            f"{case['expected_block_pct']:.4f} | "
            f"{actual_block:.4f} | "
            f"{'Yes' if reasonable else 'No'}"
        )

    print()


# ==========================================================
# Dose Response Curve
# ==========================================================

def _write_curve_plot(
    predicted_ic50_nm: dict[str, float],
    reference_concentration_nm: float,
    smiles: str,
) -> Path:

    plot_path = (
        PLOT_DIR
        / "dose_response_curve.png"
    )

    curves = {
        "hERG": {
            "ic50_nm": predicted_ic50_nm["herg"],
            "hill_coefficient": 1.0,
        },

        "Nav1.5": {
            "ic50_nm": predicted_ic50_nm["nav"],
            "hill_coefficient": 1.0,
        },

        "Cav1.2": {
            "ic50_nm": predicted_ic50_nm["cav"],
            "hill_coefficient": 1.0,
        },
    }

    fig, _ = plot_dose_response_curves(
        curves,
        reference_concentration_nm=reference_concentration_nm,
        save_path=plot_path,
    )

    if hasattr(fig, "close"):
        fig.close()

    return plot_path


# ==========================================================
# Prediction Output
# ==========================================================

def _print_prediction_block(
    preds: dict[str, dict[str, float]],
) -> None:

    print("\nPredicted IC50 Values\n")

    for task, values in preds.items():

        print(task.upper())

        print(
            f"pIC50   : "
            f"{values['pIC50']:.4f}"
        )

        print(
            f"IC50 nM : "
            f"{_get_ic50_nm(values):.2e}\n"
        )


# ==========================================================
# Dose Response Output
# ==========================================================

def _print_dose_response_output(
    payload: dict[str, float],
) -> None:

    print("Dose Response Output\n")

    print(
        f"Concentration : "
        f"{payload['concentration']:.4f} nM"
    )

    print(
        f"hERG block    : "
        f"{payload['herg_block']:.4f}%"
    )

    print(
        f"Nav1.5 block  : "
        f"{payload['nav_block']:.4f}%"
    )

    print(
        f"Cav1.2 block  : "
        f"{payload['cav_block']:.4f}%"
    )

    print(
        "\n-----------------------------\n"
    )


# ==========================================================
# Run Interactive Test
# ==========================================================

def run_test():

    print(
        "\n===== Dose Response Test =====\n"
    )

    # ------------------------------------------------------
    # Interactive Prediction
    # ------------------------------------------------------

    while True:

        smiles = input(
            "Enter SMILES (or type 'exit'): "
        ).strip()

        if smiles.lower() == "exit":
            break

        # --------------------------------------------------
        # Regression Prediction
        # --------------------------------------------------

        try:

            preds = predict(smiles)

        except Exception as exc:

            print(
                "Prediction failed:",
                exc,
            )

            continue

        # --------------------------------------------------
        # Concentration Input
        # --------------------------------------------------

        try:

            concentration_value = input(
                "Enter concentration in nM: "
            ).strip()

            concentration_nm = float(
                concentration_value
            )

        except ValueError:

            print(
                "Please enter a valid numeric "
                "concentration.\n"
            )

            continue

        # --------------------------------------------------
        # Get Predicted IC50 Values
        # --------------------------------------------------

        ic50_inputs = ChannelIC50Inputs(

            herg_ic50_nm=_get_ic50_nm(
                preds["herg"]
            ),

            nav_ic50_nm=_get_ic50_nm(
                preds["nav"]
            ),

            cav_ic50_nm=_get_ic50_nm(
                preds["cav"]
            ),
        )

        # --------------------------------------------------
        # Generate Dose Response
        # --------------------------------------------------

        generator = ChannelBlockGenerator(
            ic50_inputs
        )

        try:

            payload = generator.to_ord_payload(
                concentration_nm
            )

        except Exception as exc:

            print(
                "Dose-response calculation failed:",
                exc,
            )

            continue

        # --------------------------------------------------
        # Plot Dose Response Curves
        # --------------------------------------------------

        predicted_ic50_nm = {

            "herg": _get_ic50_nm(
                preds["herg"]
            ),

            "nav": _get_ic50_nm(
                preds["nav"]
            ),

            "cav": _get_ic50_nm(
                preds["cav"]
            ),
        }

        try:

            plot_path = _write_curve_plot(
                predicted_ic50_nm,
                reference_concentration_nm=concentration_nm,
                smiles=smiles,
            )

        except Exception as exc:

            print(
                "Dose-response plot generation failed:",
                exc,
            )

            plot_path = None

        # --------------------------------------------------
        # Output
        # --------------------------------------------------

        print("\nPredicted Values\n")

        _print_prediction_block(
            preds
        )

        _print_dose_response_output(
            payload
        )

        if plot_path is not None:

            print(
                f"Dose-response plot saved to: "
                f"{plot_path}"
            )


# ==========================================================
# Tests
# ==========================================================

def test_dose_response_prediction_schema():

    test_inputs = ChannelIC50Inputs(
        herg_ic50_nm=800.0,
        nav_ic50_nm=5000.0,
        cav_ic50_nm=2000.0,
    )

    generator = ChannelBlockGenerator(
        test_inputs
    )

    payload = generator.to_ord_payload(
        test_inputs.herg_ic50_nm
    )

    assert set(payload.keys()) == {
        "concentration",
        "herg_block",
        "nav_block",
        "cav_block",
    }

    assert (
        payload["concentration"]
        == test_inputs.herg_ic50_nm
    )

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

        payload = generator.to_ord_payload(
            case["concentration_nm"]
        )

        assert (
            abs(
                payload["herg_block"]
                - case["expected_block_pct"]
            )
            < 1e-9
        )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    run_test()