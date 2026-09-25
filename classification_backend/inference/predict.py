"""Interactive CLI for the complete classification-backend inference pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.prediction_pipeline import PredictionPipeline


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT_DIR / "classification_backend" / "dataset" / "classifier_dataset_labeled.csv"


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_reference_dataset() -> pd.DataFrame:
    dataset = pd.read_csv(DATASET_PATH)
    dataset["smiles"] = dataset["smiles"].astype(str)
    return dataset


def find_reference_row(dataset: pd.DataFrame, smiles: str, dose_nm: float) -> pd.Series | None:
    rows = dataset[dataset["smiles"] == smiles]
    if rows.empty:
        return None
    return rows.loc[(rows["dose_nm"] - dose_nm).abs().idxmin()]


def print_regression(result: dict[str, Any], reference: pd.Series | None) -> None:
    print_section("Regression Output")
    prediction = result["ic50_prediction"]
    predicted = {
        "IKr": prediction["herg"]["IC50_nM"],
        "INa": prediction["nav"]["IC50_nM"],
        "ICaL": prediction["cav"]["IC50_nM"],
    }
    for channel, value in predicted.items():
        print(f"{channel:<5}: {value:.2f} nM")

    print_section("IC50 Predicted vs Output")
    if reference is None:
        print("No reference output found for this SMILES.")
        return
    observed = {
        "IKr": reference["IC50_IKr"],
        "INa": reference["IC50_INa"],
        "ICaL": reference["IC50_ICaL"],
    }
    for channel, value in predicted.items():
        print(
            f"{channel:<5}: predicted={value:.2f} nM, "
            f"output={observed[channel]:.2f} nM, "
            f"error={abs(value - observed[channel]):.2f} nM"
        )


def print_dosage(result: dict[str, Any]) -> None:
    print_section("Dosage")
    dosage = result["input"]
    print(f"Concentration : {dosage['concentration_nm']:.2f} nM")
    print(f"Type          : {dosage['concentration_type']}")


def print_hill_equation(result: dict[str, Any]) -> None:
    print_section("Hill Equation")
    response = result["dose_response"]
    print(f"Concentration : {response['concentration_nm']:.2f} nM")
    print(f"hERG block    : {response['herg_block']:.2f}%")
    print(f"Nav block     : {response['nav_block']:.2f}%")
    print(f"Cav block     : {response['cav_block']:.2f}%")


def print_channel_block(result: dict[str, Any]) -> None:
    print_section("Channel Block")
    response = result["dose_response"]
    print(f"hERG          : {response['herg_block']:.2f}%")
    print(f"Nav1.5        : {response['nav_block']:.2f}%")
    print(f"Cav1.2        : {response['cav_block']:.2f}%")


def print_safety_margin(result: dict[str, Any]) -> None:
    print_section("Safety Margin")
    for margin in result["safety_margins"]:
        print(
            f"{margin['channel']:<8}: {margin['margin']:.4f} "
            f"({margin['risk_class']})"
        )


def print_ord(result: dict[str, Any]) -> None:
    print_section("ORd Outputs")
    simulation = result["simulation"]
    print(f"Status: {simulation['status']}")
    if simulation["status"] == "complete":
        for name, value in simulation["features"].items():
            print(f"{name:<15}: {value:.4f}")
    elif simulation.get("error"):
        print(f"Reason: {simulation['error']}")


def print_artifacts(result: dict[str, Any]) -> None:
    print_section("Generated Results")
    for category, artifacts in result.get("artifacts", {}).items():
        if not artifacts:
            continue
        print(f"{category}:")
        for name, path in artifacts.items():
            print(f"  {name}: {path}")


def print_classification(result: dict[str, Any]) -> None:
    print_section("Classification Result")
    print(f"Class           : {result['mechanistic_classification']}")
    risk = result["mechanistic_risk"]
    print(f"Risk level      : {risk.get('level', 'unavailable')}")
    print(f"Dominant channel: {risk.get('dominant_channel', 'unavailable')}")


def print_interpretation(result: dict[str, Any]) -> None:
    print_section("Interpretation")
    interpretation = result["interpretation"]
    level = interpretation.get("level", "unknown")
    risk = result["mechanistic_risk"]
    dominant = risk.get("dominant_channel", "the measured channels")
    margins = result.get("safety_margins", [])
    matches = result.get("similarity", {}).get("matches", [])

    if level == "low_mechanistic_concern":
        print(
            f"The predicted concern is low. The strongest signal comes from {dominant}, "
            "but the measured channel block is low at this dose."
        )
    elif level == "moderate_concern":
        print(
            f"The predicted concern is moderate. The results show meaningful activity "
            f"at {dominant}, so this compound should be reviewed further."
        )
    elif level in {"high_concern", "review_required"}:
        print(
            f"The results require review. The strongest signal is associated with {dominant} "
            "or the available evidence is conflicting."
        )
    else:
        print("There is not enough evidence to make a reliable interpretation.")

    if margins:
        safe_count = sum(margin.get("risk_class") == "Safe" for margin in margins)
        print(f"Safety margin summary: {safe_count} of {len(margins)} channels are in the safe range.")
    if matches:
        print(f"Similarity summary: {len(matches)} related compounds were found for comparison.")
    else:
        print("Similarity summary: no sufficiently similar reference compounds were found.")
    if interpretation.get("requires_review"):
        print("Follow-up: expert review is recommended.")


def print_similarity(result: dict[str, Any]) -> None:
    print_section("Similarity Result")
    similarity = result["similarity"]
    print(f"Status: {similarity['status']}")
    print(f"Method: {similarity.get('method', 'unavailable')}")
    for index, match in enumerate(similarity.get("matches", []), start=1):
        name = match["name_or_id"] or match["reference_id"]
        print(f"{index}. {name} ({match['similarity']:.4f})")


def print_xai(result: dict[str, Any], key: str) -> None:
    title = "Chemical XAI Result" if key == "chemical" else "Classification XAI Result"
    print_section(title)
    xai = result["xai"][key]
    print(f"Status: {xai['status']}")
    if xai["status"] == "complete":
        explanation = xai["result"]
        if key == "chemical":
            print(f"Substructures: {len(explanation.get('substructures', []))}")
            print("Explanation:")
            for item in explanation.get("explanation", []):
                print(f"  - {item.get('message', 'No explanation available.')}")
            print(f"SVG          : {explanation.get('svg_path', 'unavailable')}")
        else:
            print(f"Prediction   : {explanation['prediction']}")
            print(f"Confidence   : {explanation['confidence']:.4f}")
            print(f"Top features : {explanation.get('top_features', [])}")
            print(f"Report       : {explanation.get('report', 'unavailable')}")
    elif xai.get("error"):
        print(f"Reason: {xai['error']}")


def print_warnings(result: dict[str, Any]) -> None:
    warnings = result.get("warnings", [])
    if warnings:
        print_section("Warnings")
        for warning in warnings:
            print(f"- {warning}")


def main() -> None:
    dataset = load_reference_dataset()
    pipeline = PredictionPipeline()

    print("=" * 60)
    print("Cardiotoxicity Classification Predictor")
    print("=" * 60)

    while True:
        smiles = input("\nEnter SMILES (or exit): ").strip()
        if smiles.lower() == "exit":
            break
        try:
            dose_nm = float(input("Enter Dose (nM): "))
        except ValueError:
            print("Invalid dose.")
            continue

        try:
            result = pipeline.run({"smiles": smiles, "dose_nm": dose_nm})
        except Exception as exc:
            print(f"\nPrediction failed: {exc}")
            continue

        reference = find_reference_row(dataset, smiles, dose_nm)
        print_regression(result, reference)
        print_dosage(result)
        print_hill_equation(result)
        print_xai(result, key="chemical")
        print_channel_block(result)
        print_safety_margin(result)
        print_classification(result)
        print_similarity(result)
        print_interpretation(result)
        print_xai(result, key="classification")
        print_ord(result)
        print_artifacts(result)
        print_warnings(result)

    print("\nExiting Predictor...")


if __name__ == "__main__":
    main()
