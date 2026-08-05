import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pytest

from pipeline.prediction_pipeline import PredictionPipeline


OUTPUT_DIR = os.path.join("outputs")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def classification_label(pred_class: int):
    mapping = {
        1: ("Low", "Low blocking"),
        2: ("Moderate", "Moderate blocking"),
        3: ("High", "High blocking"),
        4: ("Very High", "Very high blocking"),
    }
    return mapping.get(pred_class, ("Unknown", "Unknown blocking"))


def run_and_report(smiles: str, dose_nm: float, output_dir: str, try_full_simulation: bool = True):
    """Run the pipeline end-to-end and produce images + markdown report.

    Returns (result_dict, md_file_path, summary_txt_path).
    """
    pipeline = PredictionPipeline()
    used_skip = False
    # Try full simulation first; if missing heavy deps, fall back to skip
    try:
        result = pipeline.run({"smiles": smiles, "dose_nm": dose_nm})
    except Exception as exc:
        msg = str(exc).lower()
        if try_full_simulation and ("myokit" in msg or "transformers" in msg or "ic50 prediction failed" in msg):
            result = pipeline.run({"smiles": smiles, "dose_nm": dose_nm, "skip_simulation": True})
            used_skip = True
        else:
            raise

    # Basic structural assertions
    assert "ic50_prediction" in result
    assert "dose_response" in result
    assert "classification" in result
    assert "features_used" in result

    # Plot dose-response curve by sweeping doses using ChannelBlockGenerator
    try:
        from classification_backend.dose_response.channel_block_generator import (
            ChannelBlockGenerator,
            ChannelIC50Inputs,
        )

        ic50s = result["ic50_prediction"]
        ic50_inputs = ChannelIC50Inputs(
            herg_ic50_nm=float(ic50s["herg"]["IC50_nM"]),
            nav_ic50_nm=float(ic50s["nav"]["IC50_nM"]),
            cav_ic50_nm=float(ic50s["cav"]["IC50_nM"]),
        )
        gen = ChannelBlockGenerator(ic50_inputs)
        doses = np.logspace(-2, 6, 200)
        herg_blocks = [gen.to_ord_payload(d)["herg_block"] for d in doses]
        nav_blocks = [gen.to_ord_payload(d)["nav_block"] for d in doses]
        cav_blocks = [gen.to_ord_payload(d)["cav_block"] for d in doses]

        plt.figure(figsize=(6, 4))
        plt.semilogx(doses, np.array(herg_blocks) * 100, label="hERG (IKr) % block")
        plt.semilogx(doses, np.array(nav_blocks) * 100, label="Nav (INa) % block")
        plt.semilogx(doses, np.array(cav_blocks) * 100, label="Cav (ICaL) % block")
        plt.xlabel("Concentration (nM)")
        plt.ylabel("Percent block (%)")
        plt.legend()
        plt.tight_layout()
        dose_plot = os.path.join(output_dir, "dose_response_curve.png")
        plt.savefig(dose_plot)
        plt.close()
    except Exception:
        dose_plot = None

    # Attempt ORd simulation waveform
    sim_plot = None
    try:
        from classification_backend.simulation.ord_simulator import ORDSimulator

        dose_payload = result.get("dose_response", {})
        herg_block = float(dose_payload.get("herg_block", 0.0))
        nav_block = float(dose_payload.get("nav_block", 0.0))
        cav_block = float(dose_payload.get("cav_block", 0.0))

        sim = ORDSimulator()
        sim.apply_channel_blocks(ikr=herg_block, ina=nav_block, inal=0.0, ical=cav_block, iks=0.0, ik1=0.0, ito=0.0)
        simres = sim.run()
        time = simres["environment.time"]
        voltage = simres["membrane.v"]

        plt.figure(figsize=(6, 3))
        plt.plot(time, voltage)
        plt.xlabel("time (ms)")
        plt.ylabel("membrane.v (mV)")
        plt.tight_layout()
        sim_plot = os.path.join(output_dir, "ord_simulation_voltage.png")
        plt.savefig(sim_plot)
        plt.close()
    except Exception:
        sim_plot = None

    # Write classification human-readable summary
    clf = result["classification"]
    pred_class = int(clf.get("predicted_class", clf.get("raw_prediction", -1)))
    label, desc = classification_label(pred_class)
    summary_txt = os.path.join(output_dir, "classification_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"input_smiles: {smiles}\n")
        f.write(f"input_dose_nm: {dose_nm}\n")
        f.write(f"predicted_class: {pred_class}\n")
        f.write(f"label: {label}\n")
        f.write(f"description: {desc}\n")
        f.write(f"probabilities: {json.dumps(clf.get('probabilities', {}))}\n")

    # Create Markdown report
    md_lines = []
    md_lines.append(f"# Pipeline Report\n")
    md_lines.append(f"## Input\n")
    md_lines.append(f"- SMILES: {smiles}\n")
    md_lines.append(f"- Dose (nM): {dose_nm}\n")
    md_lines.append("## IC50 Predictions\n")
    ic50s = result.get("ic50_prediction", {})
    if ic50s:
        md_lines.append("| Channel | pIC50 | IC50 (nM) |\n|---|---:|---:|\n")
        for ch in ("herg", "nav", "cav"):
            chd = ic50s.get(ch, {})
            md_lines.append(f"| {ch.upper()} | {chd.get('pIC50', 'NA')} | {chd.get('IC50_nM', 'NA')} |\n")
    else:
        md_lines.append("IC50 predictions not available.\n")

    md_lines.append("\n## Dose-response\n")
    if dose_plot:
        md_lines.append(f"![Dose-response]({os.path.basename(dose_plot)})\n")
    else:
        md_lines.append("Dose-response plot not generated.\n")

    md_lines.append("\n## ORd Simulation\n")
    if sim_plot:
        md_lines.append(f"![ORd voltage]({os.path.basename(sim_plot)})\n")
    else:
        md_lines.append("ORd voltage plot not generated.\n")

    md_lines.append("\n## Simulation Features\n")
    sim_features = result.get("simulation", {})
    for k, v in sim_features.items():
        md_lines.append(f"- **{k}**: {v}\n")

    md_lines.append("\n## Classification\n")
    md_lines.append(f"- Predicted class: **{pred_class}** ({label})\n")
    md_lines.append(f"- Description: {desc}\n")
    md_lines.append(f"- Probabilities: {json.dumps(clf.get('probabilities', {}))}\n")

    md_file = os.path.join(output_dir, "pipeline_report.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.writelines([l if l.endswith("\n") else l + "\n" for l in md_lines])

    # Print a concise terminal summary for interactive use
    print("\n--- Pipeline Summary ---")
    print(f"SMILES: {smiles}")
    print(f"Dose (nM): {dose_nm}")
    print(f"Predicted class: {pred_class} ({label})")
    print(f"Probabilities: {json.dumps(clf.get('probabilities', {}))}")
    print(f"Markdown report: {md_file}")
    if dose_plot:
        print(f"Dose-response image: {dose_plot}")
    if sim_plot:
        print(f"ORd simulation image: {sim_plot}")

    return result, md_file, summary_txt


def test_pipeline_end_to_end():
    """End-to-end pipeline test: IC50 -> dose-response -> ORd sim -> features -> classifier

    The test will attempt a full run (real simulation). If heavy deps are missing
    it will fall back to `skip_simulation` but will still verify structure and
    save outputs under `src/outputs/` for inspection.
    """
    ensure_output_dir()

    # Use helper to run the pipeline and generate outputs
    from typing import Tuple

    def run_and_report(smiles: str, dose_nm: float, output_dir: str, try_full_simulation: bool = True) -> Tuple[dict, str, str]:
        pipeline = PredictionPipeline()
        used_skip = False
        # Try full simulation first; if missing heavy deps, fall back to skip
        try:
            result = pipeline.run({"smiles": smiles, "dose_nm": dose_nm})
        except Exception as exc:
            msg = str(exc).lower()
            if try_full_simulation and ("myokit" in msg or "transformers" in msg or "ic50 prediction failed" in msg):
                result = pipeline.run({"smiles": smiles, "dose_nm": dose_nm, "skip_simulation": True})
                used_skip = True
            else:
                raise

        # Basic structural assertions
        assert "ic50_prediction" in result
        assert "dose_response" in result
        assert "classification" in result
        assert "features_used" in result

        # Plot dose-response curve by sweeping doses using ChannelBlockGenerator
        try:
            from classification_backend.dose_response.channel_block_generator import (
                ChannelBlockGenerator,
                ChannelIC50Inputs,
            )

            ic50s = result["ic50_prediction"]
            ic50_inputs = ChannelIC50Inputs(
                herg_ic50_nm=float(ic50s["herg"]["IC50_nM"]),
                nav_ic50_nm=float(ic50s["nav"]["IC50_nM"]),
                cav_ic50_nm=float(ic50s["cav"]["IC50_nM"]),
            )
            gen = ChannelBlockGenerator(ic50_inputs)
            doses = np.logspace(-2, 6, 200)
            herg_blocks = [gen.to_ord_payload(d)["herg_block"] for d in doses]
            nav_blocks = [gen.to_ord_payload(d)["nav_block"] for d in doses]
            cav_blocks = [gen.to_ord_payload(d)["cav_block"] for d in doses]

            plt.figure(figsize=(6, 4))
            plt.semilogx(doses, np.array(herg_blocks) * 100, label="hERG (IKr) % block")
            plt.semilogx(doses, np.array(nav_blocks) * 100, label="Nav (INa) % block")
            plt.semilogx(doses, np.array(cav_blocks) * 100, label="Cav (ICaL) % block")
            plt.xlabel("Concentration (nM)")
            plt.ylabel("Percent block (%)")
            plt.legend()
            plt.tight_layout()
            dose_plot = os.path.join(output_dir, "dose_response_curve.png")
            plt.savefig(dose_plot)
            plt.close()
        except Exception:
            dose_plot = None

        # Attempt ORd simulation waveform
        sim_plot = None
        try:
            from classification_backend.simulation.ord_simulator import ORDSimulator

            dose_payload = result.get("dose_response", {})
            herg_block = float(dose_payload.get("herg_block", 0.0))
            nav_block = float(dose_payload.get("nav_block", 0.0))
            cav_block = float(dose_payload.get("cav_block", 0.0))

            sim = ORDSimulator()
            sim.apply_channel_blocks(ikr=herg_block, ina=nav_block, inal=0.0, ical=cav_block, iks=0.0, ik1=0.0, ito=0.0)
            simres = sim.run()
            time = simres["environment.time"]
            voltage = simres["membrane.v"]

            plt.figure(figsize=(6, 3))
            plt.plot(time, voltage)
            plt.xlabel("time (ms)")
            plt.ylabel("membrane.v (mV)")
            plt.tight_layout()
            sim_plot = os.path.join(output_dir, "ord_simulation_voltage.png")
            plt.savefig(sim_plot)
            plt.close()
        except Exception:
            sim_plot = None

        # Write classification human-readable summary
        clf = result["classification"]
        pred_class = int(clf.get("predicted_class", clf.get("raw_prediction", -1)))
        label, desc = classification_label(pred_class)
        summary_txt = os.path.join(output_dir, "classification_summary.txt")
        with open(summary_txt, "w", encoding="utf-8") as f:
            f.write(f"input_smiles: {smiles}\n")
            f.write(f"input_dose_nm: {dose_nm}\n")
            f.write(f"predicted_class: {pred_class}\n")
            f.write(f"label: {label}\n")
            f.write(f"description: {desc}\n")
            f.write(f"probabilities: {json.dumps(clf.get('probabilities', {}))}\n")

        # Create Markdown report
        md_lines = []
        md_lines.append(f"# Pipeline Report\n")
        md_lines.append(f"## Input\n")
        md_lines.append(f"- SMILES: {smiles}\n")
        md_lines.append(f"- Dose (nM): {dose_nm}\n")
        md_lines.append("## IC50 Predictions\n")
        ic50s = result.get("ic50_prediction", {})
        if ic50s:
            md_lines.append("| Channel | pIC50 | IC50 (nM) |\n|---|---:|---:|\n")
            for ch in ("herg", "nav", "cav"):
                chd = ic50s.get(ch, {})
                md_lines.append(f"| {ch.upper()} | {chd.get('pIC50', 'NA')} | {chd.get('IC50_nM', 'NA')} |\n")
        else:
            md_lines.append("IC50 predictions not available.\n")

        md_lines.append("\n## Dose-response\n")
        if dose_plot:
            md_lines.append(f"![Dose-response]({os.path.basename(dose_plot)})\n")
        else:
            md_lines.append("Dose-response plot not generated.\n")

        md_lines.append("\n## ORd Simulation\n")
        if sim_plot:
            md_lines.append(f"![ORd voltage]({os.path.basename(sim_plot)})\n")
        else:
            md_lines.append("ORd voltage plot not generated.\n")

        md_lines.append("\n## Simulation Features\n")
        sim_features = result.get("simulation", {})
        for k, v in sim_features.items():
            md_lines.append(f"- **{k}**: {v}\n")

        md_lines.append("\n## Classification\n")
        md_lines.append(f"- Predicted class: **{pred_class}** ({label})\n")
        md_lines.append(f"- Description: {desc}\n")
        md_lines.append(f"- Probabilities: {json.dumps(clf.get('probabilities', {}))}\n")

        md_file = os.path.join(output_dir, "pipeline_report.md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.writelines([l if l.endswith("\n") else l + "\n" for l in md_lines])

        # Print a concise terminal summary for interactive use
        print("\n--- Pipeline Summary ---")
        print(f"SMILES: {smiles}")
        print(f"Dose (nM): {dose_nm}")
        print(f"Predicted class: {pred_class} ({label})")
        print(f"Probabilities: {json.dumps(clf.get('probabilities', {}))}")
        print(f"Markdown report: {md_file}")
        if dose_plot:
            print(f"Dose-response image: {dose_plot}")
        if sim_plot:
            print(f"ORd simulation image: {sim_plot}")

        return result, md_file, summary_txt

    # Run the pipeline once as the pytest assertion
    res, md_path, summary_txt = run_and_report("CCO", 100.0, OUTPUT_DIR)

    # Final assertions: ensure outputs were written and remove legacy JSON
    legacy_json = os.path.join(OUTPUT_DIR, "pipeline_result.json")
    if os.path.exists(legacy_json):
        try:
            os.remove(legacy_json)
        except Exception:
            pass

    assert os.path.exists(md_path)
    assert os.path.exists(summary_txt)


if __name__ == "__main__":
    # Interactive mode: prompt user for SMILES and dose, then run pipeline
    ensure_output_dir()
    print("Interactive pipeline runner — enter SMILES and dose (nM). Press Enter to use defaults.")
    in_smiles = input("SMILES [CCO]: ") or "CCO"
    in_dose = input("Dose (nM) [100]: ") or "100"
    try:
        dose_val = float(in_dose)
    except Exception:
        print("Invalid dose, using 100")
        dose_val = 100.0

    run_and_report(in_smiles, dose_val, OUTPUT_DIR)
