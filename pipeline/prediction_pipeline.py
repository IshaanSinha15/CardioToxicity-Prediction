"""End-to-end regression, classification, simulation, similarity, and XAI pipeline."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Any

import pandas as pd

from classification_backend.assessment.similarity.retriever import SimilarityRetriever
from classification_backend.classification_service import ClassificationEvidenceService
from prediction_backend.inference.predict import predict as predict_ic50

from .utils import PipelineError, PipelineInput, PipelineResult


CLASSIFICATION_XAI_FEATURES = (
    "RMP", "Peak", "APD50", "APD90", "Triangulation", "APA",
    "Block_IKr", "Block_INa", "Block_INaL", "Block_ICaL",
    "Block_IKs", "Block_IK1", "Block_Ito", "IC50_IKr",
    "IC50_INa", "IC50_ICaL",
)
RESULTS_DIR = Path(__file__).resolve().parents[1] / "classification_backend" / "inference" / "results"
REFERENCE_DATASET = Path(__file__).resolve().parents[1] / "classification_backend" / "dataset" / "classifier_dataset_labeled.csv"


class PredictionPipeline:
    """Run the complete classification-backend inference workflow."""

    def __init__(
        self,
        similarity_retriever: SimilarityRetriever | None = None,
        run_simulation: bool = True,
        run_xai: bool = True,
        output_dir: str | Path | None = None,
    ) -> None:
        self.run_simulation = run_simulation
        self.run_xai = run_xai
        self.results_dir = Path(output_dir) if output_dir else RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_retriever = similarity_retriever or self._load_similarity_retriever()
        self.evidence_service = ClassificationEvidenceService(self.similarity_retriever)

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_similarity_retriever() -> SimilarityRetriever | None:
        reference_path = (
            Path(__file__).resolve().parents[1]
            / "classification_backend"
            / "dataset"
            / "similarity_reference.csv"
        )
        if not reference_path.is_file():
            return None
        try:
            return SimilarityRetriever.build(str(reference_path))
        except Exception:
            return None

    @staticmethod
    def _validate_input(payload: PipelineInput) -> PipelineInput:
        if not isinstance(payload, dict):
            raise PipelineError("Pipeline input must be a mapping.")

        smiles = payload.get("smiles")
        if not isinstance(smiles, str) or not smiles.strip():
            raise PipelineError("SMILES must be a non-empty string.")

        if "concentration_nm" not in payload and "dose_nm" not in payload and "dose_mg" not in payload:
            raise PipelineError("Missing concentration_nm, dose_nm, or dose_mg.")

        validated = dict(payload)
        validated["smiles"] = smiles.strip()
        for field in ("concentration_nm", "dose_nm", "dose_mg"):
            if field in validated:
                try:
                    validated[field] = float(validated[field])
                except (TypeError, ValueError) as exc:
                    raise PipelineError(f"{field} must be numeric.") from exc
        return validated

    @staticmethod
    def _ic50_values(prediction: dict[str, Any]) -> dict[str, float]:
        try:
            return {
                "herg_ic50_nm": float(prediction["herg"]["IC50_nM"]),
                "nav_ic50_nm": float(prediction["nav"]["IC50_nM"]),
                "cav_ic50_nm": float(prediction["cav"]["IC50_nM"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineError("Regression output does not contain valid channel IC50 values.") from exc

    @staticmethod
    def _reference_comparison(
        smiles: str,
        concentration_nm: float,
        ic50_prediction: dict[str, Any],
    ) -> dict[str, Any]:
        channels = {
            "IKr": ("herg", "IC50_IKr"),
            "INa": ("nav", "IC50_INa"),
            "ICaL": ("cav", "IC50_ICaL"),
        }
        comparison: dict[str, Any] = {
            "available": False,
            "smiles": smiles,
            "requested_concentration_nm": concentration_nm,
            "channels": [],
        }
        if not REFERENCE_DATASET.is_file():
            return comparison

        dataset = pd.read_csv(REFERENCE_DATASET)
        rows = dataset[dataset["smiles"].astype(str) == smiles]
        if rows.empty:
            return comparison

        row = rows.loc[(rows["dose_nm"] - concentration_nm).abs().idxmin()]
        comparison["available"] = True
        comparison["reference_dose_nm"] = float(row["dose_nm"])
        for channel, (prediction_key, output_key) in channels.items():
            predicted = float(ic50_prediction[prediction_key]["IC50_nM"])
            observed = float(row[output_key])
            comparison["channels"].append({
                "channel": channel,
                "predicted_ic50_nm": predicted,
                "output_ic50_nm": observed,
                "absolute_error_nm": abs(predicted - observed),
            })
        return comparison

    @staticmethod
    def _plain_english_interpretation(
        interpretation: dict[str, Any],
        mechanistic_risk: dict[str, Any],
        safety_margins: list[dict[str, Any]],
        similarity: dict[str, Any],
    ) -> dict[str, Any]:
        level = interpretation.get("level", "unknown")
        dominant = mechanistic_risk.get("dominant_channel", "the measured channels")
        if level == "low_mechanistic_concern":
            summary = (
                f"The predicted concern is low. The strongest signal comes from {dominant}, "
                "but the measured channel block is low at this dose."
            )
        elif level == "moderate_concern":
            summary = (
                f"The predicted concern is moderate. The results show meaningful activity "
                f"at {dominant}, so this compound should be reviewed further."
            )
        elif level in {"high_concern", "review_required"}:
            summary = (
                f"The results require review. The strongest signal is associated with {dominant} "
                "or the available evidence is conflicting."
            )
        else:
            summary = "There is not enough evidence to make a reliable interpretation."

        safe_count = sum(item.get("risk_class") == "Safe" for item in safety_margins)
        similarity_count = len(similarity.get("matches", []))
        details = [
            f"{safe_count} of {len(safety_margins)} channels are in the safe margin range.",
            (
                f"{similarity_count} related compounds were found for comparison."
                if similarity_count
                else "No sufficiently similar reference compounds were found for comparison."
            ),
        ]
        if interpretation.get("requires_review"):
            details.append("Expert review is recommended.")
        return {
            "summary": summary,
            "details": details,
            "requires_review": bool(interpretation.get("requires_review")),
        }

    def run(self, payload: PipelineInput) -> PipelineResult:
        validated = self._validate_input(payload)
        smiles = validated["smiles"]

        try:
            ic50_prediction = predict_ic50(smiles)
            ic50_values = self._ic50_values(ic50_prediction)
        except Exception as exc:
            raise PipelineError(f"IC50 prediction failed: {exc}") from exc

        exposure_input = {
            key: validated[key]
            for key in (
                "concentration_nm", "dose_nm", "dose_mg",
                "molecular_weight_g_mol", "volume_distribution_l",
                "bioavailability", "absorption_rate_per_h",
                "elimination_rate_per_h", "concentration_type",
                "exposure_context",
            )
            if key in validated
        }
        try:
            evidence = self.evidence_service.evaluate(
                smiles,
                exposure_input,
                ic50_values,
            )
        except Exception as exc:
            raise PipelineError(f"Classification and similarity assessment failed: {exc}") from exc

        simulation = self._run_ord(evidence["dose_response"], validated)
        dosage_artifacts = self._save_dosage_plots(ic50_values, evidence)
        xai = self._run_xai(smiles, ic50_prediction, evidence["dose_response"], simulation)
        concentration_nm = float(evidence["input"]["concentration_nm"])
        ic50_comparison = self._reference_comparison(
            smiles,
            concentration_nm,
            ic50_prediction,
        )
        interpretation_text = self._plain_english_interpretation(
            evidence["interpretation"],
            evidence["mechanistic_risk"],
            evidence["safety_margins"],
            evidence["similarity"],
        )
        warnings = list(evidence["warnings"])
        for name, result in xai.items():
            if result.get("status") != "complete":
                warnings.append(f"{name}_unavailable")

        classification_xai = xai["classification"].get("result", {})
        classification = {
            "predicted_class": classification_xai.get("prediction"),
            "probabilities": classification_xai.get("probabilities", {}),
            "raw_prediction": classification_xai.get("prediction"),
        }

        return {
            "input": {
                **evidence["input"],
                "drug_name": validated.get("drug_name"),
            },
            "ic50_prediction": ic50_prediction,
            "ic50_comparison": ic50_comparison,
            "dose_response": evidence["dose_response"],
            "safety_margins": evidence["safety_margins"],
            "mechanistic_classification": evidence["mechanistic_classification"],
            "mechanistic_risk": evidence["mechanistic_risk"],
            "similarity": evidence["similarity"],
            "interpretation": evidence["interpretation"],
            "simulation": simulation,
            "artifacts": {
                "dosage": dosage_artifacts,
                "ord": simulation.get("artifacts", {}),
                "chemical_xai": xai["chemical"].get("artifacts", {}),
                "classification_xai": xai["classification"].get("artifacts", {}),
            },
            "xai": xai,
            "classification": classification,
            "features_used": list(CLASSIFICATION_XAI_FEATURES),
            "warnings": warnings,
            "frontend": {
                "input": {
                    "smiles": smiles,
                    "concentration_nm": concentration_nm,
                    "concentration_type": evidence["input"]["concentration_type"],
                },
                "regression": {
                    "ic50_prediction": ic50_prediction,
                    "ic50_comparison": ic50_comparison,
                },
                "dosage": evidence["dose_response"],
                "compound_xai": xai["chemical"],
                "channel_block": evidence["dose_response"],
                "safety_margin": evidence["safety_margins"],
                "classification": evidence["mechanistic_classification"],
                "similarity": evidence["similarity"],
                "interpretation": interpretation_text,
                "classification_xai": xai["classification"],
                "ord": simulation,
                "artifacts": {
                    "dosage": dosage_artifacts,
                    "ord": simulation.get("artifacts", {}),
                    "chemical_xai": xai["chemical"].get("artifacts", {}),
                    "classification_xai": xai["classification"].get("artifacts", {}),
                },
                "warnings": warnings,
            },
        }

    def _run_ord(self, dose_response: dict[str, Any], payload: PipelineInput) -> dict[str, Any]:
        if payload.get("skip_simulation", False) or not self.run_simulation:
            return {"status": "skipped", "features": {}}
        try:
            from classification_backend.feature_extraction.ap_features import APFeatureExtractor
            from classification_backend.simulation.ord_simulator import ORDSimulator

            simulator = ORDSimulator()
            simulator.apply_channel_blocks(
                ikr=float(dose_response["herg_block"]),
                ina=float(dose_response["nav_block"]),
                ical=float(dose_response["cav_block"]),
            )
            data = simulator.run()
            import matplotlib.pyplot as plt

            ord_plot = self.results_dir / "ord_voltage.png"
            figure, axis = plt.subplots(figsize=(10, 4), dpi=150)
            axis.plot(data["environment.time"], data["membrane.v"], linewidth=1.0)
            axis.set_xlabel("Time (ms)")
            axis.set_ylabel("Membrane voltage (mV)")
            axis.set_title("ORd Action Potential")
            axis.grid(True, alpha=0.25)
            figure.tight_layout()
            figure.savefig(ord_plot, bbox_inches="tight")
            plt.close(figure)
            features = APFeatureExtractor(
                data["environment.time"],
                data["membrane.v"],
            ).extract_features()
            return {
                "status": "complete",
                "features": {key: float(value) for key, value in features.items()},
                "artifacts": {"voltage_plot": str(ord_plot.resolve())},
            }
        except Exception as exc:
            return {"status": "unavailable", "features": {}, "artifacts": {}, "error": str(exc)}

    def _save_dosage_plots(
        self,
        ic50_values: dict[str, float],
        evidence: dict[str, Any],
    ) -> dict[str, str]:
        try:
            import matplotlib.pyplot as plt
            from classification_backend.dose_response.dose_response_curve import (
                plot_channel_comparison,
                plot_dose_response_curves,
                plot_safety_margin_bars,
            )

            curves_path = self.results_dir / "dose_response_curves.png"
            plot_dose_response_curves(
                {
                    "hERG": {"ic50_nm": ic50_values["herg_ic50_nm"]},
                    "Nav1.5": {"ic50_nm": ic50_values["nav_ic50_nm"]},
                    "Cav1.2": {"ic50_nm": ic50_values["cav_ic50_nm"]},
                },
                reference_concentration_nm=evidence["dose_response"]["concentration_nm"],
                save_path=curves_path,
            )[0].clf()

            block_path = self.results_dir / "channel_block.png"
            block_frame = pd.DataFrame([{
                "concentration": evidence["dose_response"]["concentration_nm"],
                "herg_block": evidence["dose_response"]["herg_block"],
                "nav_block": evidence["dose_response"]["nav_block"],
                "cav_block": evidence["dose_response"]["cav_block"],
            }])
            plot_channel_comparison(block_frame, save_path=block_path)[0].clf()

            margin_path = self.results_dir / "safety_margin.png"
            margins = pd.DataFrame(evidence["safety_margins"])
            plot_safety_margin_bars(margins, save_path=margin_path)[0].clf()
            plt.close("all")
            return {
                "dose_response_curves": str(curves_path.resolve()),
                "channel_block": str(block_path.resolve()),
                "safety_margin": str(margin_path.resolve()),
            }
        except Exception as exc:
            return {"status": "unavailable", "error": str(exc)}

    def _run_xai(
        self,
        smiles: str,
        ic50_prediction: dict[str, Any],
        dose_response: dict[str, Any],
        simulation: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        if not self.run_xai:
            return {
                "chemical": {"status": "skipped"},
                "classification": {"status": "skipped"},
            }

        chemical = self._run_chemical_xai(smiles)
        classification = self._run_classification_xai(
            ic50_prediction,
            dose_response,
            simulation,
        )
        return {"chemical": chemical, "classification": classification}

    def _run_chemical_xai(self, smiles: str) -> dict[str, Any]:
        try:
            from classification_backend.xai.chemical_pipeline import ChemicalXAIPipeline

            result = ChemicalXAIPipeline(self.results_dir).explain(smiles)
            return {
                "status": "complete",
                "result": result,
                "artifacts": {"molecule_svg": result["svg_path"]},
            }
        except Exception as exc:
            return {"status": "unavailable", "error": str(exc)}

    def _run_classification_xai(
        self,
        ic50_prediction: dict[str, Any],
        dose_response: dict[str, Any],
        simulation: dict[str, Any],
    ) -> dict[str, Any]:
        if simulation["status"] != "complete":
            return {
                "status": "unavailable",
                "error": "Completed ORd features are required for classification XAI.",
            }

        try:
            from classification_backend.xai.run_Xai import XAIPipeline

            features = simulation["features"]
            feature_vector = [
                features["RMP"],
                features["Peak"],
                features["APD50"],
                features["APD90"],
                features["Triangulation"],
                features["Peak"] - features["RMP"],
            ] + [
                dose_response["herg_block"],
                dose_response["nav_block"],
                0.0,
                dose_response["cav_block"],
                0.0,
                0.0,
                0.0,
                ic50_prediction["herg"]["IC50_nM"],
                ic50_prediction["nav"]["IC50_nM"],
                ic50_prediction["cav"]["IC50_nM"],
            ]
            result = XAIPipeline(self.results_dir).explain(feature_vector)
            return {
                "status": "complete",
                "result": result,
                "artifacts": {
                    "bar_plot": result["bar_plot"],
                    "waterfall_plot": result["waterfall_plot"],
                    "report": result["report"],
                },
            }
        except Exception as exc:
            return {"status": "unavailable", "artifacts": {}, "error": str(exc)}
