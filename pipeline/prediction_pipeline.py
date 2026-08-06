from pyexpat import features
from typing import Dict, Any

from .utils import PipelineInput, PipelineResult, PipelineError
import pandas as pd

from prediction_backend.inference.predict import predict as predict_ic50
from classification_backend.dose_response.channel_block_generator import ChannelBlockGenerator, ChannelIC50Inputs
import warnings
from typing import Optional
from .feature_builder import build_features_from_simulation
from .classifier import ClassifierService
from classification_backend.dose_response.hill_equation import HillEquation
from classification_backend.xai.run_xai import XAIPipeline

FEATURE_ORDER = [
    "RMP",
    "Peak",
    "APD50",
    "APD90",
    "Triangulation",
    "APA",
    "Block_IKr",
    "Block_INa",
    "Block_INaL",
    "Block_ICaL",
    "Block_IKs",
    "Block_IK1",
    "Block_Ito",
    "IC50_IKr",
    "IC50_INa",
    "IC50_ICaL",
]




class PredictionPipeline:
    def __init__(self, classifier_model_path: str = "saved_models/random_forest_classifier.pkl") -> None:
        self.classifier = ClassifierService(classifier_model_path)
        self.xai = XAIPipeline()
        
    def _validate_input(self, payload: PipelineInput) -> PipelineInput:
        if "smiles" not in payload or not payload["smiles"]:
            raise PipelineError("Missing SMILES string")
        try:
            dose = float(payload.get("dose_nm", 0.0))
        except Exception:
            raise PipelineError("Invalid dose_nm; must be numeric")
        return {"smiles": payload["smiles"], "dose_nm": dose, "drug_name": payload.get("drug_name")}

    def run(self, payload: Dict[str, Any]) -> PipelineResult:
        validated = self._validate_input(payload)

        smiles = validated["smiles"]
        dose_nm = float(validated["dose_nm"])

        # Step 1: IC50 prediction
        try:
            ic50_preds = predict_ic50(smiles)
        except Exception as exc:
            raise PipelineError(f"IC50 prediction failed: {exc}")

        # Step 2: Dose response
        try:
            ic50_inputs = ChannelIC50Inputs(
                herg_ic50_nm=float(ic50_preds["herg"]["IC50_nM"]),
                nav_ic50_nm=float(ic50_preds["nav"]["IC50_nM"]),
                cav_ic50_nm=float(ic50_preds["cav"]["IC50_nM"]),
            )
            generator = ChannelBlockGenerator(ic50_inputs)
            dose_payload = generator.to_ord_payload(dose_nm)
        except Exception as exc:
            raise PipelineError(f"Dose-response calculation failed: {exc}")

        # Step 3: ORd Simulation (run control and drug simulations)
        time = None
        voltage = None
        control_time = None
        control_voltage = None
        try:
            # Import simulator lazily so missing optional deps don't break package import
            from classification_backend.simulation.ord_simulator import ORDSimulator
            # Run control (no block)
            ctrl_sim = ORDSimulator()
            ctrl_result = ctrl_sim.run()
            control_time = ctrl_result["environment.time"]
            control_voltage = ctrl_result["membrane.v"]

            # Run drug simulation on a fresh simulator instance
            drug_sim = ORDSimulator()
            # Map blocks: herg -> IKr, nav -> INa, cav -> ICaL
            herg_block = float(dose_payload.get("herg_block", 0.0))
            nav_block = float(dose_payload.get("nav_block", 0.0))
            cav_block = float(dose_payload.get("cav_block", 0.0))

            drug_sim.apply_channel_blocks(
                ikr=herg_block,
                ina=nav_block,
                inal=0.0,
                ical=cav_block,
                iks=0.0,
                ik1=0.0,
                ito=0.0,
            )
            sim_result = drug_sim.run()
            time = sim_result["environment.time"]
            voltage = sim_result["membrane.v"]

        except ImportError as exc:
            # Optional dependency missing (myokit). Provide helpful message and
            # allow tests to run if caller set skip_simulation flag.
            if payload.get("skip_simulation"):
                warnings.warn("myokit not installed — using synthetic AP waveform for testing")
                import numpy as _np

                control_time = _np.linspace(0, 1000, 10001)
                control_voltage = -90 + 100 * _np.exp(-((control_time - 50) / 5) ** 2)

                # Create drug waveform by scaling amplitude using HillEquation conductance scaling
                herg_hill = HillEquation(ic50_nm=float(ic50_preds["herg"]["IC50_nM"]))
                nav_hill = HillEquation(ic50_nm=float(ic50_preds["nav"]["IC50_nM"]))
                cav_hill = HillEquation(ic50_nm=float(ic50_preds["cav"]["IC50_nM"]))

                herg_scale = herg_hill.conductance_scaling(dose_nm)
                nav_scale = nav_hill.conductance_scaling(dose_nm)
                cav_scale = cav_hill.conductance_scaling(dose_nm)

                avg_scale = float((herg_scale + nav_scale + cav_scale) / 3.0)

                time = control_time
                voltage = control_voltage * avg_scale
            else:
                raise PipelineError("ORd simulator dependency missing: myokit. Install myokit or run with skip_simulation=True")
        except Exception as exc:
            raise PipelineError(f"ORd simulation failed: {exc}")

        # Ensure block variables exist (set from dose_payload)
        herg_block = float(dose_payload.get("herg_block", 0.0))
        nav_block = float(dose_payload.get("nav_block", 0.0))
        cav_block = float(dose_payload.get("cav_block", 0.0))

        # Step 4: Feature construction
        try:
            block_mapping = {
                "IKr": herg_block,
                "INa": nav_block,
                "INaL": 0.0,
                "ICaL": cav_block,
                "IKs": 0.0,
                "IK1": 0.0,
                "Ito": 0.0,
            }

            # Drug features (from drug simulation)
            features_drug = build_features_from_simulation(time, voltage, block_mapping)

            # Control features (from control simulation) - fall back to zeros if control not available
            if control_time is not None and control_voltage is not None:
                features_control = build_features_from_simulation(control_time, control_voltage, {k: 0.0 for k in block_mapping})
            else:
                features_control = {k: 0.0 for k in features_drug.keys()}

            # Delta features (drug - control) for AP metrics
            delta_features = {}
            for k in ["RMP", "Peak", "APD50", "APD90", "Triangulation", "APA"]:
                delta_features[k] = float(features_drug[k] - features_control.get(k, 0.0))

            # Prepare final features dict (keeping Block_* as percent blocks)
            features = {
                "RMP": float(features_drug["RMP"]),
                "Peak": float(features_drug["Peak"]),
                "APD50": float(features_drug["APD50"]),
                "APD90": float(features_drug["APD90"]),
                "Triangulation": float(features_drug["Triangulation"]),
                "APA": float(features_drug["APA"]),

                "Block_IKr": float(block_mapping.get("IKr", 0.0)),
                "Block_INa": float(block_mapping.get("INa", 0.0)),
                "Block_INaL": float(block_mapping.get("INaL", 0.0)),
                "Block_ICaL": float(block_mapping.get("ICaL", 0.0)),
                "Block_IKs": float(block_mapping.get("IKs", 0.0)),
                "Block_IK1": float(block_mapping.get("IK1", 0.0)),
                "Block_Ito": float(block_mapping.get("Ito", 0.0)),

                # NEW
                "IC50_IKr": float(ic50_preds["herg"]["IC50_nM"]),
                "IC50_INa": float(ic50_preds["nav"]["IC50_nM"]),
                "IC50_ICaL": float(ic50_preds["cav"]["IC50_nM"]),
            }

            # attach delta features under a nested key for reporting
            features_df = pd.DataFrame([[features[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
        except Exception as exc:
            raise PipelineError(f"Feature construction failed: {exc}")

        feature_order = list(features_df.columns)
        
        feature_values = [
                float(features[name])
                for name in feature_order
                ]

        # Step 5: Classification
        try:
            classification = self.classifier.predict(features_df)
            xai_result = self.xai.explain(feature_values)

            # Add advisory when blocks are very small
            max_block = max(herg_block, nav_block, cav_block)
            if max_block < 5.0:
                classification["low_block_warning"] = True
                classification["low_block_message"] = "Maximum channel block <5% at tested dose — interpret classification with caution. Consider higher-dose or safety-margin features."
            else:
                classification["low_block_warning"] = False
        except Exception as exc:
            raise PipelineError(f"Classification failed: {exc}")

        # Build final output
        # -------------------------------------------------
        # Build XAI input
        # -------------------------------------------------

      

        xai_input = {
            "model_type": "RandomForestClassifier",
            "feature_names": feature_order,
            "feature_values": feature_values,
            "feature_dataframe": features_df.to_dict(orient="records")[0],
        }

        # -------------------------------------------------
        # Final Result
        # -------------------------------------------------

        result: PipelineResult = {

            "input": {
                "smiles": smiles,
                "dose_nm": dose_nm,
                "drug_name": validated.get("drug_name"),
            },

            "ic50_prediction": ic50_preds,

            "dose_response": dose_payload,

            "simulation": {
                "RMP": features["RMP"],
                "Peak": features["Peak"],
                "APD50": features["APD50"],
                "APD90": features["APD90"],
                "Triangulation": features["Triangulation"],
                "APA": features["APA"],
            },

            "classification": classification,

            "features_used": features,

            "delta_features": delta_features,

            "xai_input": xai_input,

            "xai": xai_result,
        }

        return result
