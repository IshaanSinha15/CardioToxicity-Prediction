from typing import Dict, Any

from .utils import PipelineInput, PipelineResult, PipelineError

from prediction_backend.inference.predict import predict as predict_ic50

from classification_backend.dose_response.channel_block_generator import (
    ChannelBlockGenerator,
    ChannelIC50Inputs,
)


class PredictionPipeline:
    """
    Regression Pipeline

    Input
        ↓
    Validate
        ↓
    ChemBERTa Regression
        ↓
    Predicted IC50
        ↓
    Hill Equation
        ↓
    Channel Block
        ↓
    API Response
    """

    def __init__(self):
        pass

    # -----------------------------------------------------
    # Input Validation
    # -----------------------------------------------------

    def _validate_input(self, payload: PipelineInput) -> PipelineInput:

        if "smiles" not in payload:
            raise PipelineError("Missing SMILES string.")

        smiles = str(payload["smiles"]).strip()

        if smiles == "":
            raise PipelineError("SMILES cannot be empty.")

        try:
            dose_nm = float(payload.get("dose_nm", 0))
        except Exception:
            raise PipelineError("dose_nm must be numeric.")

        if dose_nm <= 0:
            raise PipelineError("dose_nm must be greater than zero.")

        return {
            "smiles": smiles,
            "dose_nm": dose_nm,
            "drug_name": payload.get("drug_name"),
        }

    # -----------------------------------------------------
    # Main Pipeline
    # -----------------------------------------------------

    def run(self, payload: Dict[str, Any]) -> PipelineResult:

        validated = self._validate_input(payload)

        smiles = validated["smiles"]
        dose_nm = validated["dose_nm"]

        # -------------------------------------------------
        # Step 1
        # ChemBERTa Regression
        # -------------------------------------------------

        try:

            ic50_prediction = predict_ic50(smiles)

        except Exception as exc:

            raise PipelineError(
                f"IC50 prediction failed: {exc}"
            )

        # -------------------------------------------------
        # Step 2
        # Dose Response
        # -------------------------------------------------

        try:

            ic50_inputs = ChannelIC50Inputs(

                herg_ic50_nm=float(
                    ic50_prediction["herg"]["IC50_nM"]
                ),

                nav_ic50_nm=float(
                    ic50_prediction["nav"]["IC50_nM"]
                ),

                cav_ic50_nm=float(
                    ic50_prediction["cav"]["IC50_nM"]
                ),

            )

            generator = ChannelBlockGenerator(ic50_inputs)

            dose_response = generator.to_ord_payload(dose_nm)

        except Exception as exc:

            raise PipelineError(
                f"Dose-response calculation failed: {exc}"
            )

        # -------------------------------------------------
        # Final Response
        # -------------------------------------------------

        result: PipelineResult = {

            "input": {

                "smiles": smiles,

                "dose_nm": dose_nm,

                "drug_name": validated.get("drug_name"),

            },

            "ic50_prediction": {

                "herg": {
                    "IC50_nM": float(
                        ic50_prediction["herg"]["IC50_nM"]
                    )
                },

                "nav": {
                    "IC50_nM": float(
                        ic50_prediction["nav"]["IC50_nM"]
                    )
                },

                "cav": {
                    "IC50_nM": float(
                        ic50_prediction["cav"]["IC50_nM"]
                    )
                },

            },

            "dose_response": {

                "dose_nm": dose_nm,

                "herg_block": float(
                    dose_response["herg_block"]
                ),

                "nav_block": float(
                    dose_response["nav_block"]
                ),

                "cav_block": float(
                    dose_response["cav_block"]
                ),

            },

        }

        return result