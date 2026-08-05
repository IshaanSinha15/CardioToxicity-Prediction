from typing import TypedDict, Dict, Any


class PipelineInput(TypedDict):
    smiles: str
    dose_nm: float
    drug_name: str | None


class PipelineResult(TypedDict):
    input: Dict[str, Any]
    ic50_prediction: Dict[str, Dict[str, float]]
    dose_response: Dict[str, float]
    simulation: Dict[str, float]
    classification: Dict[str, Any]
    features_used: Dict[str, float]


class PipelineError(RuntimeError):
    pass
