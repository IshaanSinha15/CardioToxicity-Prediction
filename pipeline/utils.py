from typing import TypedDict, Dict, Any


class PipelineInput(TypedDict):
    smiles: str
    dose_nm: float
    drug_name: str | None


class PipelineResult(TypedDict, total=False):
    input: Dict[str, Any]
    ic50_prediction: Dict[str, Dict[str, float]]
    ic50_comparison: Dict[str, Any]
    dose_response: Dict[str, Any]
    safety_margins: list[Dict[str, Any]]
    mechanistic_classification: Dict[str, Any]
    mechanistic_risk: Dict[str, Any]
    similarity: Dict[str, Any]
    interpretation: Dict[str, Any]
    simulation: Dict[str, Any]
    artifacts: Dict[str, Any]
    xai: Dict[str, Any]
    classification: Dict[str, Any]
    features_used: list[str]
    warnings: list[str]
    frontend: Dict[str, Any]


class PipelineError(RuntimeError):
    pass
