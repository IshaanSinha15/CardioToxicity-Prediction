from __future__ import annotations

from typing import Mapping

from .dose_response import parse_typed_exposure
from .dose_response.channel_block_generator import ChannelBlockGenerator, ChannelIC50Inputs
from .dose_response.safety_margin import SafetyMarginAnalyzer
from .evidence_interpreter import EvidenceInterpreter
from .mechanistic_risk import MechanisticRiskAssessor
from .assessment.similarity.fingerprints import canonicalize_smiles, parse_smiles
from .assessment.similarity.retriever import SimilarityRetriever


class ClassificationEvidenceService:
    def __init__(
        self,
        similarity_retriever: SimilarityRetriever | None = None,
        risk_assessor: MechanisticRiskAssessor | None = None,
    ):
        self.similarity_retriever = similarity_retriever
        self.risk_assessor = risk_assessor or MechanisticRiskAssessor()
        self.evidence_interpreter = EvidenceInterpreter()

    def evaluate(
        self,
        smiles: str,
        exposure_input: Mapping[str, object],
        ic50_values_nm: Mapping[str, float],
        top_k: int = 5,
        minimum_similarity: float = 0.4,
    ) -> dict[str, object]:
        molecule = parse_smiles(smiles)
        canonical_smiles = canonicalize_smiles(molecule)
        exposure = parse_typed_exposure(exposure_input)
        channel_inputs = ChannelIC50Inputs.from_mapping(ic50_values_nm)
        generator = ChannelBlockGenerator(channel_inputs)
        block = generator.block_at_concentration(exposure.concentration_nm)
        block_payload = block.to_dict()
        block_values = {
            "herg": block.herg_block,
            "nav": block.nav_block,
            "cav": block.cav_block,
        }
        normalized_ic50 = channel_inputs.to_dict()
        margin_inputs = {
            "herg": normalized_ic50["herg_ic50_nm"],
            "nav": normalized_ic50["nav_ic50_nm"],
            "cav": normalized_ic50["cav_ic50_nm"],
        }
        margins = SafetyMarginAnalyzer().build_table(margin_inputs, exposure.concentration_nm).to_dict(orient="records")
        risk = self.risk_assessor.assess(normalized_ic50, block_values, exposure)
        if self.similarity_retriever is None:
            similarity_payload = {
                "method": "Morgan radius 2, 2048 bits, Tanimoto",
                "top_k": top_k,
                "minimum_similarity": minimum_similarity,
                "matches": [],
                "label_summary": {"known_labels": 0, "high_risk_labels": 0, "conflicting_labels": False},
                "status": "reference_database_unavailable",
            }
        else:
            similarity_payload = self.similarity_retriever.search(
                canonical_smiles,
                top_k=top_k,
                minimum_similarity=minimum_similarity,
            ).to_dict()
        interpretation = self.evidence_interpreter.interpret(risk.to_dict(), similarity_payload)
        warnings = list(risk.warnings)
        if exposure.concentration_type.value == "unspecified":
            warnings.append("unspecified_concentration_type")
        if similarity_payload["status"] == "reference_database_unavailable":
            warnings.append("similarity_reference_database_unavailable")
        return {
            "input": {
                "smiles": smiles,
                "canonical_smiles": canonical_smiles,
                **exposure.to_dict(),
            },
            "dose_response": {
                "dose_nm": exposure.concentration_nm,
                "concentration_nm": exposure.concentration_nm,
                "concentration_type": exposure.concentration_type.value,
                "herg_block": block_payload["herg_block"],
                "nav_block": block_payload["nav_block"],
                "cav_block": block_payload["cav_block"],
            },
            "safety_margins": margins,
            "mechanistic_classification": risk.mechanistic_classification,
            "mechanistic_risk": risk.to_dict(),
            "similarity": similarity_payload,
            "interpretation": interpretation.to_dict(),
            "warnings": warnings,
        }