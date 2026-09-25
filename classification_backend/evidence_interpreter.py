from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping


@dataclass(frozen=True)
class InterpretationResult:
    level: str
    status: str
    reasons: tuple[str, ...]
    requires_review: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


class EvidenceInterpreter:
    def interpret(
        self,
        mechanistic: Mapping[str, object],
        similarity: Mapping[str, object] | None = None,
    ) -> InterpretationResult:
        similarity = similarity or {}
        level = str(mechanistic.get("level", "unknown"))
        labels = self._labels(similarity)
        conflicting = bool(similarity.get("label_summary", {}).get("conflicting_labels", False))
        high_label = any(label.lower() in {"high", "known_high", "high risk", "known risk"} for label in labels)

        if conflicting:
            return InterpretationResult("review_required", "conflicting_evidence", ("retrieved analogue labels conflict",), True)
        if level == "high":
            return InterpretationResult("high_concern", "high_concern_model_derived", ("mechanistic_evidence",), False)
        if level == "moderate" and high_label:
            return InterpretationResult("high_concern", "mechanistic_and_analogue_support", ("mechanistic_evidence", "curated_analogue_evidence"), True)
        if level == "moderate":
            return InterpretationResult("moderate_concern", "model_derived_no_curated_analogue_support", ("mechanistic_evidence",), False)
        if level == "low" and high_label:
            return InterpretationResult("review_required", "conflicting_evidence", ("low mechanistic concern conflicts with a high-risk curated analogue",), True)
        if level == "low":
            return InterpretationResult("low_mechanistic_concern", "model_derived_no_curated_analogue_support", ("mechanistic_evidence",), False)
        if high_label:
            return InterpretationResult("analogue_supported_review", "analogue_evidence_without_mechanistic_result", ("curated_analogue_evidence",), True)
        return InterpretationResult("insufficient_evidence", "insufficient_evidence", ("no usable mechanistic or curated analogue evidence",), True)

    @staticmethod
    def _labels(similarity: Mapping[str, object]) -> list[str]:
        labels = []
        for match in similarity.get("matches", []):
            if isinstance(match, Mapping) and match.get("risk_label") is not None:
                labels.append(str(match["risk_label"]))
        return labels