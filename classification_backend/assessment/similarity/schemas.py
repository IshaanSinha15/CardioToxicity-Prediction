from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class SimilarityMatch:
    reference_id: str
    name_or_id: str | None
    similarity: float
    original_smiles: str
    canonical_smiles: str
    ic50_ikr: float | None = None
    ic50_ina: float | None = None
    ic50_ical: float | None = None
    ic50_source: str | None = None
    risk_label: str | None = None
    risk_label_source: str | None = None
    assay_context: str | None = None
    reference: str | None = None
    record_version: str | None = None
    self_match: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SimilarityResult:
    query_smiles: str
    query_canonical_smiles: str
    top_k: int
    minimum_similarity: float
    method: str
    matches: tuple[SimilarityMatch, ...] = field(default_factory=tuple)
    status: str = "ok"

    @property
    def label_summary(self) -> dict[str, object]:
        labels = [match.risk_label for match in self.matches if match.risk_label is not None]
        high_labels = {"high", "known_high", "known risk", "high risk"}
        normalized = {label.lower() for label in labels}
        return {
            "known_labels": len(labels),
            "high_risk_labels": sum(label in high_labels for label in normalized),
            "conflicting_labels": len(normalized) > 1,
        }

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["matches"] = [match.to_dict() for match in self.matches]
        payload["label_summary"] = self.label_summary
        return payload