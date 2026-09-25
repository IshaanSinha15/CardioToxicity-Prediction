from __future__ import annotations

from .fingerprints import canonicalize_smiles, morgan_fingerprint, parse_smiles, tanimoto_similarity
from .reference_database import ReferenceDatabase
from .schemas import SimilarityMatch, SimilarityResult


class SimilarityRetriever:
    def __init__(self, database: ReferenceDatabase):
        self.database = database

    @classmethod
    def build(cls, reference_path: str) -> "SimilarityRetriever":
        return cls(ReferenceDatabase.from_csv(reference_path))

    def search(
        self,
        smiles: str,
        top_k: int = 5,
        minimum_similarity: float = 0.4,
        exclude_reference_id: str | None = None,
    ) -> SimilarityResult:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0 <= minimum_similarity <= 1:
            raise ValueError("minimum_similarity must be between 0 and 1")
        molecule = parse_smiles(smiles)
        canonical = canonicalize_smiles(molecule)
        query_fingerprint = morgan_fingerprint(molecule, self.database.config)
        candidates = []
        for record in self.database.records:
            score = tanimoto_similarity(query_fingerprint, record.fingerprint)
            if score < minimum_similarity:
                continue
            candidates.append((record, score))
        candidates.sort(key=lambda item: (-item[1], item[0].reference_id))
        matches = tuple(
            SimilarityMatch(
                reference_id=record.reference_id,
                name_or_id=record.name_or_id,
                similarity=score,
                original_smiles=record.original_smiles,
                canonical_smiles=record.canonical_smiles,
                ic50_ikr=record.ic50_ikr,
                ic50_ina=record.ic50_ina,
                ic50_ical=record.ic50_ical,
                ic50_source=record.ic50_source,
                risk_label=record.risk_label,
                risk_label_source=record.risk_label_source,
                assay_context=record.assay_context,
                reference=record.reference,
                record_version=record.record_version,
                self_match=record.reference_id == exclude_reference_id,
            )
            for record, score in candidates
            if record.reference_id != exclude_reference_id
        )[:top_k]
        status = "ok" if matches else "no_reference_matches"
        if matches and len(matches) < top_k:
            status = "partial_top_k"
        return SimilarityResult(
            query_smiles=smiles,
            query_canonical_smiles=canonical,
            top_k=top_k,
            minimum_similarity=minimum_similarity,
            method=self.database.config.method(),
            matches=matches,
            status=status,
        )