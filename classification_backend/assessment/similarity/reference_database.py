from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .fingerprints import FingerprintConfig, canonicalize_smiles, morgan_fingerprint, parse_smiles


@dataclass(frozen=True)
class ReferenceRecord:
    reference_id: str
    name_or_id: str | None
    original_smiles: str
    canonical_smiles: str
    fingerprint: object
    ic50_ikr: float | None = None
    ic50_ina: float | None = None
    ic50_ical: float | None = None
    ic50_source: str | None = None
    risk_label: str | None = None
    risk_label_source: str | None = None
    assay_context: str | None = None
    reference: str | None = None
    record_version: str | None = None


class ReferenceDatabaseError(ValueError):
    pass


class ReferenceDatabase:
    def __init__(self, records: Iterable[ReferenceRecord], config: FingerprintConfig | None = None):
        self.config = config or FingerprintConfig()
        self.records = tuple(records)
        if not self.records:
            raise ReferenceDatabaseError("reference database contains no valid records")

    @classmethod
    def from_csv(cls, path: str | Path, config: FingerprintConfig | None = None) -> "ReferenceDatabase":
        source = Path(path)
        if not source.is_file():
            raise ReferenceDatabaseError(f"reference database not found: {source}")
        return cls.from_frame(pd.read_csv(source), config=config)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, config: FingerprintConfig | None = None) -> "ReferenceDatabase":
        config = config or FingerprintConfig()
        smiles_column = "smiles" if "smiles" in frame.columns else "original_smiles"
        if smiles_column not in frame.columns:
            raise ReferenceDatabaseError("reference data must contain a smiles or original_smiles column")
        records: list[ReferenceRecord] = []
        seen: set[str] = set()
        for row in frame.to_dict(orient="records"):
            original = row.get(smiles_column)
            try:
                molecule = parse_smiles(original)
                canonical = canonicalize_smiles(molecule)
            except (TypeError, ValueError):
                continue
            if canonical in seen:
                continue
            seen.add(canonical)
            reference_id = str(row.get("reference_id") or _stable_reference_id(canonical))
            records.append(
                ReferenceRecord(
                    reference_id=reference_id,
                    name_or_id=_text(row.get("name_or_id")),
                    original_smiles=str(original),
                    canonical_smiles=canonical,
                    fingerprint=morgan_fingerprint(molecule, config),
                    ic50_ikr=_number(row.get("IC50_IKr")),
                    ic50_ina=_number(row.get("IC50_INa")),
                    ic50_ical=_number(row.get("IC50_ICaL")),
                    ic50_source=_text(row.get("ic50_source")),
                    risk_label=_text(row.get("risk_label")),
                    risk_label_source=_text(row.get("risk_label_source")),
                    assay_context=_text(row.get("assay_context")),
                    reference=_text(row.get("reference")),
                    record_version=_text(row.get("record_version")),
                )
            )
        return cls(records, config=config)


def _text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _stable_reference_id(canonical_smiles: str) -> str:
    digest = hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest()[:16]
    return f"mol_{digest}"