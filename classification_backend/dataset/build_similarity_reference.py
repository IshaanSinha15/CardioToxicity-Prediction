from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from classification_backend.assessment.similarity.fingerprints import FingerprintConfig, canonicalize_smiles, parse_smiles

REFERENCE_COLUMNS = [
    "reference_id",
    "name_or_id",
    "original_smiles",
    "canonical_smiles",
    "parent_smiles",
    "IC50_IKr",
    "IC50_INa",
    "IC50_ICaL",
    "ic50_source",
    "risk_label",
    "risk_label_source",
    "assay_context",
    "reference",
    "record_version",
]


def build_reference(source_path: Path, output_path: Path, manifest_path: Path) -> dict[str, object]:
    source = pd.read_csv(source_path)
    if "smiles" not in source.columns:
        raise ValueError("source data must contain a smiles column")

    accepted: dict[str, dict[str, object]] = {}
    rejected = 0
    duplicates = 0
    for row in source.to_dict(orient="records"):
        original = row.get("smiles")
        try:
            molecule = parse_smiles(original)
            canonical = canonicalize_smiles(molecule)
        except (TypeError, ValueError):
            rejected += 1
            continue
        if canonical in accepted:
            duplicates += 1
            accepted[canonical] = _merge_record(accepted[canonical], row)
            continue
        accepted[canonical] = _record(canonical, original, row)

    output = pd.DataFrame(accepted.values(), columns=REFERENCE_COLUMNS).sort_values("reference_id")
    output.to_csv(output_path, index=False)
    manifest = {
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "output_path": str(output_path),
        "accepted_count": len(output),
        "rejected_count": rejected,
        "duplicate_count": duplicates,
        "fingerprint": FingerprintConfig().to_dict(),
        "source_policy": "molecule-level canonical structure; synthetic dose rows excluded",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _record(canonical: str, original: object, row: dict[str, object]) -> dict[str, object]:
    return {
        "reference_id": _reference_id(canonical),
        "name_or_id": row.get("name_or_id"),
        "original_smiles": str(original),
        "canonical_smiles": canonical,
        "parent_smiles": canonical,
        "IC50_IKr": row.get("IC50_IKr"),
        "IC50_INa": row.get("IC50_INa"),
        "IC50_ICaL": row.get("IC50_ICaL"),
        "ic50_source": row.get("ic50_source"),
        "risk_label": row.get("risk_label"),
        "risk_label_source": row.get("risk_label_source"),
        "assay_context": row.get("assay_context"),
        "reference": row.get("reference"),
        "record_version": row.get("record_version"),
    }


def _merge_record(record: dict[str, object], row: dict[str, object]) -> dict[str, object]:
    for field in REFERENCE_COLUMNS:
        if pd.isna(record.get(field)) and not pd.isna(row.get(field)):
            record[field] = row.get(field)
    return record


def _reference_id(canonical: str) -> str:
    return f"mol_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_reference(args.source, args.output, args.manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()