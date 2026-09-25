import json

import pandas as pd

from classification_backend.dataset.build_similarity_reference import build_reference


def test_builder_deduplicates_and_records_manifest(tmp_path):
    source = tmp_path / "source.csv"
    output = tmp_path / "reference.csv"
    manifest_path = tmp_path / "manifest.json"
    pd.DataFrame(
        [
            {"smiles": "C(C)O", "IC50_IKr": 10.0},
            {"smiles": "CCO", "IC50_INa": 20.0},
            {"smiles": "bad smiles", "IC50_ICaL": 30.0},
        ]
    ).to_csv(source, index=False)

    manifest = build_reference(source, output, manifest_path)

    assert manifest["accepted_count"] == 1
    assert manifest["rejected_count"] == 1
    assert manifest["duplicate_count"] == 1
    result = pd.read_csv(output)
    assert len(result) == 1
    assert result.loc[0, "IC50_IKr"] == 10.0
    assert result.loc[0, "IC50_INa"] == 20.0
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["source_sha256"]