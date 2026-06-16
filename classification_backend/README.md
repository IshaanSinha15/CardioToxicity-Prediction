# Classification Backend

This package contains the post-IC50 dose-response layer.

Pipeline position:

`SMILES -> ChemBERTa -> GNN -> XGBoost -> Meta Model -> IC50 -> Dose Response -> Channel Block % -> ORd`

The implementation lives under `classification_backend/dose_response/` and does not modify the existing phase-1 predictor.

Expected IC50 input:

```json
{
  "herg_ic50": 800.0,
  "nav_ic50": 5000.0,
  "cav_ic50": 2000.0
}
```

Expected ORd-ready output:

```json
{
  "concentration": 90.0,
  "herg_block": 10.1,
  "nav_block": 1.8,
  "cav_block": 4.5
}
```

Default concentration levels:

- 0.01x
- 0.1x
- 1x
- 10x
- 100x

The package uses the Hill equation as the default model and treats concentrations as free concentrations in nM.
# Classification Backend

This package contains the downstream dose-response layer that starts after IC50 prediction.

Pipeline position:

SMILES -> ChemBERTa -> GNN -> XGBoost -> Meta Model -> IC50 -> Dose Response -> Channel Block % -> ORd

The code in this folder does not modify the existing prediction pipeline. It converts IC50 values for hERG, Nav1.5, and Cav1.2 into channel block outputs that can be consumed by a future ORd simulation module.
