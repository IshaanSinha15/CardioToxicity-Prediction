# Repository Restructuring: CardioToxicity-Prediction

## Overview

This document describes the major restructuring of the CardioToxicity-Prediction repository to better organize the codebase by separating backend prediction modules from other components.

## Date of Restructuring

March 26, 2026

## Motivation

The original repository structure had ML inference modules scattered at the root level alongside training scripts, data processing, and other utilities. This restructuring consolidates all prediction-related modules into a dedicated `prediction_backend/` package for better organization and maintainability.

## Changes Made

### 1. Directory Structure Changes

**Before:**
```
cardiotoxicity-prediction/
├── embeddings/
├── features/
├── molecular_processing/
├── models/
├── inference/
├── evaluation/
├── tests/
├── training/
├── pipeline/
├── scripts/
├── data/
└── ...
```

**After:**
```
cardiotoxicity-prediction/
├── prediction_backend/          # NEW: Consolidated backend package
│   ├── embeddings/
│   ├── features/
│   ├── molecular_processing/
│   ├── models/
│   ├── inference/
│   ├── evaluation/
│   ├── tests/
│   └── __init__.py
├── training/                    # Unchanged
├── pipeline/                    # Unchanged
├── scripts/                     # Modified: Updated imports
├── data/                        # Modified: Updated imports
├── backend/                     # Future: CiPA simulation pipeline
├── frontend/                    # Future: UI/API
└── ...
```

### 2. Files Moved

The following directories and all their contents were moved from root to `prediction_backend/`:

- `embeddings/` → `prediction_backend/embeddings/`
- `features/` → `prediction_backend/features/`
- `molecular_processing/` → `prediction_backend/molecular_processing/`
- `models/` → `prediction_backend/models/`
- `inference/` → `prediction_backend/inference/`
- `evaluation/` → `prediction_backend/evaluation/`
- `tests/` → `prediction_backend/tests/`

### 3. Import Updates

All Python files throughout the repository were updated to reflect the new import paths:

**Examples of changes:**
```python
# Before
from embeddings.chemberta_embedding import ChemBERTaEncoder
from models.fusion_single_task import FusionSingleTask
from inference.predict import predict

# After
from prediction_backend.embeddings.chemberta_embedding import ChemBERTaEncoder
from prediction_backend.models.fusion_single_task import FusionSingleTask
from prediction_backend.inference.predict import predict
```

**Files with import updates:**
- `data/feature_builder.py`
- `scripts/generate_features.py`
- `scripts/generate_fusion_predictions.py`
- `scripts/generate_gnn_embeddings.py`
- `scripts/generate_xgb_predictions.py`
- `scripts/precompute_chemberta.py`
- `training/train_fusion_single.py`
- `training/train_gnn.py`
- `prediction_backend/inference/predict.py` (model loading paths)
- All files within `prediction_backend/` (internal imports)

### 4. New Files Created

- `prediction_backend/__init__.py` - Package initialization
- `restructure_repo.py` - Automated restructuring script

## Restructuring Script

A Python script `restructure_repo.py` was created to automate this restructuring process. The script:

1. Creates the `prediction_backend/` directory
2. Moves all specified directories into it
3. Adds the `__init__.py` file
4. Scans and updates all Python import statements
5. Is idempotent (safe to run multiple times)

**Usage:**
```bash
python restructure_repo.py
```

## Verification

After restructuring, the `predict.py` script was tested and confirmed to work correctly:

```bash
python prediction_backend/inference/predict.py
```

**Sample output:**
```
Predictions

HERG
pIC50   : 1.1743
IC50 nM : 6.69e+07

NAV
pIC50   : 0.2177
IC50 nM : 6.06e+08

CAV
pIC50   : 0.6579
IC50 nM : 2.20e+08
```

## Dependencies Installed

The following Python packages were installed to support the prediction functionality:

- torch>=2.2
- torch-geometric
- xgboost
- scikit-learn
- transformers
- numpy
- pandas
- rdkit
- joblib

## Future Structure

The restructured repository now has clear separation for future development:

- `prediction_backend/` - Core ML prediction modules
- `backend/` - Reserved for future CiPA simulation pipeline
- `frontend/` - Reserved for future UI/API components
- `training/`, `pipeline/`, `scripts/`, `data/` - Supporting infrastructure

## Migration Notes

- All existing functionality is preserved
- Import paths in user code will need to be updated to use `prediction_backend.*`
- The restructuring script can be safely re-run if needed
- All model files and data remain intact

## Contact

For questions about this restructuring, refer to the commit history or contact the development team.