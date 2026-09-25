# Implementation Progress

This document records one completed implementation phase at a time. The active plan remains [similarity_module_implementation_plan.md](similarity_module_implementation_plan.md).

## Phase 1: Typed Exposure Contract

Status: complete

Implemented:

- Added `TypedExposure` with positive finite nM validation.
- Added explicit `free_plasma`, `total_plasma`, `measured_assay`, `nominal_assay`, and `unspecified` types.
- Added therapeutic, supratherapeutic, toxic, and unknown exposure contexts.
- Added `parse_typed_exposure` with structured legacy `dose_nm` handling.
- Exported the new contract from `classification_backend.dose_response`.
- Added focused unit coverage for valid input, legacy warnings, invalid values, ambiguous units, and conflicting fields.

Results:

- The legacy `dose_nm` alias is never treated as free plasma; it emits `DeprecationWarning` and remains `unspecified`.
- Unit-bearing values such as `uM` are rejected until an explicit conversion contract is added.
- Existing pipeline and ORd code were not modified.

Validation:

- Focused command: `pytest classification_backend/tests/unit/test_typed_exposure.py`
- Result: 8 passed.
- Regression command: `pytest classification_backend/tests/unit/test_dose_response.py classification_backend/tests/unit/test_channel_block_generator.py classification_backend/tests/unit/test_safety_margin.py classification_backend/tests/unit/test_validation.py`
- Regression result: 17 passed.
- Diagnostics: no errors reported; `git diff --check` is clean.

Next phase: deterministic RDKit parsing, canonicalization, and fingerprint utilities.

## Phase 2-3: Similarity Reference and Retrieval

Status: complete

Implemented:

- Added Morgan radius-2, 2048-bit fingerprint generation with explicit chirality configuration.
- Added RDKit SMILES parsing and canonicalization with structured invalid-input errors.
- Added molecule-level reference records and immutable in-memory loading.
- Added exact Tanimoto Top-K retrieval with thresholding, stable score/ID ordering, provenance, missing labels, and self-match exclusion.
- Added a reference builder that rejects invalid structures, deduplicates canonical structures, preserves channel evidence, and writes a SHA-256 manifest.
- Built `classification_backend/dataset/similarity_reference.csv` from `combined_ic50_dataset.csv`; synthetic classifier datasets were not used.

Results:

- Reference source rows: 22,068.
- Accepted canonical molecules: 15,132.
- Rejected structures: 30.
- Canonical duplicates: 6,906.
- Generated fingerprint configuration: Morgan radius 2, 2,048 bits, chirality enabled.
- Generated artifact manifest: `classification_backend/dataset/similarity_reference_manifest.json`.
- A query absent from the reference set returns `no_reference_matches` without inventing labels.

Validation:

- Focused command: `pytest classification_backend/tests/unit/test_similarity_fingerprints.py classification_backend/tests/unit/test_similarity_retriever.py classification_backend/tests/unit/test_similarity_reference_builder.py`
- Result: 11 passed.
- Artifact smoke check: 15,132 records loaded and retrieval returned an explicit `no_reference_matches` state for an unmatched query.
- Diagnostics: no errors reported; `git diff --check` is clean.
- Rule 0 check: no `xai`, `ord_engine`, `pipeline`, or `prediction_backend` files were changed.

Next phase: mechanistic blockage bands, safety-margin rules, concentration-domain status, and evidence interpretation.

## Phase 4-6: Mechanistic Risk, Evidence Interpretation, and Backend Service

Status: complete within Rule 0 boundary

Implemented:

- Added transparent channel blockage bands: Minimal, Low, Moderate, High, and Near-maximal.
- Added margin-based mechanistic levels with explicit high/moderate/low/unknown precedence.
- Added supported, extrapolated, and unknown concentration-domain states.
- Added invalid dose-response handling without converting failures to Low risk.
- Added evidence interpretation for mechanistic-only, curated analogue support, conflicts, and insufficient evidence.
- Added `ClassificationEvidenceService` inside `classification_backend` to compose the new result contract.
- Preserved existing `ChannelBlockGenerator`, Hill equation, safety-margin, ORd-ready payload, and regression prediction code.

Results:

- Similarity is evidence only; no classifier probability or similarity score is averaged.
- Missing curated labels remain missing and cannot create a toxicity label.
- Low mechanistic concern is reported as `low_mechanistic_concern`, never `safe` or `not_toxic`.
- High-risk analogue evidence conflicting with low mechanistic evidence requires review.

Validation:

- Risk/evidence command: `pytest classification_backend/tests/unit/test_mechanistic_risk.py classification_backend/tests/unit/test_evidence_interpreter.py`
- Result: 13 passed.
- Service command: `pytest classification_backend/tests/unit/test_classification_service.py`
- Result: 2 passed; the expected legacy `dose_nm` deprecation warning was emitted.
- Rule 0 check: no `xai`, `ord_engine`, `pipeline`, or `prediction_backend` files were changed.

Explicit boundary:

- The plan's `pipeline/prediction_pipeline.py` integration is intentionally not performed because Rule 0 explicitly prohibits changing the original regression pipeline. The classification-backend service is the ready integration boundary for a later approved adapter.

Next phase: full classification-backend test validation and final implementation audit.

## Final Validation Audit

Status: complete with documented legacy failures

- New implementation tests and existing dose-response tests pass.
- Full `classification_backend/tests/unit` run: 57 passed, 3 failed.
- The 3 failures are pre-existing legacy pipeline/XAI contract tests expecting the removed classifier result and `ClassifierService.get_prediction_model()`; they are outside the Rule 0 change boundary.
- No files under `classification_backend/xai`, `pipeline`, `prediction_backend`, or `ord_engine` were changed by this implementation.
- The old classifier remains available for legacy reproducibility and was not used by the new service.
- `git diff --check` is clean and diagnostics report no errors in new implementation files.

## Top-Level Pipeline Finalization

Status: complete with XAI excluded

The top-level pipeline now runs:

```text
SMILES + typed exposure
	-> IC50 regression
	-> Hill dose response and safety margins
	-> optional ORd simulation
	-> Morgan/Tanimoto similarity retrieval
	-> mechanistic risk and evidence interpretation
```

Implemented:

- Replaced the old classifier-oriented `PredictionPipeline` response with the new evidence contract.
- Added typed `concentration_nm` input while retaining legacy `dose_nm` warning behavior.
- Loaded the validated similarity reference once per pipeline instance.
- Added optional ORd execution with structured `complete`, `skipped`, or `unavailable` status.
- Removed classifier and XAI output from the authoritative pipeline response.
- Updated pipeline type definitions for safety margins, similarity, mechanistic risk, interpretation, and warnings.
- Added a deterministic end-to-end contract test with mocked regression output.

Validation:

- End-to-end contract: 1 passed.
- Full allowed classification-backend suite excluding legacy classifier/XAI contract tests: 58 passed.
- Real ORd branch: executed and returned `unavailable` with `No module named 'myokit'`; this is an environment dependency limitation, not a silent failure.
- XAI modules were not opened or modified.

## Assessment Module Restructure

Status: complete

- Unified active similarity and legacy classifier workflows under `classification_backend/assessment/`.
- Active similarity code now lives under `assessment/similarity/`.
- Legacy synthetic classifier scripts now live under `assessment/legacy_classifier/`.
- Removed the broken duplicate legacy evaluator after capturing its missing-dataset failure in [pre_restructure_outputs.md](pre_restructure_outputs.md).
- Added the single active evaluator at `assessment/evaluate.py`.
- Updated runtime, builder, test, and pipeline imports.
- Preserved the distinction between mechanistic/similarity evidence and legacy synthetic labels.
- XAI remains outside the restructure and unchanged.
- Unified active suite: 58 passed.

## Simplified Dose Testing

Status: complete

- Kept the simple `dose_nm` input and restored it in the dose-response output.
- Positive nM values such as 60, 6,000, and 60,000 are calculated normally.
- Removed the deprecation warning for `dose_nm`.
- Changed the result wording from a hard out-of-range state to `calculable` plus `training_range: inside/outside`.
- Replaced the detailed paracetamol report with a direct module-output comparison in [paracetamol_cardio_ai_report.md](paracetamol_cardio_ai_report.md).
- Final non-XAI suite: 61 passed.
- Direct Hill check: 60 nM = 5.660%, 6,000 nM = 85.714%, 60,000 nM = 98.361% for IC50 1,000 nM.