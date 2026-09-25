# Final Similarity, Mechanistic Classification, and Risk Evidence Implementation Plan

## Final Decision

The final system is a **classification plus similarity evidence system**, but the classification is a transparent, rule-based **mechanistic blockage classification**, not the current trained Random Forest/XGBoost classifier.

The authoritative architecture is:

```text
SMILES + typed exposure
  -> IC50 regression
  -> Hill channel blockage
  -> safety-margin and mechanistic classification

SMILES
  -> Morgan fingerprint
  -> Tanimoto Top-K similarity evidence

mechanistic classification + similarity evidence
  -> overall interpretation, warnings, or abstention
```

The current classifier is legacy only. It must not be blended numerically with Tanimoto similarity, because its synthetic labels are generated from the same blockage features it receives. Its approximately 99% generated-dataset score and approximately 27.8% exploratory agreement on independent-style benchmark cases do not justify using it as a cardiotoxicity classifier.

The Hill equation is retained. Its implementation is mathematically correct; the correction is to make exposure semantics explicit and to stop treating a bare nominal nM value as automatically therapeutic or clinically meaningful.

## 1. Purpose and Product Semantics

The project should answer this question:

> Given a molecule, an exposure concentration, and the available evidence, is there a defensible reason to consider the molecule potentially cardiotoxic?

The system must not reduce that question to an unsupported class label. It should return three distinguishable evidence layers:

1. **Mechanistic exposure evidence**: predicted hERG, Nav1.5, and Cav1.2 IC50 values; calculated channel block at the requested concentration; safety margins; and an exposure-domain warning.
2. **Chemical analogue evidence**: structurally similar reference molecules, Tanimoto scores, known channel measurements, curated cardiotoxicity labels, and citations when available.
3. **Interpretation**: a transparent rule-based concern level with reasons, uncertainty, conflicts, and abstention state.

The current Random Forest/XGBoost classifier must not be the authoritative decision model. Its labels are derived from the same synthetic channel-block features that it receives as input, and its 99%+ generated-dataset score does not transfer to independent drug-level evidence.

## 2. Existing Pipeline to Preserve

The existing mechanistic path remains the backbone:

```text
SMILES + typed exposure
    -> input validation
    -> ChemBERTa/GNN/XGBoost/meta IC50 prediction
    -> ChannelBlockGenerator
    -> Hill channel block
    -> safety margins and optional ORd simulation
```

The new evidence branch runs in parallel:

```text
SMILES
    -> RDKit parse and canonicalize
    -> Morgan radius-2, 2048-bit fingerprint
    -> exact Tanimoto Top-K retrieval
    -> analogue evidence and provenance
```

The final pipeline combines the mechanistic classification and similarity evidence without averaging a legacy classifier probability with a similarity score.

Important current boundary:

- `prediction_backend.inference.predict.predict` remains responsible for IC50 prediction.
- `classification_backend.dose_response.ChannelBlockGenerator` remains responsible for Hill block and ORd-ready channel output.
- `classification_backend.dose_response.SafetyMarginAnalyzer` remains responsible for margin calculations.
- `pipeline.prediction_pipeline.PredictionPipeline` becomes the orchestration boundary for the new result.
- Existing classifier training, cluster labeling, and SHAP code are placed on a legacy path and are not used to make the final harmfulness decision.

## 3. Non-Negotiable Rules

### Rule 0: No changes in areas/ files which do not require changes , changes to be made in only dosgae response and classfication module / model training and dataset in classification backend folder as needed. Xai , ord and original regression pipeline should not be touched for now , and after every phase implementation recheck if all rules are being followed internally , I dont want unnecessary comments , line paragraphs , limit the usage of tokens as needed , wasteage of tokens should not be there.

### Rule 1: Never infer a known label from similarity alone

A retrieved label is evidence attached to the reference record. It must carry its source, date/version, assay context, and reference citation. If no label exists, return `null` and state that no known label is available.



### Rule 2: Never average unlike quantities

Do not calculate a score such as:

```text
classifier_probability + Tanimoto_similarity
```

Tanimoto similarity is structural overlap, not a probability of toxicity. Mechanistic block, safety margin, and analogue evidence must remain separate fields.

### Rule 3: The Hill block is mechanistic evidence, not a clinical diagnosis

Use the existing Hill equation to calculate concentration-dependent block. Report the result as model-derived evidence and retain uncertainty or extrapolation warnings.

### Rule 4: Expose the concentration semantics and model-domain boundary

The public input must identify the concentration domain. Supported values are:

- `free_plasma`: unbound therapeutic or supratherapeutic exposure;
- `total_plasma`: total plasma exposure, not directly interchangeable with free plasma;
- `measured_assay`: analytically measured bath/assay concentration; and
- `nominal_assay`: intended assay concentration, potentially affected by drug loss or nonspecific binding.

The old `dose_nm` field remains a temporary alias with `concentration_type: unspecified` and a warning. It must not be silently interpreted as free plasma.

The generated classifier dataset used doses from `0.3` to `3,000 nM`, but that is not a scientifically valid universal therapeutic range. The new output should mark `extrapolated_concentration` when the requested concentration is outside the configured evidence/model domain. The Hill equation may still calculate a response, but the interpretation must disclose the extrapolation.

The system may still calculate the Hill response outside that range, but it must not silently present the result as equally validated. A future model version may replace the domain configuration only with documented training and validation evidence.

### Rule 5: Structural retrieval is deterministic

For fixed reference data, fingerprint configuration, query SMILES, `top_k`, and threshold, retrieval must produce the same ordered results. Sort ties by stable `reference_id`.

### Rule 6: Preserve provenance and missingness

Never replace missing IC50 values, labels, names, assay conditions, or references with guessed values. Missing evidence is a first-class state.

### Rule 7: Invalid input fails clearly

Empty, malformed, non-string, or unparseable SMILES must return a structured validation error. They must not reach the IC50 model or similarity engine.

### Rule 8: No strong conclusion from contradictory evidence

If mechanistic evidence indicates low concern but a high-confidence curated analogue indicates a known high-risk compound, return `review_required` or `conflicting_evidence`, not a silently averaged class.

### Rule 9: Classification means blockage classification

The initial mechanistic classification is a display and triage layer:

| Channel block | Classification |
|---:|---|
| `<5%` | Minimal |
| `5% to <20%` | Low |
| `20% to <50%` | Moderate |
| `>=50%` | High |
| `>=90%` | Near-maximal |

These bands describe model-predicted channel inhibition. They are not clinical torsades thresholds and must not be presented as `safe`, `not toxic`, or a calibrated probability.

For $n=1$, the corresponding concentration-to-IC50 ratios are approximately `0.01`, `0.053`, `0.25`, `1`, and `9`. The displayed classification must be calculated from continuous blockage and safety margin, not from the legacy `RiskClass` field.

### Rule 10: Use literature-backed exposure comparisons

Reference comparisons must prefer unbound effective therapeutic concentration or measured assay concentration matched to the IC50 source. The Redfern et al. analysis supports a provisional 30-fold hERG IC50-to-unbound-Cmax margin for many lower-risk development cases, while also showing important exceptions and the need for integrated multi-channel evidence. This is a review boundary, not a universal clinical rule.

## 4. Target Result Contract

The pipeline should return the existing fields plus `exposure`, `similarity`, `mechanistic_classification`, `mechanistic_risk`, `interpretation`, and `warnings`.

```json
{
  "input": {
    "smiles": "...",
    "canonical_smiles": "...",
    "concentration_nm": 100.0,
    "concentration_type": "free_plasma",
    "exposure_context": "therapeutic",
    "drug_name": null
  },
  "ic50_prediction": {
    "herg": {"IC50_nM": 1000.0, "pIC50": 6.0},
    "nav": {"IC50_nM": 5000.0, "pIC50": 5.3},
    "cav": {"IC50_nM": 2000.0, "pIC50": 5.7}
  },
  "dose_response": {
    "concentration_nm": 100.0,
    "concentration_type": "free_plasma",
    "herg_block": 9.09,
    "nav_block": 1.96,
    "cav_block": 4.76
  },
  "safety_margins": [
    {
      "channel": "herg",
      "ic50_nm": 1000.0,
      "concentration_nm": 100.0,
      "margin": 10.0,
      "risk_class": "Moderate"
    }
  ],
  "mechanistic_risk": {
    "level": "moderate",
    "status": "model_derived",
    "reasons": ["hERG safety margin is below the safe threshold"],
    "dominant_channel": "hERG",
    "concentration_domain": "supported"
  },
  "mechanistic_classification": {
    "channels": {
      "hERG": {"block_pct": 9.09, "band": "Low"},
      "Nav1.5": {"block_pct": 1.96, "band": "Minimal"},
      "Cav1.2": {"block_pct": 4.76, "band": "Minimal"}
    },
    "dominant_channel": "hERG",
    "overall_level": "moderate",
    "status": "model_derived"
  },
  "similarity": {
    "method": "Morgan radius 2, 2048 bits, Tanimoto",
    "top_k": 5,
    "minimum_similarity": 0.4,
    "matches": [],
    "label_summary": {
      "known_labels": 0,
      "high_risk_labels": 0,
      "conflicting_labels": false
    },
    "status": "no_reference_matches"
  },
  "interpretation": {
    "level": "moderate",
    "status": "model_derived_no_curated_analogue_support",
    "reasons": ["Mechanistic evidence indicates moderate concern"],
    "requires_review": false
  },
  "warnings": []
}
```

The contract must preserve backwards-compatible `ic50_prediction` and `dose_response` fields. New consumers should use `mechanistic_risk`, `similarity`, and `interpretation` rather than `classification`.

## 5. Implementation Phases

### Phase 0: Freeze and document the current behavior

1. Record the current active artifacts and versions:
   - IC50 model files under `prediction_backend/models/saved_models/`.
   - classifier artifacts under `classification_backend/saved_models/`.
   - dataset hashes and row counts.
   - Python, RDKit, PyTorch, XGBoost, and scikit-learn versions.
2. Preserve the actual-classifier report as a baseline.
3. Keep the current classifier files untouched while the replacement is developed.
4. Add a small regression fixture for the existing Hill equation and `ChannelBlockGenerator` outputs.

**Exit criteria:** the current mechanistic outputs remain reproducible, and the baseline report identifies the classifier as legacy evidence.

### Phase 1: Define and build the reference database

Add a dataset builder such as:

`classification_backend/dataset/build_similarity_reference.py`

Input sources:

- `classification_backend/dataset/combined_ic50_dataset.csv` as the initial structure/channel source.
- A future curated drug table containing names, external IDs, risk labels, citations, assay context, and exposure metadata.

Build one molecule-level record per canonical structure. Do not use `classifier_dataset.csv` or `classifier_dataset_labeled.csv` as the similarity reference because those contain repeated synthetic dose rows and derived labels.

For every source record:

1. Read the original SMILES.
2. Parse with RDKit.
3. Reject and log invalid SMILES.
4. Generate canonical SMILES.
5. Decide and document salt handling. Preserve the original multi-component SMILES, and store a normalized parent structure only when that policy is explicit.
6. Deduplicate by canonical structure using a stable deterministic rule.
7. Preserve all available IC50 values, source fields, names, labels, references, assay conditions, units, and timestamps.
8. Assign a stable `reference_id`; never use row position as the ID.
9. Record a manifest with source file hashes, accepted/rejected counts, duplicate counts, and fingerprint configuration.

Minimum reference schema:

```text
reference_id
name_or_id
original_smiles
canonical_smiles
parent_smiles
IC50_IKr
IC50_INa
IC50_ICaL
ic50_source
risk_label
risk_label_source
assay_context
reference
record_version
```

At the first release, most name and risk-label fields will be null. That is acceptable if the API reports the missingness clearly.

**Exit criteria:** a reproducible molecule-level CSV or Parquet reference table exists, invalid structures are logged, duplicate policy is tested, and no synthetic classifier labels are treated as independent ground truth.

### Phase 2: Implement fingerprint generation

Add a focused module such as:

`classification_backend/similarity/fingerprints.py`

Implement:

1. `parse_smiles(smiles) -> RDKit Mol`.
2. `canonicalize_smiles(mol) -> str`.
3. `morgan_fingerprint(mol, radius=2, n_bits=2048)`.
4. Configuration validation and serialization.

Use RDKit's generator API when available and a compatibility fallback only when required by the installed version. Persist:

```text
fingerprint_family = Morgan
radius = 2
n_bits = 2048
use_chirality = explicit and tested
```

Do not silently change fingerprint settings between database build and query time.

Unit tests must cover valid SMILES, invalid SMILES, stereochemistry, salts, aromatic structures, duplicate canonical forms, and deterministic fingerprints.

**Exit criteria:** the same canonical molecule produces the same fingerprint and configuration metadata is available in every retrieval result.

### Phase 3: Implement the exact Top-K retriever

Add modules such as:

- `classification_backend/similarity/reference_database.py`
- `classification_backend/similarity/retriever.py`
- `classification_backend/similarity/schemas.py`

Implement a `SimilarityRetriever` with:

```text
build(reference_path)
search(smiles, top_k=5, minimum_similarity=0.4, exclude_reference_id=None)
```

Search behavior:

1. Validate and canonicalize the query.
2. Generate the query fingerprint.
3. Compare against every valid reference fingerprint using RDKit Tanimoto similarity.
4. Exclude the exact same reference when requested to avoid self-retrieval leakage.
5. Apply the optional minimum threshold.
6. Sort descending by similarity and then ascending by stable ID.
7. Return names/IDs, scores, IC50 values, labels, provenance, and match status.
8. Return `no_reference_matches` when no record passes the threshold.

Use exact brute-force retrieval initially. The current 22,068-row source is appropriate for this version. Benchmark construction and query latency before considering an indexed search implementation.

Initial operational defaults:

- `top_k = 5`
- provisional `minimum_similarity = 0.4`
- strong analogue evidence requires a curated label and a score threshold validated on a holdout set; do not infer this threshold from the synthetic classifier dataset.

**Exit criteria:** retrieval is deterministic, tie-stable, duplicate-safe, threshold-aware, and independently unit-tested.

### Phase 4: Implement mechanistic risk assessment

Add a rule-based service such as:

`classification_backend/mechanistic_risk.py`

Inputs:

- predicted IC50 values;
- requested concentration;
- `ChannelBlockResult`;
- `SafetyMarginAnalyzer` output;
- optional uncertainty and concentration-domain metadata.

Use existing rules as the initial transparent vocabulary. Keep the blockage classification separate from the overall interpretation:

#### Channel-block descriptors

- `< 5%`: minimal block
- `5% to <20%`: mild block
- `20% to <50%`: moderate block
- `>=50%`: strong block

- `>=90%`: near-maximal block

#### Safety-margin descriptors

Reuse the current thresholds from `SafetyMarginAnalyzer`:

- margin `>=30`: Safe
- margin `>=10 and <30`: Moderate
- margin `<10`: High

#### Initial overall mechanistic interpretation

Use explicit precedence, not a learned model:

1. `high` if any channel has safety margin `<10` or hERG block `>=50%`, with a reason naming the channel and concentration type.
2. `moderate` if any channel has safety margin `<30`, hERG block `>=20%`, any non-hERG block `>=50%`, or at least two channels have block `>=20%`.
3. `low` if all channels have safety margin `>=30` and all channel blocks are `<20%`.
4. `unknown` if required IC50/block values are missing, non-finite, outside validation limits, or the inference path fails.

The 30-fold margin is a provisional literature-informed review boundary, not a validated clinical safety threshold. The implementation must expose the continuous block and margin values so that the rule cannot hide the underlying evidence.

The project-specific mathematical check, blockage bands, medicine fixtures, and literature sources are documented in [dose_response_literature_validation_report.md](dose_response_literature_validation_report.md).

#### Dose-response invariants

For every valid positive IC50 and Hill coefficient:

- block at zero concentration is `0%`;
- block is monotonic non-decreasing with concentration;
- block at `C = IC50` is `50%` when `n=1` and `Emax=100`;
- block remains in `[0, 100]`; and
- increasing IC50 lowers block at the same concentration.

If any invariant fails, stop interpretation and return `dose_response_invalid` rather than a risk class.

These are initial engineering rules, not validated clinical thresholds. Name the result `mechanistic_risk`, include the triggered reasons, and make thresholds configuration values so they can be revised after independent validation.

Add a separate `concentration_domain` status:

- `supported`: concentration is inside the configured evidence/model range and its type is known.
- `extrapolated`: concentration is outside the configured evidence/model range.
- `unknown`: supported range is unavailable.

An extrapolated result may still be calculated, but its interpretation must include a warning and should not be represented as a validated certainty.

**Exit criteria:** identical inputs produce deterministic reasons, rules are unit-tested at every boundary, and the Hill calculation remains untouched.

### Phase 5: Implement evidence fusion without score averaging

Add a small orchestration component such as:

`classification_backend/evidence_interpreter.py`

Inputs:

- mechanistic risk object;
- similarity result;
- optional curated label confidence/provenance.

Rules:

1. Mechanistic `high` -> overall `high_concern`, with reason `mechanistic_evidence`.
2. Mechanistic `moderate` plus a high-confidence labeled analogue -> `elevated_concern` or `high_concern`, depending on the documented policy.
3. Mechanistic `low` plus no labeled analogue -> `low_mechanistic_concern`, not `not_toxic`.
4. Mechanistic `low` plus a high-confidence analogue with a known high-risk label -> `review_required` or `conflicting_evidence`.
5. No usable mechanistic result but labeled analogue support -> `analogue_supported_review`; never claim a mechanistic prediction.
6. No usable mechanistic result and no usable analogue -> `insufficient_evidence`.
7. Any contradictory labels among close, well-provenanced analogues -> `conflicting_evidence`.

The interpreter must return evidence provenance in every reason. It must never synthesize a cardiotoxicity label for an unseen drug solely because its nearest neighbour has one.

**Exit criteria:** fusion tests cover every matrix branch and demonstrate that similarity values are not treated as calibrated probabilities.

### Phase 6: Integrate into the actual pipeline

Modify `pipeline/prediction_pipeline.py` in this order:

1. Replace the bare `dose_nm` requirement with typed exposure fields: `concentration_nm`, `concentration_type`, and optional `exposure_context`. Keep `dose_nm` as a deprecated alias that produces an `unspecified_concentration_type` warning.
2. Strengthen `_validate_input` to reject non-finite/non-positive concentrations and malformed SMILES before expensive model loading where possible. Validate explicit unit conversions such as `uM -> nM`; never infer units from magnitude.
3. Canonicalize the SMILES once and use the canonical form for similarity and the original form for user display.
4. Run `predict_ic50(smiles)` exactly as the current pipeline does.
5. Build `ChannelIC50Inputs` and call `ChannelBlockGenerator.to_ord_payload(concentration_nm)`, retaining the exact continuous block values.
6. Build safety margins with `SafetyMarginAnalyzer` using the same concentration domain as the IC50 evidence, or mark the margin incomparable when domains differ.
7. Call `MechanisticRiskAssessor` to produce channel blockage bands and the overall rule-based mechanistic class.
8. Call `SimilarityRetriever.search` using the canonical SMILES.
9. Call `EvidenceInterpreter` to combine mechanistic classification and analogue evidence without score averaging.
10. Preserve current `input`, `ic50_prediction`, and `dose_response` fields, adding typed exposure metadata.
11. Add `safety_margins`, `mechanistic_classification`, `similarity`, `mechanistic_risk`, `interpretation`, and `warnings`.
12. Convert expected recoverable issues into structured statuses rather than silently dropping them.

Update `pipeline/utils.py` type definitions to include the new response objects. Do not make the reference database or fingerprint construction happen on every query; load a validated immutable database once per service instance.

The current `PredictionPipeline` does not yet include a classifier result even though older tests expect `classification`, `features_used`, and SHAP fields. Resolve that contract mismatch explicitly:

- new tests should assert the new fields;
- legacy classifier tests should be marked or moved to legacy coverage;
- do not add a compatibility `classification` field that falsely implies the old classifier remains authoritative.

**Exit criteria:** one pipeline call returns typed exposure, IC50, continuous dose-response, blockage classification, safety margins, similarity evidence, interpretation, and warnings without changing existing ORd-ready values.

### Phase 7: Define all edge-case behavior

#### Input and chemistry

- Empty SMILES: `invalid_input`, no model call.
- Malformed SMILES: `invalid_smiles`, no retrieval or IC50 call.
- Non-string SMILES: reject or explicitly coerce only at the public boundary; record the behavior.
- Stereochemical SMILES: preserve stereochemistry in the query and document fingerprint settings.
- Salts and dot-disconnected SMILES: preserve original structure, apply a documented parent/salt policy, and return both forms.
- Tautomers: do not silently normalize unless a tested normalization policy is introduced.
- Duplicate canonical structures: deduplicate reference records deterministically.
- Very large molecules or unsupported elements: return a structured RDKit validation issue.

#### Reference retrieval

- Empty database: `reference_database_unavailable`.
- No valid reference structures: `reference_database_invalid`.
- No match above threshold: `no_reference_matches`.
- Fewer than K matches: return available matches and `partial_top_k`.
- Exact query match: mark `self_match`; exclude it when evaluating analogue generalization.
- Missing name or ID: return stable reference ID.
- Missing label: return `known_cardiotoxicity_label: null`.
- Conflicting labels: preserve all labels and set `conflicting_evidence`.
- Missing IC50 channel: return `null`; never impute in the baseline.
- Duplicate assay records: preserve assay-level records or apply a documented aggregation policy.

#### Dose and model

- Zero or negative dose: reject at the pipeline boundary.
- Non-finite concentration: reject.
- Missing or unsupported `concentration_type`: calculate only with an explicit warning; do not compare it to free-plasma references.
- Concentration below or above the configured evidence/model range: calculate only if valid, set `extrapolated`, and lower interpretation confidence.
- `uM`, `mg/L`, or dose values supplied without an explicit conversion: reject rather than infer nM from magnitude.
- Missing/non-finite/negative IC50: return `unknown` mechanistic risk and a validation error.
- IC50 outside expected channel ranges: preserve the result but add a warning from `validate_ic50_range`.
- Model file missing: return `ic50_model_unavailable`, not a fabricated result.
- Model inference exception: return structured failure with no risk label.
- RDKit unavailable: fail the similarity branch explicitly while allowing the mechanistic branch to be returned if it is valid.
- ORd simulation failure: preserve IC50, dose-response, and similarity results; mark simulation unavailable.

#### Evidence interpretation

- Low mechanistic result without curated analogue evidence: `low_mechanistic_concern`, never `safe` or `not toxic`.
- High mechanistic result without analogue evidence: `high_concern_model_derived` with extrapolation/uncertainty warnings as appropriate.
- High-risk analogue plus low mechanism: `review_required`, not automatic high confidence.
- No evidence: `insufficient_evidence`.

**Exit criteria:** every edge case has a stable status, no exception is converted into a false Low result, and tests assert the exact response state.

### Phase 8: Build the independent validation set

The current derived classifier dataset cannot be used as the primary truth set. Create a separate curated table, for example:

`classification_backend/dataset/curated_cardiotoxicity_validation.csv`

Required fields:

```text
record_id
drug_name
smiles
canonical_smiles
reference_category
reference_system
reference_source
reference_url_or_citation
concentration_nm
concentration_type
exposure_context
nominal_assay_concentration_nm
measured_assay_concentration_nm
free_plasma_concentration_nm
assay_context
IC50_IKr
IC50_INa
IC50_ICaL
label_date
```

Rules:

1. Store source citations for every label.
2. Separate known risk, conditional risk, possible risk, and no established risk if the source system uses those categories; define the mapping to internal output explicitly.
3. Do not collapse unknown into Low.
4. Keep assay conditions and exposure type.
5. Split by scaffold or chemical series so close analogues do not appear in both development and validation partitions.
6. Include the benchmark compounds that exposed failures: paracetamol, azithromycin, dofetilide, terfenadine, cisapride, sotalol, chloroquine, hydroxychloroquine, verapamil, a macrolide comparator, and common low-risk controls.
7. Add literature-backed source-specific fixtures, retaining assay temperature, measured-versus-nominal concentration status, free/total exposure status, and channel identity.

Evaluate separately:

- IC50 regression error by channel.
- dose-response block error against measured channel data where available.
- safety-margin category agreement.
- similarity Top-K retrieval and scaffold recall.
- mechanistic risk agreement.
- final interpretation agreement.
- abstention coverage and unsafe false-negative rate.

Do not report a single accuracy number without the denominator, label system, exposure definition, and abstention policy.

### Phase 9: Test and quality gates

Add focused tests before broad integration tests:

1. Fingerprint determinism tests.
2. Canonicalization and invalid-SMILES tests.
3. Tanimoto ranking tests with hand-checkable molecules.
4. Top-K tie ordering tests.
5. Threshold and no-match tests.
6. Missing label/IC50/provenance tests.
7. Hill block boundary and monotonicity tests.
8. Mechanistic risk rule boundary tests for 5%, 20%, 50%, margin 10, and margin 30.
9. Concentration-type and unit-conversion tests for free plasma, total plasma, measured assay, nominal assay, and unknown type.
10. Evidence-domain warning tests at the configured boundaries and just outside them.
11. Evidence conflict and abstention tests.
12. Pipeline contract tests verifying existing IC50/dose-response values are unchanged.
13. Regression tests for invalid model/database/simulation dependencies.
14. Benchmark tests for the known drug cases, with expected categories stored as curated test data rather than hard-coded undocumented assumptions.

Run in this order:

```text
unit similarity tests
unit mechanistic-risk tests
unit pipeline contract tests
curated benchmark tests
full existing test suite
```

### Phase 10: Migrate and deprecate the classifier

1. Introduce a feature flag such as `CARDIOTOXICITY_DECISION_MODE=mechanistic_classification_similarity`.
2. Run old and new paths in shadow mode and log both outputs without exposing the old class as truth.
3. Compare new outputs against the curated validation set.
4. Update reports and API consumers to use the new response fields.
5. Remove SHAP from the primary path; replace it with retrieval explanations and rule reasons.
6. Mark `train_classifier.py`, `cluster_dataset.py`, `label_clusters.py`, and old classifier artifacts as legacy.
7. Keep the files for reproducibility until the migration is accepted, then archive them rather than silently deleting them.
8. Remove the feature flag only after the new contract and validation gates pass.

## 6. Exact File Change Plan

### New files

- `classification_backend/similarity/__init__.py`
- `classification_backend/similarity/fingerprints.py`
- `classification_backend/similarity/reference_database.py`
- `classification_backend/similarity/retriever.py`
- `classification_backend/similarity/schemas.py`
- `classification_backend/dataset/build_similarity_reference.py`
- `classification_backend/mechanistic_risk.py`
- `classification_backend/evidence_interpreter.py`
- `classification_backend/tests/unit/test_similarity_fingerprints.py`
- `classification_backend/tests/unit/test_similarity_retriever.py`
- `classification_backend/tests/unit/test_mechanistic_risk.py`
- `classification_backend/tests/unit/test_evidence_interpreter.py`
- `classification_backend/tests/unit/test_similarity_pipeline.py`
- `classification_backend/dataset/curated_cardiotoxicity_validation.csv` once independently curated

### Files to modify

- `pipeline/prediction_pipeline.py`
- `pipeline/utils.py`
- `classification_backend/dose_response/validation.py` only where new warnings or domain metadata are needed
- `classification_backend/dose_response/safety_margin.py` only if configuration/provenance fields are needed
- `classification_backend/dose_response/concentration_profiles.py` to enforce typed free/total/assay exposure handling
- `requirements.txt` only if the installed RDKit API requires a compatible package adjustment
- project documentation and API/report generation code that currently expects `classification` or SHAP output

### Files to isolate as legacy

- `classification_backend/assessment/legacy_classifier/train_classifier.py`
- `classification_backend/assessment/legacy_classifier/cluster_dataset.py`
- `classification_backend/assessment/legacy_classifier/label_clusters.py`
- `classification_backend/assessment/legacy_classifier/generate_validation_dataset.py`
- `classification_backend/assessment/evaluate.py` as the single active evaluator
- `classification_backend/xai/`
- old classifier artifacts under `classification_backend/saved_models/`
- the separate older classifier wrapper under `pipeline/classifier.py` and `pipeline/models.py`

Do not delete or rewrite these in the first implementation phase. They document the prior experiment and may be needed for comparison.

## 7. Recommended Delivery Order

1. Add schemas and status enums.
2. Add deterministic RDKit parsing/canonicalization/fingerprints.
3. Build and validate the molecule-level reference database.
4. Add exact Top-K retrieval and unit tests.
5. Add mechanistic risk rules around existing dose-response and safety-margin utilities.
6. Add evidence interpretation and abstention rules.
7. Integrate the new objects into `PredictionPipeline` while preserving existing output fields.
8. Add edge-case and contract tests.
9. Build the curated independent validation set.
10. Run the live benchmark and scaffold-held-out evaluation.
11. Run shadow mode against the legacy classifier.
12. Switch the user-facing decision to the new path.
13. Archive the classifier path after acceptance.

## 8. Definition of Done

The implementation is complete only when:

- a valid SMILES and typed concentration return IC50, dose-response, blockage classification, safety-margin, similarity, mechanistic-risk, interpretation, and warning objects;
- existing ORd-ready dose-response values are unchanged;
- a molecule absent from the reference database still receives a mechanistic result but clearly reports no analogue evidence;
- invalid SMILES, invalid dose, missing models, missing database, extrapolated doses, and missing labels have explicit statuses;
- similarity results are deterministic and provenance-rich;
- no synthetic `RiskClass` is used as independent ground truth;
- no classifier probability is averaged with Tanimoto similarity;
- concentration-domain extrapolation is visible;
- Hill blockage is monotonic and boundary-tested;
- blockage bands and mechanistic classification are distinct from clinical diagnosis;
- free, total, measured-assay, and nominal-assay concentrations are not mixed;
- the curated validation set includes independent labels and citations;
- accuracy, false negatives, abstention, and coverage are reported separately;
- the old classifier is clearly labeled legacy and is not presented as validated cardiotoxicity prediction.

## 9. Final Architecture

```text
                         +-----------------------------+
SMILES + typed exposure -> Input validation              |
                         +--------------+--------------+
                                        |
                 +----------------------+----------------------+
                 |                                             |
                 v                                             v
      IC50 regression path                            Similarity path
                 |                                             |
                 v                                             v
      ChannelBlockGenerator                       RDKit canonicalization
                 |                                             |
                 v                                             v
      Hill channel block                           Morgan fingerprint
                 |                                             |
                 v                                             v
      Safety margins                         Exact Tanimoto Top-K retrieval
                 |                                             |
                 +----------------------+----------------------+
                                        |
                                        v
                           Evidence interpreter
                                        |
                                        v
               blockage classification + mechanistic risk
                 + analogue evidence + warnings
                                        |
                                        v
                         optional ORd simulation output
```

The result is not a replacement of the project's mechanistic science with a nearest-neighbor guess, and it is not the old synthetic classifier relabeled. It is a transparent classification-plus-evidence system that keeps the existing physiological pipeline, adds chemical context, and refuses to claim more certainty than the available data supports.
