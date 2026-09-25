# Dose-Response and Literature Validation Report

## Executive Conclusion

The implementation of the Hill blockage equation is mathematically correct. The current code calculates:

$$
B(C) = 100 \times \frac{C^n}{IC_{50}^n + C^n}
$$

where:

- $C$ is the concentration passed to the function;
- $IC_{50}$ is in the same concentration unit;
- $n$ is the Hill coefficient, currently defaulted to `1.0`; and
- $B$ is percentage channel block.

With $n=1$:

- $C = 0.01 \times IC_{50}$ gives approximately `0.99%` block;
- $C = 0.1 \times IC_{50}$ gives approximately `9.09%` block;
- $C = 0.3 \times IC_{50}$ gives approximately `23.08%` block;
- $C = IC_{50}$ gives exactly `50%` block;
- $C = 3 \times IC_{50}$ gives `75%` block; and
- $C = 10 \times IC_{50}$ gives approximately `90.91%` block.

Therefore, a low dose can legitimately produce high block only when the predicted or measured IC50 is even lower than that dose. There is no reversal in `hill_equation.py` and no mathematical scaling error in `ChannelBlockGenerator`.

The real concern is **concentration meaning and model validity**:

1. The pipeline treats `dose_nm` as a direct concentration in nM.
2. It does not currently require the caller to specify whether that value is total plasma, free plasma, intracellular, bath, or nominal assay concentration.
3. A clinical dose in mg is not interchangeable with a plasma concentration in nM.
4. A nominal 10,000 nM input is not automatically a normal therapeutic cardiac exposure.
5. Published cardiac safety assessments commonly compare **unbound effective therapeutic plasma concentration** with channel IC50, not an arbitrary total concentration.

## What the Literature Actually Supports

### Redfern et al. 2003: broad clinical drug comparison

Redfern et al. compared preclinical electrophysiology, clinical QT effects, and torsades de pointes evidence for 100 drugs. The analysis explicitly set hERG/IKr data against **unbound effective therapeutic plasma concentration** (`ETPC_unbound`).

The key findings were:

- For the highest-risk antiarrhythmic category, hERG/IKr IC50 values were generally close to or overlapped unbound therapeutic exposure.
- Ratios of hERG/IKr IC50 to maximum unbound therapeutic concentration ranged from `0.1` to `31` fold for category 1 drugs, `0.31` to `13` fold for withdrawn drugs, and `0.03` to `35` fold for drugs with numerous human TdP reports.
- Most drugs without human TdP reports had more than a `30-fold` separation between hERG activity and unbound therapeutic exposure, with important exceptions such as verapamil because multi-channel effects matter.
- The authors proposed that a `30-fold` hERG IC50-to-unbound-Cmax margin may be sufficient as a provisional development margin, while emphasizing that integrated evidence is required and hERG block alone is not equivalent to torsades risk.

This supports using the ratio:

$$
R_{hERG} = \frac{IC_{50,hERG}}{C_{free,therapeutic,max}}
$$

as a safety-margin descriptor. It does **not** support a universal concentration such as 10,000 nM for every medicine.

### ICH E14 and S7B

The ICH E14 guideline covers clinical QT/QTc evaluation and proarrhythmic potential. The combined E14/S7B framework supports an integrated nonclinical and clinical assessment rather than relying on a single hERG number. The guidance is relevant to this project because it emphasizes concentration-response interpretation and the relationship between nonclinical assay data and clinical exposure.

The 2022 FDA-led hERG positive-control work also emphasizes that nominal concentrations in patch-clamp experiments can differ from measured concentrations because of nonspecific binding and drug loss. Dofetilide, cisapride, terfenadine, sotalol, and E-4031 are used as hERG assay sensitivity controls; the first four are clinical drugs associated with high TdP risk.

### Delaunois et al. 2021: real medicine examples

In an integrated CiPA assessment of antimalarial and macrolide drugs:

- Chloroquine inhibited IKr with an IC50 around `1 uM`, or approximately `1,000 nM`.
- Hydroxychloroquine IKr IC50 was approximately `3-7 uM`, or `3,000-7,000 nM`.
- The macrolides tested had weak effects on the measured currents with IC50 values above approximately `300-1,000 uM`, or above `300,000-1,000,000 nM`.
- The study generated concentration-response curves encompassing and exceeding therapeutic free plasma levels and concluded that chloroquine and hydroxychloroquine carried proarrhythmic risk in the tested context, while the macrolides had much weaker direct ion-current effects.

These examples show why both exposure and IC50 are required. A drug with a 1,000 nM IC50 may have low block at a 10 nM free exposure but substantial block at a 1,000 nM exposure.

## Recommended Blockage Bands

These are engineering display bands for the Hill output, not clinical diagnostic thresholds:

| Block percentage | Display band | Interpretation |
|---:|---|---|
| `< 5%` | Minimal | Little direct channel inhibition predicted by this model. |
| `5% to <20%` | Low | Detectable/mild inhibition; interpret with channel, exposure, and uncertainty. |
| `20% to <50%` | Moderate | Meaningful model-predicted inhibition; warrants review of exposure and safety margin. |
| `>= 50%` | High | At or above the IC50-equivalent response; strong concern for that channel, not an automatic TdP diagnosis. |
| `>= 90%` | Near-maximal | Concentration is about 10 times IC50 or greater for $n=1$. |

The `5%`, `20%`, and `50%` values are useful explanatory bands because they correspond to interpretable Hill ratios for $n=1$:

| Block | Approximate concentration / IC50 ratio |
|---:|---:|
| `1%` | `0.01` |
| `5%` | `0.053` |
| `10%` | `0.111` |
| `20%` | `0.25` |
| `50%` | `1.0` |
| `80%` | `4.0` |
| `90%` | `9.0` |

These ranges should be applied channel by channel. A 50% hERG block and a 50% Nav1.5 block should not automatically be interpreted identically, and multi-channel effects can alter electrophysiological risk.

## Genuine Medicine Benchmark Pairs

The following are suitable reference cases for demonstrating the model, provided the exposure is explicitly labeled as free concentration, total concentration, or nominal assay concentration. The table uses the published risk context and the Hill ratio to define what should be demonstrated; it does not claim that every listed exposure is a measured patient Cmax.

| Reference medicine or control | Literature context | Demonstration concentration | Representative IC50 / exposure relationship | Expected Hill display |
|---|---|---:|---|---|
| Dofetilide | Clinical high-TdP-risk hERG positive control | `10 nM` | Use a measured assay IC50 from the selected source; choose a demonstration exposure near or below that IC50 | Low to high depending on ratio; high-risk evidence should remain visible even if the chosen exposure is low |
| Cisapride | Withdrawn/high-TdP-risk hERG positive control | `10 nM` | Use source-specific hERG IC50 and matched assay concentration | Low to high depending on ratio; do not infer safety from one low concentration |
| Terfenadine | Withdrawn/high-TdP-risk hERG positive control | `10 nM` | Use source-specific hERG IC50 and matched assay concentration | Low to high depending on ratio; verify concentration because nonspecific binding can lower free assay exposure |
| Sotalol | Clinical high-TdP-risk hERG positive control | `10,000 nM` | Demonstrates the importance of exposure/IC50 ratio for a hydrophilic drug | Use measured IC50; do not assume the model's predicted IC50 is correct |
| Chloroquine | CiPA assessment; IKr IC50 approximately `1,000 nM` | `10 nM` | $C/IC50 \approx 0.01$ | Approximately `1%` IKr block for $n=1$ |
| Chloroquine | CiPA assessment; IKr IC50 approximately `1,000 nM` | `1,000 nM` | $C/IC50 \approx 1$ | Approximately `50%` IKr block |
| Hydroxychloroquine | CiPA assessment; IKr IC50 approximately `3,000-7,000 nM` | `100 nM` | $C/IC50 \approx 0.014-0.033$ | Approximately `1.4-3.2%` IKr block |
| Hydroxychloroquine | CiPA assessment; IKr IC50 approximately `3,000-7,000 nM` | `3,000-7,000 nM` | $C/IC50 \approx 1$ | Approximately `50%` IKr block |
| A low-risk macrolide comparator from Delaunois et al. | Weak direct ion-current effects; IC50 above approximately `300,000-1,000,000 nM` | `100-1,000 nM` | $C/IC50 \le 0.0033$ | Approximately `<0.33%` block for $n=1$ |
| Verapamil | Redfern exception; multi-channel effects can separate hERG block from clinical QT risk | `10-100 nM` | Interpret hERG and other channels together | Demonstrates why hERG block alone must not define final risk |

The first four rows are positive-control medicines, not “safe” and “unsafe” labels to be inferred from a single concentration. The chloroquine/hydroxychloroquine/macrolide rows provide numerical examples from a published integrated assessment. The benchmark should store the source, assay temperature, concentration verification status, free/total status, and channel name for every value.

## Why 10,000 nM Is Not a Universal Normal Exposure

`10,000 nM` equals `10 uM`. That can be a valid in-vitro test concentration or a high exposure for some medicines, but it is not a general therapeutic plasma concentration for all medicines.

Examples from the project benchmark illustrate the problem:

- Atorvastatin was tested at `0.05 nM` in the earlier exploratory benchmark.
- Amlodipine was tested at `10 nM`.
- Losartan was tested at `100 nM`.
- Omeprazole was tested at `500 nM`.
- Other compounds may have therapeutic or assay exposures in the low nM, hundreds of nM, several uM, or higher ranges.

The correct comparison is not:

```text
medicine -> assume 10,000 nM -> calculate block
```

It is:

```text
medicine -> obtain measured exposure definition -> convert to free nM if possible -> compare with channel-specific IC50 -> calculate block
```

For a clinical safety interpretation, use unbound Cmax or another justified free exposure. For an in-vitro assay interpretation, use the measured bath concentration when available. Do not mix nominal bath concentration, total plasma concentration, and free plasma concentration in the same reference column.

## Diagnosis of the Current Project Behavior

The current `hill_equation.py` and `channel_block_generator.py` are directionally correct. The observed surprising results can come from four other sources:

1. **Predicted IC50 error:** the IC50 regression may predict an unrealistically low IC50 for a molecule/channel.
2. **Exposure semantic error:** the caller may pass a dose in nM that is not the relevant free concentration.
3. **Unit error upstream:** a value in uM, mg/L, or mg may be passed as though it were nM.
4. **Classifier artifact:** the old XGBoost classifier was trained on synthetic doses only through `3,000 nM` and can produce non-monotonic classes outside that range.

The Hill block itself should always satisfy these tests for positive IC50 and Hill coefficient:

- block at zero concentration is `0%`;
- block increases monotonically as concentration increases;
- block at IC50 is `50%` when the Hill coefficient is positive and Emax is `100`;
- block remains within `[0, 100]`;
- increasing IC50 lowers block at the same concentration.

## Required Code/Data Changes

### 1. Make concentration semantics explicit

Change the public input contract from only:

```json
{"dose_nm": 10000}
```

to a structure that records:

```json
{
  "concentration_nm": 100.0,
  "concentration_type": "free_plasma",
  "exposure_context": "therapeutic",
  "source": "curated_reference_or_user_input"
}
```

Keep `dose_nm` temporarily as a backwards-compatible alias, but mark its concentration type as `unspecified` and emit a warning.

### 2. Add unit and range validation

Reject non-finite and non-positive concentration values. Preserve uM conversion explicitly:

- `1 uM = 1,000 nM`
- `10 uM = 10,000 nM`

Never infer this conversion from magnitude alone.

### 3. Separate three concentration domains

Store these as distinct fields:

- `nominal_assay_concentration_nm`
- `measured_assay_concentration_nm`
- `free_plasma_concentration_nm`

Only compare values within the same domain in automated interpretation.

### 4. Add monotonicity tests

For every channel and fixed IC50:

```text
block(0.1 * IC50) < block(IC50) < block(10 * IC50)
```

For a concentration series, run `validate_block_curve` and fail if block decreases.

### 5. Retire the old classifier from dose-response interpretation

The old classifier can output a legacy label for comparison, but it must not override the continuous channel block or safety-margin results. It should not be used outside its trained concentration domain.

### 6. Use literature-derived reference fixtures

Add a curated fixture with source-specific values for:

- dofetilide;
- cisapride;
- terfenadine;
- sotalol;
- chloroquine;
- hydroxychloroquine;
- a macrolide comparator; and
- verapamil.

Each fixture must include drug name, SMILES, channel, IC50, concentration type, assay conditions, reference DOI/PMID, and evidence category.

## Recommended Display for the Project

For every query, show:

1. concentration and its semantic type;
2. each channel IC50;
3. each channel block percentage;
4. each channel safety margin $IC50/C$;
5. a Low/Moderate/High block band per channel;
6. `supported` or `extrapolated` dose-domain status;
7. analogue matches and their evidence source; and
8. a disclaimer that channel block is not equivalent to clinical torsades risk.

A good example output is:

```text
Exposure: 100 nM free plasma equivalent
hERG IC50: 3,000 nM
hERG block: 3.23% (minimal/low)
Safety margin: 30-fold (provisional safe-margin boundary)
Evidence status: model-derived; no curated analogue label
```

A high-block example is:

```text
Exposure: 3,000 nM measured assay concentration
hERG IC50: 3,000 nM
hERG block: 50.0% (high band)
Safety margin: 1-fold
Evidence status: high mechanistic concern; review integrated multi-channel and clinical evidence
```

## Sources

1. Redfern WS, Carlsson L, Davis AS, et al. *Relationships between preclinical cardiac electrophysiology, clinical QT interval prolongation and torsade de pointes for a broad range of drugs: evidence for a provisional safety margin in drug development.* Cardiovascular Research. 2003;58(1):32-45. PMID: 12667944. DOI: `10.1016/S0008-6363(02)00846-5`. Europe PMC record: https://europepmc.org/article/MED/12667944
2. International Council for Harmonisation. *E14: Clinical Evaluation of QT/QTc Interval Prolongation and Proarrhythmic Potential for Non-Antiarrhythmic Drugs* and E14/S7B Q&A materials. https://database.ich.org/sites/default/files/E14_Guideline.pdf
3. Alvarez Baron C, Thiebaud N, Ren M, et al. *hERG block potencies for 5 positive control drugs obtained per ICH E14/S7B Q&As best practices: Impact of recording temperature and drug loss.* Journal of Pharmacological and Toxicological Methods. 2022;117:107193. PMID: 35792285. DOI: `10.1016/j.vascn.2022.107193`.
4. King TI, Indapurkar A, Tariq I, et al. *Determination of five positive control drugs in hERG external solution (buffer) by LC-MS/MS to support in vitro hERG assay as recommended by ICH S7B.* Journal of Pharmacological and Toxicological Methods. 2022;118:107229. PMID: 36334898. DOI: `10.1016/j.vascn.2022.107229`.
5. Delaunois A, Abernathy M, Anderson WD, et al. *Applying the CiPA approach to evaluate cardiac proarrhythmia risk of some antimalarials used off-label in the first wave of COVID-19.* Clinical and Translational Science. 2021;14(3):1133-1146. PMID: 33620150. DOI: `10.1111/cts.13011`.

## Final Recommendation

Keep the Hill equation. Change the meaning and validation of its inputs.

Use published, source-specific, exposure-matched examples to demonstrate low and high block. Do not create a universal “normal medicine concentration” of `10,000 nM`. The project should compare free or measured concentrations with channel-specific IC50 values, expose the concentration domain, and report continuous block plus safety margin before any overall interpretation.
