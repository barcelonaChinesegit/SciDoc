# Table 2/3 gap investigation and adapter correction

> **2026-09-20 provenance clarification:** [PROVENANCE_RECONCILIATION.md](PROVENANCE_RECONCILIATION.md) is the latest interpretation. Historical experiments and their aggregates are traceable. Current-release strict diagnostics use different input/protocol combinations and do not disprove those experiments. Earlier “cannot reproduce” wording applies only to those diagnostic assumptions; newly generated Judge labels are not human ground truth.

Date: 2026-09-20. Current QA and PDFs remain immutable. This investigation
supersedes the earlier adapter's treatment of all reference revisions as input
failures. The paper's raw-output rules and semantic prompt are unchanged.

## What counts as a small difference

We report percentage-point differences, task/discipline differences and affected
QA counts. One answer changes All accuracy by 100/2200 = 0.0454545 percentage
points; the manuscript prints two decimals. Different fixed-input deterministic
metrics are not explained by sampling error. A small overall delta can also hide
large subgroup shifts. We do not invent a new official tolerance or certify
agreement merely because rounded or aggregate values look close.

The stored MiniCPM-V 2.6 / InternVL2.5 / Gemma predictions have large failures
under the paper contract. These are not plausible rounding differences or small
Judge fluctuations:

| Baseline | Paper All | Previously completed strict answer diagnostic | Gap (pp) |
| --- | ---: | ---: | ---: |
| MiniCPM-V 2.6 | 15.59 | 0.00 | -15.59 |
| InternVL2.5-8B | 22.32 | 0.00 | -22.32 |
| Gemma 3 27B | 47.32 | 0.045454545454545456 | -47.27454545454545 |

The adapter correction does not change the legal semantic inputs of those three
completed diagnostics. It changes how some malformed Gemma rows are classified
(illegal rather than reference-version failure); the answer totals remain the same.
These are assessments of the stored outputs, not claims about model capability.

## A concrete bug found and fixed

`evaluation/reproduce.py` previously marked every embedded reference-answer or
evidence change as `technical_failure`, even when neither field had been input
to the evaluated model. This conflated inference inputs with scoring references
and overstated the number of unavailable predictions. The earlier audit should
not have treated all such records alike.

The corrected adapter permits a new current-reference rescore only when:

1. The actual question is byte-identical to current gold.
2. Both `gold_answer_sent` and `gold_evidence_pages_sent` are explicitly false.
3. The recorded PDF path maps to the current asset's canonical/legacy path in the
   existing manifest; any recorded page count/hash also agrees.
4. The original raw output independently passes or fails the unchanged parser.

It never substitutes the old gold or cached old decision. It does not prove
historical PDF-byte/model/prompt provenance when those hashes were not recorded.
Changed questions and insufficient input provenance remain unresolved failures.
Qwen3-VL-8B recovers 162 reference-only records, of which 159 raw outputs are
legal and three remain illegal. Its current counts are 2,122 legal, 23 illegal,
43 input-version failures and 12 missing predictions: **denominator 2,200**.

The updated no-Judge feasibility analysis still proves at least **27 Table 2
answer cells and 42 Table 3 cells** cannot match; these replace the earlier
30/47 bounds. All-model counts are in
[model_gap_summary.csv](gap_investigation/model_gap_summary.csv), and every QA
has a source-bound [attribution row](gap_investigation/item_attribution.csv).

## Why the largest gaps occur

The source-level defects are concrete:

- `sxz/internvl2_5_8b_pdfqa_all5_gpu2345_parallel_v17_sdpa_docbudget96_promptv1_failfast/common.py:151`
  strips Markdown fences and, after a JSON error, extracts text between braces.
  `worker.py:298` consumes this parsed result. Consequently a fenced raw output
  can pass the historical parser without the paper's required format correction.
- `sxz/gemma3_27b_pdfqa_all5_v2/run_gemma3_pdfqa.py:299` similarly strips fences;
  `extract_json_candidate` also tries Python literal parsing / brace extraction.
- `sxz/evaluate_11models_5datasets_calibrated_fixeddenom_seed42_gpu2345_parallel_v4.py:1417`
  prefers normalized `answer_pre` before raw output. Its `is_illegal_answer` does
  not enforce the complete current two-field JSON contract. Its prompt at line
  228 requests CORRECT/PARTIAL/WRONG with a different calibration rubric.
- The paper explicitly permits only outer whitespace and evidence set ordering /
  deduplication; extra explanations or Markdown must be counted incorrect if still
  illegal after bounded correction. No recovered successful correction bytes may
  be fabricated by stripping the offending text.

There are **489** historically correct InternVL2.5 rows whose original raw output
is now illegal, and **1,027** such Gemma rows. InternVL2.5 has 2,190 fenced raw
outputs in its mapped, otherwise eligible set; Gemma has 2,155. This explains
large losses before any new semantic Judge decision is made. All old scripts are
collaborator-owned read-only references. The maintained evaluator already rejects
these repairs; this task does not edit `sxz/` or relax the paper to copy them.

Remaining question changes are substantive enough to require new inference. For
example, QA0051's stored question asks "What modifies the drift in the
barrier-constrained SDE?"; current gold requests the penalty term in SDE (4.5)
applied to a specified drift. QA0031 adds conditioning on first-passage time;
QA0078 adds uncertainty/asymmetry conditions. We do not silently deem these
prompts identical. An additional search for unique exact `(PDF, question)` matches
found no alternative current IDs to recover the mismatched/missing records.

## Full Qwen3-VL-8B semantic rescore

The new command uses the unchanged paper prompt and local Qwen/Qwen3.6-27B.
All 1,934 eligible answerable predictions are judged; Unanswerable items use the
exact canonical label rule. Every one of the 2,200 benchmark slots remains in the
report. The run is `data/results/qwen8b_current_gold_rescore_20260920/`.

Batching is an execution optimization, not a scoring change. New-run settings:
one UUID-bound idle A800, BF16/SDPA, greedy decoding, seed 42, thinking disabled,
32 output tokens, two total attempts, batches up to 32 with left padding, no
prompt truncation, no CPU/disk offload. Inputs, checkpoint artifacts, tokenizer,
chat template, defaults and versions are recorded. These are not claimed to be
recovered historical settings. Original replies and every cached decision are
retained. The complete semantic rescore finished in 599.93 seconds (including checkpoint
hashing), with zero Judge failures and no retry. Its cache-only replay yields
identical answer, evidence-status and discipline results.

## Validation and reproduction commands

```bash
python -m pytest -q tests/test_reference_revision.py tests/test_reproduction_bounds.py tests/test_judge.py
python evaluation/investigate_gaps.py --output-dir data/results/gap_audit_new
python scripts/rescore_baseline_local.py --model Qwen3-VL-8B \
  --output-dir data/results/qwen8b_rescore_new --gpu 2 --batch-size 32
python scripts/rescore_baseline_local.py --model Qwen3-VL-8B \
  --output-dir data/results/qwen8b_rescore_new --offline
python .agents/skills/pku-qa-maintainer/scripts/sync_project_docs.py --sync
python -m pytest -q
```

A comparison exit code of 2 reports discrepancies/unresolved evidence, while the
machine-readable report preserves every item and its failure category. The fixed
paper protocol is not replaced with a historical-compatibility scoring mode.


## Actual Qwen3-VL-8B differences

| Table 2 answer metric (%) | Paper | New current-gold diagnostic | New − paper (pp) |
| --- | ---: | ---: | ---: |
| All | 68.32 | 55.31818181818182 | -13.001818181818173 |
| General | 83.20 | 74.60 | -8.60 |
| Unanswerable | 42.50 | 42.50 | 0.00 |
| Reasoning | 71.50 | 64.50 | -7.00 |
| Multi-Document | 55.38 | 32.125 | -23.255 |

This is a large discrepancy and should not be described as acceptable ordinary
run variation. [Table 2 CSV](gap_investigation/qwen8b/table2.csv) preserves full
precision; [Table 3 CSV](gap_investigation/qwen8b/table3.csv) covers all eight
disciplines. Discipline gaps range from -5.27 pp (Quantitative Biology) to
-23.82 pp (Electrical Engineering and Systems Science). They are far larger
than changes of one or two discipline assignments.

The historical workbook has 1,503 correct mapped answers; the new run has 1,217.
The **286-answer net loss** is exactly accounted for:

| Exclusive cause group | Historical correct | New correct | Net lost |
| --- | ---: | ---: | ---: |
| Legal outputs, unchanged stored references | 1328 | 1099 | 229 |
| Legal outputs, reference-only revisions | 132 | 118 | 14 |
| Illegal original outputs | 11 | 0 | 11 |
| Changed/unsupported question input | 32 | 0 | 32 |
| Total | 1503 | 1217 | 286 |

Within the reference-revision group, 18 formerly correct predictions become
incorrect and four formerly incorrect ones become correct. The adapter fix
recovers 118 correct answers that would otherwise have been incorrectly zeroed;
it does not guarantee that a revised reference preserves the old verdict.
All 12 missing current IDs remain in the denominator. See the source-bound
[attribution](gap_investigation/qwen8b_attribution.json).

There are 234 old-correct/new-incorrect cases where the workbook's question,
reference answer and model answer **all match the new Judge's inputs exactly**.
This isolates a semantic-judging disagreement from dataset/input changes.
For example, QA1409's reference contains a tidal circularization timescale of
`1.1 × 10^6 yr`, while the prediction says `8.4 × 10^3 years`. The historical
workbook labels this CORRECT; the paper-prompt Judge labels it INCORRECT. The
paper explicitly requires changed numerical facts to be rejected. Changing the
new Judge to accept it merely to match the old score would violate the paper.

Evidence headlines stay unresolved for the 23 illegal rows; no page-recovery
rule was invented. Even allowing every unresolved item's PRF contribution any
value in [0,1], fixed-denominator audit bounds are:

| Evidence diagnostic bound | Lower | Upper | Paper |
| --- | ---: | ---: | ---: |
| E-Precision | 43.183963451964786 | 44.22941799741933 | 43.34 |
| E-Recall | 50.32016317016318 | 51.365617715617724 | 49.63 |
| E-F1 | 40.992802856367966 | 42.03825740182251 | 40.66 |

These are uncertainty bounds, not alternative headline scores or an instruction
to award illegal outputs credit. The smaller evidence gaps do not resolve the
13-pp answer gap.

## What is fixed, and what remains impossible to certify

Fixed and tested: reference-only revisions no longer cause automatic zeroing;
new strict raw parsing, the exact paper prompt, full denominator and content-bound
binary cache remain intact. New reported scores can be reproduced by offline
cache replay. No QA bytes, model answers, evidence pages or paper numbers changed.

Remaining: existing raw records and historical judgments do not implement the
current manuscript protocol. Old noncompliant final responses cannot be repaired
by inventing missing generation attempts. A new inference run with the paper
prompt and bounded correction is required for those records. The remaining
models need their own complete new binary rescore; that cannot turn the already
verified Qwen and format-related discrepancies into small differences.

The code can reproduce a frozen, fully recorded run. It cannot guarantee these
old table values under contradictory historical inputs/rules. We have not
changed the paper or introduced a compatibility scoring mode to claim success.


## Single-item and cache verification

Twelve deterministically selected old-correct/new-incorrect cases with identical
workbook question/reference/prediction text were checked again **one at a time**
using the same local checkpoint and decoding parameters. All 12 remained
INCORRECT. Six controls (reference answer and unrelated answer for each answerable
task) all returned their expected label. This targeted check found no batch-only
error in the selected disagreements; it is not an unbiased Judge accuracy estimate.
Full records are in [single-item checks](gap_investigation/qwen8b/single_item_checks/summary.json).

Combining the new full Qwen rescore with the other models' conservative bounds
now proves **31 Table 2 answer-cell differences and 51 Table 3 differences**.
The earlier 27/42 counts in this report describe the pre-semantic-score bounds.
The combined [Table 2](gap_investigation/table2_current_comparison.csv) and
[Table 3](gap_investigation/table3_current_comparison.csv) explicitly distinguish
exact scores from unjudged bounds. No unjudged answer was assigned a simulated
semantic verdict.

The fixed-input cache replay preserves all answer/evidence-status/discipline
results. Full project tests, immutable QA hashes and public-file checks are
recorded in [verification.json](gap_investigation/verification.json).
