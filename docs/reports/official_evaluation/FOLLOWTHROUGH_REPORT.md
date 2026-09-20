# Paper-aligned evaluation follow-through

Update: the later [gap investigation](GAP_INVESTIGATION.md) corrects the historical
adapter's over-rejection of reference-only revisions and reports a complete
Qwen3-VL-8B local rescore. Earlier bounds/counts below describe the pre-fix audit;
use the new investigation for current values. QA and raw-output rules are unchanged.


2026-09-20. This implements the [execution plan](IMPLEMENTATION_PLAN.md).
The manuscript is authoritative. QA, PDFs and collaborator-owned `sxz/` were
read-only. This report supplements the original [A–K audit](FINAL_REPORT.md).

## A–B. Audit and recovered protocol

The previous audit located all 55 baseline result files, historical v4/v10
judges, caches, Table 2 CSV and Table 3 workbook. Their permissive parsing and
tri-class judging do not implement the current binary paper prompt. The public
package uses the paper's strict JSON, exact refusal and per-item macro metrics.

The additional [recovery inventory](RECOVERY_ARTIFACT_INVENTORY.json) examines
183 JSON files containing raw predictions and inventories 42 JSONL retry/journal
files under `sxz/models/`. Each candidate has its source SHA-256, legacy/current
ID coverage, strict parsing counts and current gold comparison. The archived
manifest contains 6,913 records, primarily old challenge/ablation checkpoints.
No archive or historical output was mutated or promoted automatically.

Concrete example: Qwen3-VL-4B strict-full Cross-PDF candidates contain 118 + 129
current-gold-matching records marked `retry2gpu`. They cannot certify the paper's
single-A800 run. Other candidates retain question/answer/evidence mismatches,
illegal raw responses, or incomplete prompt/PDF/checkpoint bindings. An absent
retry flag alone is not proof of single-GPU provenance.

## C–D. Implementation changes

Created:

- `scripts/run_local_pipeline.py`: real PDF inference, persistent raw submission,
  explicit new-run configuration, local Qwen Judge, public scoring and cache replay.
- `evaluation/prompts/pdf_inference.txt` and `pdf_provenance.json`: verbatim paper
  PDF prompt, SHA-256 `c73e58cc79cd96479be735f17dccd3a1005e08e937081c3b3033a841ace7a3da`.
- `tests/test_generation_audit.py`: original-byte preservation, tamper rejection,
  interruption/resume, fixed retry budgets and prompt identity.
- Plan, recovery inventory, preflight and this report.

Modified:

- `run_inference.py`: removed permissive nested-JSON/LaTeX repair and repetition
  heuristics; retains exact original and retry output bytes, hashes, triggers and
  failures. Retry settings stay fixed. Completed illegal/technical failures do
  not receive a new budget on resume. Failed generation retains the QA slot.
- `evaluation/validation.py`: verifies optional submission-level retry audit,
  hashes, status/order and final binding; represents technical generation failure.
- `eval_framework.py`: correct local Qwen3.6 image/text model class and explicit
  evaluation mode; preserves raw provider output.
- `evaluation_protocol.py`: inference/scoring versions 8/7; PDF prompt hash and
  generation audit/status bindings invalidate stale checkpoints.
- `run_judge.py`: propagates generation provenance into the Judge binding.
- `run_report.py`: internal diagnostic scope, no publication/reproduction claim.
- Regression tests, evaluation README and current project documentation.

No semantic Judge prompt or metric definition changed. No gold content changed.

## E–G. Dataset, local model and tests

[Preflight](followthrough_preflight.json) confirms 1,000 General, 200
Unanswerable, 200 Reasoning and 800 Multi-Document; 2,200 unique continuous IDs,
no duplicate question/answer/PDF tuples, positive/in-range gold evidence,
canonical empty-evidence refusals and 712 readable, hash-verified evaluation
PDFs. All 27 JSON files under `data/qa/` match the original task snapshot.

The final real run is `data/results/local_pdf_pipeline_20260920_final/`.
It selects QA0311/QA1136/QA1340/QA1465 solely by the shortest full PDF per task
then QA ID: 5/4/6/10 pages. All 25 pages are rendered at 144 DPI. The selection
is a diagnostic sample, not a representative performance or throughput estimate.

New inference uses Qwen3-VL-4B-Instruct; the Judge remains Qwen/Qwen3.6-27B.
One idle A800 is UUID-bound, BF16/SDPA, no CPU/disk offload, seed 42, greedy
sampling disabled. Inference: 512 tokens, three total attempts; Judge: 32 tokens,
two total attempts. Temperature/top-p/top-k are explicitly disabled. These are
new diagnostic settings, never asserted to be the missing historical settings.
The exact local weights, tokenizer/chat template, generation defaults, code,
inputs, prompts and runtime versions are hashed in the run directory.
Semantic prompt SHA-256 remains
`45785c6f699520192874355527135b437bf2d0793d88b38b7246c592d3dcbe70`.

The run has four legal predictions, no generation or Judge failure, three exact
binary Judge decisions, and deterministic refusal scoring. Two answers are
correct; the Unanswerable and Multi-Document predictions are wrong. Incorrect
answers are retained. The full report denominator is 2,200, including 2,196
missing predictions. Cache-only replay preserves all headline metrics.
Final verification: **517 project tests and 56 isolated public-evaluator tests passed**;
13 generation-audit tests cover the new retry/resume contract. The final run took
18.65 seconds for model loading/inference and 17.43 seconds for model loading/judging,
excluding input/checkpoint hashing. Identical-command resume made zero inference
or Judge calls and loaded no models. Independent CLI/cache metrics match exactly.
See [verification](followthrough_verification.json) and the
[public pilot summary](local_pdf_pipeline/summary.json).

## H–J. Tables and remaining discrepancies

The requested equality has been disproved for the available historical inputs.
The exhaustive earlier audit proves 30 Table 2 answer cells and 47 Table 3 cells
outside even their most favorable possible strict-score bounds. Four real local
Judge calls completed the remaining answer decisions for three whole-baseline
diagnostics (each retaining 2,200 slots):

| Model / Table 2 All (%) | Paper | Strict stored-output diagnostic | Difference (pp) |
| --- | ---: | ---: | ---: |
| MiniCPM-V 2.6 | 15.59 | 0.00 | -15.59 |
| InternVL2.5-8B | 22.32 | 0.00 | -22.32 |
| Gemma 3 27B | 47.32 | 0.045454545454545456 | -47.27454545454545 |

[Table 2 per-cell comparison](fast_reproduction_check/table2_local.csv) and
[Table 3 per-cell comparison](fast_reproduction_check/table3_local.csv) retain
exact new values, paper values and differences. All 11 models' bounds are in
[Table 2 bounds](fast_reproduction_check/table2_bounds.csv) and
[Table 3 bounds](fast_reproduction_check/table3_bounds.csv).
These values evaluate the stored artifacts under the current strict contract;
they do not imply that a model has no ability to answer the questions.

The new four-item PDF pilot verifies the production chain; it cannot change
those historical discrepancies or prove a full benchmark score. We did not
spend 11,550 extra semantic judgments trying to establish a false equality.
Historical full-precision paper values, binary-Judge configuration/decisions and
complete final correction records remain unresolved. The paper's illegal-output
evidence rule also remains incomplete; the public scorer exposes unresolved
headlines when illegal items occur, rather than inventing page recovery rules.
New full inference would produce a new experiment, with no guarantee of matching
the old table. API baselines additionally require their exact serving identities
and access configurations. No QA or table numbers were changed to hide this.

## K. Commands

Minimal public evaluator environment (Python 3.10+):

```bash
python -m venv .venv-eval
.venv-eval/bin/python -m pip install -r requirements-eval.txt
.venv-eval/bin/python evaluation/preflight.py --output /tmp/sciencedoc-preflight.json
.venv-eval/bin/python scripts/validate_submission.py --predictions examples/predictions.example.jsonl
.venv-eval/bin/python -m pytest -q tests/test_validation.py tests/test_metrics.py tests/test_judge.py tests/test_e2e.py
```

A partial example reports missing IDs; it is not a complete submission. Real
judging requires an explicitly configured paper-model server or the local ML
environment. In the installed local environment:

```bash
python scripts/run_local_pipeline.py --output-dir data/results/local_pdf_new --gpu 2
# Identical command verifies terminal inference records and reuses judge cache.
python scripts/run_local_pipeline.py --output-dir data/results/local_pdf_new --gpu 2
python evaluation/evaluate.py \
  --predictions data/results/local_pdf_new/predictions.jsonl \
  --judge-config data/results/local_pdf_new/judge_config.json \
  --judge-cache data/results/local_pdf_new/judge_cache.jsonl --offline \
  --output data/results/local_pdf_new/cli_report.json
```

The CLI exits 2 for the deliberately partial submission while saving the report.
The ML environment used here is PyTorch 2.8.0+cu128, Transformers 5.6.2, NumPy,
pypdfium2 and the local checkpoint directories; the minimal evaluator requirements
do not download models or CUDA libraries.

Historical comparison, explicitly expected to report discrepancies (exit 2):

```bash
python evaluation/check_reproduction.py --output-dir data/results/reproduction_new
python scripts/judge_reproduction_jobs.py --audit-dir data/results/reproduction_new \
  --model-dir models/Qwen3.6-27B --gpu 2
python .agents/skills/pku-qa-maintainer/scripts/sync_project_docs.py --sync
python -m pytest -q
```

## Other requested deliverables

The English root README, contributor acknowledgments and relocated experiment
console were already published in earlier commits. Their verification is in the
original audit. GitHub's automatic Contributors graph requires the individuals'
actual commits attributed to their accounts; acknowledgments alone do not populate
that graph, and contributor commit authorship must not be fabricated.

The Google Drive upload is ready at
`data/exports/google_drive/ScienceDoc_PDFs_complete_20260920.zip` (6,774,998,969
bytes, 1,717 PDFs including all 712 evaluation PDFs, unchanged release QA copies
and manifests). Its SHA-256 is
`540012e238d8abda1eb55e4906eaff57c308aa46b770739e922c46b0a38f8fcf`.
Upload the ZIP and its `.sha256` sidecar to Drive. The 6.8 GB ZIP stays outside
Git; source, tests, documentation and small verification artifacts are published
to GitHub.
