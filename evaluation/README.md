# ScienceDoc evaluation

This package implements the current paper's PDF-mode submission contract and
Appendix B.12 macro metrics. The paper takes precedence over historical code.
It does not change QA, gold answers, evidence, or PDF bytes. It is a submission
scorer within the existing `pdf` protocol, not a third model protocol.

**Release status:** historical Table 2 reaggregation matches 99/99 displayed
cells; Table 3 matches 98/99. A new binary-judge rescore reproducing those tables
has **not** been established. The recovered historical evaluator uses different
prompts, permissive parsing, embedded old gold, and partly different questions.
See [the audit](../docs/reports/official_evaluation/FINAL_REPORT.md) and
[unresolved items](reconciliation.json). No historical tri-class decision is
silently reused as a paper-prompt binary decision.

The subsequent [exhaustive feasibility check and real local Judge verification](../docs/reports/official_evaluation/REPRODUCTION_DISCREPANCY.md)
prove that available stored outputs cannot reproduce all current table cells.
It checks all eleven models, derives explicit bounds, and completes three model
diagnostics with four real local Qwen3.6-27B judgments. It does not claim a full
eleven-model rescore. The internal rule-first scoring helpers have been removed.

## Installation and frozen inputs

Python 3.10+ is sufficient for this standalone package. The internal `pku_qa`
package requires Python 3.12+ and is not imported by this evaluator.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-eval.txt
python evaluation/preflight.py --output preflight.json
```

Download the frozen PDFs separately and place them under `data/pdfs/`, or pass
`--pdf-dir /path/to/pdfs`. The four official JSON files and existing manifests
are checked by hash, count, schema fields, identity, PDF readability, and actual
physical page bounds. The final release has 1,000 General, 200 Unanswerable,
200 Reasoning, and 800 Multi-Document questions; IDs run from QA0001 to QA2200.
Gold is loaded independently of any prediction file. Task names come from the
four component files: the real `question_type` field is `Literal/Inferential`.

## Raw model output versus submission records

Each **raw model output** is exactly one JSON object with exactly two fields:

```json
{"answer_pre":"The concise final answer","evidence_pages":[2,5]}
```

No Markdown fences, explanations outside JSON, chain-of-thought, or extra
fields are allowed. `answer_pre` must be a nonempty string. `evidence_pages`
must be an array of genuine JSON integers, excluding booleans, floats, strings,
zero, negatives, and indices exceeding the corresponding PDF's page count.
Pages are **1-based physical PDF indices**, using externally supplied page
labels. Printed page numbers and source-document page numbers are not accepted.
For Multi-Document inputs, the supplied frozen PDF concatenates source papers
in a fixed order: count pages continuously from 1 in that merged file. Do not
reorder or reconstruct the bundle during evaluation.

The recommended **submission file** is UTF-8 JSONL:

```json
{"qa_id":"QA0001","answer_pre":"First maximize worst-case payoff, then maximize expected payoff under conjectured model","evidence_pages":[1,9]}
{"qa_id":"QA1001","answer_pre":"Unanswerable","evidence_pages":[]}
```

`qa_id` is submission-level binding metadata added by the submission writer,
not an extra raw-model-output field. A JSON array of the same records is also
accepted. Duplicate or unknown IDs are fatal. Omitted IDs become explicit
missing predictions and remain in the denominator. The example above and
[`predictions.example.jsonl`](../examples/predictions.example.jsonl) contain only
two records, so completeness validation correctly returns nonzero.

For an audit of actual raw bytes, the loader additionally accepts an envelope
with exactly `qa_id` and a string `raw_model_output`. An invalid raw string then
has a known ID and is scored as illegal. Malformed submission JSON without a
reliable binding is fatal; the validator never guesses its ID or discards it.
Existing nested internal inference files are examined by `reproduce.py`, not
mistaken for public submission records.

Only outer whitespace and page sorting/deduplication are normalized; each is
logged in audit metadata. The answer string is never stripped, lowercased,
unit-converted, fuzzily matched, or rewritten. `[7,3,7]` becomes `[3,7]`;
`"page 3"`, `"07"`, and `"3-5"` remain invalid.

## Refusal and answer accuracy

The only valid refusal is:

```json
{"answer_pre":"Unanswerable","evidence_pages":[]}
```

`Unanswerable` with nonempty pages is illegal. Any other answer with empty pages
violates the PDF response contract. Noncanonical refusal labels such as
`unanswerable`, `N/A`, and `Unknown` are not converted into the canonical label.
For Unanswerable gold, legal exact-label refusal is checked deterministically;
the semantic judge is never called to reinterpret refusal synonyms.

Every legal General, Reasoning, and Multi-Document prediction is submitted to
the semantic judge with Question, Reference answer, and Model answer. There
is no exact-match, alias, numerical-tolerance, BLEU, ROUGE, or fuzzy shortcut.
The judge does not receive PDFs. Its final response must be exactly `CORRECT`
or `INCORRECT`; lowercase, explanations, and even added output whitespace are
not accepted by the strict parser. Invalid replies retry only up to the
explicit `max_attempts` setting, with every reply retained. Exhaustion is a
technical failure with zero answer contribution and an unchanged denominator.

The paper specifies **Qwen/Qwen3.6-27B**. Its original prompt is preserved
byte-for-byte in [prompts/semantic_judge.txt](prompts/semantic_judge.txt), with
source attribution and SHA-256 in [prompts/provenance.json](prompts/provenance.json).
The exact binary-prompt run's checkpoint revision, serving configuration,
generation limits, seed, timeouts, and retries have not been recovered.
Historical v4's seed 42 and 32-token limit belong to a different prompt and
are not silently used as defaults.

For a **new explicitly configured run**, provide a JSON configuration containing:

- `backend: "openai_compatible"`, `api_model`: the exact served model label;
- `identity`: `model: "Qwen/Qwen3.6-27B"`, `checkpoint`, `revision`,
  `tokenizer_sha256`, `chat_template_sha256`, and `serving_version`;
- `generation`: explicit `temperature`, `top_p`, `top_k`, `seed`, `max_tokens`;
- explicit positive `timeout_seconds` and integer `max_attempts`.

The configured server must support those parameters and use the declared
checkpoint/template. The evaluator cannot independently attest remote weight
bytes. An explicit null generation parameter must mean disabled/unsupported
under that server's documented contract; it must not disguise an unknown
default. No production values are invented in this repository. Credentials and
the `/chat/completions` URL are read from `SCIENCEDOC_JUDGE_API_KEY` and
`SCIENCEDOC_JUDGE_URL`; no secrets belong in configuration committed to Git.

## Evidence localization and denominator

For each item, let P and G be the predicted and gold sets of physical pages:

- Nonempty P and G: precision = |P ∩ G| / |P|, recall = |P ∩ G| / |G|,
  F1 = 2 × precision × recall / (precision + recall), or zero if the sum is zero.
- Both sets empty: precision = recall = F1 = 1.
- Exactly one set empty: precision = recall = F1 = 0.

E-Precision, E-Recall, and E-F1 are **per-item macro means × 100**, never micro
metrics. **A-Pages = sum of each item's unique predicted page count / N**.
For the full release N is always 2,200, not the number received, legal, or
answerable. Computation retains full float precision; only display rounds to
two decimals. There is no composite Overall Score.

Missing predictions contribute zero to answer, evidence, and A-Pages, including
missing Unanswerable items (they must not receive the both-empty credit).
Unrecoverable inference failures also contribute zero. A semantic judge failure
does not erase a legal model's independently computable evidence prediction.

Illegal outputs have zero answer contribution and remain in N. Historical
evidence scoring runs before format validity and uses permissively recovered
pages; the paper does not define a page-recovery rule for every illegal raw
output. **The package therefore reports evidence headlines as null/UNRESOLVED
when any illegal prediction requires this unresolved policy**, instead of
choosing an undocumented zeroing or recovery rule. Legal per-item evidence
remains available in the JSON. This limitation prevents publication claims.

Submission evaluation does not rerun inference or apply format correction.
Raw text, its hash, parse/format/contract statuses, normalization, and the final
normalized object are retained. A flat public record cannot reconstruct the
original model's bytes or retry history, and is marked accordingly. Historical
pipelines have different bounded-retry behavior; missing original attempts are
reported as unavailable, never fabricated.

## Commands and output

```bash
# Validation only; missing records return 1, fatal binding errors return 2.
python scripts/validate_submission.py --predictions predictions.jsonl

# New explicit judge run (configuration must be supplied by the run owner).
python evaluation/evaluate.py --predictions predictions.jsonl \
  --judge-config judge-config.json --judge-cache .cache/sciencedoc_judge.jsonl \
  --output results.json

# Cache-only run; use the identical config to reuse its bound cache entries.
python evaluation/evaluate.py --predictions predictions.jsonl \
  --judge-config judge-config.json --judge-cache .cache/sciencedoc_judge.jsonl \
  --offline --output results.json

# Runnable two-record diagnostic, without a model or API credentials.
python evaluation/evaluate.py --predictions examples/predictions.example.jsonl \
  --offline --output example-results.json

# All unit tests for this package use mocks, never real model inference.
python -m pytest -q tests/test_validation.py tests/test_metrics.py tests/test_judge.py tests/test_e2e.py

# Historical artifacts must be obtained separately; no files are written to sxz.
python evaluation/reproduce.py --results-dir data/results --history-dir sxz \
  --output-dir data/results/reproduction_audit
```

The evaluator returns 0 for a complete new paper-contract run (without claiming
historical reproduction), 2 for unresolved/provisional reports, and 1 for fatal
evaluation errors. Historical audit returns 2 until an actual paper-protocol
rescore is established; this is not a successful reproduction exit status.
Reproduction needs all 55 baseline JSONs, the summary CSV and detail workbook
under `data/results/evaluations/`, and the historical subject workbook under
`sxz/`. These are not included in a clean Git clone. Their hashes and derived
audit results are published under `docs/reports/official_evaluation/`.

The two-record diagnostic prints, among the other fields:

```text
ScienceDoc Official Evaluation
Release status: provisional_pending_paper_history_reconciliation
Dataset:
  Total: 2200
  Received: 2
  Missing: 2198
  Legal: 2
  Illegal: 0
  Technical Failures: 1
Answer Accuracy (%):
  All: 0.05
Evidence Localization:
  E-Precision: 0.09
  E-Recall: 0.09
  E-F1: 0.09
  A-Pages: 0.00
```

Legal counts refer to prediction validity; technical failures also include
unavailable judge decisions, so those categories can overlap. The console
also reports every task and all eight disciplines, illegal-output rate, exact
page-set match, and joint correctness. Those last three are audit diagnostics.
`--detailed` adds the 61 fine-grained fields. JSON stores full-precision metrics,
per-item outcomes, dataset/PDF/prediction hashes, timestamp, evaluator version,
judge identity/config hash, prompt hash, and unresolved items.

## Persistent judge cache

Every JSONL cache entry binds the QA ID, question hash, reference hash, complete
prediction hash (including original raw hash), judge identity, prompt hash, and
configuration hash. Only an exact binding reuses a decision. Editing any of
those inputs causes a miss. The cache preserves raw judge attempts and a
timestamp. Corrupt bindings and conflicting decisions fail loudly. Do not share
one append-only cache among simultaneous writers; use a cache per run/process.
Offline misses become visible technical failures; they are never removed from
the denominator. Reporting an existing JSON file does not call the judge.

## Real local-model smoke test

The completed [local Qwen3.6-27B test and replayable artifacts](../docs/reports/official_evaluation/LOCAL_MODEL_TEST.md)
record 15 actual model calls, six successful controls, and cache-only CLI replay.

The opt-in diagnostic below uses local Qwen3.6-27B weights with the unchanged
paper prompt and strict binary parser. It requires the internal Python 3.12+
ML environment (tested with PyTorch 2.8.0 and Transformers 5.6.2), an idle A800,
the frozen PDFs, and historical `data/results/Qwen3-VL-8B/` files. These large
assets and ML dependencies are not part of `requirements-eval.txt`.

```bash
python scripts/smoke_local_evaluation.py \
  --model-dir models/Qwen3.6-27B --gpu 2 \
  --output-dir data/results/local_evaluation_smoke

# Recompute the full-denominator partial report entirely from the real cache.
python evaluation/evaluate.py \
  --predictions data/results/local_evaluation_smoke/predictions.jsonl \
  --judge-config data/results/local_evaluation_smoke/judge_config.json \
  --judge-cache data/results/local_evaluation_smoke/judge_cache.jsonl \
  --offline --output data/results/local_evaluation_smoke/offline_report.json
```

Use a new output directory for each diagnostic. The script selects the first
three legal historical raw outputs per task whose question, reference, and
evidence exactly match current gold. This selection tests execution, not model
performance; illegal/changed historical records remain accounted for in the
separate reproduction audit. All 2,200 gold items remain in the scoring report,
so omitted predictions contribute zero. The offline command returns **2**
because this deliberately partial submission is incomplete.

Six additional, explicitly constructed predictions check positive/negative
Judge behavior across the three answerable tasks; they never alter gold.
Unanswerable scoring remains deterministic. The script records raw Judge
replies and token IDs, cache bindings, configuration, every checkpoint shard's
SHA-256, full PDF validation metadata, and before/after hashes of all QA JSONs.
It checks cache-only replay, fixed denominator, expected controls, and no QA
changes, returning nonzero on a failed check.

Its greedy decoding, seed 42, 32-token limit, disabled thinking, and two-attempt
limit are **explicit new diagnostic choices**, not recovered paper settings.
Generation is synchronous and token-bounded, without a wall-clock timeout.
The local checkpoint is identified by its complete artifact hashes; the
unavailable upstream revision is recorded as null. This diagnostic config is
used with `Judge(generate=...)` and offline replay, not the API backend.
This test does not establish historical Table 2/3 reproduction or rerun a
baseline's PDF inference.
