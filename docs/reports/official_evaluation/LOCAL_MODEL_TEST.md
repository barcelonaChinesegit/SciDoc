# Real local-model evaluation test

Completed 2026-09-20 using local **Qwen/Qwen3.6-27B** on one NVIDIA A800 80 GB
(physical index 2, selected by UUID). This tests the public evaluation package
with actual model generation. It does **not** establish Table 2/3 reproduction
or rerun baseline PDF inference. The paper prompt and scoring code are unchanged.

## Inputs and configuration

- Frozen gold: 1,000 General, 200 Unanswerable, 200 Reasoning, 800 Multi-Document;
  2,200 continuous unique IDs; 712 readable, hash-matching PDFs.
- Historical predictions: three per task from `data/results/Qwen3-VL-8B/`, selected
  by ascending global ID among legal raw outputs with exact current question,
  reference-answer, and evidence matches. This intentionally selected diagnostic
  is not a representative performance estimate. No historical raw text was repaired.
- Additional controls: the reference answer and an unrelated incorrect answer for
  one item in each answerable task. Only predictions were constructed; gold was
  neither modified nor replaced.
- PyTorch 2.8.0+cu128, Transformers 5.6.2, BF16, SDPA, single-GPU placement,
  no CPU/disk offload. Transformers reported the PyTorch implementation of linear
  attention because optional accelerated kernels were unavailable.
- Explicit new test choices: greedy decoding, seed 42, disabled thinking,
  32 new tokens, two maximum attempts; temperature/top-p/top-k disabled.
  Local generation is token-bounded without a wall-clock timeout.
- Every local weight shard and configuration/tokenizer/template file is hashed.
  Upstream checkpoint revision remains unknown and is recorded as null; these
  local content hashes identify the checkpoint used in this test.
- Unchanged paper prompt SHA-256:
  `45785c6f699520192874355527135b437bf2d0793d88b38b7246c592d3dcbe70`.

These settings identify a **new diagnostic run**, not recovered historical
binary-Judge settings. See [the exact config](local_smoke/judge_config.json)
and [checkpoint artifact hashes](local_smoke/checkpoint_artifacts.json).

## Observed results

| Check | Observed result |
| --- | --- |
| Selected historical predictions | 12 legal, no technical failures |
| Real local model calls | 15: 9 historical answerable predictions + 6 controls |
| Strict Judge responses | 15/15 exact `CORRECT` or `INCORRECT`; no retry needed |
| Positive/negative controls | 6/6 as expected |
| Unanswerable predictions | 3 deterministic exact-label decisions; no Judge calls |
| Cache-only replay | 9/9 answerable cache hits, identical full-precision metrics |
| Independent public CLI replay | Same metrics, zero Judge failures; expected exit 2 for incomplete submission |
| Fixed denominator | 2,200; 12 received, 2,188 missing, missing contributions zero |
| GPU peak allocated tensor memory | 55,266,330,624 bytes; GPU released at process exit |
| QA byte preservation | All 27 QA JSON hashes unchanged, also equal to the original task snapshot |
| Automated regression tests | 486 passed, including 62 evaluation/diagnostic tests |

Historical sample answer decisions were General 3/3, Unanswerable 2/3,
Reasoning 2/3, and Multi-Document 2/3. These are observations, not pass targets;
the evaluator correctly retains wrong baseline answers. On the deliberately
partial full-denominator report the answer contribution is **9/2200**, not 9/12.
Evidence and discipline results remain at full precision in the
[report excerpt](local_smoke/report_excerpt.json). Its 2,188 omitted JSON rows
are explicitly identified as missing; they were included during aggregation.

The initial environment check exposed an optional PEFT/Transformers trainer
import conflict. The diagnostic now seeds Python/NumPy/PyTorch directly, without
changing shared packages. Monitoring also exposed different CUDA ordinal and
physical GPU ordering; the script now binds a UUID to the same physical device
used by admission checks. The final published run used this correction. Earlier
diagnostic attempts are retained locally, not merged into this run's cache.

## Repeat or inspect

See [the evaluation guide](../../../evaluation/README.md#real-local-model-smoke-test)
for a fresh model run. To inspect the published real decisions without loading a
model, first install `requirements-eval.txt` and provide the frozen PDFs, then:

```bash
python evaluation/evaluate.py \
  --predictions docs/reports/official_evaluation/local_smoke/predictions.jsonl \
  --judge-config docs/reports/official_evaluation/local_smoke/judge_config.json \
  --judge-cache docs/reports/official_evaluation/local_smoke/judge_cache.jsonl \
  --offline --output data/results/local_smoke_replay.json
```

Exit **2** is expected: the selected submission is deliberately incomplete.
The real [raw replies/token IDs](local_smoke/model_calls.jsonl),
[bound cache](local_smoke/judge_cache.jsonl), [controls](local_smoke/controls.json),
[summary](local_smoke/summary.json), and [verification](local_smoke/verification.json)
are published alongside this report. Runtime source directory:
`data/results/official_local_smoke_20260920_v3/`.

The limitations in the [original audit](FINAL_REPORT.md) remain: historical
binary-Judge configuration/decisions are unresolved, and illegal-output evidence
handling still prevents an unconditional official reproduction claim. This
test does not resolve those historical ambiguities.
