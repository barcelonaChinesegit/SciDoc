# Evaluation documentation update — 2026-09-21

The repository documentation now describes the `sxz` v4 evaluation as the
current public scoring path. The update covers the root English and Chinese
README files, the evaluator README, current maintainer guides, the report
index, the QA/release guides, and the internal console guide.

The new Quick Start has four steps:

1. install the lightweight evaluator dependencies and run the CPU tests;
2. replay one model from its five historical result files and the original
   input-bound Judge cache;
3. inspect `report.json`, per-item details and the run binding;
4. optionally start a new local Judge run with an A800 and Qwen3.6-27B.

The documentation now distinguishes three workflows:

- `evaluation/evaluate.py --offline` replays recorded `sxz` v4 decisions and
  needs no GPU, PDF or network access;
- `evaluation/evaluate.py --gpu` judges existing model answers with a new,
  separately bound local run and does not regenerate PDF answers;
- the internal `src/pku_qa` workflows generate answers and run current-release
  diagnostics against the four canonical QA files.

The original five result components and v4 cache remain external experiment
artifacts. The current four-file QA release and its PDF package are not silently
substituted into a historical replay. Historical audit reports retain their
original evidence, but now link to this current guide and the [sxz v4 alignment
report](official_evaluation/SXZ_V4_ALIGNMENT.md).

Verification completed after the update:

- the Qwen3-VL-8B Quick Start replay returned exit code 0 and Table 2 matched
  9/9 displayed cells;
- the full replay remains 1,540/1,540 historical summary values, Table 2
  99/99 and Table 3 98/99;
- documentation inventory synchronization passed;
- the full repository test suite passed 553 tests.
