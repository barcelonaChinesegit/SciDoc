# Real PDF-to-score pilot artifacts

Final code run: `data/results/local_pdf_pipeline_20260920_final/`.
These are public diagnostic copies, not a complete benchmark submission.
All four predictions were generated from complete 144-DPI PDFs by the local
Qwen3-VL-4B-Instruct; answerable items were judged by Qwen3.6-27B with the
unchanged paper prompt. All model/input/code/configuration hashes are included.

`summary.json` retains the full 2,200 denominator and the four received rows;
`predictions.jsonl` preserves exact raw outputs and ordered generation audits.
Stage timings include model load and execution, excluding dataset/checkpoint
hashing. The shortest-PDF sample is unsuitable for estimating full-run cost.

See [the complete follow-through report](../FOLLOWTHROUGH_REPORT.md) for
limitations, tests and the already-proven historical Table 2/3 discrepancies.
