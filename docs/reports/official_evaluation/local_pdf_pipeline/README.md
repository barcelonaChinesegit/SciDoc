# Real PDF-to-score pilot artifacts

> **历史记录，现行入口已更新（2026-09-21）：** 本文保留当时的二分类/严格格式诊断、命令和测试结果，
> 不描述现行评分器；文中的旧诊断命令及旧 `--predictions` 评分接口已停用。
> 当前 `evaluation/` 使用 sxz v4 原始提示词与评分规则；55 文件及原始缓存回放匹配
> 历史汇总 1,540/1,540 个数值、Table 2 的 99/99 和 Table 3 的 98/99 个显示值。
> 唯一论文差异是 Claude Table 3 All（68.05% 对 69.32%）。见
> [当前验证结果](../SXZ_V4_ALIGNMENT.md)及[可执行 Quick Start](../../../../evaluation/README.md#quick-start)。

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
