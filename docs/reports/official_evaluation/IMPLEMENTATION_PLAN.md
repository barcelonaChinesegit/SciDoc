# Paper-aligned evaluation follow-through plan

> **历史记录，现行入口已更新（2026-09-21）：** 本文保留当时的二分类/严格格式诊断、命令和测试结果，
> 不描述现行评分器；文中的旧诊断命令及旧 `--predictions` 评分接口已停用。
> 当前 `evaluation/` 使用 sxz v4 原始提示词与评分规则；55 文件及原始缓存回放匹配
> 历史汇总 1,540/1,540 个数值、Table 2 的 99/99 和 Table 3 的 98/99 个显示值。
> 唯一论文差异是 Claude Table 3 All（68.05% 对 69.32%）。见
> [当前验证结果](SXZ_V4_ALIGNMENT.md)及[可执行 Quick Start](../../../evaluation/README.md#quick-start)。

Started 2026-09-20 from commit `1049a6a`. The manuscript is the authority.
QA/PDF contents and `sxz/` are read-only. Work does not promise equality with
historical table values already contradicted by the stored predictions.

## Execution order and acceptance checks

1. **Audit remaining inference/report paths and recovery artifacts.** Inspect
   `run_inference.py`, provider loaders, internal reporting, the public scorer,
   `data/results/`, archived inventories and relevant read-only `sxz/` retry
   records. Record concrete defects and any recoverable final predictions.
2. **Close protocol gaps.** Preserve the paper's exact PDF prompt; retain every
   original/retry output and its trigger/hash; reject unsupported semantic
   repairs. Correct model loading as necessary. Remove unused permissive
   parsing. Ensure reporting delegates headline metrics to the public scorer,
   retains failures and never presents a partial run as a complete benchmark.
3. **Make new outputs reviewable and resumable.** Bind immutable gold/PDFs,
   model, prompt, configuration and QA IDs; export canonical raw submissions.
   Recover old records only with verifiable bindings, without reconstructing
   absent original bytes. Document which baselines require new inference.
4. **Run a timed local end-to-end pilot, then assess further execution.** Use
   the smallest locally available paper baseline on an idle A800, full PDFs
   at 144 DPI, the original prompt, and Qwen3.6-27B Judge. Choose and record new
   generation/retry settings explicitly rather than inventing historical ones.
   Check actual inference → validation → semantic judging → fixed-denominator
   report → cache/resume. A failed resource/provider check is recorded, never
   worked around by downsampling or dropping pages. Full runs require working
   providers and all input bindings; missing API/checkpoint information remains
   an explicit unresolved item.
5. **Validate and publish.** Run focused regression tests, all project tests,
   release preflight, documentation synchronization and before/after QA hashes.
   Publish the code, plan outcome, timings and diagnostics to GitHub. Distinguish
   verified implementation from pilot coverage and unavailable historical data.

## Initial status

- [x] Remaining protocol and artifact audit
- [x] Protocol fixes and regression tests
- [x] Submission export and resumable bindings
- [x] Real local inference/Judge/report pilot
- [x] Final checks and report; publish with this change set

Findings, commands and final status will be appended as work proceeds. Already
completed historical audits and local Judge tests are referenced, not rerun
without a changed input or newly identified concern.


## Execution outcome

See [FOLLOWTHROUGH_REPORT.md](FOLLOWTHROUGH_REPORT.md) for the complete findings,
file changes, tests, timings, per-cell comparison links and runnable commands.
The real pilot covers 25 physical PDF pages across all four tasks, with all raw
attempts preserved. Current gold remains unchanged. The prior mathematical and
local-model counterexamples to Table 2/3 equality stand; additional historical
artifacts do not establish matching final single-A800/binary-Judge provenance.
Full new inference for 11 models is a separate experiment and cannot prove that
old stored predictions reproduce incompatible paper values. No extra thousands
of semantic calls were launched for that false-equality claim.
