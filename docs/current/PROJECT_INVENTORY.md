# 项目文件清单（自动生成）

> 请勿手工编辑。由维护脚本生成；明确排除 `sxz/` 和本地开发工具；SHA-256 仅显示前 12 位。

## 顶层目录

| 路径 | 内容 |
| --- | --- |
| `src/pku_qa/` | Python 包：评测、服务和分类工作流。 |
| `tests/` | unit、integration、system 三层测试。 |
| `data/pdfs/` | 平铺 PDF 资产；`paper_*`、`source_*`、`z_cross_*` 依次排列。 |
| `data/qa/` | QA 数据；一级目录按生产顺序编号，`7.final_2200/` 是四文件正式发布集。 |
| `data/results/` | 模型推理、Judge 输出及技术报告。 |
| `data/web/` | Web 控制台的账户、审核、任务队列、日志和健康状态。 |
| `models/` | Qwen、Gemma、Mistral、InternVL、MiniCPM 九套本地模型的唯一真实目录。 |
| `docs/` | 全部项目说明、指南、报告说明与提示词。 |
| `deploy/systemd/` | 本地服务的可复现 systemd 单元。 |
| `tools/internal/experiment_console/web/` | 本地任务与数据审核 Web 控制台。 |
| `schemas/` | 数据结构 JSON Schema。 |
| `sxz/` | 其他同学独立目录；本清单和维护工具永远排除。 |

## 源码、配置、文档与测试逐文件清单

| 路径 | 用途 | 行数 | SHA-256 |
| --- | --- | ---: | --- |
| `.gitignore` | 项目配置或入口：.gitignore | 178 | `ae85f31cd193` |
| `CONTRIBUTORS.md` | 项目配置或入口：CONTRIBUTORS | 20 | `b84c7afd6165` |
| `README.en.md` | 项目配置或入口：README.en | 4 | `d61ad80987bb` |
| `README.md` | 项目配置或入口：README | 110 | `622bc0b58ab8` |
| `README.zh-CN.md` | 项目配置或入口：README.zh-CN | 125 | `b3cd95516fff` |
| `deploy/systemd/pku-data-manager-api.service` | 本地 systemd 服务与健康守护部署单元 | 17 | `c6dac49af915` |
| `deploy/systemd/pku-task-queue-api.service` | 本地 systemd 服务与健康守护部署单元 | 15 | `27796a52b576` |
| `deploy/systemd/pku-task-queue-daemon.service` | 本地 systemd 服务与健康守护部署单元 | 15 | `0e84f267087a` |
| `deploy/systemd/pku-task-queue-web.service` | 本地 systemd 服务与健康守护部署单元 | 18 | `60533b690c37` |
| `deploy/systemd/pku-web-health.service` | 本地 systemd 服务与健康守护部署单元 | 9 | `d1efa505031c` |
| `deploy/systemd/pku-web-health.timer` | 本地 systemd 服务与健康守护部署单元 | 10 | `4874bd250b0d` |
| `deploy/systemd/pku-web-tunnel.service` | 本地 systemd 服务与健康守护部署单元 | 14 | `7ef0af37b291` |
| `docs/current/ARCHITECTURE.md` | 项目文档：ARCHITECTURE | 415 | `315117c4679e` |
| `docs/current/CURRENT_STATUS.md` | 项目文档：CURRENT STATUS | 69 | `b77445197732` |
| `docs/current/DATASET_CATALOG.md` | 项目文档：DATASET CATALOG | 33 | `a8b83f7b7a4a` |
| `docs/current/GETTING_STARTED.md` | 项目文档：GETTING STARTED | 261 | `37ecc507e615` |
| `docs/current/MANUAL_REVIEW_GUIDE.md` | 项目文档：MANUAL REVIEW GUIDE | 678 | `cf611e006c26` |
| `docs/current/README.md` | 项目文档：README | 126 | `307579b7ddb5` |
| `docs/current/WEB_CONSOLE.md` | 项目文档：WEB CONSOLE | 173 | `7ab162dbb736` |
| `docs/prompts/MAINTENANCE_REMEDIATION.md` | 项目文档：MAINTENANCE REMEDIATION | 26 | `7640aa7c9528` |
| `docs/releases/PDF_DISTRIBUTION.md` | 项目文档：PDF DISTRIBUTION | 50 | `2c5375f7eb60` |
| `docs/reports/CROSS_PDF_EVIDENCE_AUDIT.md` | 项目文档：CROSS PDF EVIDENCE AUDIT | 305 | `f1b33be43190` |
| `docs/reports/DOCUMENTATION_AUDIT.md` | 项目文档：DOCUMENTATION AUDIT | 52 | `67582a68d2eb` |
| `docs/reports/FIGURE_RECOVERY_20260916.md` | 项目文档：FIGURE RECOVERY 20260916 | 38 | `17b41b8be827` |
| `docs/reports/README.md` | 项目文档：README | 14 | `ce204a6418e8` |
| `docs/reports/SCIDOC_MIGRATION_20260920.md` | 项目文档：SCIDOC MIGRATION 20260920 | 78 | `e4073f071a5c` |
| `docs/reports/SCIDOC_PATH_AUDIT_20260920.md` | 项目文档：SCIDOC PATH AUDIT 20260920 | 69 | `0ee565793a91` |
| `docs/reports/official_evaluation/FINAL_REPORT.md` | 项目文档：FINAL REPORT | 208 | `2bf5f74296d0` |
| `docs/reports/official_evaluation/LOCAL_MODEL_TEST.md` | 项目文档：LOCAL MODEL TEST | 90 | `16c8baa69bac` |
| `docs/reports/official_evaluation/claude_subject_discrepancy.json` | 项目文档：claude subject discrepancy | 118 | `4fd535e724eb` |
| `docs/reports/official_evaluation/dataset_preflight.json` | 项目文档：dataset preflight | 1624 | `2fcf32ae8f4e` |
| `docs/reports/official_evaluation/local_smoke/checkpoint_artifacts.json` | 项目文档：checkpoint artifacts | 28 | `c7f21f2e8f4b` |
| `docs/reports/official_evaluation/local_smoke/controls.json` | 项目文档：controls | 206 | `ea6abc83bd7a` |
| `docs/reports/official_evaluation/local_smoke/judge_config.json` | 项目文档：judge config | 117 | `158cdeca0fd7` |
| `docs/reports/official_evaluation/local_smoke/preflight.json` | 项目文档：preflight | 1624 | `e5481707eae9` |
| `docs/reports/official_evaluation/local_smoke/qa_after.json` | 项目文档：qa after | 29 | `29fdb517e43d` |
| `docs/reports/official_evaluation/local_smoke/qa_before.json` | 项目文档：qa before | 29 | `29fdb517e43d` |
| `docs/reports/official_evaluation/local_smoke/report_excerpt.json` | 项目文档：report excerpt | 986 | `50159e1eab74` |
| `docs/reports/official_evaluation/local_smoke/selection.json` | 项目文档：selection | 17 | `a9f307d24afa` |
| `docs/reports/official_evaluation/local_smoke/summary.json` | 项目文档：summary | 47 | `9bcbf653ffaa` |
| `docs/reports/official_evaluation/local_smoke/verification.json` | 项目文档：verification | 11 | `1118dfef59aa` |
| `docs/reports/official_evaluation/reproduction_audit.json` | 项目文档：reproduction audit | 5120 | `0f17d4df501c` |
| `docs/reports/official_evaluation/verification.json` | 项目文档：verification | 63 | `fa808b6ac98c` |
| `environment.yml` | 项目配置或入口：environment | 8 | `1dc4df61202c` |
| `evaluation/README.md` | 项目配置或入口：README | 286 | `c0b231383c34` |
| `evaluation/__init__.py` | 项目配置或入口：  init   | 3 | `b51a76f50781` |
| `evaluation/evaluate.py` | 项目配置或入口：evaluate | 83 | `cdff69d6256c` |
| `evaluation/judge.py` | 项目配置或入口：judge | 125 | `c6a08c4ff9fd` |
| `evaluation/metrics.py` | 项目配置或入口：metrics | 87 | `328f11cbcdd3` |
| `evaluation/paper_reference.json` | 项目配置或入口：paper reference | 251 | `0ae6486571cc` |
| `evaluation/preflight.py` | 项目配置或入口：preflight | 108 | `b3d8dbf81dd3` |
| `evaluation/prompts.py` | 项目配置或入口：prompts | 12 | `694a63ed56b5` |
| `evaluation/prompts/provenance.json` | 项目配置或入口：provenance | 7 | `069ca54dd7b3` |
| `evaluation/reconciliation.json` | 项目配置或入口：reconciliation | 13 | `f13a46c70ff1` |
| `evaluation/reproduce.py` | 项目配置或入口：reproduce | 234 | `9dd83e97d104` |
| `evaluation/validation.py` | 项目配置或入口：validation | 255 | `db009a6b4d27` |
| `provider_config.example.json` | 项目配置或入口：provider config.example | 16 | `1ed1d13e6268` |
| `pyproject.toml` | 项目配置或入口：pyproject | 27 | `7678f6ba4aa5` |
| `requirements.txt` | 项目配置或入口：requirements | 34 | `b1d620c6158e` |
| `schemas/final_2200_qa.schema.json` | QA 数据结构约束 | 56 | `8ac0c9e1049b` |
| `schemas/single_pdf_qa.schema.json` | QA 数据结构约束 | 110 | `b4e168585abf` |
| `scripts/dataset_construction/phase1_paper_acquisition/README.md` | 项目配置或入口：README | 62 | `1d9e8bfeca84` |
| `scripts/dataset_construction/phase1_paper_acquisition/arxiv_download_backend.py` | 项目配置或入口：arxiv download backend | 86 | `a67bf7499717` |
| `scripts/generate_dataset_catalog.py` | 项目配置或入口：generate dataset catalog | 52 | `58d7107827ed` |
| `scripts/smoke_local_evaluation.py` | 项目配置或入口：smoke local evaluation | 213 | `05629f8cac9b` |
| `scripts/validate_submission.py` | 项目配置或入口：validate submission | 29 | `7684e9e25bb3` |
| `src/pku_qa/__init__.py` | 项目配置或入口：  init   | 23 | `5ec76212d2b6` |
| `src/pku_qa/evaluation/__init__.py` | 统一评测核心：  init   | 0 | `e3b0c44298fc` |
| `src/pku_qa/evaluation/adaptive_gpu_pool.py` | 统一评测核心：adaptive gpu pool | 611 | `c4aedbbb59d5` |
| `src/pku_qa/evaluation/calculate_evidence_weighted_accuracy.py` | 统一评测核心：calculate evidence weighted accuracy | 360 | `7e394ed0574e` |
| `src/pku_qa/evaluation/durable_work_queue.py` | 统一评测核心：durable work queue | 510 | `0e38dd574f56` |
| `src/pku_qa/evaluation/eval_framework.py` | 统一评测核心：eval framework | 881 | `5f663ecd2ee2` |
| `src/pku_qa/evaluation/evaluation_protocol.py` | 统一评测核心：evaluation protocol | 794 | `0048f0898b3f` |
| `src/pku_qa/evaluation/gpu_reservation.py` | 统一评测核心：gpu reservation | 355 | `daa7cce39404` |
| `src/pku_qa/evaluation/model_paths.py` | 统一评测核心：model paths | 69 | `732aac67f3bb` |
| `src/pku_qa/evaluation/progress_logging.py` | 统一评测核心：progress logging | 22 | `39a607f715b4` |
| `src/pku_qa/evaluation/qa_scoring.py` | 统一评测核心：qa scoring | 179 | `975e65aa3f40` |
| `src/pku_qa/evaluation/run_eval_pipeline.py` | 统一评测核心：run eval pipeline | 309 | `715d05ade3ba` |
| `src/pku_qa/evaluation/run_hard_eval.py` | 统一评测核心：run hard eval | 894 | `bad05dbe21b6` |
| `src/pku_qa/evaluation/run_inference.py` | 统一评测核心：run inference | 1434 | `e04401b045c0` |
| `src/pku_qa/evaluation/run_judge.py` | 统一评测核心：run judge | 875 | `88f5ffbb6b9b` |
| `src/pku_qa/evaluation/run_report.py` | 统一评测核心：run report | 1005 | `8c6d3eee64c0` |
| `src/pku_qa/pdf_assets.py` | 项目配置或入口：pdf assets | 155 | `3d0339ec71b7` |
| `src/pku_qa/services/__init__.py` | 本地服务：  init   | 0 | `e3b0c44298fc` |
| `src/pku_qa/services/review/__init__.py` | 本地服务：  init   | 0 | `e3b0c44298fc` |
| `src/pku_qa/services/review/data_manager_api.py` | 本地服务：data manager api | 661 | `9c0ff036b0a5` |
| `src/pku_qa/services/review/data_review_manager.py` | 本地服务：data review manager | 2838 | `c2fd95c32ad3` |
| `src/pku_qa/services/task_queue/__init__.py` | 本地服务：  init   | 0 | `e3b0c44298fc` |
| `src/pku_qa/services/task_queue/task_queue_api.py` | 本地服务：task queue api | 481 | `316025ffc225` |
| `src/pku_qa/services/task_queue/task_queue_cli.py` | 本地服务：task queue cli | 382 | `5600cb7db6d2` |
| `src/pku_qa/services/task_queue/task_queue_daemon.py` | 本地服务：task queue daemon | 701 | `13d22122d64a` |
| `src/pku_qa/services/task_queue/task_queue_manager.py` | 本地服务：task queue manager | 1809 | `0d2698edebe7` |
| `src/pku_qa/services/task_queue/task_queue_runner.py` | 本地服务：task queue runner | 161 | `3005a2be54c6` |
| `src/pku_qa/services/task_queue/task_queue_supervisor.py` | 本地服务：task queue supervisor | 58 | `e5c69271a7b9` |
| `src/pku_qa/services/task_queue/task_queue_tui.py` | 本地服务：task queue tui | 146 | `1dd3a926ef12` |
| `src/pku_qa/workflows/__init__.py` | 工作流：  init   | 0 | `e3b0c44298fc` |
| `src/pku_qa/workflows/cleaning/__init__.py` | 数据清洗与校验：  init   | 0 | `e3b0c44298fc` |
| `src/pku_qa/workflows/cleaning/apply_hard_cross_pdf_review_fixes.py` | 数据清洗与校验：apply hard cross pdf review fixes | 209 | `71093e6f5828` |
| `src/pku_qa/workflows/cleaning/audit_reasoning_pdf_dependency.py` | 数据清洗与校验：audit reasoning pdf dependency | 628 | `c318c9097432` |
| `src/pku_qa/workflows/cleaning/build_final_2200_modality_review.py` | 数据清洗与校验：build final 2200 modality review | 138 | `f8605cb66358` |
| `src/pku_qa/workflows/cleaning/filter_reasoning_closed_book.py` | 数据清洗与校验：filter reasoning closed book | 202 | `f68d506d32c4` |
| `src/pku_qa/workflows/cleaning/finalize_cross_pdf_manual_audit.py` | 数据清洗与校验：finalize cross pdf manual audit | 243 | `3a530cdf8e6a` |
| `src/pku_qa/workflows/cleaning/finalize_cross_pdf_semantic_reaudit.py` | 数据清洗与校验：finalize cross pdf semantic reaudit | 583 | `f18408499b55` |
| `src/pku_qa/workflows/cleaning/normalize_human_reviewed_evidence_pages.py` | 数据清洗与校验：normalize human reviewed evidence pages | 124 | `e3ec22d42337` |
| `src/pku_qa/workflows/cleaning/reclassify_cross_pdf_modalities_api.py` | 数据清洗与校验：reclassify cross pdf modalities api | 590 | `9adb2feef62f` |
| `src/pku_qa/workflows/cleaning/restore_reasoning_qa_evidence.py` | 数据清洗与校验：restore reasoning qa evidence | 457 | `9acf44053259` |
| `src/pku_qa/workflows/cleaning/sync_human_reviewed_qa.py` | 数据清洗与校验：sync human reviewed qa | 508 | `3f689dcb6eef` |
| `src/pku_qa/workflows/cleaning/validate_single_pdf_qa_schema.py` | 数据清洗与校验：validate single pdf qa schema | 57 | `f92fd5e75532` |
| `src/pku_qa/workflows/generation/__init__.py` | QA 生成与构造：  init   | 0 | `e3b0c44298fc` |
| `src/pku_qa/workflows/generation/build_cross_pdf_challenge_v3.py` | QA 生成与构造：build cross pdf challenge v3 | 1689 | `7d58da59893b` |
| `src/pku_qa/workflows/generation/build_cross_pdf_claude_fix_rereview.py` | QA 生成与构造：build cross pdf claude fix rereview | 120 | `5916c603ca8a` |
| `src/pku_qa/workflows/generation/build_cross_pdf_gemini_fix_rereview.py` | QA 生成与构造：build cross pdf gemini fix rereview | 203 | `1f3b01cd32c3` |
| `src/pku_qa/workflows/generation/build_cross_pdf_quality_v2.py` | QA 生成与构造：build cross pdf quality v2 | 1462 | `dba7b42a720f` |
| `src/pku_qa/workflows/generation/build_cross_pdf_review_calibration.py` | QA 生成与构造：build cross pdf review calibration | 168 | `558c5313490e` |
| `src/pku_qa/workflows/generation/build_cross_pdf_review_complement.py` | QA 生成与构造：build cross pdf review complement | 139 | `8a7060a5f995` |
| `src/pku_qa/workflows/generation/build_cross_pdf_second_stage_fix_rereview.py` | QA 生成与构造：build cross pdf second stage fix rereview | 122 | `4cb5487a02ee` |
| `src/pku_qa/workflows/generation/build_fulltext_cross_pdf_qa_api.py` | QA 生成与构造：build fulltext cross pdf qa api | 422 | `75dc0ca4a506` |
| `src/pku_qa/workflows/generation/build_global_cross_pdf_bundle_plan.py` | QA 生成与构造：build global cross pdf bundle plan | 552 | `03c3cb742d31` |
| `src/pku_qa/workflows/generation/build_hard_cross_pdf_qa.py` | QA 生成与构造：build hard cross pdf qa | 2248 | `f479165c974f` |
| `src/pku_qa/workflows/generation/build_single_pdf_reasoning_qa.py` | QA 生成与构造：build single pdf reasoning qa | 2526 | `3ac30784a305` |
| `src/pku_qa/workflows/generation/prepare_cross_pdf_manual_audit.py` | QA 生成与构造：prepare cross pdf manual audit | 263 | `d1dc961b3927` |
| `src/pku_qa/workflows/generation/wait_and_build_hard_cross_pdf_qa.py` | QA 生成与构造：wait and build hard cross pdf qa | 164 | `fae714c671a5` |
| `src/pku_qa/workflows/generation/wait_and_build_single_pdf_reasoning_qa.py` | QA 生成与构造：wait and build single pdf reasoning qa | 167 | `ad4afadadfde` |
| `src/pku_qa/workflows/operations/__init__.py` | 服务运维：  init   | 0 | `e3b0c44298fc` |
| `src/pku_qa/workflows/operations/bootstrap_data_review_service.py` | 服务运维：bootstrap data review service | 80 | `67e42d8efee2` |
| `src/pku_qa/workflows/operations/check_web_stack.py` | 服务运维：check web stack | 129 | `27ff5a6194ee` |
| `src/pku_qa/workflows/operations/flatten_pdf_assets.py` | 服务运维：flatten pdf assets | 291 | `b24fccf322bd` |
| `src/pku_qa/workflows/operations/inspect_final_2200.py` | 服务运维：inspect final 2200 | 81 | `976f39598435` |
| `src/pku_qa/workflows/operations/migrate_final_2200_review_state.py` | 服务运维：migrate final 2200 review state | 426 | `06ef97319115` |
| `src/pku_qa/workflows/reporting/__init__.py` | 实验执行与报告：  init   | 0 | `e3b0c44298fc` |
| `src/pku_qa/workflows/reporting/build_final_2200_classification.py` | 实验执行与报告：build final 2200 classification | 438 | `73080e1bcaed` |
| `src/pku_qa/workflows/reporting/diagnose_external_answer_accuracy.py` | 实验执行与报告：diagnose external answer accuracy | 243 | `d82bdc614dce` |
| `src/pku_qa/workflows/reporting/prepare_hard_expansion_pilot_eval.py` | 实验执行与报告：prepare hard expansion pilot eval | 165 | `7908a649bde0` |
| `src/pku_qa/workflows/reporting/report_cross_pdf_eval.py` | 实验执行与报告：report cross pdf eval | 735 | `4567ec837254` |
| `src/pku_qa/workflows/reporting/report_hard_expansion_pilot_eval.py` | 实验执行与报告：report hard expansion pilot eval | 73 | `4f1e8b44502a` |
| `src/pku_qa/workflows/reporting/report_reasoning_qa_eval.py` | 实验执行与报告：report reasoning qa eval | 1159 | `b2c32d709ab3` |
| `src/pku_qa/workflows/reporting/report_single_pdf_challenge_eval.py` | 实验执行与报告：report single pdf challenge eval | 1491 | `151d487e43de` |
| `src/pku_qa/workflows/reporting/run_claude_pdf_selection.py` | 实验执行与报告：run claude pdf selection | 293 | `5e2c06ab9ea0` |
| `src/pku_qa/workflows/reporting/run_final_2200_evaluation.py` | 实验执行与报告：run final 2200 evaluation | 136 | `f986fca652e1` |
| `src/pku_qa/workflows/reporting/run_single_pdf_challenge_gpu_queue.py` | 实验执行与报告：run single pdf challenge gpu queue | 897 | `333eb1d9a890` |
| `src/pku_qa/workflows/reporting/summarize_cross_pdf_dual_reviews.py` | 实验执行与报告：summarize cross pdf dual reviews | 334 | `bfdb7019fe37` |
| `src/pku_qa/workflows/reporting/summarize_cross_pdf_quality_v2.py` | 实验执行与报告：summarize cross pdf quality v2 | 212 | `c29772703555` |
| `src/pku_qa/workflows/reporting/summarize_cross_pdf_review_calibration.py` | 实验执行与报告：summarize cross pdf review calibration | 324 | `8a0433b87b2b` |
| `src/pku_qa/workflows/review/__init__.py` | 数据复审：  init   | 0 | `e3b0c44298fc` |
| `src/pku_qa/workflows/review/finalize_hard_cross_pdf_pool_set.py` | 数据复审：finalize hard cross pdf pool set | 220 | `9aea1d17f4ba` |
| `src/pku_qa/workflows/review/finalize_hard_cross_pdf_qa.py` | 数据复审：finalize hard cross pdf qa | 351 | `e8eb3bcf1ee1` |
| `src/pku_qa/workflows/review/review_cross_pdf_qa_api.py` | 数据复审：review cross pdf qa api | 1316 | `3e5919a819ce` |
| `src/pku_qa/workflows/review/review_single_pdf_reasoning_qa_api.py` | 数据复审：review single pdf reasoning qa api | 1118 | `7a7bc6ea03f2` |
| `src/pku_qa/workflows/review/wait_and_review_single_pdf_reasoning_qa.py` | 数据复审：wait and review single pdf reasoning qa | 133 | `45e50bf5317e` |
| `src/pku_qa/workflows/selection/build_hard_benchmark_selection_pool.py` | selection：build hard benchmark selection pool | 234 | `fa4272d6ac13` |
| `src/pku_qa/workflows/selection/final_2200_contract.py` | selection：final 2200 contract | 217 | `b12aed89e819` |
| `src/pku_qa/workflows/selection/normalize_final_2200_release.py` | selection：normalize final 2200 release | 284 | `7e90ea9e8455` |
| `src/pku_qa/workflows/selection/single_pdf_release_views.py` | selection：single pdf release views | 202 | `b5214a5ce86f` |
| `src/pku_qa/workflows/selection/sync_final_2200_manifest.py` | selection：sync final 2200 manifest | 149 | `d5d3b086f0e6` |
| `tests/__init__.py` | 公共支持 测试：  init   | 0 | `e3b0c44298fc` |
| `tests/conftest.py` | 公共支持 测试：conftest | 18 | `22580724449b` |
| `tests/integration/__init__.py` | integration 测试：  init   | 0 | `e3b0c44298fc` |
| `tests/integration/test_adaptive_gpu_queue.py` | integration 测试：adaptive gpu queue | 1307 | `c2956615a052` |
| `tests/integration/test_apply_hard_cross_pdf_review_fixes.py` | integration 测试：apply hard cross pdf review fixes | 46 | `a0e823eb3677` |
| `tests/integration/test_audit_reasoning_pdf_dependency.py` | integration 测试：audit reasoning pdf dependency | 152 | `25c39d89341c` |
| `tests/integration/test_build_cross_pdf_gemini_fix_rereview.py` | integration 测试：build cross pdf gemini fix rereview | 60 | `2c5707a4827e` |
| `tests/integration/test_build_global_cross_pdf_bundle_plan.py` | integration 测试：build global cross pdf bundle plan | 206 | `273caffd6c89` |
| `tests/integration/test_build_hard_benchmark_selection_pool.py` | integration 测试：build hard benchmark selection pool | 66 | `d3fa1ed00876` |
| `tests/integration/test_build_hard_cross_pdf_qa.py` | integration 测试：build hard cross pdf qa | 724 | `4898677a66ab` |
| `tests/integration/test_build_single_pdf_reasoning_qa.py` | integration 测试：build single pdf reasoning qa | 1056 | `39c0e85f47be` |
| `tests/integration/test_cross_pdf_challenge_v3.py` | integration 测试：cross pdf challenge v3 | 159 | `112258282f02` |
| `tests/integration/test_cross_pdf_eval_report.py` | integration 测试：cross pdf eval report | 233 | `d4ec1bc961d0` |
| `tests/integration/test_cross_pdf_quality_v2.py` | integration 测试：cross pdf quality v2 | 94 | `b26da1f7b023` |
| `tests/integration/test_filter_reasoning_closed_book.py` | integration 测试：filter reasoning closed book | 65 | `fd9c2d13d173` |
| `tests/integration/test_finalize_cross_pdf_semantic_reaudit.py` | integration 测试：finalize cross pdf semantic reaudit | 104 | `1256eeb69d72` |
| `tests/integration/test_finalize_hard_cross_pdf_pool_set.py` | integration 测试：finalize hard cross pdf pool set | 70 | `0771f31a3038` |
| `tests/integration/test_finalize_hard_cross_pdf_qa.py` | integration 测试：finalize hard cross pdf qa | 66 | `f12015a0450d` |
| `tests/integration/test_migrate_final_2200_review_state.py` | integration 测试：migrate final 2200 review state | 52 | `83e8ea42d184` |
| `tests/integration/test_normalize_human_reviewed_evidence_pages.py` | integration 测试：normalize human reviewed evidence pages | 30 | `4b11468eb0d6` |
| `tests/integration/test_prepare_hard_expansion_pilot_eval.py` | integration 测试：prepare hard expansion pilot eval | 51 | `6127ec146442` |
| `tests/integration/test_reasoning_qa_eval_report.py` | integration 测试：reasoning qa eval report | 423 | `d2cb44ee7d15` |
| `tests/integration/test_report_hard_expansion_pilot_eval.py` | integration 测试：report hard expansion pilot eval | 45 | `d54dc8aa8a57` |
| `tests/integration/test_restore_reasoning_qa_evidence.py` | integration 测试：restore reasoning qa evidence | 257 | `01a9e8969e15` |
| `tests/integration/test_review_cross_pdf_qa_api.py` | integration 测试：review cross pdf qa api | 362 | `65b292392302` |
| `tests/integration/test_review_single_pdf_reasoning_qa_api.py` | integration 测试：review single pdf reasoning qa api | 290 | `13ca2e2ede86` |
| `tests/integration/test_run_report.py` | integration 测试：run report | 227 | `38211cb95c4d` |
| `tests/integration/test_single_pdf_challenge_v2.py` | integration 测试：single pdf challenge v2 | 794 | `c60b848c4a27` |
| `tests/integration/test_sync_human_reviewed_qa.py` | integration 测试：sync human reviewed qa | 187 | `2c353da12431` |
| `tests/integration/test_task_queue.py` | integration 测试：task queue | 1175 | `c4b49f0939d9` |
| `tests/integration/test_wait_and_build_hard_cross_pdf_qa.py` | integration 测试：wait and build hard cross pdf qa | 49 | `6a7d4ae7046f` |
| `tests/integration/test_wait_and_build_single_pdf_reasoning_qa.py` | integration 测试：wait and build single pdf reasoning qa | 51 | `3fcaa55e49cb` |
| `tests/system/__init__.py` | system 测试：  init   | 0 | `e3b0c44298fc` |
| `tests/system/test_project_docs.py` | system 测试：project docs | 24 | `6b25945572f2` |
| `tests/system/test_relocated_checkout.py` | system 测试：relocated checkout | 48 | `d002c12a5563` |
| `tests/system/test_web_stack_health.py` | system 测试：web stack health | 15 | `8feea5f78e59` |
| `tests/test_e2e.py` | 公共支持 测试：e2e | 50 | `86d45e9716ed` |
| `tests/test_judge.py` | 公共支持 测试：judge | 109 | `5db380bf1138` |
| `tests/test_local_smoke.py` | 公共支持 测试：local smoke | 47 | `6c50957d7743` |
| `tests/test_metrics.py` | 公共支持 测试：metrics | 11 | `4fe3911315ab` |
| `tests/test_validation.py` | 公共支持 测试：validation | 73 | `f9c76c857561` |
| `tests/unit/__init__.py` | unit 测试：  init   | 0 | `e3b0c44298fc` |
| `tests/unit/test_bootstrap_data_review_service.py` | unit 测试：bootstrap data review service | 34 | `8915b1841caa` |
| `tests/unit/test_build_final_2200_classification.py` | unit 测试：build final 2200 classification | 60 | `90a847f8c37a` |
| `tests/unit/test_build_final_2200_modality_review.py` | unit 测试：build final 2200 modality review | 14 | `c1db78369293` |
| `tests/unit/test_data_review_manager.py` | unit 测试：data review manager | 905 | `73bb5fd17b02` |
| `tests/unit/test_diagnose_external_answer_accuracy.py` | unit 测试：diagnose external answer accuracy | 44 | `b058ac3ef5d3` |
| `tests/unit/test_evaluation_protocol.py` | unit 测试：evaluation protocol | 179 | `692b1bebc3ae` |
| `tests/unit/test_evidence_weighted_accuracy.py` | unit 测试：evidence weighted accuracy | 236 | `7d948ce241a6` |
| `tests/unit/test_final_2200_contract.py` | unit 测试：final 2200 contract | 74 | `12a8d43552fa` |
| `tests/unit/test_hard_eval_report_commands.py` | unit 测试：hard eval report commands | 56 | `3d94d5a6b194` |
| `tests/unit/test_inspect_final_2200.py` | unit 测试：inspect final 2200 | 57 | `2b40d156e2e5` |
| `tests/unit/test_judge_rules.py` | unit 测试：judge rules | 411 | `fb41aa3efb06` |
| `tests/unit/test_model_paths.py` | unit 测试：model paths | 100 | `8c66cfcfeede` |
| `tests/unit/test_normalize_final_2200_release.py` | unit 测试：normalize final 2200 release | 33 | `9430c309387d` |
| `tests/unit/test_pdf_assets.py` | unit 测试：pdf assets | 51 | `e4e565121d23` |
| `tests/unit/test_progress_logging.py` | unit 测试：progress logging | 23 | `d3bfe0099a66` |
| `tests/unit/test_protocol_fingerprints.py` | unit 测试：protocol fingerprints | 130 | `8a54890c63c7` |
| `tests/unit/test_qa_scoring.py` | unit 测试：qa scoring | 113 | `2da749336b6f` |
| `tests/unit/test_reclassify_cross_pdf_modalities_api.py` | unit 测试：reclassify cross pdf modalities api | 112 | `f5a8cbab74a9` |
| `tests/unit/test_run_claude_pdf_selection.py` | unit 测试：run claude pdf selection | 25 | `af8b1f8e56f4` |
| `tests/unit/test_run_final_2200_evaluation.py` | unit 测试：run final 2200 evaluation | 39 | `0b4168e3d025` |
| `tests/unit/test_sync_final_2200_manifest.py` | unit 测试：sync final 2200 manifest | 59 | `3ad5f544bf8c` |
| `tests/unit/test_task_queue_api.py` | unit 测试：task queue api | 60 | `312dda8c4d37` |
| `tools/internal/experiment_console/README.md` | 项目配置或入口：README | 23 | `a2646004dd6f` |
| `tools/internal/experiment_console/web/.openai/hosting.json` | 本地任务与数据审核控制台组件 | 4 | `d2841f8a91a9` |
| `tools/internal/experiment_console/web/app/admin/users/page.tsx` | 本地任务与数据审核控制台组件 | 580 | `31bc4421bfd4` |
| `tools/internal/experiment_console/web/app/api/[...path]/route.ts` | 本地任务与数据审核控制台组件 | 90 | `c096d908057d` |
| `tools/internal/experiment_console/web/app/components/ConsoleNav.tsx` | 本地任务与数据审核控制台组件 | 64 | `e1e4ff0b39f2` |
| `tools/internal/experiment_console/web/app/components/LoginScreen.tsx` | 本地任务与数据审核控制台组件 | 100 | `c2da92fe5e54` |
| `tools/internal/experiment_console/web/app/data-api/[...path]/route.ts` | 本地任务与数据审核控制台组件 | 77 | `a8d49c3298c0` |
| `tools/internal/experiment_console/web/app/data/page.tsx` | 本地任务与数据审核控制台组件 | 921 | `0d3b6b5631ba` |
| `tools/internal/experiment_console/web/app/globals.css` | 本地任务与数据审核控制台组件 | 641 | `da234cf094d3` |
| `tools/internal/experiment_console/web/app/guide/page.tsx` | 本地任务与数据审核控制台组件 | 97 | `2a3d322def47` |
| `tools/internal/experiment_console/web/app/layout.tsx` | 本地任务与数据审核控制台组件 | 38 | `60bac5933866` |
| `tools/internal/experiment_console/web/app/lib/auth.ts` | 本地任务与数据审核控制台组件 | 83 | `9120042a949e` |
| `tools/internal/experiment_console/web/app/login/page.tsx` | 本地任务与数据审核控制台组件 | 5 | `803d942e469e` |
| `tools/internal/experiment_console/web/app/page.tsx` | 本地任务与数据审核控制台组件 | 72 | `027971c9c720` |
| `tools/internal/experiment_console/web/app/profile/page.tsx` | 本地任务与数据审核控制台组件 | 119 | `9f798102618e` |
| `tools/internal/experiment_console/web/app/queue/page.tsx` | 本地任务与数据审核控制台组件 | 905 | `bdc7efa36968` |
| `tools/internal/experiment_console/web/app/register/page.tsx` | 本地任务与数据审核控制台组件 | 5 | `c40a3749a7fc` |
| `tools/internal/experiment_console/web/build/font-cache.mjs` | 本地任务与数据审核控制台组件 | 40 | `d95fee0156dd` |
| `tools/internal/experiment_console/web/build/sites-vite-plugin.ts` | 本地任务与数据审核控制台组件 | 45 | `0c788fe80191` |
| `tools/internal/experiment_console/web/eslint.config.mjs` | 本地任务与数据审核控制台组件 | 18 | `275a07c13fc7` |
| `tools/internal/experiment_console/web/next.config.ts` | 本地任务与数据审核控制台组件 | 7 | `a972c4f0ffa6` |
| `tools/internal/experiment_console/web/package-lock.json` | 本地任务与数据审核控制台组件 | 10152 | `0b81ee0ffc1b` |
| `tools/internal/experiment_console/web/package.json` | 本地任务与数据审核控制台组件 | 40 | `6d2fe6196940` |
| `tools/internal/experiment_console/web/postcss.config.mjs` | 本地任务与数据审核控制台组件 | 7 | `7b299d3d3b16` |
| `tools/internal/experiment_console/web/tests/rendered-html.test.mjs` | 本地任务与数据审核控制台组件 | 196 | `7fa85f54cf6d` |
| `tools/internal/experiment_console/web/tsconfig.json` | 本地任务与数据审核控制台组件 | 34 | `fbff01604d6c` |
| `tools/internal/experiment_console/web/vite.config.ts` | 本地任务与数据审核控制台组件 | 62 | `c43063c7c594` |
| `tools/internal/experiment_console/web/worker/index.ts` | 本地任务与数据审核控制台组件 | 47 | `5f9565e5505c` |

## 正式任务的规范数据

| 路径 | QA 数 | 字节 | SHA-256 |
| --- | ---: | ---: | --- |
| `data/pdf_assets_manifest.json` | 审计账本 | 605277 | `c00a419a56bf8713c349b4c13a0456c224ef86436cdbf91caf7ec8e953d04a10` |
| `data/qa/7.final_2200/ordinary_qa.json` | 1000 | 3848180 | `706377d66ef456d69cd90f4246b3a4c0f2221b3fc39d57eba9fd692e034d8dbc` |
| `data/qa/7.final_2200/unanswerable_qa.json` | 200 | 698894 | `96f0702104f7e16ba8f067f18df0956ce8e96ccc80f7ac33e80d5f179d2053e4` |
| `data/qa/7.final_2200/reasoning_qa.json` | 200 | 1801460 | `ed77ded5c266ae47c053275c258cceeeefc220ec10f79fee77b17485ec5b84fd` |
| `data/qa/7.final_2200/cross_pdf_qa.json` | 800 | 6202661 | `6bdd3e73a3288ba3d3d84a20de7fd30981e1b41bd7b358ebc76ec15795239c82` |
| `data/qa/7.final_2200/rel__collection__final_2200__manifest.json` | 审计账本 | 1730 | `a104ab0e614ef01bfeabc27b603eb017bf6c33a32706e6cbd7440a5edab89546` |
| `data/qa/5.human_reviewed/rel__human_reviewed__authority__batch01__n983.json` | 983 | 471661 | `b3b38c1a3b51be4e20d2a40f23d60a85dc0e6b1a0589d0f85fb2ac7e8bb876a2` |
| `data/qa/3.reasoning/work__reasoning__historical_clean__batch00__n100.json` | 100 | 414681 | `6647533d2d80a2402c88235c113b4854673c8646b2b3e01a9f3f8e528bb4daea` |
| `data/qa/3.reasoning/dual_review_v2/cleaning_ledger.json` | 审计账本 | 2615408 | `818342942bff6b2df96f9678c32daff66498b2d13b7e06cedf8af1aed05175bf` |
| `data/qa/1.base/rel__single_pdf__ordinary__batch01__n1000.json` | 1000 | 3848180 | `706377d66ef456d69cd90f4246b3a4c0f2221b3fc39d57eba9fd692e034d8dbc` |
| `data/qa/2.unanswerable/rel__single_pdf__unanswerable__batch01__n200.json` | 200 | 698894 | `96f0702104f7e16ba8f067f18df0956ce8e96ccc80f7ac33e80d5f179d2053e4` |

## 大型外部资产与运行时目录

- `data/pdfs/`：平铺且按类型命名的 PDF，共 1717 个不同内容文件；主论文 703、生成源论文 583、Cross-PDF 合订本 431。
- `data/pdf_assets_manifest.json`：保存全部 1946 个原路径、规范文件名和 SHA-256；已归并 229 个重复副本。
- `models/Qwen3-VL-4B-Instruct/`、`models/Qwen3-VL-8B-Instruct/`、`models/Qwen3.6-27B/`、`models/Gemma-3-27B-IT/`、`models/Mistral-Small-3.1-24B-Instruct-2503/`、`models/InternVL2_5-8B/`、`models/InternVL3_5-8B/`、`models/MiniCPM-V-2_6/`、`models/MiniCPM-V-4_5/`：本地模型权重的唯一真实位置，不纳入 Git。Hugging Face、ModelScope 和 Torch 模型缓存统一位于 `models/.cache/`；项目根目录和用户缓存目录不保留模型链接或副本。
- `data/results/evaluations/`：规范评测输出和断点；不纳入 Git。
- `data/results/reports/`：严格报告；可从结果重建，不纳入 Git。
- `data/web/review/`：Web 人工校验账户、会话、事件和撤销快照；不纳入 Git。
- `data/web/task_queue/`：队列数据库、任务日志、incident 和 Web 健康状态；不纳入 Git。
- `sxz/`：其他同学所有的独立代码，禁止项目自动化触碰。
