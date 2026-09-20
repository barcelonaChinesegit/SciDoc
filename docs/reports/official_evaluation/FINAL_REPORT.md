# ScienceDoc official evaluation — final audit report

Audit date: 2026-09-20. The manuscript is authoritative; QA and PDF contents are immutable.

Latest implementation, PDF-to-Judge pilot, test results and publication status:
[FOLLOWTHROUGH_REPORT.md](FOLLOWTHROUGH_REPORT.md). Test counts below describe
the earlier audit stages; the linked follow-through contains the final checks.

Follow-up: [real local Qwen3.6-27B evaluation test](LOCAL_MODEL_TEST.md) passed
15 actual model calls, six controls, and independent cache-only CLI replay.
The updated regression suite has 486 passing tests. This diagnostic does not
change the historical reproduction status below.

Later follow-up: [exhaustive feasibility checks and local discrepancy verification](REPRODUCTION_DISCREPANCY.md)
prove that the available predictions cannot match all current paper cells.
The old main-project rule-first scoring helpers listed in this historical audit
have since been removed; the collaborator-owned `sxz/` references remain read-only.

**Status: engineering implementation and historical audit delivered; an official rescore reproducing the paper is not established.** Historical reaggregation is explicitly distinguished from new semantic evaluation. No numbers, gold answers, or evidence pages were changed to conceal discrepancies.

## A. Repository audit

The user-designated baseline source is `data/results/`: 11 model directories × five JSON files. Their exact file hashes are in [reproduction_audit.json](reproduction_audit.json). The summary CSV and detail XLSX in `data/results/evaluations/` are byte-identical to their v4 copies under `sxz/`.

| Role | Recovered source |
| --- | --- |
| Prediction parsing / permissive page extraction | `sxz/evaluate_11models_5datasets_calibrated_fixeddenom_seed42_gpu2345_parallel_v4.py`: `get_pred_answer_from_qa`, `extract_pages`, `get_pred_pages_from_qa` |
| Historical output validation | Same file: `is_illegal_answer`; primarily checks unrecoverable/error answers, not the current strict PDF contract |
| Table 2 semantic judge and prompt | Same file: `JUDGE_RULES`, `build_judge_prompt`, `run_qwen36_judge` |
| Evidence / aggregation | Same file: `calc_evidence_metrics`, `build_fixed_denominator_summary` |
| Actual Table 2 data | `data/results/evaluations/summary_11models_77rows_calibrated_fixed_denominator.csv` |
| Table 3 data | `data/results/evaluations/detail_11models_5datasets.xlsx`; `sxz/final_2200_classification_statistics.xlsx`; `sxz/build_table3_by_subject.py` |
| Cached historical judge decisions | `sxz/evaluation_11models_5datasets_qwen36_calibrated_fixeddenom_v4/qwen36_answer_judge_cache.json` and worker caches |
| Alternate historical judge | `sxz/evaluate_11models_cross_reasoning_directrubric_qwen36_gpu2345_v10.py`, its evaluation manifest and direct-rubric caches |
| Hybrid, not the source matching Table 2 | `sxz/build_final_hybrid_11models_v3.py`; `sxz/evaluation_11models_hybrid_final/` |
| Main-project historical implementation | `src/pku_qa/evaluation/{evaluation_protocol,qa_scoring,run_inference,run_judge,run_report}.py` |
| Exact current paper prompt | `论文/写作材料/图/附录/提示词方框_drawio/source_text/06_semantic_judge.txt`, copied byte-for-byte to `evaluation/prompts/semantic_judge.txt` |

`sxz/` was inspected as read-only reference; no collaborator files or caches were written. Source searches covered Python/shell scripts, notebooks, configs, logs, raw JSONs, judge caches, CSVs, workbooks, archive inventories, and the manuscript. Historical archive binaries and third-party model/vendor Markdown are not rewritten.

## B. Protocol recovered

Current paper: strict two-field PDF JSON; exact canonical refusal; only outer whitespace and evidence ordering/duplicates may normalize. Answerable predictions use binary Qwen3.6-27B semantic judgment. Evidence PRF is per-item macro averaging with the stated empty-set cases; A-Pages uses unique predicted pages; the full denominator remains 2,200. Exact-set and joint correctness are diagnostics.

Historical v4: permissive fence/field/page extraction, whitespace normalization of question/answer strings, embedded gold from each result JSON, tri-class CORRECT/PARTIAL/WRONG semantic review, and fixed 1200+400+400+100+100 denominators. Only CORRECT contributes to answer accuracy. Evidence is calculated before answer validity checks, so parseable/recovered page sets may earn evidence credit even when the answer is invalid. This behavior is recorded, not copied into the strict evaluator.

Historical v10 is a different direct-rubric JSON judge with selective second and tie-breaking third passes. The hybrid output mixes v4/v10 model-dataset pairs; it does not match all current Table 2 accuracies. **The complete v4 summary does match all 99 displayed Table 2 cells.**

## C. Files created

The complete per-file add/modify/rename list with reasons is [changed_files.csv](changed_files.csv).

- `evaluation/`: strict validator, macro metrics, binary judge/cache, CLI, preflight, historical audit, paper prompt/provenance, reference table extraction, reconciliation record, README.
- `scripts/validate_submission.py`, `examples/predictions.example.jsonl`, `requirements-eval.txt`.
- `tests/test_validation.py`, `tests/test_metrics.py`, `tests/test_judge.py`, `tests/test_e2e.py`.
- `CONTRIBUTORS.md`, `README.zh-CN.md`, internal console README, PDF inventories/packaging guide.
- This audit directory: full-precision comparisons, dataset preflight, per-item baseline differences and source hashes.

## D. Files modified and reasons

- Moved the complete `task_queue_web/` tree to `tools/internal/experiment_console/web/`; updated ignore rules, local skill inventory configuration, deployed and repository Web service paths.
- Web home and navigation show the experiment queue only to administrators; corrected the home-page dataset count from five to four. Authentication, loopback ports, tunnel, HTTPS, and session boundaries remain intact.
- `README.md` is English; `README.en.md` links to it; the Chinese version remains available. Added the two requested contributor acknowledgments without fabricating Git authorship or sending access invitations.
- All 37 project-owned Markdown files were reviewed (see [markdown_audit.csv](markdown_audit.csv)); current architecture, getting-started, status, manual-review, and Web docs now distinguish paper metrics from historical scoring; old dated reports retain their evidence with explicit historical scope notices.
- Local manuscript figure README marks historical figure/case descriptions as generation history. Manuscript TeX and gold JSON are not edited.

## E. Dataset validation

| Actual path | QA count | ID field |
| --- | ---: | --- |
| `data/qa/7.final_2200/ordinary_qa.json` | 1000 | `paper["QA"]` mapping key |
| `data/qa/7.final_2200/unanswerable_qa.json` | 200 | `paper["QA"]` mapping key |
| `data/qa/7.final_2200/reasoning_qa.json` | 200 | `paper["QA"]` mapping key |
| `data/qa/7.final_2200/cross_pdf_qa.json` | 800 | `paper["QA"]` mapping key |

Counts, unique continuous QA0001–QA2200 IDs, final Multi-Document QA1401–QA2200 range, manifest hashes, all positive integer gold evidence, physical PDF bounds, and all 200 exact `Unanswerable` answers with empty gold evidence passed. There are no duplicate `(question, answer, PDF)` records. All 712 evaluation PDFs passed SHA-256 and readability checks. The 61 fine-grained fields were counted from actual paper metadata.

| Requested field | Actual representation / availability |
| --- | --- |
| Question / reference answer | QA `question` / `answer` |
| Gold evidence | QA `evidence_pages` |
| Task component | Official filename; **not** `question_type` |
| Question type / category | QA `question_type` (`Literal` or `Inferential`) / `question_category` |
| Discipline / fine field | Paper `primary_category` / `secondary_category` |
| PDF identity / path | Top-level paper ID, paper `paper`, `pdf_path`; canonical filename and SHA in `data/pdf_assets_manifest.json` |
| Answer format | QA `answer_format`, present on 972 General, 200 Unanswerable, 400 Multi-Document, zero Reasoning items; not a scoring branch |
| Modality | QA `modal_types` on all 2,200 items |
| Multi-Document sources | `source_doc_numbers` / `source_document_count` on 402 items, `source_paper_ids` / `evidence_source_docs` / `evidence_items` on 400; review ledger may provide `doc_number` and `merged_page` |
| Complete Multi-Document interval map | No universal source-map field in the four release files. Frozen merged PDF bytes are authoritative; do not rebuild from partial fields |
| Reasoning fields | `source_qa_ids`, `reasoning_type`, `relation`, `derivation`, `construction_quality_check`, `evidence_provenance`; 198/200 also contain focus/conclusion/intermediate facts/necessity tests/source evidence |
| Provenance IDs | `annotation_provenance.final_2200_identity` binds global and legacy paper/QA IDs; nested legacy `qa_id`/`qa_uid` are not global submission IDs |

The real schema differs from the task prompt's assumed `question_type == Unanswerable`; the loader uses the official component identity and independently verifies the canonical answer/evidence contract. All 27 JSON files anywhere under `data/qa/` retain their pre-task SHA-256. No fields were added or corrected in gold. Physical bounds are read from the hashed evaluation PDF, not trusted from historical embedded metadata.

## F. Judge reproducibility

| Setting | Recovered fact |
| --- | --- |
| Model identifier | Local model README identifies `Qwen/Qwen3.6-27B` |
| Historical checkpoint path | `/data/czj/pku/models/Qwen3.6-27B`, now the filesystem alias for `models/Qwen3.6-27B` |
| Checkpoint revision / historical weight hashes | **UNRESOLVED REPRODUCIBILITY ITEM**; not bound in the historical run/caches |
| Backend | Local PyTorch/Transformers; v4 loads AutoProcessor and an available AutoModel vision/text class; bfloat16, `device_map="auto"`, `local_files_only=True`, `trust_remote_code=True` |
| Chat template | `processor.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`; exact historical tokenizer/template bytes not bound |
| v4 generation | `seed=42`, `do_sample=False`, `max_new_tokens=32`; first branch passes `temperature=None`, `top_p=None`, `top_k=None`; TypeError fallback omits those arguments |
| v10 generation | `seed=42`, `do_sample=False`, thinking disabled, 256-token review / 48-token constrained decision fallback |
| v4 judge parse/retry | Regex label extraction accepting CORRECT/PARTIAL/WRONG (and INCORRECT as WRONG); malformed judge response raises and stops worker, no uniform timeout or strict binary retry policy |
| Current binary prompt run settings | **UNRESOLVED REPRODUCIBILITY ITEM**: no matching historical bound config/decisions. Do not transfer v4/v10 settings as defaults |
| New evaluator configuration | Requires explicit identity, generation, timeout and attempts; optional OpenAI-compatible endpoint, env-only credentials; mocks/offline audit require no model |
| Exact paper prompt SHA-256 | `45785c6f699520192874355527135b437bf2d0793d88b38b7246c592d3dcbe70` |
| v4 historical prompt SHA-256 | `d4365c671f3b69ba98a197691a8eb89e03bd89684fabc462127a78bbf6c13278` |

## G. Validation and tests

Focused tests: **56 passed** in an isolated clean environment. They cover the six required evidence cases, asymmetric PRF, fixed four-item denominator with a missing item, strict types/refusal/JSON, judge labels/retries/failures, and cache invalidation. The final full repository suite passed **480 tests** in 13.52 seconds; the isolated evaluator suite passed **56 tests**. See [verification.json](verification.json). Clean-environment and final command results are recorded in `verification.json`.

Web relocation: build and all **8 frontend tests passed**. Following service restart, `check_web_stack` reported `healthy: true`: local data API, queue API, Web, VPS tunnel backend and public HTTPS front all returned 200 with the application login page. No Nginx Basic Auth was introduced.

## H. Table 2 reproduction

Every model was validated against the current release. No binary-prompt historical cache matches the required judge contract. The later local run completed strict stored-output answer diagnostics for three models; these conclusively differ from the paper and are not official reproduction. The other eight models are bounded but not fully rescored. See [the discrepancy report](REPRODUCTION_DISCREPANCY.md) and its per-cell CSVs.

Columns below show All accuracy. All nine requested metrics for all 11 models are in [table2_comparison.csv](table2_comparison.csv), including full-precision historical values, paper values, rounded differences, new-score availability and reasons.

| Model | Paper All | Historical reaggregation | New official evaluator | New − Paper |
| --- | ---: | ---: | --- | --- |
| MiniCPM-V 2.6 | 15.59 | 15.590909090909 | strict diagnostic 0.00 | -15.59 |
| InternVL2.5-8B | 22.32 | 22.318181818182 | strict diagnostic 0.00 | -22.32 |
| MiniCPM-V 4.5 | 34.32 | 34.318181818182 | unavailable | unavailable |
| InternVL3.5-8B | 34.95 | 34.954545454545 | unavailable | unavailable |
| Gemma 3 27B | 47.32 | 47.318181818182 | strict diagnostic 0.045454545454545456 | -47.27454545454545 |
| Mistral-Small-3.1-24B | 49.00 | 49.000000000000 | unavailable | unavailable |
| Qwen3-VL-4B | 64.91 | 64.909090909091 | unavailable | unavailable |
| Qwen3-VL-8B | 68.32 | 68.318181818182 | unavailable | unavailable |
| Claude Sonnet 5 | 69.32 | 69.318181818182 | unavailable | unavailable |
| DeepSeek-V4-Flash | 79.45 | 79.454545454545 | unavailable | unavailable |
| GLM-4.6V | 80.95 | 80.954545454545 | unavailable | unavailable |

Historical Table 2 matches **99/99** cells after two-decimal formatting. Paper full-precision references are unavailable: the manuscript only contains rounded numbers. Historical artifacts retain full precision and are not rounded before aggregation.

### Baseline-to-current-gold discrepancies

| Model | Stored records | Mapped current IDs | Missing current IDs | Changed question/answer/evidence items | Strict illegal raw outputs¹ |
| --- | ---: | ---: | ---: | ---: | ---: |
| MiniCPM-V 2.6 | 2198 | 2198 | 2 | 0 | 2198 |
| InternVL2.5-8B | 2198 | 2198 | 2 | 5 | 2195 |
| MiniCPM-V 4.5 | 2200 | 2198 | 2 | 208 | 752 |
| InternVL3.5-8B | 2190 | 2188 | 12 | 205 | 481 |
| Gemma 3 27B | 2198 | 2198 | 2 | 139 | 2196 |
| Mistral-Small-3.1-24B | 2198 | 2198 | 2 | 5 | 1192 |
| Qwen3-VL-4B | 2190 | 2188 | 12 | 205 | 25 |
| Qwen3-VL-8B | 2190 | 2188 | 12 | 205 | 25 |
| Claude Sonnet 5 | 2190 | 2160 | 40 | 197 | 558 |
| DeepSeek-V4-Flash | 2198 | 2198 | 2 | 139 | 43 |
| GLM-4.6V | 2198 | 2198 | 2 | 136 | 274 |

¹ Strict raw status is counted before excluding a dataset-version mismatch; columns overlap. Per-item identities, changed fields and raw hashes are in [baseline_item_audit.csv](baseline_item_audit.csv). Some historical raw responses include explanations or JSON strings that their old parsers repaired; they are illegal under the current paper, even if historical answer accuracy was nonzero.

Qwen3-VL-8B specifically has 2,190 stored records, 2,188 matching legacy identities, 12 missing current IDs, and 205 mapped items whose embedded question, gold answer, or gold evidence differs. Its 68.31818181818181 historical All, 40.664217879496846 historical E-F1 and 8.442272727272726 historical A-Pages explain the printed 68.32/40.66/8.44; they cannot certify a new run on current immutable gold.

## I. Table 3 reproduction

[table3_comparison.csv](table3_comparison.csv) contains all 99 cells (All plus eight disciplines for 11 models). Reaggregation matches **98/99**, including all 88 discipline values. The discrepancy is:

| Model / metric | Paper | Reaggregated historical detail | Difference of displayed values | New official evaluator |
| --- | ---: | ---: | ---: | --- |
| Claude Sonnet 5 / All | 69.32 | 68.04545454545455 → 68.05 | −1.27 percentage points | unavailable |

The historical subject-identity join excludes 28 correct Claude records from the numerator (855 ordinary workbook correct records versus 827 after joining; see [the identity audit](claude_subject_discrepancy.json)) while retaining the 2,200 denominator: the raw ordinary record identities differ from the subject workbook for those items. The table's eight discipline values match that 68.05 aggregate, while the displayed All was 69.32. No paper or data number is changed by this task.

| Discipline | Historical subject denominator | Current release denominator |
| --- | ---: | ---: |
| Computer Science | 745 | 743 |
| Economics | 137 | 137 |
| Electrical Engineering and Systems Science | 128 | 129 |
| Mathematics | 245 | 245 |
| Physics | 379 | 379 |
| Quantitative Biology | 114 | 114 |
| Quantitative Finance | 86 | 86 |
| Statistics | 366 | 367 |

## J. Remaining ambiguities / release blockers

1. The binary paper-prompt execution config and decisions are not recoverable from the historical tri-class records. A fresh fully recorded binary-judge run is needed; its scores may legitimately differ from the existing paper table.
2. Some predictions were generated for different questions/answers/evidence or missing legacy IDs. Bind new inference to the unchanged current gold/PDF hashes; do not transplant historical decisions or change QA to fit them.
3. The paper does not settle evidence recovery for every illegal raw output. The old evaluator recovers pages permissively. New illegal answer accuracy is zero; evidence headline values remain null until that narrow policy is specified rather than guessed.
4. Exact historical checkpoint revision, template/tokenizer bytes, dependency versions, fallback branch and per-baseline correction trails are incompletely bound. Current local weights/configs cannot prove historical bytes.
5. The complete source interval map is not uniformly embedded in final Multi-Document QA. Preserve the frozen merged PDFs; do not invent source offsets.
6. Published historical artifacts and the current release have different discipline counts; Table 3's Claude All is internally inconsistent with its historical discipline aggregation.
7. PDFs and historical raw/workbook artifacts are external to Git. The Drive asset lists are prepared, but uploading and publishing a link still require access to the owner's chosen Drive destination.

## K. Exact commands

Run from a fresh clone after placing the frozen PDFs under `data/pdfs/`:

```bash
git clone https://github.com/barcelonaChinesegit/SciDoc.git
cd SciDoc
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-eval.txt
python evaluation/preflight.py --output preflight.json
python scripts/validate_submission.py --predictions predictions.jsonl
python evaluation/evaluate.py --predictions predictions.jsonl --judge-config judge-config.json --judge-cache .cache/sciencedoc_judge.jsonl --output results.json
python -m pytest -q tests/test_validation.py tests/test_metrics.py tests/test_judge.py tests/test_e2e.py
# Obtain the historical artifacts separately before this audit:
python evaluation/reproduce.py --results-dir data/results --history-dir sxz --output-dir data/results/reproduction_audit
```

`judge-config.json` must supply the verified settings described in [evaluation/README.md](../../../evaluation/README.md); no invented template values are provided. Set `SCIENCEDOC_JUDGE_URL` and, if required, `SCIENCEDOC_JUDGE_API_KEY` in the environment. `--offline` runs without network calls and labels unavailable decisions as technical failures. The current provisional evaluator/audit exit code is 2; that must not be interpreted as successful official reproduction.

For the full internal repository suite use Python 3.12+, install `requirements.txt`, then run `python -m pytest -q`. No test invokes real semantic-model inference.

## PDF distribution and contributors

Upload inventory: [all 1,717 PDFs](../../releases/pdf_upload_all.csv) or [712 benchmark PDFs](../../releases/pdf_upload_benchmark.csv), with exact paths/bytes/hashes; [packaging guide](../../releases/PDF_DISTRIBUTION.md). The asset sizes are 7,645,305,946 and 3,262,178,765 bytes respectively.

`yuetanbupt` and `ElephantsGit` are acknowledged in the English homepage and CONTRIBUTORS.md. No manual action is needed for that list. GitHub's automatic contributor graph requires qualifying real authored/coauthored commits on the default branch; write access is separately granted by Settings → Collaborators → Add people, followed by accepting the invitation. This task does not fabricate commits or invite anyone.
