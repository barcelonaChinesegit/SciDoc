# `data/qa` directory

The directory prefixes record the dataset production order. The final manually
reviewed release is in `7.final_2200/`. Other directories
contain categorized read-only views, upstream sources, or release evidence.
Historical build artifacts have been moved to the classified archive under
`data/archive/final_2200_cleanup_20260911/`.

## Final delivery

`7.final_2200/` is the user-facing delivery directory. It contains exactly the
four batch-independent JSON files and the current classification workbook:

| File | QA count | Purpose |
| --- | ---: | --- |
| `7.final_2200/ordinary_qa.json` | 1,000 | Ordinary single-PDF QA. |
| `7.final_2200/unanswerable_qa.json` | 200 | Single-PDF QA whose exact answer is `Unanswerable`. |
| `7.final_2200/reasoning_qa.json` | 200 | Reasoning QA. |
| `7.final_2200/cross_pdf_qa.json` | 800 | Cross-PDF QA. |
| `7.final_2200/final_2200_classification_statistics.xlsx` | 2,200 rows | Full classification statistics and per-question detail. |

The workbook is generated only from those four JSON files. Its 2,200 detail
rows and all aggregate sheets were checked against the current JSON content.
The four JSON files use collection-wide IDs `QA0001` through `QA2200`.
Cross-PDF questions occupy `QA1401` through `QA2200`; each record preserves
its old paper and QA IDs in `annotation_provenance.final_2200_identity`.

## Categorized final views

The four-file release is the only authority. Two byte-identical, read-only views
make the single-PDF components easier to find by purpose:

| Component | Categorized view |
| --- | --- |
| Ordinary 1,000 | `1.base/rel__single_pdf__ordinary__batch01__n1000.json` |
| Unanswerable 200 | `2.unanswerable/rel__single_pdf__unanswerable__batch01__n200.json` |

These files are synchronized from `7.final_2200/ordinary_qa.json` and
`7.final_2200/unanswerable_qa.json`; they must never be edited independently.
The old 1,000/1,190/1,200 single-PDF files and the old 850-row ablation file
have been removed from the active QA tree. Historical copies remain in the
classified archive. Current ablation inputs are generated at evaluation time
from the final ordinary file.

## Retained release provenance

The remaining pre-merge components document how the other two final files were
assembled:

| Component | Canonical path |
| --- | --- |
| Reasoning batch 1 | `3.reasoning/rel__reasoning__refreshed__batch01__n100__claude_hard__v1.json` |
| Reasoning batch 2 | `3.reasoning/hard_expansion/rel__reasoning__incremental__batch02__n100__claude_hard__v1.json` |
| Cross-PDF batch 1 | `4.cross_pdf/challenge/rel__cross_pdf__challenge__batch01__n400__v1.json` |
| Cross-PDF batch 2 | `4.cross_pdf/hard_expansion/rel__cross_pdf__strict_dual_review__batch02__n400__v1.json` |
| Collection manifest | `7.final_2200/rel__collection__final_2200__manifest.json` |

The Cross-PDF final selection ledger, Reasoning cleaning/calibration ledgers,
batch-1 manual review packet, and final modality audits are retained because
they directly substantiate the released questions. Completed pilots, large
candidate pools, one-off supplement inputs, stale progress files, and partial
API-review errors were deleted.

## Directory roles

| Directory | Current role |
| --- | --- |
| `1.base/` | Upstream single-PDF datasets and the read-only final ordinary view. |
| `3.reasoning/` | Two retained Reasoning components and their direct evidence/calibration lineage. |
| `4.cross_pdf/` | Two retained Cross-PDF components and their direct selection/supplement lineage. |
| `5.human_reviewed/` | Human-review authority still used by synchronization workflows. |
| `6.review/final_2200_modalities/` | Modality audit JSON referenced by final QA provenance. |
| `6.review/cross_pdf/` | Batch-1 Cross-PDF manual-review evidence; not a second formal dataset. |
| `2.unanswerable/` | Read-only categorized view of the final 200 unanswerable QA. |
| `7.final_2200/` | Final four JSON files, manifest, and classification workbook. |

Directories containing only completed pilot data, raw response shards,
historical reviews, locks, logs, or superseded candidates were removed after
their contents were verified and archived.

## Classified archive

`data/archive/final_2200_cleanup_20260911/` stores historical material in
purpose-specific directories. Each category has a `manifest.csv` containing
the original path, size, and SHA-256, plus a verified `files.tar.zst` archive.
The categories separate evaluation results, checkpoints, raw generation
shards, selection/provenance data, historical datasets, historical QA,
model/cleaning reviews, model outputs, evaluation companion files, and QA
non-JSON provenance.

The Web review database and its undo snapshots now live under
`data/web/review/`; task queue state lives under `data/web/task_queue/`.
These runtime records are outside the QA dataset tree and this QA archive.

## Maintenance rules

- Do not use archived candidates as evaluation inputs without restoring and
  validating the complete original workflow context.
- Do not delete a classified archive unless its member list and SHA-256
  manifest have been independently preserved.
- Do not treat retained pre-merge components or review packets as additional
  publication datasets; the four files under `7.final_2200/` are authoritative.
- Keep `sxz/` strictly read-only; it is outside this cleanup and archive.

Some provenance fields preserve paths as recorded at generation time (including pre-migration directory names). Resolve those historical references through the archive manifests and migration records; they are not runnable current input paths.
