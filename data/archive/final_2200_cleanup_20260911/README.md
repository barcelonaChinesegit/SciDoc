# Final 2,200 cleanup archive (2026-09-11)

This archive groups historical JSON artifacts by purpose after the final
2,200 QA release was manually reviewed. The migration-time cleanup inventory
was removed from the live documentation after the four-file migration finished;
the manifests in this archive remain the durable file-level record.

Each category directory contains:

- `manifest.csv`: original project-relative path, byte size, SHA-256,
  classification, and audit description;
- `files.tar.zst`: a Zstandard-compressed tar archive whose member paths are
  the original project-relative paths.

`archive_index.csv` records the member count, uncompressed source size,
compressed size, archive SHA-256, and manifest path for all categories.

Categories:

| Directory | Contents |
| --- | --- |
| `evaluation_result/` | Historical inference, Judge, diagnostic, and report JSON. |
| `evaluation_non_json/` | Logs, CSV, Markdown, HTML, text reports, and review workbooks stored with historical evaluations. |
| `generation_checkpoint/` | Resumable generation and inference checkpoints. |
| `generation_raw_shard/` | Per-paper, per-bundle, and per-round raw model responses. |
| `generation_selection_or_provenance/` | Selection ledgers, bundle mappings, and provenance. |
| `historical_or_upstream_dataset/` | Historical baseline and upstream QA datasets. |
| `historical_qa_artifact/` | Superseded QA candidates, summaries, and cleaning records. |
| `model_or_cleaning_review/` | Claude/Gemini review and deterministic cleaning records. |
| `model_output/` | Historical answering-model and Judge output JSON. |
| `qa_non_json_provenance/` | Completed-task locks, logs, progress text, CSV/JSONL ledgers, and historical workbooks from the QA tree. |

The live manual-review snapshots remain in
`data/web/review/snapshots/` because SQLite review events reference their exact
paths. The former five release-source JSON files and source manifest remain only
for generation provenance; Web review, SQLite state, and final evaluation use
the four files under `data/qa/7.final_2200/`.

The cleanup removed 1,145 hash-verified regenerable or superseded JSON files
(15,299,515 bytes) without archiving them. It then archived 50,086 historical
JSON files, 123 evaluation companion files, and 9,295 QA companion files. Of
those archived members, 63 JSON files and one review packet remain at their
canonical paths because active project references still use them. The other
59,440 archived members were removed from their original locations.

To inspect one category, extract into a separate directory and verify the member paths and hashes before any selective restoration:

```bash
mkdir -p /tmp/pku-archive-inspection
tar --zstd -xf data/archive/final_2200_cleanup_20260911/<category>/files.tar.zst -C /tmp/pku-archive-inspection
```
