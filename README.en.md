# SciDoc: Multimodal Question Answering over Scientific Papers

[简体中文](README.md) | **English**

SciDoc evaluates reading comprehension, evidence localization, and reasoning across scientific papers in PDF format. Questions draw on text, images, tables, and formulas. Models must provide short answers and identify the physical PDF pages supporting them, or abstain when the papers do not provide sufficient evidence.

The project provides **2,200 short-answer QA items with completed human review**, data validation tools, resumable inference and scoring pipelines, and a collaborative review console. Its construction process corresponds to the question generation, independent review, human verification, and evidence evaluation described in the paper's Dataset section and appendix.

## Dataset

The current release consists of four QA files under [`data/qa/7.final_2200/`](data/qa/7.final_2200/):

| File | QA items | PDFs¹ | Task |
| --- | ---: | ---: | --- |
| [ordinary_qa.json](data/qa/7.final_2200/ordinary_qa.json) | 1,000 | 398 | Factual understanding, comparison, and calculation within one paper |
| [unanswerable_qa.json](data/qa/7.final_2200/unanswerable_qa.json) | 200 | 132 | Recognizing insufficient evidence and abstaining correctly |
| [reasoning_qa.json](data/qa/7.final_2200/reasoning_qa.json) | 200 | 87 | Combining multiple necessary facts for reasoning within one paper |
| [cross_pdf_qa.json](data/qa/7.final_2200/cross_pdf_qa.json) | 800 | 238 | Answering questions that require information from two or three papers |

¹ These are per-component PDF counts, with overlap between components. The final evaluation uses **712 distinct PDFs: 474 single-paper PDFs and 238 merged PDFs**. A merged PDF is not an additional original paper. Of the Cross-PDF questions, 621 require two papers and 179 require three.

All questions have global IDs from `QA0001` to `QA2200`, with Cross-PDF occupying the last 800 IDs. The release covers eight primary disciplines and 21 question categories. It contains 1,053 items requiring one modality and 1,147 requiring multiple modalities. Exact modality combinations include 982 text-only items, 609 text-and-formula items, 192 text-and-image items, and 189 text-and-table items; the remaining items use other combinations.

The [release manifest](data/qa/7.final_2200/rel__collection__final_2200__manifest.json) fixes the four files' counts, identities, and content hashes. The [classification workbook](data/qa/7.final_2200/final_2200_classification_statistics.xlsx) provides component statistics and per-question details. The upstream pool of 6,204 raw QA items, the cleaned baseline of 4,211 items, and the source batches document construction history; their counts are not added to the final 2,200 items.

## Quick Start

### 1. Clone the repository and validate the release

Use Python 3.12 or later. This first check requires no GPU, PDFs, API keys, or third-party Python packages:

```bash
git clone https://github.com/barcelonaChinesegit/SciDoc.git
cd SciDoc
PYTHONPATH=src python -m pku_qa.workflows.operations.inspect_final_2200
```

A successful check prints `"status": "valid"`, `"qa_count": 2200`, the four component counts, and `"evaluation_pdfs": 712`. The command reads the four release files without modifying them, validates shared fields, global IDs, counts, and manifest hashes, and recomputes modality and document-count statistics. `pdf_check: "not_requested"` means that external PDF assets have not yet been checked.

Read a real question:

```python
import json
from pathlib import Path

data = json.loads(Path("data/qa/7.final_2200/ordinary_qa.json").read_text())
paper_id, paper = next(iter(data.items()))
qa_id, qa = next(iter(paper["QA"].items()))
print(paper_id, qa_id)
print(qa["question"])
print(qa["answer"])
print(qa["evidence_pages"])
```

### 2. Obtain the PDFs and validate evidence pages

PDFs are managed separately; cloning the repository does not download PDFs or model weights. Obtain files matching the [PDF asset manifest](data/pdf_assets_manifest.json) from the project maintainers and place them directly under `data/pdfs/`. Preserve the manifest's `paper_*`, `source_*`, and `z_cross_*` filenames. A newly downloaded version of a paper cannot substitute for its frozen PDF.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install pypdf==6.10.2
PYTHONPATH=src python -m pku_qa.workflows.operations.inspect_final_2200 --check-pdfs
```

This verifies SHA-256 hashes, readability, and all gold evidence-page bounds for the 712 evaluation PDFs. On success, `pdf_check` is `hashes_readability_and_evidence_bounds_valid`.

### 3. Run the evaluation

Full inference requires CUDA and local model weights. See the [Getting Started guide](docs/current/GETTING_STARTED.md) for installation and GPU memory requirements. Prepare `models/Qwen3-VL-4B-Instruct/`, `models/Qwen3-VL-8B-Instruct/`, and `models/Qwen3.6-27B/`, then print the evaluation plan:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
PYTHONPATH=src python -m pku_qa.workflows.reporting.run_final_2200_evaluation \
  --print-plan -- --dynamic-a800 --a800-gpus 2 3 4 5
```

Adjust the GPU IDs to match your machine. After checking resources and inputs, remove `--print-plan` to execute the plan. Use `--component reasoning` to select a single component. By default, the runner processes ordinary, unanswerable, reasoning, and cross_pdf in that order. Each component runs 4B → 8B → Judge → strict report, writing results to `data/results/evaluations/final_2200/<component>/`. Printing the plan does not start inference or call paid APIs.

## Dataset Construction

Construction begins with arXiv retrieval, discipline classification, and PDF validation, yielding 6,204 raw QA items from 703 papers. Format cleaning, deduplication, conversion to short answers, and evidence correction produce a single-paper baseline of 4,211 items from 693 papers. Further stages construct questions that are relevant to a paper but unanswerable from its evidence, reasoning questions combining multiple facts, and questions requiring multiple papers.

Generation includes both composition from verified QA facts and generation from full PDFs. Automated checks cover fields, page numbers, duplicates, answer leakage, and necessary dependencies. Specialized candidates undergo independent review by Claude Sonnet 5 and Gemini 2.5 Flash, evidence recovery, and human verification of each item. All 2,200 final items also undergo modality review using PDF page images. Generation, review, and historical diagnostic records support traceability; the current independent gold QA files remain authoritative.

## Data Format and Evaluation Protocols

The top-level JSON structure is `{paper_id: paper}`, and each paper contains a `QA` mapping of `{qa_id: qa}`. The six shared required fields are `question`, `answer`, `evidence_pages`, `modal_types`, `question_type`, and `question_category`. See the [final QA schema](schemas/final_2200_qa.schema.json) for the full definition. Final items contain no multiple-choice `options`; task-specific provenance and review fields are retained.

| Protocol | Input | Model output |
| --- | --- | --- |
| `question_only` | Question only, for closed-book controls and difficulty checks | Plain answer text |
| `pdf` | Question and page images labeled `[Page N]` | `{"answer_pre":"…","evidence_pages":[1,3]}` |

Full, Oracle, and evidence-page ablation change only page selection within the `pdf` protocol. All page numbers are 1-based physical PDF pages; Cross-PDF uses page numbers in the merged file. The exact abstention label is `Unanswerable`. In PDF mode, abstention must return `{"answer_pre":"Unanswerable","evidence_pages":[]}`.

Evaluation reports answer correctness, exact evidence-page-set match, and joint correctness separately. Evidence precision, recall, and F1 provide additional localization analysis. Deterministic scoring rules run first, followed by a shared semantic Judge when needed. Invalid outputs count as errors. Strict reports require complete question coverage and consistent bindings among QA files, PDFs, protocols, raw outputs, and Judge records. Results from older versions and manually scored samples must identify their inputs and denominators separately; they do not represent results for the current full 2,200-item release.

## Human Review and Documentation

[Open the Web console](https://pku.chenzijian.com/). Sign in with an individual application account. Administrators assign review scopes using stable QA identities; reviewers verify questions, answers, and evidence against the PDFs. Edits and deletions retain snapshots and audit records. The public entry point connects to the school's internal service through a VPS HTTPS reverse tunnel.

The detailed guides below are currently in Chinese; the Web console interface is in English.

- [Architecture](docs/current/ARCHITECTURE.md): protocols, modules, directories, and field constraints.
- [Getting Started](docs/current/GETTING_STARTED.md): installation, preflight checks, evaluation, and troubleshooting.
- [Human Review Guide](docs/current/MANUAL_REVIEW_GUIDE.md): per-question criteria, interface reference, and review workflow.
- [Web Console](docs/current/WEB_CONSOLE.md): accounts, assignments, permissions, and deployment.
- [QA Dataset Catalog](docs/current/DATASET_CATALOG.md): purposes and permissions of final inputs, upstream sources, and process artifacts.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `data/qa/7.final_2200/` | Four-file release, manifest, and classification workbook |
| `data/qa/1.base/` through `6.review/` | Base sources, unanswerable mirror, Reasoning/Cross-PDF source components, human review sources, and review evidence |
| `src/pku_qa/evaluation/` | Protocols, inference, Judge, reporting, and checkpoint recovery |
| `src/pku_qa/workflows/` | Generation, cleaning, review, selection, statistics, and operations |
| `task_queue_web/`, `src/pku_qa/services/` | Web console and backend services |
| `tests/`, `schemas/` | Automated tests and data schema constraints |
| `scripts/dataset_construction/` | Early paper acquisition, base QA notebooks, and prompts |
| `data/archive/` | Historical experiments and construction records; Git includes only documentation and inventories |

PDFs and models remain subject to their respective source licenses. Personal account databases, sessions, and API keys are not included in the repository. The `sxz/` directory belongs to another collaborator and is strictly read-only reference material.
