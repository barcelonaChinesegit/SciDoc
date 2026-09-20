# ScienceDoc: Scientific-Document Multimodal Question Answering

**English** | [简体中文](README.zh-CN.md)

ScienceDoc is a scientific-document multimodal QA benchmark being prepared for
ICLR 2027. It contains **2,200 questions across 8 disciplines and 61 scientific
fields**, covering text, figures, tables, and equations. Models return concise
answers with supporting physical PDF pages, or the exact label `Unanswerable`.

## Dataset

| Component | Release file | Questions | Global IDs |
| --- | --- | ---: | --- |
| General | [ordinary_qa.json](data/qa/7.final_2200/ordinary_qa.json) | 1,000 | QA0001–QA1000 |
| Unanswerable | [unanswerable_qa.json](data/qa/7.final_2200/unanswerable_qa.json) | 200 | QA1001–QA1200 |
| Reasoning | [reasoning_qa.json](data/qa/7.final_2200/reasoning_qa.json) | 200 | QA1201–QA1400 |
| Multi-Document | [cross_pdf_qa.json](data/qa/7.final_2200/cross_pdf_qa.json) | 800 | QA1401–QA2200 |

The [release manifest](data/qa/7.final_2200/rel__collection__final_2200__manifest.json)
binds these files by SHA-256. The final evaluation uses **712 PDF assets**:
474 single-paper PDFs and 238 merged PDFs. Merged PDFs are evaluation inputs,
not additional original papers. Multi-Document evidence refers to physical page
indices in the frozen merged PDF, not page numbers within a source paper.

PDFs and model weights are stored separately from Git. See the exact
[benchmark PDF inventory](docs/releases/pdf_upload_benchmark.csv) and
[Google Drive packaging guide](docs/releases/PDF_DISTRIBUTION.md).
A public Drive download link has not yet been configured. Obtain the frozen
assets from the maintainers and preserve their filenames under `data/pdfs/`.

## Quick start

The standalone evaluation package supports Python 3.10+ and requires no GPU
for validation. Run these commands from the repository root:

```bash
git clone https://github.com/barcelonaChinesegit/SciDoc.git
cd SciDoc
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-eval.txt
# After placing the frozen PDFs in data/pdfs/:
python evaluation/preflight.py --output preflight.json
python scripts/validate_submission.py --predictions predictions.jsonl
```

Recommended submission format: one record per line, with a binding `qa_id`:

```json
{"qa_id":"QA0001","answer_pre":"First maximize worst-case payoff, then maximize expected payoff under conjectured model","evidence_pages":[1,9]}
```

The model's **raw output** contains only `answer_pre` and `evidence_pages`.
`qa_id` is added by the submission writer. Page values must be positive JSON
integers in the PDF's physical range; booleans and page strings are illegal.
The only refusal is `{"answer_pre":"Unanswerable","evidence_pages":[]}`.
See the full [evaluation contract and commands](evaluation/README.md).

## Evaluation and reproduction status

The paper defines semantic **Answer Accuracy**, macro **E-Precision**, **E-Recall**,
**E-F1**, and **A-Pages**, with a fixed full-benchmark denominator of 2,200.
Answerable items use the paper's Qwen3.6-27B binary semantic judge; Unanswerable
items use deterministic exact-label validation. Exact page-set match and joint
correctness are audit diagnostics. There is no composite Overall Score.

The new package preserves the paper prompt verbatim and does not normalize
answer meanings. Historical experiment artifacts use a different tri-class
judge and permissive parsing, and some predictions target older gold versions.
**An official rescore reproducing the paper has not been established.**
Reaggregating historical v4 records matches all 99 displayed Table 2 cells;
Table 3 matches 98/99 cells, with a discrepancy in Claude's All value.
See the [complete audit and remaining reproducibility items](docs/reports/official_evaluation/FINAL_REPORT.md).

The [subsequent full-artifact check and local Judge verification](docs/reports/official_evaluation/REPRODUCTION_DISCREPANCY.md)
demonstrate concrete mismatches with the current paper protocol. Old rule-first
answer scoring has been removed; QA and historical artifacts remain unchanged.

```bash
python evaluation/evaluate.py --predictions predictions.jsonl \
  --judge-config judge-config.json --judge-cache .cache/sciencedoc_judge.jsonl \
  --output results.json
python -m pytest -q tests/test_validation.py tests/test_metrics.py tests/test_judge.py tests/test_e2e.py
```

Judge configuration must be explicitly supplied; unavailable historical settings
are not guessed. Use `--offline` for cache-only diagnostics. Incomplete or
unresolved evaluations return nonzero and are clearly labeled.

## Repository layout

| Path | Purpose |
| --- | --- |
| `data/qa/7.final_2200/` | Immutable evaluation inputs and existing release manifest |
| `evaluation/` | Paper-contract validation, semantic judge cache, macro metrics, preflight, and reproduction audit |
| `scripts/validate_submission.py` | Submission checking without semantic judging |
| `examples/` | Public submission example |
| `src/pku_qa/` | Internal construction, inference, review, and historical experiment workflows |
| `tests/`, `schemas/` | Tests and dataset schemas |
| `docs/releases/` | External PDF packaging and file inventories |
| `docs/current/` | Maintainer and review documentation |

The optional internal experiment console is documented separately in the
[maintainer guide](docs/current/WEB_CONSOLE.md). It is not needed to evaluate
benchmark submissions.

## Contributors

Thanks to [yuetanbupt](https://github.com/yuetanbupt) and
[ElephantsGit](https://github.com/ElephantsGit) for contributing to ScienceDoc.
See [CONTRIBUTORS.md](CONTRIBUTORS.md) for GitHub attribution and access details.

PDFs and model weights remain subject to their original licenses. Account
databases, session material, API keys, and local model weights are not published.
