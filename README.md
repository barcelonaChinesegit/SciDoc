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

## Download the PDFs

PDFs are hosted separately because of GitHub repository size limits.
**Download: [public Google Drive folder](https://drive.google.com/drive/folders/1J7l5HHPlKyjjegxZ2c0_plOnk9CdMVbY).**
Download them before reproducing PDF-based answer generation or validating the
release's PDF inputs.

1. Download the PDF resources from the folder and extract any archives.
2. Place the PDFs directly under `data/pdfs/` in your checkout, preserving their
   original filenames: `data/pdfs/<filename>.pdf`. Avoid an extra nested
   `data/pdfs/` directory. Keep the supplied merged PDFs unchanged.
3. After installing the Quick Start dependencies below, run from the repository root:

   ```bash
   python evaluation/preflight.py --output data/results/preflight.json
   ```

Success reports `status: valid` and `pdf_count: 712`, with PDF hashes,
readability and gold evidence-page bounds checked for the current release.
The full asset inventory contains 1,717 PDFs; the benchmark uses 712 of them.
See the [benchmark PDF inventory](docs/releases/pdf_upload_benchmark.csv) and
[PDF distribution guide](docs/releases/PDF_DISTRIBUTION.md) for exact files and
alternative storage paths. Model weights and original experiment results/Judge
caches are separate resources. The recorded-cache score replay below does not
read PDFs.

## Quick Start

### 1. Install and check the evaluator

Use Python 3.10+ for the standalone evaluator. From a fresh checkout:

```bash
git clone https://github.com/barcelonaChinesegit/SciDoc.git
cd SciDoc
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-eval.txt
python evaluation/evaluate.py --help
python -m pytest -q tests/test_sxz_evaluation.py tests/test_judge.py tests/test_e2e.py tests/test_metrics.py
```

These checks need no GPU, PDFs, weights, API key or historical results. The
comparison test against the local `sxz/` source skips if that directory is absent;
the bundled source-hash and scoring tests still run.

### 2. Replay one model's recorded evaluation

Obtain these original experiment artifacts from the maintainers; Git does not
include them:

| Asset | Expected location |
| --- | --- |
| Five Qwen3-VL-8B result JSONs, retaining reference and prediction fields | `data/results/Qwen3-VL-8B/` |
| Original v4 Judge cache | `sxz/evaluation_11models_5datasets_qwen36_calibrated_fixeddenom_v4/qwen36_answer_judge_cache.json` |

The five filenames and record schema are in the
[input guide](evaluation/README.md#输入与安装). Paths below match the maintained
workspace; `--results-dir` and `--judge-cache` may point to copies elsewhere.
The scorer does not import or write to `sxz/`.

```bash
python evaluation/evaluate.py \
  --results-dir data/results --model Qwen3-VL-8B --offline \
  --judge-cache sxz/evaluation_11models_5datasets_qwen36_calibrated_fixeddenom_v4/qwen36_answer_judge_cache.json \
  --output-dir data/results/quick_start_qwen8b
```

This uses no GPU, PDF files or network calls. It recovers predictions, reuses
input-bound recorded Judge labels and recomputes the metrics. With the original
artifacts, expect `table2_display_matches: 9`, `paper_differences: []` and exit
code **0**.

### 3. Read the result

```bash
python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path("data/results/quick_start_qwen8b/report.json").read_text())
print(report["execution"])
print(report["table2"]["Qwen3-VL-8B"])
PY
```

The execution is `recorded_cache_replay`. Rounded results are **68.32%** All
accuracy, **40.66%** E-F1 and **8.44** A-Pages. Per-item results are in
`data/results/quick_start_qwen8b/Qwen3-VL-8B/details.csv`; input/code/cache hashes
are in `run_binding.json`. Repeating the command verifies the same binding;
changed inputs or code require a new output directory.

Remove `--model` for all 11 supported baselines. Add the original
`--subject-xlsx` for Table 3 and `--reference-summary` to compare the historical
CSV; see the [full replay command](evaluation/README.md#原实验缓存回放无需-gpu).

### 4. Start a new Judge run (optional, GPU required)

With the internal Python 3.12+ ML environment, a free A800 and local
Qwen3.6-27B weights installed:

```bash
python evaluation/evaluate.py \
  --results-dir data/results --model Qwen3-VL-8B \
  --gpu 2 --model-dir models/Qwen3.6-27B \
  --output-dir data/results/quick_start_qwen8b_new_judge
```

Choose the GPU index for your machine. This judges existing predictions using
the v4 prompt and settings; it does not rerun PDF answer generation. Do not pass
`--judge-cache` for a new run. Dependencies and resume rules are in the
[new-run guide](evaluation/README.md#新本地-judge-运行).

## Dataset and output-format checks

After [downloading the frozen PDFs](#download-the-pdfs), validate the current four-file release and its
712 PDF assets:

```bash
python evaluation/preflight.py --output data/results/preflight.json
```

The model-generation contract is JSON with `answer_pre` and `evidence_pages`,
using 1-based physical PDF pages. A refusal is
`{"answer_pre":"Unanswerable","evidence_pages":[]}`. For a **format diagnostic**,
`scripts/validate_submission.py --predictions predictions.jsonl` accepts flat
records with an added `qa_id` and requires the release PDFs. The bundled
`examples/predictions.example.jsonl` has only two records and correctly returns
1 for incompleteness. The v4 scorer consumes five result files, not flat JSONL.

Current-release validation and original-experiment replay use different input
versions. Preserve the references stored with the historical results when
reproducing that experiment.

## Evaluation and reproduction status

`evaluation/` now uses the **sxz v4 experiment's exact parsing, calibrated
CORRECT/PARTIAL/WRONG Judge prompt, embedded historical references and fixed
2,200 denominator**. Only CORRECT contributes to answer accuracy. Evidence is
scored independently before answer validity checks. The strict producer-format
validator above is a diagnostic, not a scoring gate.

Replaying all 55 result files with the original bound Judge cache matches all
1,540 numeric cells of the historical 77-row summary. Table 2 matches **99/99**
displayed cells; Table 3 matches **98/99**. Claude's Table 3 All remains 68.05%
from the subject cohort versus 69.32% in the paper. QA and paper text are unchanged.

The scorer implementation and recorded-cache replay are verified. Fresh Judge
generation is a separate run: historical weight identity was not fully recorded,
so identical newly generated labels cannot be guaranteed. The manuscript still
contains a binary Judge prompt that differs from the adopted experiment rules;
this code update has not revised the manuscript. See the
[alignment verification](docs/reports/official_evaluation/SXZ_V4_ALIGNMENT.md).

## Repository layout

| Path | Purpose |
| --- | --- |
| `data/qa/7.final_2200/` | Current four-file release and manifest; original experiment references stay in result files |
| `evaluation/` | sxz v4 experiment scoring, recorded-cache replay, macro metrics and format diagnostics |
| `scripts/validate_submission.py` | Submission checking without semantic judging |
| `examples/` | Two-record output-format diagnostic example |
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
