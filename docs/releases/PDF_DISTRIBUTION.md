# PDF assets for Google Drive

## Download and validate

The PDFs are available in the
[public Google Drive folder](https://drive.google.com/drive/folders/1J7l5HHPlKyjjegxZ2c0_plOnk9CdMVbY)
and are excluded from Git because of repository size limits. Download the
resources and extract any archives before PDF-based answer generation or
current-release PDF validation. Preserve the filenames and place the PDFs
directly at `data/pdfs/<filename>.pdf` in your checkout, without an extra nested
`data/pdfs/` directory.

After installing `requirements-eval.txt`, run from the repository root:

```bash
python evaluation/preflight.py --output data/results/preflight.json
```

Success reports `status: valid` and `pdf_count: 712`; the command verifies the
current benchmark's PDF hashes, readability and gold evidence-page bounds.
If you keep the PDFs elsewhere, add `--pdf-dir /absolute/path/to/pdfs` pointing
directly to the directory containing the PDF files.

## Asset inventories and packaging

Upload the existing frozen files without renaming, recompressing individual
PDFs, re-downloading newer paper versions, or rebuilding merged documents.
The existing `data/pdf_assets_manifest.json` remains the hash authority.
The CSV files here are upload inventories derived from that manifest, not a
second release manifest.

| Package | Files | Bytes | Contents |
| --- | ---: | ---: | --- |
| Full PDF backup | 1,717 | 7,645,305,946 | All files currently in the asset manifest: 703 `paper_*`, 583 `source_*`, 431 `z_cross_*` |
| Benchmark inputs | 712 | 3,262,178,765 | 474 single-paper PDFs plus 238 frozen merged evaluation PDFs |

To move **all PDFs**, use [pdf_upload_all.csv](pdf_upload_all.csv). To distribute
only what a researcher needs to evaluate ScienceDoc, use
[pdf_upload_benchmark.csv](pdf_upload_benchmark.csv). Every row lists the exact
relative path, byte count, SHA-256, and whether the benchmark needs it.

Keep the following alongside the PDF package:

- `data/pdf_assets_manifest.json`;
- the four original files under `data/qa/7.final_2200/` and
  `rel__collection__final_2200__manifest.json` (already versioned in Git;
  including byte-identical copies makes the Drive release self-contained);
- this guide and the relevant CSV upload inventory;
- optional `final_2200_classification_statistics.xlsx` for descriptive statistics.

Preserve the layout `data/pdfs/<filename>.pdf` on extraction. The final QA's
historical absolute `pdf_path` values are provenance; `evaluation/preflight.py` resolves
PDF identities through the manifest and `--pdf-dir`, without changing QA data.
Use the merged PDFs as supplied. Some final Multi-Document records do not carry
a complete source interval map, so recreating bundles from source IDs alone is
not an interchangeable substitute.

For future package updates, preserve public download access and keep the root
README's download link current. Do not include `models/`, `data/web/`,
credentials, or GPU caches in the public PDF package.

Regenerate and verify these lists without modifying PDFs or QA:

```bash
python evaluation/preflight.py --output preflight.json --pdf-inventory-dir docs/releases
```

The full asset list retains original PDFs needed for construction history.
The benchmark subset includes the PDFs needed for current-release answer
generation and PDF validation; source PDFs outside it are construction references. Manuscript drafts, screenshots,
and third-party PDFs outside the asset manifest are not benchmark inputs.

## Materials for score replay

The PDF package does not include the original five result components per model,
v4 Judge cache or historical Table 3 subject workbook. Those are separate
experiment artifacts listed in the [evaluation Quick Start](../../evaluation/README.md#quick-start).
Recorded-cache replay reads no PDF bytes; it recomputes scores from the frozen
result fields and bound Judge decisions. Current-release PDF validation is not
proof that a historical model run used the same inputs.
