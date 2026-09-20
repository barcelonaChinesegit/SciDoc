# Data archive

This directory contains immutable archives of historical data and audit
artifacts that are not direct inputs to the current final 2,200 QA release.
Each cleanup archive preserves project-relative paths and includes a CSV
manifest with the original size and SHA-256 of every archived file.

Current release data remains in `data/qa/7.final_2200/`. Files that are still
referenced by the review database, release manifest, or active runtime state
remain at their canonical paths and are not stored only in this archive.

