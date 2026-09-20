# Unanswerable QA

This directory holds the categorized, read-only view of the final 200
single-PDF questions whose answer is exactly `Unanswerable` and whose evidence
pages are empty.

`data/qa/7.final_2200/unanswerable_qa.json` is the only authority. Run
`PYTHONPATH=src python -m pku_qa.workflows.selection.sync_final_2200_manifest`
after changing the final file; the command refreshes this view and the final
collection manifest together.
