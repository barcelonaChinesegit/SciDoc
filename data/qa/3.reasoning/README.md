# Reasoning retained data

This directory retains the two 100-question source components and the evidence
needed to audit their PDF-dependency repair, dual review, and Claude difficulty
calibration.

- `rel__reasoning__refreshed__batch01__n100__claude_hard__v1.json` is batch 1.
- `hard_expansion/rel__reasoning__incremental__batch02__n100__claude_hard__v1.json`
  is batch 2.
- The non-`claude_hard` versions, historical clean set, dependency audit,
  cleaning ledger, and 11-item calibration record preserve the transformation
  lineage.

The canonical 200-question publication and Web-review file is
`data/qa/7.final_2200/reasoning_qa.json`. Candidate pools, pilots, and stale
progress files are not retained here.
