# Current paper and stored predictions: conclusive reproduction discrepancies

> **2026-09-20 provenance clarification:** [PROVENANCE_RECONCILIATION.md](PROVENANCE_RECONCILIATION.md) is the latest interpretation. Historical experiments and their aggregates are traceable. Current-release strict diagnostics use different input/protocol combinations and do not disprove those experiments. Earlier “cannot reproduce” wording applies only to those diagnostic assumptions; newly generated Judge labels are not human ground truth.

Update: the later [gap investigation](GAP_INVESTIGATION.md) corrects the historical
adapter's over-rejection of reference-only revisions and reports a complete
Qwen3-VL-8B local rescore. Earlier bounds/counts below describe the pre-fix audit;
use the new investigation for current values. QA and raw-output rules are unchanged.


Date: 2026-09-20. The current paper is authoritative; all 27 QA JSON files are
unchanged. **The stored predictions cannot reproduce both paper tables under
the paper's strict contract.** This is a verified counterexample to equality,
not a claim that all eleven models have been fully rescored.

## Fast exhaustive checks, followed by actual local judging

The checker read all 55 baseline files under `data/results/`, bound legacy IDs
to the immutable final release, and validated original raw outputs without
stripping Markdown, extracting embedded JSON, repairing refusals, or changing
answers/pages. Every model retains all 2,200 gold slots. Predictions with
different embedded question/reference/evidence versions are identified as
technical failures, not treated as predictions to the current release.

For each task and discipline, the lower bound counts known correct answers;
the upper bound additionally counts every legal answerable item awaiting its
Judge as correct. Missing, illegal, and input-version failures stay in the
denominator. These bounds prove **30 Table 2 answer cells and 47 Table 3 cells
cannot equal the paper**, even if every pending semantic decision is favorable.
Comparison allows the paper's two-decimal rounding interval; raw full-precision
paper scores remain unavailable. Bounds are not guessed Judge results.

Three models together require only four semantic decisions after exhaustive
format and input checks. All four were actually run in one local batch using
**Qwen/Qwen3.6-27B**, the unchanged paper prompt, BF16 on an idle A800, left
padding, SDPA, greedy decoding, seed 42, disabled thinking, and a 32-token limit.
All four returned exact `INCORRECT`; no invalid Judge output or retry occurred.
Offline cache replay then scored all 2,200 slots for each of these three models.
The GPU was released. This new execution configuration is recorded explicitly;
it is not asserted to be the historical binary-Judge configuration.

| Table 2 All (%) | Paper | Strict current-release diagnostic | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| MiniCPM-V 2.6 | 15.59 | 0.00 | -15.59 |
| InternVL2.5-8B | 22.32 | 0.00 | -22.32 |
| Gemma 3 27B | 47.32 | 0.045454545454545456 (display 0.05) | -47.27454545454545 |

These scores describe the available artifacts under strict validation, not
the models' underlying answering ability. MiniCPM-V 2.6 has 2,198 illegal raw
outputs and two missing IDs. Its first raw output includes explanatory prose,
a JSON object without `evidence_pages`, and further explanation. InternVL2.5
and Gemma frequently wrap JSON in Markdown fences. The paper permits bounded
correction during inference, but no missing successful correction output can
be reconstructed or invented by the scorer. If separately retained final
corrected outputs exist, they must be supplied and independently audited.

| Table 3 discipline | Model | Paper (%) | Strict diagnostic (%) |
| --- | --- | ---: | ---: |
| All eight disciplines | MiniCPM-V 2.6 | See per-cell CSV | 0.00 in each |
| All eight disciplines | InternVL2.5-8B | See per-cell CSV | 0.00 in each |

The complete per-cell **Paper / exact value or bound / Difference / status**
tables are published:

- [Table 2: all 11 models, bounds](fast_reproduction_check/table2_bounds.csv)
- [Table 3: all 11 models, bounds](fast_reproduction_check/table3_bounds.csv)
- [Table 2: three locally completed model diagnostics](fast_reproduction_check/table2_local.csv)
- [Table 3: three locally completed model diagnostics](fast_reproduction_check/table3_local.csv)
- [All source hashes, status counts and raw-output witnesses](fast_reproduction_check/feasibility.json)
- [Local run summary](fast_reproduction_check/local_summary.json),
  [configuration](fast_reproduction_check/local_judge_config.json),
  [raw replies/token IDs](fast_reproduction_check/local_raw_calls.jsonl),
  [bound decisions](fast_reproduction_check/local_judge_cache.jsonl)

There are 11,550 remaining semantic jobs across the other eight models. They
were **not run**: their results cannot reverse the already proven discrepancies.
This early stop avoids a costly full rescore while answering the requested
equality question conclusively. Evidence headlines remain unresolved for
illegal outputs because the paper does not specify how to recover their page
sets; no new illegal-evidence policy was invented to force agreement.

## Removed or corrected outdated scoring behavior

- Deleted `src/pku_qa/evaluation/qa_scoring.py`: normalized text/alias matching,
  numeric tolerance, list matching, and ANLS can no longer determine correctness.
- Removed MCQ-letter extraction and rule-first answerable scoring from the
  internal Judge. Every legal answerable prediction uses the paper prompt.
- Internal Judge now reads the same verbatim prompt file as the public package.
  Its strict parser no longer trims or uppercases replies; attempts retain raw
  replies, and exhausted failures are retained as zero answer contributions.
- Local/API providers preserve output whitespace. Internal PDF parsing preserves
  answer text, permits only external JSON whitespace and page sorting/deduplication,
  and rejects duplicate JSON fields, invalid types and noncanonical refusals.
- Removed the unused page-label extraction helper from the Judge.
- That audit advanced inference/scoring versions to 7/6; the subsequent
  [follow-through](FOLLOWTHROUGH_REPORT.md) advances them to 8/7; the paper prompt
  hash is part of the Judge fingerprint. Older fingerprints do not certify a
  new run. This intentionally invalidates stale cached contracts.

The independent public `evaluation/` metric definitions and original semantic
prompt did not change. Historical result files and read-only `sxz/` were not
modified or deleted. Internal historical report tools remain identified as
diagnostics; they do not certify current paper tables.

Validation: **508 tests passed**; a fresh run of the feasibility checker produced
byte-identical tables and audit records. The 2,200-QA/712-PDF preflight passed,
and all 27 QA JSON hashes match the original task snapshot. See the
[verification record](fast_reproduction_check/verification.json).

## Repeat the checks

```bash
python evaluation/check_reproduction.py --output-dir data/results/paper_check
python scripts/judge_reproduction_jobs.py --audit-dir data/results/paper_check \
  --model-dir models/Qwen3.6-27B --gpu 2
python .agents/skills/pku-qa-maintainer/scripts/sync_project_docs.py --sync
python -m pytest -q
```

Use a fresh output directory. Both check commands return **2** for a discrepancy;
this is not a successful reproduction exit code. The first requires the frozen
PDFs, current manuscript and all baseline files; the second additionally requires
the local model and the project's ML environment. Run the second command after
the first reports differences; shell `&&` would intentionally stop at exit 2.

The next scientifically valid step is to recover the matching final corrected
prediction artifacts or rerun affected baseline inference using the unchanged
current QA and paper contract. Replacing the specified Judge with a smaller
model, relaxing parsing, changing QA, or modifying table values merely to force
agreement would not be reproduction.
