from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from calculate_evidence_weighted_accuracy import (
    validate_judge_inputs,
)
from evaluation_protocol import (
    judge_inference_binding_sha256,
    pdf_corpus_sha256_from_manifest,
    sha256_file,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def gold_dataset() -> dict:
    return {
        "paper-1": {
            "QA": {
                "QA1": {
                    "question": "What is the answer?",
                    "answer": "Alpha",
                    "evidence_pages": [1],
                }
            }
        }
    }


def judge_dataset(
    *,
    mode: str,
    source_sha256: str,
    model: str,
    pdf_corpus_sha256: str | None = None,
    pdf_sha256: str = "paper-hash",
) -> dict:
    if pdf_corpus_sha256 is None:
        pdf_corpus_sha256 = pdf_corpus_sha256_from_manifest(
            {"paper-1": pdf_sha256}
        )
    if mode == "pdf":
        model_output = '{"answer_pre":"Alpha","evidence_pages":[1]}'
        require_structured = True
        require_evidence = True
        predicted_pages = [1]
    else:
        model_output = "Alpha"
        require_structured = False
        require_evidence = False
        predicted_pages = []
    dataset = {
        "paper-1": {
            "QA": {
                "QA1": {
                    "question": "What is the answer?",
                    "correct_answer": "Alpha",
                    "model_output": model_output,
                    "parsed_answer": "Alpha",
                    "reference_evidence_pages": [1],
                    "predicted_evidence_pages": predicted_pages,
                    "input_mode": mode,
                    "require_structured_output": require_structured,
                    "require_evidence_pages": require_evidence,
                    "output_is_legal": True,
                    "answer_is_correct": True,
                    "evidence_pages_is_correct": True,
                    "is_correct": True,
                    "protocol_fingerprint": f"{mode}-{model}-fingerprint",
                    "qa_source_sha256": source_sha256,
                    "pdf_corpus_sha256": (
                        pdf_corpus_sha256 if mode == "pdf" else None
                    ),
                    "pdf_sha256": pdf_sha256 if mode == "pdf" else None,
                    "shown_pdf_pages": [1] if mode == "pdf" else [],
                    "total_pdf_pages": 1 if mode == "pdf" else None,
                    "max_pdf_pages": 0,
                    "page_input_policy": "full",
                    "prompt_style": (
                        "pdf" if mode == "pdf" else "question_only"
                    ),
                    "evaluated_model": model,
                    "raw_model_output": model_output,
                    "raw_model_output_sha256": hashlib.sha256(
                        model_output.encode("utf-8")
                    ).hexdigest(),
                    "deterministic_normalizations": [],
                    "publication_eligible": True,
                    "protocol_validation": "publication_strict",
                    "judge_protocol_fingerprint": f"{mode}-judge-{model}",
                }
            }
        }
    }
    row = dataset["paper-1"]["QA"]["QA1"]
    row["inference_binding_sha256"] = judge_inference_binding_sha256(row)
    return dataset


class EvidenceWeightedValidationTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path, dict]:
        qa = gold_dataset()
        qa_path = root / "qa.json"
        write_json(qa_path, qa)
        source_sha256 = sha256_file(qa_path)
        pdf_dir = root / "pdf"
        closedbook_dir = root / "question_only"
        for model in ("4B", "8B"):
            pdf_inference = judge_dataset(
                mode="pdf",
                source_sha256=source_sha256,
                model=model,
            )
            pdf_judge = json.loads(json.dumps(pdf_inference))
            pdf_row = pdf_judge["paper-1"]["QA"]["QA1"]
            pdf_row["judge_protocol_fingerprint"] = f"pdf-judge-{model}"
            pdf_row["inference_binding_sha256"] = (
                judge_inference_binding_sha256(
                    pdf_inference["paper-1"]["QA"]["QA1"]
                )
            )
            write_json(pdf_dir / f"results_{model}.json", pdf_inference)
            write_json(
                pdf_dir / f"judge_{model}.json",
                pdf_judge,
            )
            question_inference = judge_dataset(
                mode="question_only",
                source_sha256=source_sha256,
                model=model,
            )
            question_judge = json.loads(json.dumps(question_inference))
            question_row = question_judge["paper-1"]["QA"]["QA1"]
            question_row["judge_protocol_fingerprint"] = (
                f"question-judge-{model}"
            )
            question_row["inference_binding_sha256"] = (
                judge_inference_binding_sha256(
                    question_inference["paper-1"]["QA"]["QA1"]
                )
            )
            write_json(
                closedbook_dir / f"results_{model}.json",
                question_inference,
            )
            write_json(
                closedbook_dir / f"judge_{model}.json",
                question_judge,
            )
        return qa_path, pdf_dir, closedbook_dir, qa

    def test_validates_all_modes_and_models_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            qa_path, pdf_dir, closedbook_dir, qa = self._fixture(Path(tmp))
            judged, audit = validate_judge_inputs(
                qa_path=qa_path,
                qa_data=qa,
                pdf_judge_dir=pdf_dir,
                closedbook_judge_dir=closedbook_dir,
            )
            self.assertEqual(audit["status"], "passed")
            self.assertEqual(audit["pdf_sha256_by_paper"], {"paper-1": "paper-hash"})
            self.assertEqual(set(judged["pdf"]), {"4B", "8B"})
            self.assertEqual(set(judged["question_only"]), {"4B", "8B"})

    def test_missing_judge_row_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            qa_path, pdf_dir, closedbook_dir, qa = self._fixture(Path(tmp))
            write_json(pdf_dir / "judge_8B.json", {})
            with self.assertRaisesRegex(ValueError, "judged/gold keys differ"):
                validate_judge_inputs(
                    qa_path=qa_path,
                    qa_data=qa,
                    pdf_judge_dir=pdf_dir,
                    closedbook_judge_dir=closedbook_dir,
                )

    def test_cross_model_pdf_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            qa_path, pdf_dir, closedbook_dir, qa = self._fixture(Path(tmp))
            write_json(
                pdf_dir / "judge_8B.json",
                judge_dataset(
                    mode="pdf",
                    source_sha256=sha256_file(qa_path),
                    model="8B",
                    pdf_corpus_sha256="other-corpus",
                    pdf_sha256="other-paper",
                ),
            )
            with self.assertRaisesRegex(
                ValueError, "different PDF corpus|pdf_corpus_sha256 mismatch"
            ):
                validate_judge_inputs(
                    qa_path=qa_path,
                    qa_data=qa,
                    pdf_judge_dir=pdf_dir,
                    closedbook_judge_dir=closedbook_dir,
                )

    def test_cross_model_per_paper_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            qa_path, pdf_dir, closedbook_dir, qa = self._fixture(Path(tmp))
            write_json(
                pdf_dir / "judge_8B.json",
                judge_dataset(
                    mode="pdf",
                    source_sha256=sha256_file(qa_path),
                    model="8B",
                    pdf_corpus_sha256=pdf_corpus_sha256_from_manifest(
                        {"paper-1": "paper-hash"}
                    ),
                    pdf_sha256="other-paper",
                ),
            )
            with self.assertRaisesRegex(
                ValueError, "per-paper PDF hashes|pdf_sha256 mismatch"
            ):
                validate_judge_inputs(
                    qa_path=qa_path,
                    qa_data=qa,
                    pdf_judge_dir=pdf_dir,
                    closedbook_judge_dir=closedbook_dir,
                )


if __name__ == "__main__":
    unittest.main()
