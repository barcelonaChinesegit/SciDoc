"""The exact sxz v4 rules and complete dataset-conditioned prompt."""
from pathlib import Path
import hashlib
from evaluation.sxz_v4 import build_judge_prompt

PROMPT_PATH = Path(__file__).parent / "prompts/semantic_judge.txt"
JUDGE_RULES = PROMPT_PATH.read_text(encoding="utf-8")
PROMPT_HASH = hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()
# Kept as a template for callers that explicitly supply the dataset identity.
JUDGE_PROMPT = build_judge_prompt("{dataset_id}", "{question}", "{correct}", "{model_answer}")


def build_prompt(question: str, reference: str, prediction: str, dataset_id: str) -> str:
    return build_judge_prompt(dataset_id, question, reference, prediction)
