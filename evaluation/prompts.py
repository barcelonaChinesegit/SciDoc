"""The paper's prompt, copied verbatim from its figure source text."""

from pathlib import Path
import hashlib

PROMPT_PATH = Path(__file__).parent / "prompts/semantic_judge.txt"
JUDGE_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")
PROMPT_HASH = hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()


def build_prompt(question: str, reference: str, prediction: str) -> str:
    return JUDGE_PROMPT.format(question=question, correct=reference, model_answer=prediction)
