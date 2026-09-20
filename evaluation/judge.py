"""Strict binary judge and persistent, content-bound decision cache.

No generation setting is inferred from historical tri-class runs. A supplied
configuration identifies a new run; it does not certify paper reproduction.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

from evaluation.prompts import JUDGE_PROMPT
from evaluation.validation import Gold, Prediction, digest, strict_json


def object_hash(value) -> str:
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


class Judge:
    def __init__(self, config: dict, cache: Path | None = None,
                 generate: Callable[[str], str] | None = None, prompt: str = JUDGE_PROMPT):
        self.config, self.cache, self.generate, self.prompt = config, cache, generate, prompt
        self.config_hash = object_hash(config)
        self.prompt_hash = digest(prompt)
        self.entries = {}
        if cache and cache.exists():
            for number, line in enumerate(cache.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                row = strict_json(line)
                if row.get("key") != object_hash(row.get("binding")):
                    raise ValueError(f"Judge cache line {number}: invalid input binding hash")
                if row.get("decision") not in {"CORRECT", "INCORRECT"}:
                    raise ValueError(f"Judge cache line {number}: invalid decision")
                if row["key"] in self.entries and self.entries[row["key"]]["decision"] != row["decision"]:
                    raise ValueError(f"Judge cache line {number}: conflicting decisions")
                self.entries[row["key"]] = row

    def decide(self, gold: Gold, prediction: Prediction) -> dict:
        binding = {"qa_id": gold.qa_id, "question_sha256": digest(gold.question),
                   "reference_answer_sha256": digest(gold.answer),
                   "model_prediction_sha256": object_hash({"answer_pre": prediction.answer_pre,
                                                            "evidence_pages": prediction.evidence_pages,
                                                            "raw_sha256": prediction.audit.get("raw_output_sha256")}),
                   "judge_identity": self.config.get("identity"), "prompt_sha256": self.prompt_hash,
                   "config_sha256": self.config_hash}
        key = object_hash(binding)
        if key in self.entries:
            return {**self.entries[key], "cache_hit": True}
        if self.generate is None:
            return {"decision": None, "error": "no_matching_cache_or_configured_judge", "binding": binding, "attempts": []}
        attempts = []
        max_attempts = self.config.get("max_attempts")
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("Judge max_attempts must be explicitly configured; no historical policy is assumed")
        prompt = self.prompt.format(question=gold.question, correct=gold.answer, model_answer=prediction.answer_pre)
        for _ in range(max_attempts):
            try:
                raw = self.generate(prompt)
                # No uppercasing, regex extraction, whitespace stripping or fuzzy acceptance.
                attempts.append({"raw": raw})
                if raw not in {"CORRECT", "INCORRECT"}:
                    continue
                row = {"key": key, "binding": binding, "decision": raw, "attempts": attempts,
                       "timestamp": datetime.now(timezone.utc).isoformat()}
                if self.cache:
                    self.cache.parent.mkdir(parents=True, exist_ok=True)
                    with self.cache.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                self.entries[key] = row
                return {**row, "cache_hit": False}
            except Exception as exc:
                # Do not persist server URLs, request headers or secrets in exceptions.
                attempts.append({"error_type": type(exc).__name__})
        return {"decision": None, "error": "judge_failure", "binding": binding, "attempts": attempts}


def api_generator(config: dict) -> Callable[[str], str]:
    """Explicit OpenAI-compatible serving contract; keys and endpoint stay in env."""
    required = {"identity", "backend", "max_attempts", "timeout_seconds", "generation", "api_model"}
    if not required <= config.keys() or config["backend"] != "openai_compatible":
        raise ValueError(f"Judge config requires {sorted(required)} and backend=openai_compatible")
    identity = config["identity"]
    if not isinstance(identity, dict) or identity.get("model") != "Qwen/Qwen3.6-27B":
        raise ValueError("The paper specifies Qwen/Qwen3.6-27B; provide identity.model exactly")
    for field in ("checkpoint", "revision", "tokenizer_sha256", "chat_template_sha256", "serving_version"):
        if not identity.get(field):
            raise ValueError(f"UNRESOLVED REPRODUCIBILITY ITEM: identity.{field}")
    generation = config["generation"]
    if not isinstance(generation, dict) or set(generation) - {"temperature", "top_p", "top_k", "seed", "max_tokens", "extra_body"}:
        raise ValueError("generation must contain only explicit decoding parameters, never model/messages overrides")
    # Explicit null means disabled/unsupported and must be documented by the server,
    # never an implicit library default. Sent unchanged to the configured server.
    for field in ("temperature", "top_p", "top_k", "seed", "max_tokens"):
        if field not in generation:
            raise ValueError(f"UNRESOLVED REPRODUCIBILITY ITEM: generation.{field}")
    if not isinstance(config["timeout_seconds"], (int, float)) or config["timeout_seconds"] <= 0:
        raise ValueError("timeout_seconds must be explicitly positive")
    if type(config["max_attempts"]) is not int or config["max_attempts"] < 1:
        raise ValueError("max_attempts must be an explicit positive integer")
    if type(generation["max_tokens"]) is not int or generation["max_tokens"] < 1:
        raise ValueError("generation.max_tokens must be an explicit positive integer")
    url = os.environ.get("SCIENCEDOC_JUDGE_URL")
    if not url:
        raise ValueError("Set SCIENCEDOC_JUDGE_URL to the /chat/completions endpoint")

    def generate(prompt):
        body = {"model": config["api_model"], "messages": [{"role": "user", "content": prompt}], **generation}
        headers = {"Content-Type": "application/json"}
        key = os.environ.get("SCIENCEDOC_JUDGE_API_KEY")
        if key:
            headers["Authorization"] = "Bearer " + key
        request = Request(url, data=json.dumps(body).encode(), headers=headers)
        with urlopen(request, timeout=config["timeout_seconds"]) as response:
            payload = strict_json(response.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]

    return generate
