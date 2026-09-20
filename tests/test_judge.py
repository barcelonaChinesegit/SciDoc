from dataclasses import replace
import json

import pytest

from evaluation.judge import Judge, api_generator
from evaluation.validation import Gold, validate_raw


def inputs():
    gold = Gold("QA0001", "What value?", "42", (2,), "General", "Physics", "Optics", "paper", 9)
    pred = validate_raw(gold.qa_id, '{"answer_pre":"42","evidence_pages":[2]}', 9)
    return gold, pred


@pytest.mark.parametrize("raw", ["CORRECT", "INCORRECT"])
def test_binary_labels(raw):
    judge = Judge({"max_attempts": 1}, generate=lambda _: raw)
    assert judge.decide(*inputs())["decision"] == raw


@pytest.mark.parametrize("raw", ["correct", "CORRECT\n", "It is CORRECT", "PARTIAL", "WRONG", ""])
def test_invalid_labels_fail(raw):
    judge = Judge({"max_attempts": 1}, generate=lambda _: raw)
    result = judge.decide(*inputs())
    assert result["decision"] is None
    assert result["error"] == "judge_failure"


def test_retry_and_final_failure(tmp_path):
    outputs = iter(["bad", "CORRECT"])
    path = tmp_path / "cache.jsonl"
    result = Judge({"max_attempts": 2}, path, lambda _: next(outputs)).decide(*inputs())
    assert result["decision"] == "CORRECT"
    assert [x["raw"] for x in result["attempts"]] == ["bad", "CORRECT"]
    failed = Judge({"max_attempts": 2}, generate=lambda _: "bad").decide(*inputs())
    assert len(failed["attempts"]) == 2 and failed["decision"] is None


def test_cache_reuse_and_invalidation(tmp_path):
    cache = tmp_path / "cache.jsonl"
    config = {"max_attempts": 1, "identity": {"model": "mock"}}
    gold, pred = inputs()
    Judge(config, cache, lambda _: "CORRECT").decide(gold, pred)
    judge = Judge(config, cache)
    assert judge.decide(gold, pred)["cache_hit"]
    changed = validate_raw(gold.qa_id, '{"answer_pre":"43","evidence_pages":[2]}', 9)
    assert judge.decide(gold, changed)["decision"] is None
    assert Judge(config, cache, prompt="changed").decide(gold, pred)["decision"] is None
    assert Judge({**config, "max_attempts": 2}, cache).decide(gold, pred)["decision"] is None
    assert judge.decide(replace(gold, question="Changed?"), pred)["decision"] is None
    assert judge.decide(replace(gold, answer="43"), pred)["decision"] is None
    assert judge.decide(replace(gold, qa_id="QA0002"), pred)["decision"] is None


def test_corrupt_cache_is_fatal(tmp_path):
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps({"key": "wrong", "binding": {}, "decision": "CORRECT"}))
    with pytest.raises(ValueError, match="binding"):
        Judge({}, cache)


def api_config():
    # Synthetic transport fixture, not a claimed historical configuration.
    return {"backend": "openai_compatible", "api_model": "test-serving-label",
            "identity": {"model": "Qwen/Qwen3.6-27B", "checkpoint": "test-checkpoint",
                "revision": "test-revision", "tokenizer_sha256": "test-tokenizer",
                "chat_template_sha256": "test-template", "serving_version": "test-server"},
            "max_attempts": 1, "timeout_seconds": 2,
            "generation": {"temperature": 0, "top_p": 1, "top_k": None, "seed": 7, "max_tokens": 8}}


def test_api_transport_preserves_prompt_and_explicit_settings(monkeypatch):
    monkeypatch.setenv("SCIENCEDOC_JUDGE_URL", "https://judge.invalid/chat/completions")
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b'{"choices":[{"message":{"content":"CORRECT"}}]}'

    def transport(request, timeout):
        requests.append((json.loads(request.data), timeout))
        return Response()

    monkeypatch.setattr("evaluation.judge.urlopen", transport)
    config = api_config()
    assert api_generator(config)("  exact prompt\n") == "CORRECT"
    assert requests[0][0]["messages"] == [{"role": "user", "content": "  exact prompt\n"}]
    assert requests[0][0]["seed"] == 7 and requests[0][0]["max_tokens"] == 8
    assert requests[0][1] == 2


def test_api_missing_settings_are_not_guessed():
    config = api_config()
    del config["generation"]["seed"]
    with pytest.raises(ValueError, match="UNRESOLVED.*seed"):
        api_generator(config)


def test_paper_prompt_provenance_hash():
    from evaluation.prompts import PROMPT_HASH, PROMPT_PATH
    provenance = json.loads((PROMPT_PATH.parent / "provenance.json").read_text())
    assert PROMPT_HASH == provenance["sha256"]
