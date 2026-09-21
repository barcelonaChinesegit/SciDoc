from dataclasses import replace
import json

import pytest

from evaluation.judge import Judge, api_generator
from evaluation.validation import Gold, validate_raw


def inputs():
    gold = Gold("QA0001", "What value?", "42", (2,), "General", "Physics", "Optics", "paper", 9)
    pred = validate_raw(gold.qa_id, '{"answer_pre":"42","evidence_pages":[2]}', 9)
    return gold, pred


@pytest.mark.parametrize("raw,expected", [("CORRECT", "CORRECT"), ("PARTIAL", "PARTIAL"),
    ("WRONG", "WRONG"), ("INCORRECT", "WRONG"), ("correct", "CORRECT"),
    ("CORRECT\n", "CORRECT"), ("It is CORRECT", "CORRECT"), ("CORRECT but PARTIAL", "PARTIAL")])
def test_source_label_parser(raw, expected):
    judge = Judge({"max_attempts": 1}, generate=lambda _: raw)
    assert judge.decide(*inputs())["decision"] == expected


@pytest.mark.parametrize("raw", ["", "uncertain", "correctness"])
def test_invalid_labels_abort(raw):
    with pytest.raises(ValueError, match="invalid label"):
        Judge({"max_attempts": 1}, generate=lambda _: raw).decide(*inputs())


def test_no_retry_or_cached_failure(tmp_path):
    calls = []
    def generate(prompt):
        calls.append(prompt)
        return "bad"
    cache = tmp_path / "cache.jsonl"
    with pytest.raises(ValueError, match="invalid label"):
        Judge({"max_attempts": 1}, cache, generate).decide(*inputs())
    assert len(calls) == 1 and not cache.exists()
    with pytest.raises(ValueError, match="max_attempts"):
        Judge({"max_attempts": 2})


def test_cache_reuse_and_invalidation(tmp_path):
    cache = tmp_path / "cache.jsonl"
    config = {"max_attempts": 1, "identity": {"model": "mock"}}
    gold, pred = inputs()
    Judge(config, cache, lambda _: "CORRECT").decide(gold, pred)
    judge = Judge(config, cache)
    assert judge.decide(gold, pred)["cache_hit"]
    changed = validate_raw(gold.qa_id, '{"answer_pre":"43","evidence_pages":[2]}', 9)
    for instance, g, p in [
        (judge, gold, changed), (Judge(config, cache, prompt="changed"), gold, pred),
        (Judge({**config, "identity": {"model": "changed"}}, cache), gold, pred),
        (judge, replace(gold, question="Changed?"), pred),
        (judge, replace(gold, answer="43"), pred),
        (judge, replace(gold, qa_id="QA0002"), pred),
        (judge, replace(gold, dataset_id="reasoning_old100"), pred),
    ]:
        with pytest.raises(ValueError, match="No matching"):
            instance.decide(g, p)


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
            "generation": {"temperature": 0, "top_p": 1, "top_k": None, "seed": 42, "max_tokens": 32,
                           "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}}


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
    assert requests[0][0]["seed"] == 42 and requests[0][0]["max_tokens"] == 32
    assert requests[0][1] == 2


def test_api_missing_settings_are_not_guessed():
    config = api_config()
    del config["generation"]["seed"]
    with pytest.raises(ValueError, match="UNRESOLVED.*seed"):
        api_generator(config)


def test_sxz_prompt_provenance_hash():
    from evaluation.prompts import PROMPT_HASH, PROMPT_PATH
    provenance = json.loads((PROMPT_PATH.parent / "provenance.json").read_text())
    assert PROMPT_HASH == provenance["sha256"]
