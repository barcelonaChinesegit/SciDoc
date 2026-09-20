from __future__ import annotations

import json
from pathlib import Path

import pytest

from pku_qa.evaluation import model_paths
from pku_qa.evaluation.eval_framework import (
    DEFAULT_PROVIDER_SPECS,
    load_provider_specs,
)
from pku_qa.evaluation.model_paths import (
    MODEL_ROOT,
    configure_model_cache_environment,
    resolve_local_model_path,
)


def test_default_local_providers_use_canonical_model_root() -> None:
    for spec in DEFAULT_PROVIDER_SPECS.values():
        if spec.get("provider_type") != "local_transformers":
            continue
        for field in ("model_path", "processor_path"):
            path = Path(spec[field])
            assert path.is_absolute()
            path.relative_to(MODEL_ROOT)


def test_relative_local_path_is_resolved_from_project_root() -> None:
    expected = MODEL_ROOT / "Qwen3-VL-4B-Instruct"
    assert resolve_local_model_path(
        "models/Qwen3-VL-4B-Instruct", require_exists=False
    ) == expected


def test_model_load_validation_requires_transformers_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "project"
    model_root = project_root / "models"
    incomplete_model = model_root / "incomplete"
    incomplete_model.mkdir(parents=True)
    monkeypatch.setattr(model_paths, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(model_paths, "MODEL_ROOT", model_root.resolve())

    with pytest.raises(FileNotFoundError, match="config.json"):
        resolve_local_model_path(incomplete_model, require_config=True)


def test_model_caches_are_forced_under_model_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "models"
    cache_root = model_root / ".cache"
    monkeypatch.setattr(model_paths, "MODEL_ROOT", model_root)
    monkeypatch.setattr(model_paths, "MODEL_CACHE_ROOT", cache_root)
    monkeypatch.setenv("HF_HOME", "/tmp/outside-model-root")

    configure_model_cache_environment()

    assert Path(model_paths.os.environ["HF_HOME"]) == cache_root / "huggingface"
    assert Path(model_paths.os.environ["HF_HUB_CACHE"]) == cache_root / "huggingface/hub"
    assert Path(model_paths.os.environ["MODELSCOPE_CACHE"]) == cache_root / "modelscope/hub"
    assert Path(model_paths.os.environ["TORCH_HOME"]) == cache_root / "torch"


def test_local_provider_config_rejects_path_outside_model_root(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "providers.json"
    config_path.write_text(
        json.dumps(
            {
                "outside": {
                    "provider_type": "local_transformers",
                    "model_path": str(tmp_path / "weights"),
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must resolve inside"):
        load_provider_specs(str(config_path))


def test_root_compatibility_link_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "project"
    model_root = project_root / "models"
    canonical_path = model_root / "example-model"
    canonical_path.mkdir(parents=True)
    (project_root / "example-model").symlink_to("models/example-model")
    monkeypatch.setattr(model_paths, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(model_paths, "MODEL_ROOT", model_root.resolve())

    with pytest.raises(ValueError, match="must resolve inside"):
        resolve_local_model_path(project_root / "example-model")
