"""Canonical paths for local model assets."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = (PROJECT_ROOT / "models").resolve()
MODEL_CACHE_ROOT = MODEL_ROOT / ".cache"


def configure_model_cache_environment() -> None:
    """Keep model download and dynamic-module caches under ``models/``."""
    cache_paths = {
        "HF_HOME": MODEL_CACHE_ROOT / "huggingface",
        "HF_HUB_CACHE": MODEL_CACHE_ROOT / "huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": MODEL_CACHE_ROOT / "huggingface/hub",
        "TRANSFORMERS_CACHE": MODEL_CACHE_ROOT / "huggingface/hub",
        "HF_MODULES_CACHE": MODEL_CACHE_ROOT / "huggingface/modules",
        "MODELSCOPE_CACHE": MODEL_CACHE_ROOT / "modelscope/hub",
        "TORCH_HOME": MODEL_CACHE_ROOT / "torch",
    }
    for name, path in cache_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)


def model_directory(name: str) -> Path:
    """Return the canonical directory for one top-level local model."""
    if not name or Path(name).name != name:
        raise ValueError(f"Model name must be one directory name: {name!r}")
    return MODEL_ROOT / name


def resolve_local_model_path(
    value: str | Path,
    *,
    require_exists: bool = True,
    require_config: bool = False,
) -> Path:
    """Resolve a local asset path and require it to remain under ``models/``."""
    raw_path = Path(value).expanduser()
    candidate = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
    lexical_path = Path(os.path.abspath(candidate))
    try:
        lexical_path.relative_to(MODEL_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Local model assets must resolve inside {MODEL_ROOT}; local paths "
            f"must be addressed inside that directory directly: {value}"
        ) from exc

    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(MODEL_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Local model assets must resolve inside {MODEL_ROOT}: {value}"
        ) from exc

    if require_exists and not resolved.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {resolved}")
    if require_config and not (resolved / "config.json").is_file():
        raise FileNotFoundError(
            f"Local model config not found: {resolved / 'config.json'}"
        )
    return resolved
