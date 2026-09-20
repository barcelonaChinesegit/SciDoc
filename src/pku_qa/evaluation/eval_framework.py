#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pku_qa.evaluation.model_paths import (
    configure_model_cache_environment,
    model_directory,
    resolve_local_model_path,
)

HF_MIRROR_ENDPOINT = "https://hf-mirror.com"


DEFAULT_PROVIDER_SPECS: dict[str, dict[str, Any]] = {
    "local_qwen3_vl_4b": {
        "provider_type": "local_transformers",
        "model_class": "Qwen3VLForConditionalGeneration",
        "model_path": str(model_directory("Qwen3-VL-4B-Instruct")),
        "processor_path": str(model_directory("Qwen3-VL-4B-Instruct")),
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "local_files_only": True,
        "trust_remote_code": False,
        "enable_thinking": False,
    },
    "local_qwen3_vl_8b": {
        "provider_type": "local_transformers",
        "model_class": "Qwen3VLForConditionalGeneration",
        "model_path": str(model_directory("Qwen3-VL-8B-Instruct")),
        "processor_path": str(model_directory("Qwen3-VL-8B-Instruct")),
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "local_files_only": True,
        "trust_remote_code": False,
        "enable_thinking": False,
    },
    "local_qwen3_vl_4b_pdf": {
        "provider_type": "local_transformers",
        "model_backend": "modelscope",
        "generation_mode": "pdf_qwen_vl_utils",
        "model_class": "Qwen3VLForConditionalGeneration",
        "model_path": str(model_directory("Qwen3-VL-4B-Instruct")),
        "processor_path": str(model_directory("Qwen3-VL-4B-Instruct")),
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "local_files_only": True,
        "trust_remote_code": False,
        "enable_thinking": False,
    },
    "local_qwen3_vl_8b_pdf": {
        "provider_type": "local_transformers",
        "model_backend": "modelscope",
        "generation_mode": "pdf_qwen_vl_utils",
        "model_class": "Qwen3VLForConditionalGeneration",
        "model_path": str(model_directory("Qwen3-VL-8B-Instruct")),
        "processor_path": str(model_directory("Qwen3-VL-8B-Instruct")),
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "local_files_only": True,
        "trust_remote_code": False,
        "enable_thinking": False,
    },
    "local_qwen3_6_27b_judge": {
        "provider_type": "local_transformers",
        "model_class": "AutoModelForCausalLM",  # 修复：改为纯文本模型类
        "model_path": str(model_directory("Qwen3.6-27B")),
        "processor_path": str(model_directory("Qwen3.6-27B")),
        "dtype": "auto",
        "attn_implementation": None,
        "local_files_only": True,
        "trust_remote_code": True,
        # The judge must emit only its final verdict. Qwen's chat template
        # inserts an empty thinking block when this is false.
        "enable_thinking": False,
    },
    "openai_compatible_template": {
        "provider_type": "openai_compatible",
        "api_base": "https://example.com/v1",
        "model": "replace-with-model-name",
        "api_key_env": "OPENAI_API_KEY",
        "timeout": 300,
    },
}

MODEL_ALIASES = {
    "4B": "local_qwen3_vl_4b",
    "8B": "local_qwen3_vl_8b",
    "judge": "local_qwen3_6_27b_judge",
}

class ProviderError(RuntimeError):
    pass

class BaseChatProvider(ABC):
    def __init__(self, name: str, spec: dict[str, Any]):
        self.name = name
        self.spec = spec

    @abstractmethod
    def generate(self, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
        raise NotImplementedError

class LocalTransformersProvider(BaseChatProvider):
    _ALLOWED_GENERATION_OVERRIDES = {
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "no_repeat_ngram_size",
        "repetition_penalty",
    }

    def __init__(self, name: str, spec: dict[str, Any]):
        super().__init__(name, spec)
        self.model = None
        self.processor = None
        self._generation_overrides: dict[str, Any] = {}
        self._vision_cache_key: str | None = None
        self._vision_cache_signature: tuple[Any, ...] | None = None
        self._vision_cache_output: Any = None
        self._load()

    def _load(self) -> None:
        configure_hf_environment()
        ensure_torch_float8_patch()

        import torch
        # 修复：增加 AutoTokenizer 和 AutoModelForCausalLM
        from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, Qwen3VLForConditionalGeneration

        if self.spec.get("model_backend") == "modelscope":
            from modelscope import (
                AutoProcessor as ModelScopeAutoProcessor,
                Qwen3VLForConditionalGeneration as ModelScopeQwen3VL,
            )
        else:
            ModelScopeAutoProcessor = None
            ModelScopeQwen3VL = None

        model_class_name = self.spec.get("model_class")
        model_class_map = {
            "Qwen3VLForConditionalGeneration": (
                ModelScopeQwen3VL
                if ModelScopeQwen3VL is not None
                else Qwen3VLForConditionalGeneration
            ),
            "AutoModelForCausalLM": AutoModelForCausalLM,
        }
        if model_class_name not in model_class_map:
            raise ProviderError(f"Unsupported local model class: {model_class_name}")

        model_class = model_class_map[model_class_name]
        dtype_value = self.spec.get("dtype", "auto")
        if dtype_value == "auto":
            torch_dtype = "auto"
            qwen_dtype = None
        else:
            torch_dtype = getattr(torch, dtype_value)
            qwen_dtype = torch_dtype

        load_kwargs = {
            "device_map": self.spec.get("device_map", "auto"),
            "trust_remote_code": self.spec.get("trust_remote_code", False),
            "local_files_only": self.spec.get("local_files_only", True),
        }
        attn_implementation = self.spec.get("attn_implementation")
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        model_path = str(
            resolve_local_model_path(
                self.spec["model_path"], require_config=True
            )
        )
        if model_class_name == "Qwen3VLForConditionalGeneration":
            if qwen_dtype is not None:
                load_kwargs["dtype"] = qwen_dtype
        else:
            load_kwargs["torch_dtype"] = torch_dtype

        self.model = model_class.from_pretrained(model_path, **load_kwargs)
        device_map = getattr(self.model, "hf_device_map", {}) or {}
        offloaded_modules = [
            module_name
            for module_name, device in device_map.items()
            if str(device).lower() in {"cpu", "disk"}
        ]
        if offloaded_modules and not self.spec.get("allow_cpu_offload", False):
            preview = ", ".join(offloaded_modules[:5])
            raise ProviderError(
                "CPU/disk model offload is disabled for GPU evaluation; "
                f"offloaded module(s): {preview}"
            )
        
        # 修复：文本模型使用 AutoTokenizer，视觉模型使用 AutoProcessor
        if model_class_name == "AutoModelForCausalLM":
            processor_path = str(
                resolve_local_model_path(
                    self.spec.get("processor_path", model_path)
                )
            )
            self.processor = AutoTokenizer.from_pretrained(
                processor_path,
                trust_remote_code=self.spec.get("trust_remote_code", False),
                local_files_only=self.spec.get("local_files_only", True),
            )
        else:
            processor_class = (
                ModelScopeAutoProcessor
                if ModelScopeAutoProcessor is not None
                else AutoProcessor
            )
            processor_path = str(
                resolve_local_model_path(
                    self.spec.get("processor_path", model_path)
                )
            )
            self.processor = processor_class.from_pretrained(
                processor_path,
                trust_remote_code=self.spec.get("trust_remote_code", False),
                local_files_only=self.spec.get("local_files_only", True),
            )
            self._install_vision_feature_cache()

    def _install_vision_feature_cache(self) -> None:
        """Cache Qwen3-VL image features while one PDF is reused for many QA."""
        vision_model = getattr(self.model, "model", None)
        original_get_image_features = getattr(
            vision_model, "get_image_features", None
        )
        if original_get_image_features is None:
            return

        def get_image_features_cached(
            pixel_values: Any,
            image_grid_thw: Any,
            return_dict: bool = True,
        ) -> Any:
            cache_key = self._vision_cache_key
            if cache_key is None:
                return original_get_image_features(
                    pixel_values,
                    image_grid_thw,
                    return_dict=return_dict,
                )

            grid_signature = tuple(
                tuple(int(value) for value in row)
                for row in image_grid_thw.detach().cpu().tolist()
            )
            signature = (
                cache_key,
                tuple(pixel_values.shape),
                str(pixel_values.dtype),
                grid_signature,
                bool(return_dict),
            )
            if (
                signature == self._vision_cache_signature
                and self._vision_cache_output is not None
            ):
                return self._vision_cache_output

            output = original_get_image_features(
                pixel_values,
                image_grid_thw,
                return_dict=return_dict,
            )
            self._vision_cache_signature = signature
            self._vision_cache_output = output
            return output

        vision_model.get_image_features = get_image_features_cached

    def set_vision_cache_key(self, cache_key: str | None) -> None:
        normalized = None if cache_key is None else str(cache_key)
        if normalized != self._vision_cache_key:
            self._vision_cache_key = normalized
            self._vision_cache_signature = None
            self._vision_cache_output = None

    def clear_vision_cache(self) -> None:
        self.set_vision_cache_key(None)

    def set_generation_overrides(
        self, overrides: dict[str, Any] | None
    ) -> None:
        normalized = dict(overrides or {})
        unsupported = set(normalized) - self._ALLOWED_GENERATION_OVERRIDES
        if unsupported:
            raise ValueError(
                "Unsupported generation override(s): "
                + ", ".join(sorted(unsupported))
            )
        self._generation_overrides = normalized

    def generate(self, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
        import torch

        is_causal_lm = self.spec.get("model_class") == "AutoModelForCausalLM"
        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
        }
        if "enable_thinking" in self.spec:
            template_kwargs["enable_thinking"] = bool(
                self.spec["enable_thinking"]
            )

        if self.spec.get("generation_mode") == "pdf_qwen_vl_utils":
            from qwen_vl_utils import process_vision_info

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                **template_kwargs,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
        elif is_causal_lm:
            # 文本模型不支持输入图像，我们需要将图像剔除，只保留文字传给裁判
            text_messages = []
            for m in messages:
                if isinstance(m["content"], str):
                    text_messages.append(m)
                elif isinstance(m["content"], list):
                    texts = [c["text"] for c in m["content"] if c["type"] == "text"]
                    text_messages.append({"role": m["role"], "content": "\n".join(texts)})
            
            causal_template_kwargs = {
                "tokenize": False,
                **template_kwargs,
            }

            text = self.processor.apply_chat_template(
                text_messages,
                **causal_template_kwargs,
            )
            inputs = self.processor([text], return_tensors="pt").to(self.model.device)
        else:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                **template_kwargs,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            }
            generate_kwargs.update(self._generation_overrides)
            if is_causal_lm:
                generate_kwargs["pad_token_id"] = self.processor.eos_token_id
            generated_ids = self.model.generate(**inputs, **generate_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=(
                self.spec.get("generation_mode") == "pdf_qwen_vl_utils"
            ),
        )
        return output_text[0].strip() if output_text else ""


class OpenAICompatibleProvider(BaseChatProvider):
    def __init__(self, name: str, spec: dict[str, Any]):
        super().__init__(name, spec)
        self.api_base = spec["api_base"].rstrip("/")
        self.model = spec["model"]
        self.api_key_env = spec.get("api_key_env", "OPENAI_API_KEY")
        self.timeout = int(spec.get("timeout", 300))

    def _encode_image_as_data_url(self, image: Any) -> str:
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://") or image.startswith("data:"):
                return image
            path = Path(image)
            if not path.exists():
                raise ProviderError(f"Image path does not exist: {image}")
            binary = path.read_bytes()
        else:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            binary = buffer.getvalue()

        encoded = base64.b64encode(binary).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def _serialize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, str):
                serialized.append({"role": message["role"], "content": content})
                continue

            content_blocks: list[dict[str, Any]] = []
            for item in content:
                item_type = item.get("type")
                if item_type == "text":
                    content_blocks.append({"type": "text", "text": item.get("text", "")})
                elif item_type == "image":
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": self._encode_image_as_data_url(item.get("image"))},
                        }
                    )
                else:
                    raise ProviderError(f"Unsupported content block for API provider: {item_type}")

            serialized.append({"role": message["role"], "content": content_blocks})
        return serialized

    def generate(self, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ProviderError(f"Missing API key environment variable: {self.api_key_env}")

        payload = {
            "model": self.model,
            "messages": self._serialize_messages(messages),
            "max_tokens": max_new_tokens,
        }
        request = urllib.request.Request(
            url=f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"API request failed: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"API request failed: {exc}") from exc

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Unexpected API response schema: {body}") from exc

        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                    chunks.append(item.get("text", ""))
            return "".join(chunks).strip()
        return str(content).strip()


def configure_hf_environment(use_mirror: bool = True, cuda_visible_devices: str | None = None) -> None:
    configure_model_cache_environment()
    if use_mirror:
        os.environ.setdefault("HF_ENDPOINT", HF_MIRROR_ENDPOINT)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)


def ensure_torch_float8_patch() -> None:
    import torch

    if not hasattr(torch, "float8_e8m0fnu"):
        setattr(torch, "float8_e8m0fnu", torch.float32)


def load_provider_specs(config_path: str | None = None) -> dict[str, dict[str, Any]]:
    specs = json.loads(json.dumps(DEFAULT_PROVIDER_SPECS))
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Provider config not found: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            user_specs = json.load(f)

        if not isinstance(user_specs, dict):
            raise ValueError("Provider config must be a JSON object keyed by provider name")

        for name, spec in user_specs.items():
            if not isinstance(spec, dict):
                raise ValueError(f"Provider spec for {name} must be a JSON object")
            specs[name] = spec

    for spec in specs.values():
        if spec.get("provider_type") != "local_transformers":
            continue
        for field in ("model_path", "processor_path"):
            if field in spec:
                spec[field] = str(
                    resolve_local_model_path(spec[field], require_exists=False)
                )
    return specs


def resolve_provider_name(model: str | None = None, provider_name: str | None = None) -> str:
    if provider_name:
        return provider_name
    if model and model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    raise ValueError("You must provide either --provider-name or a known --model alias")


def create_provider(provider_name: str, config_path: str | None = None) -> BaseChatProvider:
    specs = load_provider_specs(config_path)
    if provider_name not in specs:
        known = ", ".join(sorted(specs))
        raise KeyError(f"Unknown provider: {provider_name}. Known providers: {known}")

    spec = specs[provider_name]
    provider_type = spec.get("provider_type")
    if provider_type == "local_transformers":
        return LocalTransformersProvider(provider_name, spec)
    if provider_type == "openai_compatible":
        return OpenAICompatibleProvider(provider_name, spec)
    raise ProviderError(f"Unsupported provider type: {provider_type}")


def load_json(path: str | Path, default: Any | None = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, target)


def append_progress_line(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(f"{text}\n")


def sanitize_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe.strip("._") or "run"


def query_gpus() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "nvidia-smi failed")

    gpus: list[dict[str, Any]] = []
    for raw_line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 5:
            continue
        index, name, memory_total, memory_used, utilization = parts
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(memory_total),
                "memory_used_mib": int(memory_used),
                "utilization_gpu_percent": int(utilization),
            }
        )
    return gpus


def print_gpu_overview() -> None:
    try:
        gpus = query_gpus()
    except Exception as exc:
        print(f"[GPU] Failed to query nvidia-smi: {exc}")
        return

    print("[GPU] Current GPU status:")
    for gpu in gpus:
        print(
            f"  GPU {gpu['index']}: {gpu['name']} | "
            f"used {gpu['memory_used_mib']} / {gpu['memory_total_mib']} MiB | "
            f"util {gpu['utilization_gpu_percent']}%"
        )


def iter_paper_items(data: dict[str, Any]):
    for paper_id, paper_data in data.items():
        if str(paper_id).startswith("__"):
            continue
        if not isinstance(paper_data, dict):
            continue
        yield paper_id, paper_data


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\u00a0", " ").replace("\u200b", " ")
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_loose_text(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[\[\]{}()<>]", " ", text)
    text = re.sub(r"[,:;!?\"'`]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .")


_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[-+]?\d+)?%?")

# Conservative whole-answer numeric matcher.  A direct numeric verdict is only
# safe when the complete answer is one scalar with an optional short unit.
# Ratios, ranges, formulas, multiple values, comparisons, and explanations must
# fall through to the semantic judge.
_SIMPLE_NUMERIC_ANSWER_PATTERN = re.compile(
    r"""
    ^
    (?P<number>
        [-+]?
        (?:
            (?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?
            |
            \.\d+
        )
        (?:[eE][-+]?\d+)?
    )
    \s*
    (?P<unit>
        %
        |
        [A-Za-zµμ°][A-Za-zµμ°%\s/·*._-]*
    )?
    $
    """,
    re.VERBOSE,
)
_MAX_SIMPLE_NUMERIC_ANSWER_CHARS = 64
_MAX_SIMPLE_NUMERIC_UNIT_WORDS = 3


_SMALL_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_SCALE_NUMBER_WORDS = {
    "hundred": 100,
    "thousand": 1000,
    "million": 1000000,
    "billion": 1000000000,
}

_ORDINAL_NUMBER_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}

_FRACTION_PATTERN = re.compile(r"(?:\\frac\{\s*(-?\d+)\s*\}\{\s*(-?\d+)\s*\}|(-?\d+)\s*/\s*(-?\d+))")


def _parse_number_words_segment(segment: str) -> float | None:
    tokens = [token for token in re.split(r"[\s-]+", segment.lower()) if token and token != "and"]
    if not tokens:
        return None

    total = 0
    current = 0
    seen = False
    for token in tokens:
        if token in _SMALL_NUMBER_WORDS:
            current += _SMALL_NUMBER_WORDS[token]
            seen = True
        elif token in _ORDINAL_NUMBER_WORDS and len(tokens) == 1:
            current += _ORDINAL_NUMBER_WORDS[token]
            seen = True
        elif token == "hundred":
            current = (current or 1) * 100
            seen = True
        elif token in {"thousand", "million", "billion"}:
            scale = _SCALE_NUMBER_WORDS[token]
            current = (current or 1) * scale
            total += current
            current = 0
            seen = True
        else:
            return None

    if not seen:
        return None
    return float(total + current)


def extract_numbers(text: str) -> list[float]:
    raw_text = str(text or "")
    numbers: list[float] = []
    seen: set[float] = set()

    def add_number(value: float) -> None:
        rounded = round(float(value), 12)
        if rounded not in seen:
            seen.add(rounded)
            numbers.append(float(value))

    for match in _FRACTION_PATTERN.finditer(raw_text):
        numerator = match.group(1) or match.group(3)
        denominator = match.group(2) or match.group(4)
        try:
            denominator_value = float(denominator)
            if denominator_value != 0:
                add_number(float(numerator) / denominator_value)
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    for token in _NUMBER_PATTERN.findall(raw_text):
        is_percent = token.endswith("%")
        token = token.rstrip("%").replace(",", "")
        try:
            value = float(token)
        except ValueError:
            continue
        add_number(value)
        if is_percent:
            add_number(value / 100.0)

    normalized_words = raw_text.lower().replace("-", " ")
    normalized_words = re.sub(r"[^a-z\s]", " ", normalized_words)
    word_tokens = [token for token in normalized_words.split() if token]
    for start in range(len(word_tokens)):
        for end in range(min(len(word_tokens), start + 6), start, -1):
            segment = " ".join(word_tokens[start:end])
            value = _parse_number_words_segment(segment)
            if value is not None:
                add_number(value)
                break

    return numbers


def parse_simple_numeric_answer(text: str) -> tuple[float, str] | None:
    """Parse a complete short answer containing one scalar and optional unit.

    This intentionally rejects anything that needs semantic interpretation:
    ratios (``3:2``), ranges, inequalities, formulas, multiple values, units
    with ASCII numeric exponents, and prose explanations.
    """

    candidate = str(text or "").strip()
    if not candidate or len(candidate) > _MAX_SIMPLE_NUMERIC_ANSWER_CHARS:
        return None

    match = _SIMPLE_NUMERIC_ANSWER_PATTERN.fullmatch(candidate)
    if not match:
        return None

    number_token = match.group("number").replace(",", "")
    try:
        value = float(number_token)
    except ValueError:
        return None

    unit = str(match.group("unit") or "").strip()
    if unit and len(unit.split()) > _MAX_SIMPLE_NUMERIC_UNIT_WORDS:
        return None

    normalized_unit = re.sub(r"\s+", "", unit).lower()
    normalized_unit = normalized_unit.replace("μ", "µ")
    normalized_unit = normalized_unit.replace("·", "*")
    return value, normalized_unit


def is_numeric_match(
    reference: str,
    prediction: str,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-6,
) -> bool:
    """Return true only for compatible, whole-answer scalar quantities."""

    ref_quantity = parse_simple_numeric_answer(reference)
    pred_quantity = parse_simple_numeric_answer(prediction)
    if ref_quantity is None or pred_quantity is None:
        return False

    ref_value, ref_unit = ref_quantity
    pred_value, pred_unit = pred_quantity
    if ref_unit != pred_unit:
        return False

    tolerance = max(abs_tol, abs(ref_value) * rel_tol)
    return abs(ref_value - pred_value) <= tolerance


def looks_like_error_output(text: str) -> bool:
    normalized = normalize_text(text)
    return normalized.startswith("[error") or normalized.startswith("error:")


def safe_divide(numerator: int, denominator: int) -> float:
    return (numerator / denominator * 100.0) if denominator else 0.0
