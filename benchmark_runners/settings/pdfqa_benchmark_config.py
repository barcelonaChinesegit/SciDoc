from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DATA_ROOT = WORKSPACE_ROOT / "data"
PDF_ROOT = DATA_ROOT / "pdfs"
MODEL_ROOT = WORKSPACE_ROOT / "models"
from typing import Any, Dict

SXZ_ROOT = REPO_ROOT
PDF_DPI = 144

DATASETS = {'ordinary': {'task_family': 'ordinary', 'json_path': str(DATA_ROOT / "qa" / "7.final_2200" / "ordinary_qa.json"), 'pdf_root': str(PDF_ROOT), 'pdf_kind': 'single_pdf', 'prompt_key': 'ordinary', 'expected_qa_count': None, 'document_manifest_mode': 'none'}, 'unanswerable': {'task_family': 'ordinary', 'json_path': str(DATA_ROOT / "qa" / "7.final_2200" / "unanswerable_qa.json"), 'pdf_root': str(PDF_ROOT), 'pdf_kind': 'single_pdf', 'prompt_key': 'ordinary', 'expected_qa_count': None, 'document_manifest_mode': 'none'}, 'reasoning': {'task_family': 'reasoning', 'json_path': str(DATA_ROOT / "qa" / "7.final_2200" / "reasoning_qa.json"), 'pdf_root': str(PDF_ROOT), 'pdf_kind': 'single_pdf', 'prompt_key': 'reasoning', 'expected_qa_count': None, 'document_manifest_mode': 'none'}, 'cross_pdf': {'task_family': 'cross_document', 'json_path': str(DATA_ROOT / "qa" / "7.final_2200" / "cross_pdf_qa.json"), 'pdf_root': str(PDF_ROOT), 'pdf_kind': 'merged_pdf', 'prompt_key': 'cross_document', 'expected_qa_count': None, 'document_manifest_mode': 'none'}}

MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "Qwen3-VL-4B": {
        "model_path": str(MODEL_ROOT / "Qwen3-VL-4B-Instruct"),
        "max_new_tokens": 1024,
        "dtype": "auto",
        "device_map": "auto",
        "processor_pixel_policy": "native_default",
    },
    "Qwen3-VL-8B": {
        "model_path": str(MODEL_ROOT / "Qwen3-VL-8B-Instruct"),
        "max_new_tokens": 1024,
        "dtype": "auto",
        "device_map": "auto",
        "processor_pixel_policy": "native_default",
    },
    "Claude-Sonnet-5": {
        "model_id": "claude-sonnet-5",
        "inference_backend": "anthropic_compatible_messages_api",
        "base_url": "https://api.aicodemirror.ai/api/claudecode",
        "pdf_dpi": 144,
        "whole_pdf": True,
        "page_dropping": False,
        "context_truncation": False,
        "ordinary_max_output_tokens": 1536,
        "unanswerable_max_output_tokens": 1536,
        "reasoning_max_output_tokens": 2048,
        "cross_pdf_max_output_tokens": 4096,
        "request_timeout": 1800.0,
        "thinking_mode": "provider/model_default",
        "format_repair": False,
        "semantic_repair": False,
        "inference_contract": "claude-sonnet5-wholepdf-dynamiccompress-no-repair",
    },

    "Mistral-Small-3.1-24B-Instruct-2503": {
        "model_path": str(MODEL_ROOT / "Mistral-Small-3.1-24B-Instruct-2503"),
        "inference_backend": "vllm_openai_local_tp2",
        "context_limit": 128000,
        "max_new_tokens": 512,
        "request_timeout": 1800.0,
        "gpu_memory_utilization": 0.90,
        "max_images_per_prompt": 512,
        "pdf_dpi": 144,
        "render_format": "JPEG",
        "jpeg_quality": 95,
        "jpeg_subsampling": 0,
        "temperature": 0.0,
        "tensor_parallel_size": 2,
        "context_fit_enabled": True,
        "context_fit_target_prompt_tokens": 118000,
        "context_fit_min_scale": 0.35,
        "context_fit_max_attempts": 6,
        "page_dropping": False,
        "context_truncation": False,
        "processor_image_policy": "mistral_pixtral_native",
        "inference_contract": "mistral31-v16-tp2-contextfit-wholepdf-no-repair",
    },

    "Gemma-3-27B-IT": {
        "model_path": str(MODEL_ROOT / "Gemma-3-27B-IT"),
        "max_new_tokens": 512,
        "context_limit": 131072,
        "attn_implementation": "sdpa",
        "torch_dtype": "bfloat16",
        "pdf_dpi": 144,
        "render_format": "JPEG",
        "jpeg_quality": 95,
        "jpeg_subsampling": 0,
        "do_sample": False,
        "use_cache": True,
        "trust_remote_code": False,
        "processor_image_policy": "gemma3_native",
        "inference_contract": "gemma3-v2-wholepdf-jpeg95-no-repair",
    },

    "MiniCPM-V-2_6": {
        "model_path": str(MODEL_ROOT / "MiniCPM-V-2_6"),
        "max_new_tokens": 512,
        "max_input_length": 32768,
        "attn_implementation": "sdpa",
        "torch_dtype": "bfloat16",
        "pdf_dpi": 144,
        "page_width": 1008,
        "page_height": 1344,
        "page_label_height": 64,
        "max_slice_nums": 1,
        "use_image_id": False,
        "sampling": False,
        "inference_contract": "minicpm-v2.6-referencefaithful-v3",
    },
    "MiniCPM-V-4_5": {
        "model_path": str(MODEL_ROOT / "MiniCPM-V-4_5"),
        "max_new_tokens": 512,
        "max_input_length": 32768,
        "preferred_input_tokens": 30000,
        "max_slice_nums": 9,
        "expected_llm_context": 40960,
        "attn_implementation": "sdpa",
        "torch_dtype": "bfloat16",
        "slice_policy": "contextfit-v4.1",
        "use_image_id": False,
        "enable_thinking": False,
        "stream": False,
        "sampling": False,
    },
    "InternVL2_5-8B": {
        "model_path": str(MODEL_ROOT / "InternVL2_5-8B"),
        "max_new_tokens": 512,
        "input_size": 448,
        "tile_budget": 96,
        "context_safety_margin": 512,
        "device_map": "single_cuda0",
        "use_thumbnail": False,
        "use_flash_attn": False,
        "torch_dtype": "bfloat16",
    },
    "InternVL3_5-8B": {
        "model_path": str(MODEL_ROOT / "InternVL3_5-8B"),
        "max_new_tokens": 1024,
        "input_size": 448,
        "tile_budget": 96,
        "context_safety_margin": 512,
        "device_map": "balanced",
        "use_thumbnail": False,
        "use_flash_attn": False,
        "torch_dtype": "bfloat16",
    },
}


def get_dataset_spec(dataset_id: str) -> Dict[str, Any]:
    if dataset_id not in DATASETS:
        raise KeyError(f"Unknown dataset_id={dataset_id!r}. Available: {', '.join(DATASETS)}")
    return DATASETS[dataset_id]
