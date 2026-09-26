"""Inference-only SDPA adapter for the legacy InternVL2.5 InternLM2 tuple cache.

No on-disk model files, weights, rotary embedding rules, or global torch functions
are modified. CUDA math fallback is deliberately disabled.
"""
import contextlib
import hashlib
import inspect
import sys
import types

import torch
import torch.nn.functional as F


def kernel_context(device):
    if device.type != "cuda":
        return contextlib.nullcontext()  # Only used by CPU correctness tests.
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])


def compact_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    """Avoid the original O(sequence_length**2) mask allocation altogether."""
    if attention_mask is None:
        return None
    if attention_mask.ndim != 2:
        raise ValueError("Expected the legacy model's 2-D padding mask")
    expected = (input_shape[0], input_shape[1] + past_key_values_length)
    if tuple(attention_mask.shape) != expected:
        raise ValueError(f"Padding mask {attention_mask.shape} != {expected}")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)).item():
        raise ValueError("Padding mask must contain only 0/1")
    if attention_mask.bool().all().item():
        return None
    # The production runner always uses batch_size=1 without padding. Reject
    # padded CUDA batches instead of allocating another enormous dense mask.
    if inputs_embeds.is_cuda:
        raise ValueError("This full-PDF runner supports unpadded CUDA inputs only")
    return attention_mask.bool()


def sdpa_forward(self, hidden_states, attention_mask=None, position_ids=None,
                 past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
    if self.training:
        raise RuntimeError("This adapter is inference-only; call model.eval()")
    if output_attentions:
        raise ValueError("output_attentions=True is forbidden: it recreates quadratic attention weights")
    if kwargs.get("padding_mask") is not None:
        raise ValueError("Use attention_mask, not deprecated padding_mask")
    b, q_len, _ = hidden_states.shape
    groups, d = self.num_key_value_groups, self.head_dim
    # InternLM2 interleaves Q groups, K and V per KV head. A plain Q/K/V split
    # is WRONG even when shapes appear to fit.
    qkv = self.wqkv(hidden_states).view(b, q_len, self.num_key_value_heads, groups + 2, d)
    q = qkv[..., :groups, :].reshape(b, q_len, self.num_heads, d).transpose(1, 2)
    k = qkv[..., -2, :].transpose(1, 2)
    v = qkv[..., -1, :].transpose(1, 2)
    past_len = 0 if past_key_value is None else past_key_value[0].shape[-2]
    kv_len = q_len + past_len
    if position_ids is None:
        position_ids = torch.arange(past_len, kv_len, device=q.device).unsqueeze(0)
    cos, sin = self.rotary_emb(v, seq_len=kv_len)
    q, k = self._sdpa_rotary(q, k, cos, sin, position_ids)
    if past_key_value is not None:
        k = torch.cat((past_key_value[0], k), dim=2)
        v = torch.cat((past_key_value[1], v), dim=2)
    cache = (k, v) if use_cache else None  # Cache original KV heads, NOT repeated heads.
    k = self._sdpa_repeat(k, groups)
    v = self._sdpa_repeat(v, groups)
    # Explicit contiguous layout works on both fused backends in torch 2.8.
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    mask = None
    causal = q_len == kv_len and q_len > 1
    if attention_mask is not None:
        if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (b, kv_len):
            raise ValueError("Expected a compact [batch, KV length] padding mask")
        allowed = torch.arange(kv_len, device=q.device)[None, :] <= (
            past_len + torch.arange(q_len, device=q.device)[:, None])
        mask = attention_mask[:, None, None, :].bool() & allowed[None, None, :, :]
        causal = False
    elif past_len and q_len > 1:
        # Cached chunks need LOWER-RIGHT alignment. is_causal=True uses the
        # upper-left alignment and silently gives incorrect cached decoding.
        if q.is_cuda:
            from torch.nn.attention.bias import causal_lower_right
            mask = causal_lower_right(q_len, kv_len)
        else:
            mask = torch.arange(kv_len, device=q.device)[None, :] <= (
                past_len + torch.arange(q_len, device=q.device)[:, None])
        causal = False
    # q_len=1 with cache: no causal mask; attend to ALL past tokens + this token.
    with kernel_context(q.device):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                             dropout_p=0.0, is_causal=causal)
    self._sdpa_calls += 1
    self._sdpa_max_q = max(self._sdpa_max_q, q_len)
    out = out.transpose(1, 2).contiguous().reshape(b, q_len, self.hidden_size)
    return self.wo(out), None, cache


def install_sdpa(model):
    """Patch all legacy decoder layers or refuse an incompatible implementation."""
    lm = model.language_model if hasattr(model, "language_model") else model
    decoder = lm.model
    if getattr(decoder, "_pdfqa_sdpa_patched", False):
        raise RuntimeError("SDPA adapter already installed")
    if type(decoder).__name__ != "InternLM2Model":
        raise TypeError(f"Unsupported decoder: {type(decoder)}")
    src = inspect.getsource(type(decoder).forward)
    if "_prepare_decoder_attention_mask" not in src or "past_key_values[0][0]" not in src:
        raise RuntimeError("Remote-code decoder/cache API differs from the tested legacy version")
    layers = list(decoder.layers)
    if len(layers) != decoder.config.num_hidden_layers:
        raise RuntimeError("Unexpected layer count")
    for layer in layers:
        attn = layer.attention
        if type(attn).__name__ != "InternLM2Attention":
            raise TypeError(f"Load with use_flash_attn=False; got {type(attn)}")
        for key in ("wqkv", "wo", "rotary_emb", "num_heads", "num_key_value_heads", "num_key_value_groups"):
            if not hasattr(attn, key):
                raise RuntimeError(f"Incompatible attention: missing {key}")
        if attn.wqkv.out_features != (attn.num_heads + 2 * attn.num_key_value_heads) * attn.head_dim:
            raise RuntimeError("Incompatible QKV projection shape")
    module = sys.modules[type(layers[0].attention).__module__]
    for layer in layers:
        attn = layer.attention
        attn._sdpa_rotary = module.apply_rotary_pos_emb
        attn._sdpa_repeat = module.repeat_kv
        attn._sdpa_calls = attn._sdpa_max_q = 0
        attn.forward = types.MethodType(sdpa_forward, attn)
    decoder._prepare_decoder_attention_mask = types.MethodType(compact_mask, decoder)
    # This is an in-process label AFTER instantiation, not a nonexistent entry
    # passed to the original attention class registry at model construction.
    decoder.config.attn_implementation = "sdpa"
    decoder.config.output_attentions = False
    decoder._pdfqa_sdpa_patched = True
    source_path = inspect.getsourcefile(type(layers[0].attention))
    with open(source_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    report = {"patched_layers": len(layers), "source": source_path, "source_sha256": sha,
              "cuda_backends": ["FLASH_ATTENTION", "EFFICIENT_ATTENTION"], "math_fallback": False}
    print("[SDPA PATCHED]", report, flush=True)
    return report


def stats(model):
    lm = model.language_model if hasattr(model, "language_model") else model
    return [{"calls": x.attention._sdpa_calls, "max_q": x.attention._sdpa_max_q}
            for x in lm.model.layers]
