# SPDX-License-Identifier: Apache-2.0
"""Collapse Qwen3.5/3.6 verify-width attention into fused kernel calls.

mlx-vlm's ``Qwen3_5Attention`` serves a target-verify forward (q_len =
1 + draft depth) on a batch-1 dense cache with a per-row fallback: L
single-row SDPA calls over per-row K/V slices, concatenated. That costs L
dispatches plus 2L slice ops and a concat per attention layer per verify
cycle, and it is the reason deeper MTP chains stop paying on this family.

Two fused paths claim that shape, in order:

1. The NAX split-K (flash decoding) dsplit kernel from the fa256 patch:
   at long KV the verify grid (NQ = 1 x H threadgroups) starves the GPU
   on a sequential scan, so the kernel splits the KV axis across
   threadgroups and folds the partials with logsumexp weights — this
   patch routes to it once the KV scan dominates (default 8K).
2. MLX's SDPA vector kernel, which serves causal blocks up to
   ``q_len * gqa <= 32`` (rows <= 5 on the dense 24/4 layout, <= 4 on the
   16/2 MoE one) with bottom-right alignment — exactly the per-row causal
   windows the fallback loop reproduces by hand. A verify block of L rows
   needs ceil(L / limit) dispatches: rows are chunked at the
   vector-kernel row limit, each chunk c covering rows [c0, c1) against
   ``keys[: kv_len - (L - c1)]`` with ``mask="causal"``. (Same
   construction as mlx-serve's ``splitCausalSdpa``, measured +4..9%
   decode there with speculation on.)

The seam is ``_target_verify_left_padded_attention``: it runs FIRST in the
target-verify branch and its non-None result skips the row loop, while a
None keeps every existing path unchanged. This patch wraps it to claim the
batch-1 / dense-cache / head_dim-256 shape and delegate everything else
(left-padded batches, quantized caches) to the original.
"""

from __future__ import annotations

import logging
import sys

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False
_ENGAGED_LOGGED: set[int] = set()
_NAX_ENGAGED_LOGGED: set[int] = set()

# MLX fast::ScaledDotProductAttention routes to the vector kernel only while
# q_len * gqa_factor <= 32; wider blocks fall to the composed unfused path.
_VECTOR_ROW_BUDGET = 32

# The NAX split-K kernel takes over once the KV scan dominates the verify
# block's fixed dispatch costs (partial slab + reduce). Below it the chunked
# vector kernel stays cheaper.
_NAX_MIN_KV_LEN = 8192


def _log_engaged(q_len: int) -> None:
    # One log per width; a single one-shot log would only witness the first
    # width and hide whether deeper chains route.
    if q_len not in _ENGAGED_LOGGED:
        _ENGAGED_LOGGED.add(q_len)
        logger.info(
            "[verify-split] hd-256 causal vector attention engaged (q_len=%d)",
            q_len,
        )


def _log_nax_engaged(q_len: int) -> None:
    if q_len not in _NAX_ENGAGED_LOGGED:
        _NAX_ENGAGED_LOGGED.add(q_len)
        logger.info(
            "[verify-split] hd-256 NAX split-K attention engaged (q_len=%d)",
            q_len,
        )


def _nax_sdpa(queries, keys, values, scale, kv_len: int):
    if kv_len < _NAX_MIN_KV_LEN:
        return None
    # Late import: the fa256 patch owns the kernel handle, its calibrated
    # dispatch budget, and the demote-on-failure discipline.
    try:
        from omlx.patches.qwen35_fa256_attention import try_nax_attn
    except Exception:
        return None
    out = try_nax_attn(
        queries,
        keys,
        values,
        scale,
        min_q_len=2,
        min_kv_len=_NAX_MIN_KV_LEN,
    )
    if out is not None:
        _log_nax_engaged(queries.shape[-2])
    return out


def _chunked_causal_sdpa(queries, keys, values, scale, limit: int):
    q_len = queries.shape[-2]
    kv_len = keys.shape[-2]
    outs = []
    c0 = 0
    while c0 < q_len:
        c1 = min(c0 + limit, q_len)
        kv_end = kv_len - (q_len - c1)
        outs.append(
            mx.fast.scaled_dot_product_attention(
                queries[..., c0:c1, :],
                keys[..., :kv_end, :],
                values[..., :kv_end, :],
                scale=scale,
                mask="causal",
            )
        )
        c0 = c1
    if len(outs) == 1:
        return outs[0]
    return mx.concatenate(outs, axis=-2)


def _eligible(queries, keys, cache) -> int:
    """Return the vector-kernel row limit (>0) when this call is ours."""
    # A quantized cache may hand back state proxies that expose .shape but
    # deliberately not .ndim (to avoid dequantizing). That's exactly the
    # "quantized caches" case this patch already means to delegate to the
    # original path below, but the attribute access below used to run before
    # the `hasattr(cache, "bits")` guard could rule it out, so it crashed
    # instead of falling through.
    if getattr(queries, "ndim", None) != 4 or getattr(keys, "ndim", None) != 4:
        return 0
    if queries.shape[0] != 1:
        return 0
    if cache is not None and hasattr(cache, "bits"):
        return 0
    if queries.dtype not in (mx.float16, mx.bfloat16):
        return 0
    if queries.dtype != keys.dtype:
        return 0
    if queries.shape[-1] != 256 or keys.shape[-1] != 256:
        return 0
    q_heads = queries.shape[-3]
    kv_heads = keys.shape[-3]
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        return 0
    limit = _VECTOR_ROW_BUDGET // (q_heads // kv_heads)
    if limit <= 0:
        return 0
    q_len = queries.shape[-2]
    if q_len <= 1 or q_len > keys.shape[-2]:
        return 0
    return limit


def apply_qwen35_verify_sdpa_split_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if not mx.metal.is_available():
        return False

    try:
        from mlx_vlm.models.qwen3_5 import language as q35_lang
    except ImportError:
        return False

    original = getattr(q35_lang, "_target_verify_left_padded_attention", None)
    if original is None:
        logger.debug("verify-split: target-verify seam not found; patch skipped")
        return False

    def patched_target_verify_attention(
        queries,
        keys,
        values,
        *,
        cache,
        scale,
        mask,
    ):
        # Only the batch-1 dense-cache shape is ours; a batch with real left
        # padding (or anything unexpected) keeps the original behavior.
        if mask is None or (isinstance(mask, str) and mask == "causal"):
            limit = _eligible(queries, keys, cache)
            lp = getattr(cache, "left_padding", None)
            if limit and lp is None:
                kv_len = keys.shape[-2]
                try:
                    out = _nax_sdpa(queries, keys, values, scale, kv_len)
                    if out is not None:
                        return out
                    out = _chunked_causal_sdpa(
                        queries, keys, values, scale, limit
                    )
                    _log_engaged(queries.shape[-2])
                    return out
                except Exception:
                    logger.warning(
                        "verify-split attention failed; falling back",
                        exc_info=True,
                    )
            elif not getattr(sys.modules[__name__], "_probe_logged", False):
                sys.modules[__name__]._probe_logged = True
                logger.info(
                    "[verify-split] declined: limit=%d mask=%r cache=%s "
                    "bits=%s left_padding=%r q_len=%s hd=%s",
                    limit,
                    type(mask).__name__,
                    type(cache).__name__,
                    hasattr(cache, "bits"),
                    lp,
                    getattr(queries, "shape", None),
                    getattr(keys, "shape", None),
                )
        return original(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )

    q35_lang._target_verify_left_padded_attention = patched_target_verify_attention
    _PATCHED = True
    logger.info("Qwen3.5/3.6 verify-width causal vector attention patch applied")
    return True
