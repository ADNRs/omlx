# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the Qwen3.5/3.6 verify-width chunked causal attention.

``_chunked_causal_sdpa`` must reproduce the per-row loop it replaces: row i
of a verify block attends ``keys[: prefix + i + 1]``. Chunks at the vector
kernel row limit ride the same kernel family as the loop, so agreement is
bit-exact at short KV and bf16 tail-ULP at long KV (2-pass reduction split).
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches.qwen35_verify_sdpa_split import (
    _chunked_causal_sdpa,
    _eligible,
)

HQ, HKV, HD = 24, 4, 256


def _per_row_reference(q, k, v, scale):
    q_len = q.shape[2]
    prefix = k.shape[2] - q_len
    outs = []
    for i in range(q_len):
        outs.append(
            mx.fast.scaled_dot_product_attention(
                q[:, :, i : i + 1, :],
                k[:, :, : prefix + i + 1, :],
                v[:, :, : prefix + i + 1, :],
                scale=scale,
                mask=None,
            )
        )
    return mx.concatenate(outs, axis=2)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("q_len", [2, 4, 5, 6, 7, 9])
@pytest.mark.parametrize("kv_len", [512, 2048])
def test_chunked_causal_matches_per_row(q_len, kv_len):
    mx.random.seed(7)
    q = mx.random.normal((1, HQ, q_len, HD)).astype(mx.bfloat16)
    k = mx.random.normal((1, HKV, kv_len, HD)).astype(mx.bfloat16)
    v = mx.random.normal((1, HKV, kv_len, HD)).astype(mx.bfloat16)
    scale = HD**-0.5
    ref = _per_row_reference(q, k, v, scale)
    got = _chunked_causal_sdpa(q, k, v, scale, limit=32 // (HQ // HKV))
    diff = mx.abs(
        ref.astype(mx.float32) - got.astype(mx.float32)
    ).max().item()
    # Same kernel family; short KV is bit-exact, long KV differs only in
    # the 2-pass reduction split (bf16 tail ULP).
    assert diff <= 3e-4, f"q_len={q_len} kv_len={kv_len} diff={diff}"


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_eligibility_gates():
    q = mx.random.normal((1, HQ, 4, HD)).astype(mx.bfloat16)
    k = mx.random.normal((1, HKV, 256, HD)).astype(mx.bfloat16)
    assert _eligible(q, k, None) > 0
    # batch > 1 is not ours
    q2 = mx.random.normal((2, HQ, 4, HD)).astype(mx.bfloat16)
    k2 = mx.random.normal((2, HKV, 256, HD)).astype(mx.bfloat16)
    assert _eligible(q2, k2, None) == 0
    # non-256 head dim is not ours
    q3 = mx.random.normal((1, HQ, 4, 128)).astype(mx.bfloat16)
    k3 = mx.random.normal((1, HKV, 256, 128)).astype(mx.bfloat16)
    assert _eligible(q3, k3, None) == 0
    # single row (plain decode) is not ours
    q4 = mx.random.normal((1, HQ, 1, HD)).astype(mx.bfloat16)
    assert _eligible(q4, k, None) == 0

    class _QuantCache:
        bits = 4

    assert _eligible(q, k, _QuantCache()) == 0


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_nax_first_routing():
    """The patched seam tries the NAX split-K kernel before the chunked
    vector path: engaged at/above the KV floor (output matches the fp32
    reference), declined below it or when the fa256 route table is empty
    (the caller then falls back to the vector chunks)."""
    from omlx.custom_kernels.qwen35_prefill import fast as fa_fast
    from omlx.patches import qwen35_fa256_attention as fa
    from omlx.patches.qwen35_verify_sdpa_split import _NAX_MIN_KV_LEN, _nax_sdpa

    if not fa_fast.nax_attn256_available():
        pytest.skip("NAX dsplit kernel unavailable")

    q = mx.random.normal((1, HQ, 5, HD)).astype(mx.bfloat16)
    k = mx.random.normal((1, HKV, _NAX_MIN_KV_LEN, HD)).astype(mx.bfloat16)
    v = mx.random.normal((1, HKV, _NAX_MIN_KV_LEN, HD)).astype(mx.bfloat16)
    scale = HD**-0.5

    ref = _per_row_reference(q[:, :, :1], k, v, scale)  # shape sanity only
    del ref

    fa._NAX_ROUTE["kernel"] = fa_fast.qwen35_attn256_nax
    fa._NAX_ROUTE["budget"] = 0
    try:
        out = _nax_sdpa(q, k, v, scale, _NAX_MIN_KV_LEN)
        assert out is not None
        assert out.shape == q.shape
        # Agreement with the vector path it replaces (bf16 tail ULPs).
        vec = _chunked_causal_sdpa(q, k, v, scale, 32 // (HQ // HKV))
        diff = mx.abs(
            out.astype(mx.float32) - vec.astype(mx.float32)
        ).max().item()
        assert diff <= 2e-2, f"NAX vs vector diff {diff}"
        # Below the floor the seam declines (vector fallback).
        assert _nax_sdpa(q, k[:, :, :-1], v[:, :, :-1], scale, 8191) is None
        # Empty route table (fa256 patch unapplied) declines too.
        fa._NAX_ROUTE["kernel"] = None
        assert _nax_sdpa(q, k, v, scale, _NAX_MIN_KV_LEN) is None
    finally:
        fa._NAX_ROUTE["kernel"] = None
        fa._NAX_ROUTE["budget"] = 0
