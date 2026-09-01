# SPDX-License-Identifier: Apache-2.0
"""Tests for the fused NAX head-dim-256 dsplit attention kernel.

Covers the native op (correctness vs an fp32 reference across aligned /
unaligned / GQA / sliced-grid / cache-widened shapes) and the routing
gates of the fa256 patch that dispatches it.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from omlx.custom_kernels.qwen35_prefill import fast as fa_fast

HEAD_DIM = 256
SCALE = HEAD_DIM**-0.5


def _nax_ready() -> bool:
    return (
        mx.metal.is_available()
        and fa_fast.is_native_available()
        and fa_fast.nax_attn256_available()
    )


def _reference(q, k, v, scale, causal=True):
    """fp32 unfused reference with END-aligned causal (MLX convention)."""
    q_len, k_len = q.shape[2], k.shape[2]
    group = q.shape[1] // k.shape[1]
    qf = q.astype(mx.float32)
    qr = qf.reshape(q.shape[0], k.shape[1], group, q_len, HEAD_DIM)
    kr = k.astype(mx.float32).reshape(
        k.shape[0], k.shape[1], 1, k_len, HEAD_DIM
    )
    vr = v.astype(mx.float32).reshape(
        v.shape[0], v.shape[1], 1, k_len, HEAD_DIM
    )
    s = qr @ mx.swapaxes(kr, -1, -2) * scale
    off = k_len - q_len if causal else 0
    kp = mx.broadcast_to(mx.arange(k_len), (q_len, k_len))
    qp = mx.arange(off, off + q_len).reshape(q_len, 1)
    s = mx.where(kp > qp, float("-inf"), s)
    w = mx.softmax(s, axis=-1)
    out = (w @ vr).reshape(q.shape[0], q.shape[1], q_len, HEAD_DIM)
    mx.eval(out)
    return out


def _make(batch, q_heads, kv_heads, q_len, k_len, dtype=mx.bfloat16, seed=0):
    rng = np.random.default_rng(seed)
    q = mx.array(
        rng.standard_normal((batch, q_heads, q_len, HEAD_DIM)) * 0.5
    ).astype(dtype)
    k = mx.array(
        rng.standard_normal((batch, kv_heads, k_len, HEAD_DIM)) * 0.5
    ).astype(dtype)
    v = mx.array(
        rng.standard_normal((batch, kv_heads, k_len, HEAD_DIM)) * 0.5
    ).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def _assert_close(out, ref, tol):
    diff = mx.abs(out.astype(mx.float32) - ref).max().item()
    scale_ref = mx.abs(ref).max().item()
    assert diff <= tol * max(scale_ref, 1e-6), (
        f"max abs diff {diff} exceeds {tol} x {scale_ref}"
    )


@pytest.mark.skipif(not _nax_ready(), reason="NAX dsplit kernel unavailable")
class TestAttn256NaxKernel:
    def test_aligned_square(self):
        q, k, v = _make(1, 8, 4, 64, 64)
        out = fa_fast.qwen35_attn256_nax(q, k, v, SCALE, causal=True)
        _assert_close(out, _reference(q, k, v, SCALE), tol=0.02)

    def test_unaligned_q_and_k(self):
        q, k, v = _make(1, 24, 4, 65, 129)
        out = fa_fast.qwen35_attn256_nax(q, k, v, SCALE, causal=True)
        _assert_close(out, _reference(q, k, v, SCALE), tol=0.02)

    def test_verify_width_block(self):
        # MTP-verify-shaped call (also exercises q < BQ padding).
        q, k, v = _make(1, 24, 4, 5, 100)
        out = fa_fast.qwen35_attn256_nax(q, k, v, SCALE, causal=True)
        _assert_close(out, _reference(q, k, v, SCALE), tol=0.02)

    def test_single_row(self):
        q, k, v = _make(1, 24, 4, 1, 96)
        out = fa_fast.qwen35_attn256_nax(q, k, v, SCALE, causal=True)
        _assert_close(out, _reference(q, k, v, SCALE), tol=0.02)

    def test_chunked_prefill_offset(self):
        # Cached-prefix shape: qL < kL, offset = kL - qL.
        q, k, v = _make(1, 24, 4, 100, 2048)
        out = fa_fast.qwen35_attn256_nax(q, k, v, SCALE, causal=True)
        _assert_close(out, _reference(q, k, v, SCALE), tol=0.02)

    def test_batched_moe_layout(self):
        q, k, v = _make(2, 16, 2, 128, 320)
        out = fa_fast.qwen35_attn256_nax(q, k, v, SCALE, causal=True)
        _assert_close(out, _reference(q, k, v, SCALE), tol=0.02)

    def test_fp16(self):
        q, k, v = _make(1, 8, 8, 48, 96, dtype=mx.float16)
        out = fa_fast.qwen35_attn256_nax(q, k, v, SCALE, causal=True)
        _assert_close(out, _reference(q, k, v, SCALE), tol=0.01)

    def test_dispatch_budget_grid_split_exact(self):
        # The budget splits the query-block grid across dispatches; query
        # blocks are independent so the result must match single-dispatch.
        # q is wide enough that the grid already clears the split-K
        # occupancy floor, so tiny budgets take query slicing (the
        # starved-grid split-K variant is covered separately with a
        # tolerance).
        q, k, v = _make(1, 24, 4, 1024, 1024)
        single = fa_fast.qwen35_attn256_nax(q, k, v, SCALE, causal=True)
        sliced = fa_fast.qwen35_attn256_nax(
            q, k, v, SCALE, causal=True, dispatch_budget=1
        )
        mx.eval(single, sliced)
        assert mx.abs(single - sliced).max().item() == 0.0

    def test_kv_cache_padding_widening(self):
        # Unaligned kL sliced from a padded cache: the op widens kL to the
        # BK tile over backing rows (masked out by the causal diagonal).
        k_len, max_len = 2000, 2048
        q = mx.array(
            np.random.default_rng(1).standard_normal((1, 24, 100, HEAD_DIM))
            * 0.5
        ).astype(mx.bfloat16)
        k_full = mx.zeros((1, 4, max_len, HEAD_DIM), dtype=mx.bfloat16)
        v_full = mx.zeros((1, 4, max_len, HEAD_DIM), dtype=mx.bfloat16)
        kn = mx.array(
            np.random.default_rng(2).standard_normal((1, 4, k_len, HEAD_DIM))
            * 0.5
        ).astype(mx.bfloat16)
        vn = mx.array(
            np.random.default_rng(3).standard_normal((1, 4, k_len, HEAD_DIM))
            * 0.5
        ).astype(mx.bfloat16)
        k_full[:, :, :k_len, :] = kn
        v_full[:, :, :k_len, :] = vn
        k = k_full[:, :, :k_len, :]
        v = v_full[:, :, :k_len, :]
        mx.eval(q, k, v)
        out = fa_fast.qwen35_attn256_nax(q, k, v, SCALE, causal=True)
        _assert_close(out, _reference(q, kn, vn, SCALE), tol=0.02)

    def test_rejects_bad_shapes(self):
        q, k, v = _make(1, 8, 4, 64, 64)
        with pytest.raises(ValueError):
            fa_fast.qwen35_attn256_nax(
                q.astype(mx.float32), k.astype(mx.float32),
                v.astype(mx.float32), SCALE, causal=True,
            )
        with pytest.raises(ValueError):
            # head_dim != 256
            fa_fast.qwen35_attn256_nax(
                q[..., :128], k[..., :128], v[..., :128], SCALE, causal=True
            )
        with pytest.raises(ValueError):
            fa_fast.qwen35_attn256_nax(q, k, v, SCALE, causal=False)


@pytest.mark.skipif(not _nax_ready(), reason="NAX dsplit kernel unavailable")
class TestAttn256NaxSplitK:
    """KV-axis split (flash decoding) dispatch: tiny budgets force many
    splits and grouped dispatches; every path must match the fp32
    reference (the logsumexp merge reorders fp32 accumulation, so vs the
    single-dispatch result a small tolerance applies)."""

    def test_splitk_verify_width(self):
        q, k, v = _make(1, 24, 4, 5, 4096)
        out = fa_fast.qwen35_attn256_nax(
            q, k, v, SCALE, causal=True, dispatch_budget=1000
        )
        _assert_close(out, _reference(q, k, v, SCALE), tol=0.02)

    def test_splitk_decode_single_row(self):
        q, k, v = _make(1, 24, 4, 1, 8192, seed=5)
        out = fa_fast.qwen35_attn256_nax(
            q, k, v, SCALE, causal=True, dispatch_budget=1000
        )
        _assert_close(out, _reference(q, k, v, SCALE), tol=0.02)

    def test_splitk_narrow_tile_boundaries(self):
        # q=17 routes to the bq32/wm2 tile, q=33 to bq64/wm4; both with
        # and without forced splits.
        for q_len, seed in ((17, 6), (33, 7)):
            q, k, v = _make(1, 24, 4, q_len, 4096, seed=seed)
            ref = _reference(q, k, v, SCALE)
            for budget in (0, 1000):
                out = fa_fast.qwen35_attn256_nax(
                    q, k, v, SCALE, causal=True, dispatch_budget=budget
                )
                _assert_close(out, ref, tol=0.02)

    def test_splitk_batched_moe_layout(self):
        q, k, v = _make(2, 16, 2, 9, 4096, seed=8)
        out = fa_fast.qwen35_attn256_nax(
            q, k, v, SCALE, causal=True, dispatch_budget=1000
        )
        _assert_close(out, _reference(q, k, v, SCALE), tol=0.02)

    def test_splitk_kv_cache_padding_widening(self):
        k_len, max_len = 4000, 4096
        q = mx.array(
            np.random.default_rng(9).standard_normal((1, 24, 5, HEAD_DIM))
            * 0.5
        ).astype(mx.bfloat16)
        k_full = mx.zeros((1, 4, max_len, HEAD_DIM), dtype=mx.bfloat16)
        v_full = mx.zeros((1, 4, max_len, HEAD_DIM), dtype=mx.bfloat16)
        kn = mx.array(
            np.random.default_rng(10).standard_normal((1, 4, k_len, HEAD_DIM))
            * 0.5
        ).astype(mx.bfloat16)
        vn = mx.array(
            np.random.default_rng(11).standard_normal((1, 4, k_len, HEAD_DIM))
            * 0.5
        ).astype(mx.bfloat16)
        k_full[:, :, :k_len, :] = kn
        v_full[:, :, :k_len, :] = vn
        k = k_full[:, :, :k_len, :]
        v = v_full[:, :, :k_len, :]
        mx.eval(q, k, v)
        out = fa_fast.qwen35_attn256_nax(
            q, k, v, SCALE, causal=True, dispatch_budget=1000
        )
        _assert_close(out, _reference(q, kn, vn, SCALE), tol=0.02)

    def test_splitk_matches_single_dispatch(self):
        # Splits reorder the fp32 logsumexp merge but must not move the
        # result beyond rounding noise.
        q, k, v = _make(1, 24, 4, 5, 8192, seed=12)
        single = fa_fast.qwen35_attn256_nax(q, k, v, SCALE, causal=True)
        split = fa_fast.qwen35_attn256_nax(
            q, k, v, SCALE, causal=True, dispatch_budget=1000
        )
        diff = (
            mx.abs(single.astype(mx.float32) - split.astype(mx.float32))
            .max()
            .item()
        )
        assert diff <= 2e-3, f"split vs single diff {diff}"


@pytest.mark.skipif(not _nax_ready(), reason="NAX dsplit kernel unavailable")
class TestAttn256NaxGqaPacking:
    """GQA head packing (split-K path): contiguous q head blocks fold the
    gqa heads of each kv head into packed tile rows — one KV scan per kv
    head instead of per q head. The causal wedge wraps per packed head and
    the chunk-reduce unpacks slab rows (g*q_pack + t -> head h*G+g, token
    t) onto o's original strides."""

    @pytest.mark.parametrize(
        "q_heads,kv_heads,q_len,k_len",
        [
            (24, 4, 5, 8192), # production MTP verify (gqa6, pqL=30, bq32)
            (24, 4, 1, 8192), # decode row (pqL=6, bq16)
            (24, 4, 4, 4096), # pqL=24, bq32 tail
            (24, 4, 2, 4096), # pqL=12, bq16 tail
            (8, 4, 4, 4096), # gqa2, pqL=8
            (10, 2, 3, 4096), # gqa5, pqL=15
            (16, 2, 2, 4096), # gqa8, pqL=16, aligned bq16
            (2, 1, 16, 4096), # gqa2, pqL=32, aligned bq32
            (24, 4, 5, 8190), # unaligned kL
            (24, 4, 5, 64), # tiny KV: forced splits, dead split folds -inf
            (24, 4, 5, 33), # NK=2: one block per forced split
        ],
    )
    def test_packed_matches_reference(self, q_heads, kv_heads, q_len, k_len):
        q, k, v = _make(1, q_heads, kv_heads, q_len, k_len)
        ref = _reference(q, k, v, SCALE)
        for budget in (0, 1000):
            out = fa_fast.qwen35_attn256_nax(
                q, k, v, SCALE, causal=True, dispatch_budget=budget
            )
            _assert_close(out, ref, tol=0.02)

    def test_packed_fp16(self):
        q, k, v = _make(1, 24, 4, 5, 8192, dtype=mx.float16, seed=14)
        ref = _reference(q, k, v, SCALE)
        out = fa_fast.qwen35_attn256_nax(q, k, v, SCALE, causal=True)
        _assert_close(out, ref, tol=0.01)

    def test_packed_batched(self):
        q, k, v = _make(2, 16, 2, 4, 4096, seed=15)
        ref = _reference(q, k, v, SCALE)
        out = fa_fast.qwen35_attn256_nax(q, k, v, SCALE, causal=True)
        _assert_close(out, ref, tol=0.02)

    def test_packed_kv_cache_padding_widening(self):
        k_len, max_len = 4002, 4096
        rng = np.random.default_rng(16)
        q = mx.array(
            rng.standard_normal((1, 24, 5, HEAD_DIM)) * 0.5
        ).astype(mx.bfloat16)
        k_full = mx.zeros((1, 4, max_len, HEAD_DIM), dtype=mx.bfloat16)
        v_full = mx.zeros((1, 4, max_len, HEAD_DIM), dtype=mx.bfloat16)
        kn = mx.array(
            rng.standard_normal((1, 4, k_len, HEAD_DIM)) * 0.5
        ).astype(mx.bfloat16)
        vn = mx.array(
            rng.standard_normal((1, 4, k_len, HEAD_DIM)) * 0.5
        ).astype(mx.bfloat16)
        k_full[:, :, :k_len, :] = kn
        v_full[:, :, :k_len, :] = vn
        k = k_full[:, :, :k_len, :]
        v = v_full[:, :, :k_len, :]
        mx.eval(q, k, v)
        out = fa_fast.qwen35_attn256_nax(q, k, v, SCALE, causal=True)
        _assert_close(out, _reference(q, kn, vn, SCALE), tol=0.02)

    def test_packed_vs_stride_broken_q(self):
        # Same values, but q is a transposed view ([B, qL, H, D] physical),
        # so head blocks are not row-contiguous (head stride D != qL * H*D
        # token stride): packing declines while split-K still serves the
        # call. Both paths must agree beyond fp32 merge-order noise.
        q_heads, kv_heads, q_len, k_len = 24, 4, 5, 8192
        q, k, v = _make(1, q_heads, kv_heads, q_len, k_len, seed=21)
        phys = mx.zeros((1, q_len, q_heads, HEAD_DIM), dtype=q.dtype)
        phys[:, :, :, :] = q.transpose(0, 2, 1, 3)
        q_strided = phys.transpose(0, 2, 1, 3)
        mx.eval(q_strided)
        packed = fa_fast.qwen35_attn256_nax(q, k, v, SCALE, causal=True)
        unpacked = fa_fast.qwen35_attn256_nax(
            q_strided, k, v, SCALE, causal=True
        )
        mx.eval(packed, unpacked)
        diff = (
            mx.abs(packed.astype(mx.float32) - unpacked.astype(mx.float32))
            .max()
            .item()
        )
        assert diff <= 2e-3, f"packed vs stride-broken q diff {diff}"

    def test_packed_not_taken_when_pql_wide(self):
        # qL * gqa > 32 stays on the unpacked tile layout (and must still
        # be correct); guards the eligibility gate boundary.
        q, k, v = _make(1, 24, 4, 6, 4096) # pqL = 36 > 32
        ref = _reference(q, k, v, SCALE)
        out = fa_fast.qwen35_attn256_nax(q, k, v, SCALE, causal=True)
        _assert_close(out, ref, tol=0.02)


def _patch_module():
    import omlx.patches.qwen35_fa256_attention as patch

    return patch


@pytest.fixture()
def _fresh_fa256_patch(monkeypatch):
    patch = _patch_module()
    monkeypatch.setattr(patch, "_PATCHED", False, raising=False)
    monkeypatch.delenv("OMLX_FA256_STEEL", raising=False)
    monkeypatch.delenv("OMLX_FA256_NAX", raising=False)
    monkeypatch.delenv("OMLX_FA256_NAX_DISPATCH_BUDGET", raising=False)
    monkeypatch.setattr(
        patch, "_auto_nax_dispatch_budget", lambda kernel: 0, raising=False
    )
    monkeypatch.setattr(
        patch, "_auto_dispatch_budget", lambda k, q, r: 0, raising=False
    )
    yield patch


class TestFa256NaxRouting:
    def test_nax_gate_requires_prefill_width(self, _fresh_fa256_patch):
        patch = _fresh_fa256_patch
        q, k, v = _make(1, 24, 4, 63, 128)
        # Below the measured crossover (one full BQ tile) the NAX route
        # must decline.
        assert not patch._nax_should_route(q, k, None, "causal", None)
        q, k, v = _make(1, 24, 4, 64, 128)
        assert patch._nax_should_route(q, k, None, "causal", None)
        # Array masks / sinks / fp32 stay out.
        assert not patch._nax_should_route(q, k, None, q, None)
        assert not patch._nax_should_route(
            q.astype(mx.float32), k.astype(mx.float32), None, "causal", None
        )

    def test_nax_gate_rejects_quantized_cache_proxy(self, _fresh_fa256_patch):
        patch = _fresh_fa256_patch

        class _Quant:
            bits = 4

        q, k, v = _make(1, 24, 4, 128, 256)
        assert not patch._nax_should_route(q, k, _Quant(), "causal", None)

    def test_nax_gate_small_q_needs_long_kv(self, _fresh_fa256_patch):
        # Below the prefill q floor the split-K route engages only once
        # the KV scan dominates (default floor 32K); callers may tighten
        # both thresholds.
        patch = _fresh_fa256_patch
        q, k, v = _make(1, 24, 4, 5, 32768)
        assert patch._nax_should_route(q, k, None, "causal", None)
        assert patch._nax_should_route(q, k, None, None, None)
        q, k, v = _make(1, 24, 4, 5, 32767)
        assert not patch._nax_should_route(q, k, None, "causal", None)
        assert patch._nax_should_route(
            q, k, None, "causal", None, min_q_len=2, min_kv_len=1024
        )
        # Single-row decode never takes the small-q branch (the stock
        # vector kernel wins there).
        q, k, v = _make(1, 24, 4, 1, 32768)
        assert not patch._nax_should_route(q, k, None, None, None)

    def test_try_nax_attn_dispatch_and_demote(self, _fresh_fa256_patch):
        # With the route table populated the helper runs the kernel;
        # unavailable route or declined shape yields None so callers keep
        # their fallback.
        patch = _fresh_fa256_patch
        assert patch.try_nax_attn(*_make(1, 24, 4, 128, 256), SCALE) is None

        if not _nax_ready():
            pytest.skip("NAX dsplit kernel unavailable")
        # Neutralize the fast-module availability cache so the deliberate
        # failure below does not demote NAX for later tests.
        patch._fa256_fast._nax_attn_cache = True
        patch._NAX_ROUTE["kernel"] = fa_fast.qwen35_attn256_nax
        patch._NAX_ROUTE["budget"] = 0
        try:
            q, k, v = _make(1, 24, 4, 128, 4096, seed=13)
            out = patch.try_nax_attn(q, k, v, SCALE)
            _assert_close(out, _reference(q, k, v, SCALE), tol=0.02)
            # A launch failure demotes the route for the process.
            patch._NAX_ROUTE["kernel"] = lambda *a, **kw: (_ for _ in ()).throw(
                RuntimeError("boom")
            )
            assert patch.try_nax_attn(q, k, v, SCALE) is None
            assert patch._NAX_ROUTE["kernel"] is None
        finally:
            patch._NAX_ROUTE["kernel"] = None
            patch._NAX_ROUTE["budget"] = 0
            patch._fa256_fast._nax_attn_cache = None

    @pytest.mark.skipif(not _nax_ready(), reason="NAX dsplit kernel unavailable")
    def test_patched_sdpa_routes_to_nax(self, _fresh_fa256_patch):
        import sys

        import mlx_lm.models.base as mlx_base

        patch = _fresh_fa256_patch
        original = mlx_base.scaled_dot_product_attention
        assert patch.apply_qwen35_fa256_attention_patch()
        try:
            q, k, v = _make(1, 24, 4, 128, 4096, seed=4)
            out = mlx_base.scaled_dot_product_attention(
                q, k, v, None, SCALE, "causal"
            )
            mx.eval(out)
            _assert_close(out, _reference(q, k, v, SCALE), tol=0.02)
        finally:
            mlx_base.scaled_dot_product_attention = original
            # Undo the rebind loop for modules the apply touched (their
            # attribute was `original` before, so anything else was us).
            for mod_name, mod in list(sys.modules.items()):
                if mod is None or not mod_name.startswith("mlx_lm.models."):
                    continue
                current = getattr(mod, "scaled_dot_product_attention", None)
                if current is not None and current is not original:
                    mod.scaled_dot_product_attention = original
