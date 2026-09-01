# SPDX-License-Identifier: Apache-2.0
"""Patch scaled_dot_product_attention to fix head_dim=256 long-context prefill.

MLX's fused SDPA kernel supports head_dim in {64, 80, 128} only, so head_dim=256
(e.g. Qwen3.6-27B) multi-token prefill falls back to an unfused path that
materializes the full ``[n_q, query_len, kv_len]`` score matrix -> O(L^2) memory,
OOMing / tripping the prefill guard far below the context window. Decode
(query_len == 1) is unaffected (MLX has a fused vector kernel for 256).

This routes head_dim=256 causal prefill to a flash-style online-softmax pass in
pure MLX array ops (tiled over KV; running max/sum/accumulator) that never
materializes the score matrix -> peak memory O(L). It rides MLX's GEMM, so speed
is on par with the fallback; the win is memory. ``register_tiled_prefill_head_dim``
flips the prefill-guard estimator to O(L) in lockstep (else it keeps rejecting).

The route is memory-aware (issue #2204): the unfused fallback is faster
everywhere its score matrix fits (on NAX GPUs its big GEMMs run on the tensor
units; even pre-NAX it is ~2x faster than the tiled pass at long context, issue
#2155), so when the Scheduler has registered a headroom provider the tiled pass
engages only if the unfused transient would NOT fit under the prefill-guard
ceiling. Without a provider (no Scheduler, ceiling not propagated yet) the
route keeps the memory-safe default: always tiled past the kv_len threshold.
``OMLX_SDPA256_TILED=1`` forces the tiled pass whenever the shape gates match
(pre-#2204 behavior); ``OMLX_SDPA256_TILED=0`` never engages it (restores the
O(L^2) memory wall — benchmarking only).

Install mechanics: patch the module attr + rebind already-imported model
modules. The route is strictly gated (see _should_route); everything else
passes through to the original SDPA unchanged.
"""

import logging
import os
import weakref

import mlx.core as mx

from omlx.memory_monitor import estimate_unfused_sdpa_call_bytes

logger = logging.getLogger(__name__)

_PATCHED = False

HEAD_DIM = 256
# Engage the tiled kernel only once the context is long enough that the unfused
# fallback's O(L^2) score matrix becomes a memory problem. Below this, the
# fused-GEMM fallback is faster and fits comfortably. Tunable.
_SDPA256_MIN_KV_LEN = 8192
# Multi-row decode-shaped calls (MTP verify: q_len = 1 + draft depth <= 9)
# stay on the stock path: at q_len 2-8 the Q-tiled kernel's batched GEMM
# shapes underperform the stock unfused fallback (measured 22.8 vs 9.4
# ms/layer at kv=200k — the broadcast-GQA GEMM materializes poorly at
# tiny M), and no fused multi-row hd-256 kernel exists. Decode (q_len 1)
# keeps the fused vector kernel.
_SDPA256_MIN_Q_LEN = 16
# Q-tile cap for the kernel (auto-sized down by live headroom; see
# _pick_q_tile). Env-overridable: OMLX_SDPA256_Q_TILE. Measured on M5 Pro
# at q_len 2048 (2026-08-27 sweep): tile 128-256 is the sweet spot —
# 200k kv: 64 -> 1440ms, 128 -> 893ms, 256 -> 870ms (1.66x), 512+ ->
# slower or flat while transient keeps growing; 100k kv: flat 128-512.
# Capping at 256 keeps the transient bounded with the best throughput.
_Q_TILE = 256
# Per-score-element transient bytes budgeted when sizing a Q tile.
# MEASURED with mx.get_peak_memory at kv=200k (2026-08-27): marginal cost
# 5.1 bytes/elem for tiles >= 128 (score bf16 write 2 + softmax passes +
# PV read; fixed q/k/v/out buffers amortize away at larger tiles). The
# previous 10 B/elem constant was 2x conservative and capped the tile at
# 64 rows at 200k — M=64 GEMMs run at ~1/3 of tensor-core peak.
_Q_TILE_BYTES_PER_ELEM = 5.5
# Transient ceiling for one Q tile when no guard headroom is available
# (no scheduler / unit tests): keeps the kernel memory-safe standalone.
_Q_TILE_DEFAULT_BUDGET = 4 * 1024**3
# When live headroom IS available, one Q tile may claim at most this
# fraction of it (and at most this absolute cap). The first flash chunk's
# tile-pool fill is recorded by the scheduler's transient tracker as a
# positive phys delta; keeping the budget a minority share of headroom
# leaves room for that one-off charge plus KV growth, so the pre-chunk
# guard never rejects the SECOND chunk over the first's pool allocation
# (the pool is reused afterwards and the measured delta collapses back to
# KV growth, healing the prediction within one chunk).
_Q_TILE_HEADROOM_FRACTION = 0.35
_Q_TILE_BUDGET_CAP = 3 * 1024**3
# Second budget anchored to the PHYSICAL admission-limit headroom rather
# than the sizing target: the sizing target reserves headroom for PERSISTENT
# growth (KV), and at 200k context it leaves only ~2.75GB — never enough
# for a productive tile (192 rows needs ~5GB). A tile transient is
# short-lived and pool-recycled; it competes with the enforcer's hard
# watermark, not the sizing target. The fraction is deliberately a MINORITY
# share: the tile peak also carries the chunk's other activations, KV
# growth, and boundary-snapshot slop on top, and the pool itself is capped
# (8GB default) — larger fractions measurably cycle the pool against the
# enforcer at 170K+ and spiral into requeue rejections (2026-08-28
# regression: 0.80 of the raw-limit headroom capped 262K runs at ~15xK).
_Q_TILE_PHYS_FRACTION = 0.50
_Q_TILE_PHYS_BUDGET_CAP = 4 * 1024**3
_NEG_INF = -1e30  # fp32 sentinel for masked logits (exp -> 0)


def _env_int(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, "").strip())
        return v if v > 0 else default
    except Exception:
        return default


def _pick_q_tile(n_q: int, kv_len: int) -> int:
    """Size one Q tile so its score transient fits the live headroom.

    The kernel materializes [n_q, q_tile, kv_len] scores per Q tile (bf16
    GEMM out, bf16 softmax pass, bf16 weights — ~10 bytes/elem). Sizing the
    tile from the guard's live headroom keeps the peak under the same
    ceiling the scheduler enforces, lets short contexts run the whole chunk
    in one tile, and shrinks the tile automatically as context grows. The
    budget is a minority share of headroom (see _Q_TILE_HEADROOM_FRACTION)
    so the first chunk's pool allocation cannot crowd out KV growth in the
    guard's own accounting."""
    # The physical-headroom budget is currently DISABLED for production
    # sizing (2026-08-28 postmortem): any tile large enough to move the
    # needle (192-256 rows = 3.5-5GB transient) cycles the capped MLX pool
    # against the enforcer at long context, and the per-window refill
    # deltas poison the transient estimator into chunk collapse and
    # requeue rejections (262K runs died at 51K-177K across three tuning
    # attempts). The speed answer is the fused NAX kernel (O(1)
    # transient), not bigger composed-GEMM tiles. Provider kept for the
    # fused-kernel work; fraction deliberately zero.
    budget = _Q_TILE_DEFAULT_BUDGET
    live = unfused_headroom_bytes()
    if live is not None:
        budget = max(
            512 << 20,
            min(int(live * _Q_TILE_HEADROOM_FRACTION), _Q_TILE_BUDGET_CAP),
        )
    tile = int(budget // (max(n_q, 1) * max(kv_len, 1) * _Q_TILE_BYTES_PER_ELEM))
    tile = min(_Q_TILE, tile)
    tile = max(64, tile)
    return tile - (tile % 64)


def _flash_sdpa256(queries, keys, values, scale, mask):
    """Flash-style attention for head_dim=256 prefill, tiled over Q only.

    queries: [batch, n_q, q_len, head_dim]
    keys/values: [batch, n_kv, k_len, head_dim]   (n_q % n_kv == 0)
    mask: "causal" or None. Returns [batch, n_q, q_len, head_dim] in
    queries.dtype.

    Per Q tile (headroom-sized, see ``_pick_q_tile``): one bf16 QK^T GEMM
    over the FULL key span (NAX tensor cores), one fused fp32 softmax, one
    bf16 PV GEMM. The [q_tile x kv_len] score tile is transient and bounded
    by the tile sizing — peak stays O(headroom), never O(q_len x kv_len)
    for the whole chunk. Profiling on M5 Pro (29 TFLOPS bf16 GEMM) showed
    the earlier KV-tiled online-softmax kernel spends most of its time in
    per-tile eval syncs and fp32 elementwise passes (~470GB of softmax
    traffic per layer at 100k kv_len — bandwidth-bound at 6.6x off GEMM
    peak); this layout keeps the GEMMs big and the softmax fused, running
    near the GEMM roofline instead.

    MLX is lazy: each Q tile's output is eval'd to bound the live graph.
    """
    batch, n_q, q_len, head_dim = queries.shape
    _, n_kv, k_len, _ = keys.shape
    group_size = n_q // n_kv
    causal = mask == "causal"
    dtype = queries.dtype

    qr = queries.reshape(batch, n_kv, group_size, q_len, head_dim)
    kr = keys.reshape(batch, n_kv, 1, k_len, head_dim)
    vr = values.reshape(batch, n_kv, 1, k_len, head_dim)

    # MLX 'causal' aligns queries to the END of the key axis: with a cached
    # prefix (k_len > q_len, chunked prefill) local query i is global position
    # i + offset and attends keys 0..(i + offset). offset == 0 for square.
    offset = k_len - q_len

    k_pos = mx.arange(k_len).reshape(1, 1, 1, 1, k_len) if causal else None
    q_tile = _pick_q_tile(n_q, k_len)

    out_q_tiles = []
    for qi0 in range(0, q_len, q_tile):
        qi1 = min(qi0 + q_tile, q_len)
        # scale is applied to the queries up front: for head_dim 256 it is
        # 2^-4 — exactly representable in bf16, so this costs nothing in
        # precision and saves a full pass over the score tile.
        qb = qr[:, :, :, qi0:qi1, :] * scale
        qt = qi1 - qi0

        # Full-KV score GEMM in the input dtype (NAX tensor cores; measured
        # 27-31 TFLOPS at these shapes). Softmax stays in the input dtype
        # too: an fp32 upcast more than doubles tile traffic and mx.softmax
        # measured only ~119GB/s on the fp32 volume — the dominant cost of
        # the first cut of this kernel. bf16 softmax matches the numerics
        # class of stock MLX SDPA (whose weights matmul is input-dtype).
        s = qb @ mx.swapaxes(kr, -1, -2)
        if causal:
            q_pos = mx.arange(qi0 + offset, qi1 + offset).reshape(
                1, 1, 1, qt, 1
            )
            s = mx.where(k_pos > q_pos, float("-inf"), s)
        w = mx.softmax(s, axis=-1)
        out_tile = w @ vr

        mx.eval(out_tile)  # bound the live graph -> headroom-sized peak
        out_q_tiles.append(out_tile)

    out = mx.concatenate(out_q_tiles, axis=3)
    return out.reshape(batch, n_q, q_len, head_dim)

# Live guard-headroom provider for memory-aware routing (issue #2204).
# Registered by Scheduler.__init__ as a bound method returning the bytes left
# under the adaptive-prefill-throttle target (hard ceiling x headroom safety,
# clamped by the abort cap), or a negative value when no ceiling is active.
# Held as a WeakMethod so a torn-down Scheduler auto-unregisters and the route
# falls back to the memory-safe always-tiled default.
_HEADROOM_PROVIDER: "weakref.WeakMethod | None" = None
# OMLX_SDPA256_TILED override, parsed at apply time: True = always tiled,
# False = never tiled, None = memory-aware auto.
_FORCE_TILED: bool | None = None
# Tiled-route reasons already logged. The tiled pass trades substantial
# prefill throughput at long kv_len for O(L) memory, and nothing surfaced the
# route decision before (issue #2283 took an A/B repro to diagnose), so the
# first engagement per reason logs at INFO; repeats stay silent to keep the
# hot path quiet.
_TILED_ROUTE_LOGGED: "set[str]" = set()


def _note_tiled_route(reason: str, detail: str) -> None:
    if reason in _TILED_ROUTE_LOGGED:
        return
    _TILED_ROUTE_LOGGED.add(reason)
    logger.info(
        "sdpa256: head-dim-256 prefill taking the tiled (memory-safe, slower) "
        "path: %s. The unfused fast path resumes when guard headroom allows; "
        "OMLX_SDPA256_TILED=1/0 forces the route.",
        detail,
    )


def set_unfused_headroom_provider(method) -> None:
    """Register a bound method returning the prefill guard's live headroom in
    bytes (negative when no ceiling is active). Lets ``_should_route`` prefer
    the faster unfused fallback whenever its O(L^2) transient fits."""
    global _HEADROOM_PROVIDER
    _HEADROOM_PROVIDER = weakref.WeakMethod(method)


# Second provider: live headroom under the PHYSICAL hard ceiling (the
# enforcer's hard limit minus current usage). Registered by the Scheduler
# alongside _HEADROOM_PROVIDER; the Q-tile sizer uses it to let the
# short-lived tile transient use memory the sizing target reserves for
# persistent KV growth. WeakMethod for the same teardown reasons.
_PHYS_HEADROOM_PROVIDER: "weakref.WeakMethod | None" = None


def set_physical_headroom_provider(method) -> None:
    """Register a bound method returning bytes left under the physical hard
    ceiling (negative/None when no ceiling is active)."""
    global _PHYS_HEADROOM_PROVIDER
    _PHYS_HEADROOM_PROVIDER = weakref.WeakMethod(method)


def physical_headroom_bytes():
    """Live hard-ceiling headroom in bytes, or ``None`` when inactive."""
    provider = (
        _PHYS_HEADROOM_PROVIDER() if _PHYS_HEADROOM_PROVIDER is not None else None
    )
    if provider is None:
        return None
    try:
        headroom = provider()
    except Exception:
        return None
    if headroom is None or headroom < 0:
        return None
    return int(headroom)


def _parse_force_tiled_env() -> bool | None:
    value = os.environ.get("OMLX_SDPA256_TILED", "").strip()
    if value == "1":
        return True
    if value == "0":
        return False
    return None


def _sticky_tiled_enabled() -> bool:
    return os.environ.get("OMLX_SDPA256_STICKY", "1").strip() != "0"


def unfused_headroom_bytes():
    """Live guard headroom in bytes, or ``None`` when no ceiling is active.

    Shared by this module's route gate and other long-context prefill routes
    so every O(L^2)-shaped decision prices against the same scheduler
    target."""
    provider = _HEADROOM_PROVIDER() if _HEADROOM_PROVIDER is not None else None
    if provider is None:
        return None
    try:
        headroom = provider()
    except Exception:
        return None
    if headroom is None or headroom < 0:
        return None
    return int(headroom)


# The shared estimator prices ONE unfused SDPA call: the bf16 score matrix
# plus the fp32 output. The real unfused path also materializes softmax
# temporaries of the same [n_q, q_len, kv_len] shape, and MLX's async eval
# can keep more than one layer's graph live, so the measured Metal peak runs
# ~1.7-2x the raw estimate (262k-context Qwen3.8-27B repro, 2026-08-25
# server log: ~11.6GB estimate at kv_len 124928 vs ~20GB observed spike).
# Pricing only the raw matrix let the route re-engage the unfused path right
# after an eviction/reclaim re-opened headroom, overshoot the target again,
# and loop (prefill-LRU-eviction thrash, issue #2204 follow-up).
_UNFUSED_TRANSIENT_SAFETY = 3.0

# High-water kv_len at which the unfused path last proved not to fit. The
# unfused transient grows with kv_len and within one prefill current usage
# only grows, so once it did not fit at kv_len K it cannot fit again at
# kv_len >= K on the same envelope — regardless of the transient headroom a
# just-finished reclaim opened (that headroom is the buffer pool the very
# next chunk re-allocates). Without this ratchet the route flips between
# unfused and tiled every reclaim cycle. Cleared only by process restart;
# a shorter request starts below the mark and keeps the fast path.
# OMLX_SDPA256_STICKY=0 disables it (benchmarking).
_STICKY_TILED_KV_LEN: int | None = None
_ROUTE_PROBE_LOGGED = False


def _tiled_route_required(queries, keys) -> bool:
    """Decide tiled vs stock for a shape-matched prefill call (True = tiled).

    The stock unfused fallback is faster wherever its score matrix fits
    (issues #2155 / #2204), so take the tiled pass only when the unfused
    transient would not fit under the guard ceiling — or when no headroom
    info is available, keeping the memory-safe #2025 behavior. The transient
    is priced with a safety multiplier (see ``_UNFUSED_TRANSIENT_SAFETY``)
    and engagement is sticky (see ``_STICKY_TILED_KV_LEN``)."""
    global _STICKY_TILED_KV_LEN
    if _FORCE_TILED is not None:
        if _FORCE_TILED:
            _note_tiled_route("forced", "forced by OMLX_SDPA256_TILED=1")
        return _FORCE_TILED
    kv_len = int(keys.shape[-2])
    if (
        _sticky_tiled_enabled()
        and _STICKY_TILED_KV_LEN is not None
        and kv_len >= _STICKY_TILED_KV_LEN
    ):
        _note_tiled_route(
            "sticky",
            f"kv_len={kv_len} at/above the high-water mark {_STICKY_TILED_KV_LEN} "
            "where the unfused path last proved not to fit",
        )
        return True
    headroom = unfused_headroom_bytes()
    if headroom is None:
        _note_tiled_route(
            "no-ceiling",
            "memory ceiling not available (no guard headroom provider "
            "registered, or enforcer state not yet propagated)",
        )
        return True
    try:
        batch, n_q, q_len, _ = queries.shape
        transient = estimate_unfused_sdpa_call_bytes(
            batch * n_q,
            q_len,
            kv_len,
            HEAD_DIM,
            score_dtype_size=queries.dtype.size,
        ) * _UNFUSED_TRANSIENT_SAFETY
        if transient > headroom:
            _note_tiled_route(
                "insufficient-headroom",
                f"unfused transient ~{transient / 2**20:.0f}MiB exceeds live "
                f"guard headroom ~{headroom / 2**20:.0f}MiB at "
                f"kv_len={kv_len}",
            )
            _STICKY_TILED_KV_LEN = max(
                _STICKY_TILED_KV_LEN or 0, kv_len
            )
            return True
        return False
    except Exception:
        _note_tiled_route("probe-error", "guard headroom probe failed")
        logger.debug("sdpa256 headroom probe failed", exc_info=True)
        return True  # headroom info unavailable -> memory-safe default


def _should_route(queries, keys, cache, mask, sinks) -> bool:
    # Never raise: any unexpected input must fall through to the original SDPA,
    # never break a request. Worst case we decline to engage.
    # Shape gates first: this wrapper is installed unconditionally and runs
    # on every SDPA call of every decode step, so the common (decode / MTP
    # verify) case must exit on the q_len check alone (issue #2132).
    try:
        if queries.shape[-2] >= 16 and queries.shape[-1] == HEAD_DIM:
            global _ROUTE_PROBE_LOGGED
            if not _ROUTE_PROBE_LOGGED:
                _ROUTE_PROBE_LOGGED = True
                logger.info(
                    "sdpa256 route probe: cache=%s bits=%s mask=%s "
                    "kv_len=%s q_len=%s route_pending",
                    type(cache).__name__,
                    hasattr(cache, "bits"),
                    type(mask).__name__,
                    keys.shape[-2] if hasattr(keys, "shape") else "?",
                    queries.shape[-2],
                )
        if queries.shape[-2] < _SDPA256_MIN_Q_LEN:  # decode / MTP verify
            return False
        if queries.shape[-1] != HEAD_DIM:
            return False
        if keys.shape[-2] < _SDPA256_MIN_KV_LEN:
            return False
        if sinks is not None:
            return False
        # Quantized KV cache: keys/values are packed state,
        # not plain [.., kv, hd] arrays. MLX's own dispatcher detects this via
        # hasattr(cache, "bits"); let the quant-aware path handle it.
        if cache is not None and hasattr(cache, "bits"):
            return False
        mask_is_bool = isinstance(mask, mx.array) and mask.dtype == mx.bool_
        if not (
            mask is None
            or (isinstance(mask, str) and mask == "causal")
            or mask_is_bool
        ):
            return False
        n_q = queries.shape[-3]
        n_kv = keys.shape[-3]
        if n_kv <= 0 or n_q % n_kv != 0:
            return False
        return _tiled_route_required(queries, keys)
    except Exception:
        return False


def apply_sdpa256_attention_patch(min_kv_len: int = _SDPA256_MIN_KV_LEN) -> bool:
    """Monkey-patch mlx-lm's scaled_dot_product_attention for head_dim=256
    long-context prefill, and register the O(L) cost with the memory monitor."""
    global _PATCHED, _SDPA256_MIN_KV_LEN, _FORCE_TILED, _Q_TILE
    if _PATCHED:
        return False
    _SDPA256_MIN_KV_LEN = min_kv_len
    _FORCE_TILED = _parse_force_tiled_env()
    _Q_TILE = _env_int("OMLX_SDPA256_Q_TILE", _Q_TILE)

    try:
        from mlx_lm.models import base as mlx_base
    except ImportError:
        return False

    original_sdpa = mlx_base.scaled_dot_product_attention

    def patched_sdpa(
        queries,
        keys,
        values,
        cache,
        scale: float,
        mask: mx.array | None,
        sinks: mx.array | None = None,
    ) -> mx.array:
        if _should_route(queries, keys, cache, mask, sinks):
            try:
                return _flash_sdpa256(queries, keys, values, scale, mask)
            except Exception:
                logger.warning(
                    "sdpa256 prefill kernel failed; falling back to MLX SDPA",
                    exc_info=True,
                )
        return original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    mlx_base.scaled_dot_product_attention = patched_sdpa

    # Rebind already-imported model modules that did
    # `from .base import scaled_dot_product_attention` at import time. Only
    # rebind modules whose attribute IS the base function we wrapped — a model
    # that defined its own SDPA keeps it untouched (don't silently redirect a
    # model we never intended to patch).
    import sys

    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("mlx_lm.models."):
            continue
        if getattr(mod, "scaled_dot_product_attention", None) is original_sdpa:
            mod.scaled_dot_product_attention = patched_sdpa

    # mlx-vlm carries its own base SDPA (a distinct function, with its own
    # cache handling), and model modules like qwen3_5.language copy
    # the reference at import time. It needs its own capture + wrapper +
    # submodule rebind, mirroring qwen35_fa256_attention: checking mlx-vlm
    # modules against the mlx-lm original can never match, which left the VLM
    # engine on the unfused O(L^2) path and — because this patch installs
    # first — polluted the fa256 patch's "original" capture so its rebind
    # missed the VLM submodules too.
    try:
        from mlx_vlm.models import base as vlm_base
    except ImportError:
        vlm_base = None

    if vlm_base is not None:
        original_vlm_sdpa = getattr(vlm_base, "scaled_dot_product_attention", None)
        if original_vlm_sdpa is not None:

            def patched_vlm_sdpa(
                queries,
                keys,
                values,
                cache,
                scale: float,
                mask=None,
                sinks=None,
            ) -> mx.array:
                if _should_route(queries, keys, cache, mask, sinks):
                    try:
                        return _flash_sdpa256(queries, keys, values, scale, mask)
                    except Exception:
                        logger.warning(
                            "sdpa256 prefill kernel failed; falling back to "
                            "MLX SDPA",
                            exc_info=True,
                        )
                return original_vlm_sdpa(
                    queries, keys, values, cache, scale, mask, sinks
                )

            vlm_base.scaled_dot_product_attention = patched_vlm_sdpa
            for mod_name, mod in list(sys.modules.items()):
                if mod is None or not mod_name.startswith("mlx_vlm.models."):
                    continue
                if (
                    getattr(mod, "scaled_dot_product_attention", None)
                    is original_vlm_sdpa
                ):
                    mod.scaled_dot_product_attention = patched_vlm_sdpa

    # Keep the prefill memory guard in lockstep: tell the monitor head_dim 256
    # prefill no longer materializes the O(L^2) score matrix. The Q-tiled
    # kernel's true transient is bounded by the live headroom at tile-sizing
    # time (_pick_q_tile), so the estimator's one-tile charge is a floor the
    # measured EWMA terms of the guard build on top of.
    try:
        from .. import memory_monitor

        memory_monitor.register_tiled_prefill_head_dim(
            HEAD_DIM, min_kv_len=min_kv_len, kv_tile=1024
        )
    except Exception:
        logger.debug("could not register sdpa256 with memory_monitor", exc_info=True)

    _PATCHED = True
    if _FORCE_TILED is None:
        routing = "tiled only when unfused exceeds guard headroom"
    elif _FORCE_TILED:
        routing = "always tiled (OMLX_SDPA256_TILED=1)"
    else:
        routing = "never tiled (OMLX_SDPA256_TILED=0)"
    logger.info(
        "sdpa256 attention patch applied (head_dim=256 prefill, kv_len>=%d, %s)",
        min_kv_len,
        routing,
    )
    return True
