# fa256 NAX attention throughput bench. Acceptance gate: the 262K single
# point must reach >= 21 TFLOPS on M5 Pro/Max (baseline-232tps geometry:
# Hq=24, Hkv=4, D=256, qL=2048 chunks, splits=32, BK=32, dispatch budget =
# 214,391,069).
#
# Reports median-of-N kernel times per kL point; TFLOPS counted as
# 4 * qL * kL_eff * D * Hq (QK^T + PV, mul + add), where kL_eff ≈ kL for
# end-aligned causal chunks (within-chunk masking is a half-chunk effect).

import math
import os
import statistics
import sys
import time

os.environ.setdefault("OMLX_FA256_NAX_SPLITS", "64")  # M5 Pro/Max tuned (was 32)
os.environ.setdefault("OMLX_FA256_NAX_BK", "32")

import mlx.core as mx

from omlx.custom_kernels.qwen35_prefill import fast

HQ, HKV, D = 24, 4, 256
CHUNK = 2048
KLENS = [int(k) for k in os.environ.get("OMLX_BENCH_KLENS", "32768,65536,131072,262144").split(",")]
REPEATS = int(os.environ.get("OMLX_BENCH_REPEATS", "9"))
DISPATCH_BUDGET = int(
    os.environ.get("OMLX_FA256_NAX_DISPATCH_BUDGET", "214391069") or "214391069"
)
ACCEPT_TFLOPS = 21.0
SEED = 20260830


def bench_point(k_len, rng):
    rng, qk, kk, vk = mx.random.split(rng, 3 + 1)
    q = (mx.random.normal((1, HQ, CHUNK, D), key=qk) * 0.5).astype(mx.bfloat16)
    k = (mx.random.normal((1, HKV, k_len, D), key=kk) * 0.5).astype(mx.bfloat16)
    v = (mx.random.normal((1, HKV, k_len, D), key=vk) * 0.5).astype(mx.bfloat16)
    scale = 1.0 / math.sqrt(D)
    out = fast.qwen35_attn256_nax(
        q,
        k,
        v,
        scale,
        causal=True,
        dispatch_budget=DISPATCH_BUDGET,
    )
    mx.eval(out)

    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = fast.qwen35_attn256_nax(
            q,
            k,
            v,
            scale,
            causal=True,
            dispatch_budget=DISPATCH_BUDGET,
        )
        mx.eval(out)
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    flops = 4.0 * CHUNK * k_len * D * HQ
    return med, flops / med / 1e12


def main():
    rng = mx.random.key(SEED)
    print(
        f"{'kL':>8} {'qL':>6} {'ms':>9} {'TFLOPS':>8}",
        flush=True,
    )
    tf262 = None
    t0 = time.perf_counter()
    for k_len in KLENS:
        med, tf = bench_point(k_len, rng)
        if k_len == KLENS[-1]:
            tf262 = tf
        print(f"{k_len:>8} {CHUNK:>6} {med * 1e3:>9.2f} {tf:>8.2f}", flush=True)
        mx.clear_cache()
    dt = time.perf_counter() - t0

    print(f"\n262K single point: {tf262:.2f} TFLOPS ({dt:.0f}s total)")
    if tf262 < ACCEPT_TFLOPS:
        print(
            f"ACCEPTANCE FAILED: {tf262:.2f} < {ACCEPT_TFLOPS} TFLOPS",
            file=sys.stderr,
        )
        return 1
    print(f"acceptance ok (>= {ACCEPT_TFLOPS} TFLOPS)")
    return 0


if __name__ == "__main__":
    if not fast.nax_attn256_available():
        print("NAX attention kernels unavailable on this machine", file=sys.stderr)
        sys.exit(2)
    sys.exit(main())
