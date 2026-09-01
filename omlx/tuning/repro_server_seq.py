# GPU wedge gate: drive the fa256 NAX attention through the exact shape
# sequence a 262K server prefill produces — 128 chunks of 2048 tokens with
# monotonically growing kL — and verify outputs against a float32 reference
# at checkpoints along the way.
#
# The historical wedge only appeared on increasing-kL sequences (single-shape
# benches passed), so the sequence itself is the test. Any wrong result,
# Metal fault, or hang here fails the gate.
#
# Production geometry: Hq=24, Hkv=4, D=256, splits=32, BK=32,
# dispatch budget = 214,391,069 (server-calibrated).

import math
import os
import sys
import time

os.environ.setdefault("OMLX_FA256_NAX_SPLITS", "32")
os.environ.setdefault("OMLX_FA256_NAX_BK", "32")

import mlx.core as mx

from omlx.custom_kernels.qwen35_prefill import fast

HQ, HKV, D = 24, 4, 256
CHUNK = 2048
N_CHUNKS = 128
DISPATCH_BUDGET = 214_391_069
CHECK_EVERY = 16  # reference-check cadence (1-indexed chunk numbers)
RTOL = 0.05
SEED = 20260830


REF_KV_BLOCK = 16_384


def reference(q, k, v, scale):
    """fp32 unfused reference with END-aligned causal (MLX convention).

    Key axis is walked in blocks with an online-softmax accumulation so the
    scores matrix never exceeds one block (a full fp32 scores tensor at
    kL=262K would be ~45 GB, past the per-buffer cap).
    """
    q_len, k_len = q.shape[2], k.shape[2]
    group = q.shape[1] // k.shape[1]
    qf = q.astype(mx.float32)
    qr = qf.reshape(q.shape[0], k.shape[1], group, q_len, D)
    off = k_len - q_len
    qp = mx.arange(off, off + q_len).reshape(q_len, 1)

    m = mx.full((q.shape[0], k.shape[1], group, q_len, 1), mx.array(-mx.inf))
    l = mx.zeros((q.shape[0], k.shape[1], group, q_len, 1))
    acc = mx.zeros((q.shape[0], k.shape[1], group, q_len, D))
    for s0 in range(0, k_len, REF_KV_BLOCK):
        s1 = min(s0 + REF_KV_BLOCK, k_len)
        kb = k.astype(mx.float32)[:, :, s0:s1, :].reshape(
            k.shape[0], k.shape[1], 1, s1 - s0, D
        )
        vb = v.astype(mx.float32)[:, :, s0:s1, :].reshape(
            v.shape[0], v.shape[1], 1, s1 - s0, D
        )
        s = qr @ mx.swapaxes(kb, -1, -2) * scale
        kp = mx.broadcast_to(mx.arange(s0, s1), (q_len, s1 - s0))
        s = mx.where(kp[None, None, None, :, :] > qp[None, None, None], float("-inf"), s)
        m_b = mx.max(s, axis=-1, keepdims=True)
        m_new = mx.maximum(m, m_b)
        l = l * mx.exp(m - m_new) + mx.sum(mx.exp(s - m_new), axis=-1, keepdims=True)
        acc = acc * mx.exp(m - m_new) + mx.exp(s - m_new) @ vb
        m = m_new
    return (acc / l).reshape(q.shape[0], q.shape[1], q_len, D)


def check_close(out, ref, tol):
    """Repo-test style: global-scale absolute diff."""
    diff = mx.abs(out.astype(mx.float32) - ref).max().item()
    scale_ref = mx.abs(ref).max().item()
    return diff <= tol * max(scale_ref, 1e-6), diff, scale_ref


def main():
    rng = mx.random.key(SEED)
    scale = 1.0 / math.sqrt(D)
    t0 = time.perf_counter()
    worst = 0.0
    for chunk in range(1, N_CHUNKS + 1):
        k_len = chunk * CHUNK
        rng, qk, kk, vk = mx.random.split(rng, 4)
        q = (mx.random.normal((1, HQ, CHUNK, D), key=qk) * 0.5).astype(mx.bfloat16)
        k = (mx.random.normal((1, HKV, k_len, D), key=kk) * 0.5).astype(mx.bfloat16)
        v = (mx.random.normal((1, HKV, k_len, D), key=vk) * 0.5).astype(mx.bfloat16)
        out = fast.qwen35_attn256_nax(
            q,
            k,
            v,
            scale,
            causal=True,
            dispatch_budget=DISPATCH_BUDGET,
        )
        if chunk == 1 or chunk % CHECK_EVERY == 0 or chunk == N_CHUNKS:
            ref = reference(q, k, v, scale)
            ok, diff, scale_ref = check_close(out, ref, RTOL)
            rel = diff / max(scale_ref, 1e-6)
            worst = max(worst, rel)
            status = "ok" if ok else "FAIL"
            print(
                f"chunk {chunk:3d}/128  kL={k_len:>6}  diff={diff:.4f} "
                f"(x{rel:.4f} of scale)  {status}",
                flush=True,
            )
            if not ok:
                print(f"WEDGE GATE FAILED at chunk {chunk}", file=sys.stderr)
                return 1
        else:
            mx.eval(out)
        if chunk % 32 == 0:
            mx.clear_cache()
    dt = time.perf_counter() - t0
    print(
        f"repro_server_seq: {N_CHUNKS} chunks passed, worst rel_err={worst:.4f}, "
        f"{dt:.1f}s total"
    )
    return 0


if __name__ == "__main__":
    if not fast.nax_attn256_available():
        print("NAX attention kernels unavailable on this machine", file=sys.stderr)
        sys.exit(2)
    sys.exit(main())
