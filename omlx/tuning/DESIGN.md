# oMLX Phase 4 Tuner — Design

Baseline (2026-08-31, M5 Pro/Max, Qwen3.8-27B-oQ4e-mtp):
- 262K prefill E2E: 228.6 TPS / 1160.6s (cold, user-measured)
- fa256 kernel @262K: 21.24-21.99 TF (thermal-dependent)
- decode slope: MTP on ~24.8ms/tok @2K; MTP off ~61.5ms/tok @2K

## Principle

One 262K E2E run costs ~19.3 min. The tuner therefore never grid-searches
at E2E scale. It measures at three altitudes and only promotes winners:

| Level | What | Cost | Used for |
|---|---|---|---|
| L1 | standalone fa256 kernel bench (bench_fa256_sweep, 262K point) | ~12s | splits / BK / budget |
| L2 | server decode slope (2K + 240K via prompt-cache restore) | ~1.5 min | kNaxOccupancy/MinSplit/MaxSplits, MTP depth |
| L3 | full server 262K E2E prefill | ~19.3 min | final validation only |

E2E-only knobs (q8 gate, prefill tiers, pool cap) rank on a 64K proxy
(~5 min) inside L3's runner, then the winner is validated at 262K.

## Noise control

- median-of-N (N=3 default) per candidate; fixed seed in bench scripts
- one measurement per OS process (env is read once at import — the "env
  static cache" invariant); every candidate = fresh subprocess
- thermal: L1 runs sleep 45s between candidates; L2/L3 sleep 60s
- E2E claims only ever come from cold-cache L3 runs

## Guardrails

1. Any candidate that changes dispatch geometry (splits/budget/kNax*) and
   beats baseline at its level must pass `repro_server_seq.py` (growing-kL,
   128 chunks) before promotion — single-shape benches miss the wedge
   condition (2026-08-30 incident).
2. Final config: full pytest suite + peak memory <= ceiling (48GB).
3. All candidates inherit production defaults; only the swept vars differ.

## Stages

- S1 kernel: splits {16,24,32,40,48,64} x BK {16,32,64} x budget
  {0.75x,1x,1.25x,1.5x} — L1, rank, top-3 -> repro gate
- S2 decode: OMLX_NAX_{OCCUPANCY_THREADS,MIN_SPLIT_BLOCKS,MAX_SPLITS} —
  L2 decode slope, warm-cache restores
- S3 E2E: q8 gate {8K,16K,32K}, prefill tiers variants, pool cap {6,8,10} —
  64K proxy; MTP depth {2,3} on L2
- S4 final: winner vs baseline at L3 (median-of-2) + guardrails

## Output

`tmp/tuner/results/` — one JSON per candidate + `winner.json`.
Write-back is manual (or via `search.py --write-back`): winning values
land in the same places Phase 3 defaults live (C++ defaults, env defaults
in fa256/scheduler/settings, model_settings mtp depth).
