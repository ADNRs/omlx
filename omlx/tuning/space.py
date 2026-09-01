"""Tuner parameter space + stage definitions.

Every candidate is a flat dict {env_var: value}. Measurement levels:
  L1 = standalone kernel bench, L2 = server decode slope,
  L3 = full 262K E2E, L3P = 64K E2E proxy.
"""

from __future__ import annotations

import itertools
import os

BASE_BUDGET = 214_391_069

# ---------------------------------------------------------------- S1 kernel
# BK space = {32, 64}: the shipped metallib instantiates only bk32 (default)
# and bk64 (_bk) variants — bk16 pipelines do not exist.
# 2026-08-31 sweep result (M5 Pro): splits=64/BK=32 won (+2.3% kernel,
# +1.7% E2E) and is now the shipped default; the space below is kept for
# re-tunes. On higher-core dies (M5 Max/Ultra) the optimum may sit above
# 64 — set OMLX_TUNER_BIG_DIE=1 to extend the splits space to 96/128
# (kNaxMaxSplits env already allows up to 128; the partials slab grows
# ~800MB per doubling, fine on 64GB+ machines).
S1_SPLITS = [16, 24, 32, 40, 48, 64]
if os.environ.get("OMLX_TUNER_BIG_DIE"):
    S1_SPLITS += [96, 128]
S1_BK = [32, 64]
S1_BUDGET_MULT = [0.75, 1.0, 1.25, 1.5]


def s1_candidates() -> list[dict[str, str]]:
    out = []
    for splits, bk, mult in itertools.product(S1_SPLITS, S1_BK, S1_BUDGET_MULT):
        out.append(
            {
                "OMLX_FA256_NAX_SPLITS": str(splits),
                "OMLX_FA256_NAX_BK": str(bk),
                "OMLX_FA256_NAX_DISPATCH_BUDGET": str(int(BASE_BUDGET * mult)),
            }
        )
    return out


# ---------------------------------------------------------------- S2 decode
S2_OCC = [128 * 256, 192 * 256, 256 * 256]
S2_MIN_BLOCKS = [2, 4, 8]
S2_MAX_SPLITS = [32, 64, 128]


def s2_candidates() -> list[dict[str, str]]:
    out = []
    for occ, minb, maxs in itertools.product(S2_OCC, S2_MIN_BLOCKS, S2_MAX_SPLITS):
        out.append(
            {
                "OMLX_NAX_OCCUPANCY_THREADS": str(occ),
                "OMLX_NAX_MIN_SPLIT_BLOCKS": str(minb),
                "OMLX_NAX_MAX_SPLITS": str(maxs),
            }
        )
    return out


# ---------------------------------------------------------------- S3 E2E
def s3_candidates() -> list[dict[str, str]]:
    out = []
    for gate in (8192, 16384, 32768):
        out.append({"OMLX_QWEN35_Q8_LINEAR_MIN_TOKENS": str(gate)})
    for cap in ("6.0", "8.0", "10.0"):
        out.append({"OMLX_MX_CACHE_LIMIT_GB": cap})
    return out


# MTP depth rides model_settings, not env — handled by the L2 runner via
# --mtp-depth (writes a temp model_settings entry or uses the admin API).
S2_MTP_DEPTHS = [2, 3]

STAGES = {
    "S1": {"level": "L1", "candidates": s1_candidates, "median": 3},
    "S2": {"level": "L2", "candidates": s2_candidates, "median": 3},
    "S3": {"level": "L3P", "candidates": s3_candidates, "median": 1},
}
