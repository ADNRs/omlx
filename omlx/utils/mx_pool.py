# SPDX-License-Identifier: Apache-2.0
"""MLX wired buffer-cache pool sizing, shared by every server entry point.

The pool limit is set high (total GPU memory) by default so
allocator::free() never releases a Metal buffer the GPU may still be
using — kernel panics on M4 otherwise (issue #300); freed buffers leave
the pool only via mx.clear_cache(), which callers protect with
mx.synchronize().

A CAP below total memory trades that for smooth self-eviction: with the
pool uncapped, long-context prefill grows it by ~13GB, the enforcer's
soft-pressure reclaim then clears it, and the next chunk re-allocates
it — a thrash loop that flaps the footprint ±13GB per chunk, poisons
the prefill transient estimator with refill deltas, and collapses chunk
sizes to the 32-token floor (measured on a 262K bench: ~50 tok/s at
170K+ unbounded vs full-speed chunks with an 8GB cap). The cap is
configured via the ``mx_cache_limit_gb`` cache setting; the
``OMLX_MX_CACHE_LIMIT_GB`` env var overrides it for testing.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def apply_mx_cache_limit(settings_limit_gb: float = 0.0) -> int:
    """Apply the MLX buffer-cache limit; returns the limit in bytes (0 if
    the device reports no memory and nothing was set)."""
    import mlx.core as mx

    total_mem = int(mx.device_info().get("memory_size", 0) or 0)
    if total_mem <= 0:
        return 0

    cache_limit = total_mem
    limit_gb = float(settings_limit_gb or 0.0)

    raw = os.environ.get("OMLX_MX_CACHE_LIMIT_GB", "").strip()
    if raw:
        try:
            env_gb = float(raw)
            if env_gb > 0:
                limit_gb = env_gb
        except ValueError:
            logger.debug("Ignoring invalid OMLX_MX_CACHE_LIMIT_GB=%r", raw)

    if limit_gb > 0:
        cap = int(limit_gb * 1024**3)
        if 0 < cap < cache_limit:
            cache_limit = cap

    mx.set_cache_limit(cache_limit)
    if cache_limit < total_mem:
        logger.info(
            "MLX buffer-cache pool capped at %.1fGB of %.1fGB device memory",
            cache_limit / 1024**3,
            total_mem / 1024**3,
        )
    return cache_limit
