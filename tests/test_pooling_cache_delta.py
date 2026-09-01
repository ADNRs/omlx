# SPDX-License-Identifier: Apache-2.0
"""Regression tests for block-delta PoolingCache persistence."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from omlx.cache.paged_cache import BlockTable, PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.pooling_delta import (
    POOLING_CACHE_DELTA_CLASS,
    POOLING_CACHE_DELTA_FORMAT_VERSION,
    compact_pooling_cache_snapshot,
)
from omlx.cache.prefix_cache import BlockAwarePrefixCache

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

BLOCK_SIZE = 4
POOL_RATIO = 4
POOL_DIM = 8


class _MockModel:
    def __init__(self):
        self.layers = [MagicMock()]

    @property
    def args(self):
        args = MagicMock()
        args.num_hidden_layers = 1
        return args


def _pooling_layer(token_count: int, *, include_overlap_state: bool = False) -> dict:
    pooled_count = token_count // POOL_RATIO
    pooled = mx.arange(pooled_count * POOL_DIM, dtype=mx.float32).reshape(
        1, pooled_count, POOL_DIM
    )
    mx.eval(pooled)
    state = (None, None, pooled)
    if include_overlap_state:
        prev_win_kv = mx.arange(POOL_RATIO * POOL_DIM, dtype=mx.float32).reshape(
            1, 1, POOL_RATIO, POOL_DIM
        )
        prev_win_gate = prev_win_kv + 1000
        mx.eval(prev_win_kv, prev_win_gate)
        state = (*state, prev_win_kv, prev_win_gate)
    return {
        "state": [state],
        "meta_state": (["PoolingCache"], [POOL_RATIO]),
        "sub_class_names": ["PoolingCache"],
        "class_name": "CacheList",
        "cache_type": "CacheList",
    }


def _delta_pooling_layer(
    token_count: int, *, include_overlap_state: bool = False
) -> list[dict]:
    layers = [_pooling_layer(token_count, include_overlap_state=include_overlap_state)]
    compact_pooling_cache_snapshot(layers, token_count, BLOCK_SIZE)
    return layers


def _make_cache(tmp_path, *, hot_cache_only: bool = True):
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    paged = PagedCacheManager(
        block_size=BLOCK_SIZE,
        max_blocks=100,
        model_name="pooling-delta-test",
        initial_blocks=100,
    )
    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path / "ssd",
        max_size_bytes=100 * 1024**2,
        hot_cache_max_bytes=10 * 1024**2,
        hot_cache_only=hot_cache_only,
        expected_model_name="pooling-delta-test",
    )
    return (
        BlockAwarePrefixCache(
            model=_MockModel(),
            paged_cache_manager=paged,
            paged_ssd_cache_manager=ssd,
        ),
        ssd,
    )


def _wait_for_pending_writes(ssd: PagedSSDCacheManager) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with ssd._pending_write_hashes_lock:
            if not ssd._pending_write_hashes:
                return
        time.sleep(0.01)
    pytest.fail("Timed out waiting for SSD cache writes")


def test_compaction_is_linear_and_preserves_absolute_ranges():
    full_rows = 0
    delta_rows = 0
    for boundary in range(BLOCK_SIZE, 101 * BLOCK_SIZE, BLOCK_SIZE):
        layers = [_pooling_layer(boundary)]
        full_rows += layers[0]["state"][0][2].shape[1]
        compact_pooling_cache_snapshot(layers, boundary, BLOCK_SIZE)
        state = layers[0]["state"][0]
        start, end = layers[0]["pooling_delta_ranges"]["0"]
        assert (start, end) == (
            (boundary - BLOCK_SIZE) // POOL_RATIO,
            boundary // POOL_RATIO,
        )
        assert state[2].shape[1] == end - start == 1
        delta_rows += state[2].shape[1]

    assert full_rows == 5050
    assert delta_rows == 100


def test_mismatched_pool_length_keeps_legacy_full_snapshot():
    layers = [_pooling_layer(2 * BLOCK_SIZE)]
    compact_pooling_cache_snapshot(layers, BLOCK_SIZE, BLOCK_SIZE)
    assert "pooling_delta_ranges" not in layers[0]
    assert layers[0]["state"][0][2].shape[1] == 2


def test_compaction_preserves_overlap_state():
    layers = [_pooling_layer(2 * BLOCK_SIZE, include_overlap_state=True)]
    original = layers[0]["state"][0]
    compact_pooling_cache_snapshot(layers, 2 * BLOCK_SIZE, BLOCK_SIZE)
    compacted = layers[0]["state"][0]

    assert len(compacted) == 5
    assert compacted[2].shape[1] == 1
    assert compacted[3] is original[3]
    assert compacted[4] is original[4]


def test_boundary_snapshot_metadata_roundtrip(tmp_path):
    from omlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore

    store = BoundarySnapshotSSDStore(tmp_path)
    layers = [_pooling_layer(2 * BLOCK_SIZE)]
    saved = store.save(
        "req-delta",
        2 * BLOCK_SIZE,
        [object()],
        lambda _: (layers, None),
        block_size=BLOCK_SIZE,
    )
    assert saved is True
    restored = store.load("req-delta", 2 * BLOCK_SIZE)
    assert restored is not None
    assert restored[0]["pooling_delta_ranges"] == {"0": [1, 2]}
    assert restored[0]["state"][0][2].shape[1] == 1
    store.shutdown()


