# SPDX-License-Identifier: Apache-2.0
"""The telemetry patch must capture prompt-cache boundaries to SSD during
prefill and restore the longest prefix on a later miss."""

from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from omlx.cluster.telemetry import install_server_telemetry

STEP = 4
MODEL = "model-key"


class _Marker:
    def update(self, phase, **extra):
        return None


def _kv(steps=2):
    cache = KVCache()
    for _ in range(steps):
        k = mx.random.normal((1, 2, 1, 4))
        v = mx.random.normal((1, 2, 1, 4))
        cache.update_and_fetch(k, v)
    return [cache]


def _fake_stream_generate(*_args, **kwargs):
    """Stand in for MLX-LM: fire the progress callback at each prefill step."""

    callback = kwargs.get("prompt_progress_callback")
    total = len(kwargs.get("prompt", []))
    processed = 0
    while processed < total:
        processed = min(processed + STEP, total)
        if callback is not None:
            callback(processed, total)
    return
    yield  # make this a generator, matching stream_generate


def _install(tmp_path, monkeypatch):
    import mlx_lm.server as mlx_server

    monkeypatch.setattr(mlx_server, "stream_generate", _fake_stream_generate)
    return mlx_server, install_server_telemetry(
        _Marker(),
        ssd_cache_dir=str(tmp_path),
        prefill_step_size=STEP,
    )


def test_prefill_boundaries_are_snapshotted_to_ssd(tmp_path, monkeypatch):
    mlx_server, ctx = _install(tmp_path, monkeypatch)
    with ctx:
        cache = mlx_server.LRUPromptCache()
        tokens = list(range(8))  # base 0, boundaries at 4 and 8
        cache.prefetch_nearest_cache(MODEL, tokens)
        list(
            mlx_server.stream_generate(
                model=None,
                prompt=tokens,
                prompt_cache=_kv(),
                prompt_progress_callback=None,
            )
        )
        snapshots = sorted(tmp_path.glob("*.safetensors"))

    assert len(snapshots) == 2  # one at 4 tokens, one at 8


def test_a_later_miss_restores_the_longest_ssd_prefix(tmp_path, monkeypatch):
    mlx_server, ctx = _install(tmp_path, monkeypatch)
    with ctx:
        cache = mlx_server.LRUPromptCache()
        first = list(range(8))
        cache.prefetch_nearest_cache(MODEL, first)
        list(
            mlx_server.stream_generate(
                model=None,
                prompt=first,
                prompt_cache=_kv(),
                prompt_progress_callback=None,
            )
        )

        # A new request that shares the first eight tokens misses in memory and
        # is served the boundary-8 snapshot from SSD, leaving only the tail.
        longer = list(range(12))
        fresh = mlx_server.LRUPromptCache()
        restored, rest = fresh.prefetch_nearest_cache(MODEL, longer)

    assert restored is not None
    assert rest == [8, 9, 10, 11]


def test_the_fetch_path_alone_carries_the_ssd_tier(tmp_path, monkeypatch):
    """A guardless deployment never calls the preflight lookup; MLX-LM only
    calls fetch_nearest_cache, which must still capture and restore."""

    mlx_server, ctx = _install(tmp_path, monkeypatch)
    with ctx:
        cache = mlx_server.LRUPromptCache()
        first = list(range(8))
        cache.fetch_nearest_cache(MODEL, first)
        list(
            mlx_server.stream_generate(
                model=None,
                prompt=first,
                prompt_cache=_kv(),
                prompt_progress_callback=None,
            )
        )

        fresh = mlx_server.LRUPromptCache()
        restored, rest = fresh.fetch_nearest_cache(MODEL, list(range(12)))

    assert restored is not None
    assert rest == [8, 9, 10, 11]


def test_an_unaligned_base_deposits_no_snapshot(tmp_path, monkeypatch):
    """Only aligned boundaries are reusable, so an off-grid base writes nothing."""

    mlx_server, ctx = _install(tmp_path, monkeypatch)
    with ctx:
        cache = mlx_server.LRUPromptCache()
        full = list(range(10))
        cache.prefetch_nearest_cache(MODEL, full)
        # Pretend three tokens were already cached: base 3 keeps every boundary
        # off the step-4 grid.
        list(
            mlx_server.stream_generate(
                model=None,
                prompt=full[3:],
                prompt_cache=_kv(),
                prompt_progress_callback=None,
            )
        )

    assert sorted(tmp_path.glob("*.safetensors")) == []


def test_the_patch_restores_stream_generate_on_exit(tmp_path, monkeypatch):
    mlx_server, ctx = _install(tmp_path, monkeypatch)
    with ctx:
        assert mlx_server.stream_generate is not _fake_stream_generate
    assert mlx_server.stream_generate is _fake_stream_generate


def test_teardown_removes_the_snapshot_directory(tmp_path, monkeypatch):
    """Snapshots are process-lifetime: nothing may outlive the serving span."""

    mlx_server, ctx = _install(tmp_path, monkeypatch)
    with ctx:
        cache = mlx_server.LRUPromptCache()
        tokens = list(range(8))
        cache.fetch_nearest_cache(MODEL, tokens)
        list(
            mlx_server.stream_generate(
                model=None,
                prompt=tokens,
                prompt_cache=_kv(),
                prompt_progress_callback=None,
            )
        )
        assert sorted(tmp_path.glob("*.safetensors"))
    assert not tmp_path.exists()


class _FakeBaseBatchGenerator:
    """Report prefill progress at each step boundary, like BatchGenerator."""

    def __init__(self, *_args, **_kwargs):
        self._call = 0

    def insert_segments(self, *_args, **_kwargs):
        return [0]

    def remove(self, _uids):
        return None

    def next(self):
        self._call += 1
        total = 3 * STEP
        if self._call <= 3:
            processed = self._call * STEP
            return (
                [
                    SimpleNamespace(
                        uid=0,
                        progress=(processed, total),
                        end_of_prompt=processed == total,
                    )
                ],
                [],
            )
        return ([], [])

    def extract_cache(self, uids):
        return {uid: (_kv(), None) for uid in uids}


def test_batched_prefill_snapshots_at_each_boundary(tmp_path, monkeypatch):
    """The path these models actually use: BatchGenerator, not stream_generate."""

    import mlx_lm.server as mlx_server

    monkeypatch.setattr(mlx_server, "BatchGenerator", _FakeBaseBatchGenerator)
    with install_server_telemetry(
        _Marker(), ssd_cache_dir=str(tmp_path), prefill_step_size=STEP
    ):
        tokens = list(range(3 * STEP))
        # Setting snapshot context is the prompt cache's job on the same thread.
        mlx_server.LRUPromptCache().prefetch_nearest_cache(MODEL, tokens)
        batch = mlx_server.BatchGenerator()
        batch.insert_segments(segments=[[tokens]], all_tokens=[[]])
        while True:
            prompt_responses, gen_responses = batch.next()
            if not prompt_responses and not gen_responses:
                break
        snapshots = sorted(tmp_path.glob("*.safetensors"))

    assert len(snapshots) == 3  # STEP, 2*STEP, 3*STEP


def test_batched_capture_restores_on_a_later_batched_miss(tmp_path, monkeypatch):
    import mlx_lm.server as mlx_server

    monkeypatch.setattr(mlx_server, "BatchGenerator", _FakeBaseBatchGenerator)
    with install_server_telemetry(
        _Marker(), ssd_cache_dir=str(tmp_path), prefill_step_size=STEP
    ):
        first = list(range(3 * STEP))
        mlx_server.LRUPromptCache().prefetch_nearest_cache(MODEL, first)
        batch = mlx_server.BatchGenerator()
        batch.insert_segments(segments=[[first]], all_tokens=[[]])
        while True:
            prompt_responses, gen_responses = batch.next()
            if not prompt_responses and not gen_responses:
                break

        # A fresh request sharing 2*STEP tokens misses in memory and is served
        # the boundary snapshot from SSD.
        longer = list(range(3 * STEP)) + [999, 998]
        fresh = mlx_server.LRUPromptCache()
        restored, rest = fresh.prefetch_nearest_cache(MODEL, longer)

    assert restored is not None
    assert rest == [999, 998]
