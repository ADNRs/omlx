# SPDX-License-Identifier: Apache-2.0
"""SSD-backed prompt-cache snapshots for the distributed rank server.

A distributed rank runs the pinned ``mlx_lm.server`` with a rank-local
in-memory prompt cache. That cache is bounded and dies with the process, and it
cannot help a model whose per-layer state is not sliceable: a ``RotatingKVCache``
overwrites its window and a gated-delta-net keeps a single recurrent state, so
the state at an interior prefix boundary is gone the moment prefill moves past
it. This store persists whole-cache snapshots at prefix boundaries to local SSD,
using MLX-LM's own ``save_prompt_cache`` / ``load_prompt_cache`` so every cache
type serialises through its declared ``state`` / ``meta_state``.

Each rank keeps its own directory holding its own layer-slice snapshots. The
keys are a hash of ``(model, prefix tokens)`` and are therefore identical across
ranks that processed the same broadcast request, while the bytes under a key are
this rank's shard alone. Eviction is a deterministic count-bounded LRU keyed on
the sequence of operations rather than a wall clock, so ranks that see the same
requests keep identical key sets without any coordination. Coordinating the
*hit* across ranks (so a disk write that failed on one rank cannot desync the
pipeline) is the caller's job and lives in the telemetry integration, which has
the collective; this module stays pure and unit-testable.
"""

from __future__ import annotations

import hashlib
import os
import struct
import tempfile
import threading
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MAX_ENTRIES_DEFAULT = 64


class PoolingCacheSnapshot:
    """Serialisation stand-in for the DeepSeek sparse-attention pool cache.

    ``save_prompt_cache`` serialises through ``state`` / ``meta_state`` and
    reconstructs via ``globals()[name].from_state`` inside
    ``mlx_lm.models.cache``. A ``PoolingCache`` breaks both directions: its
    state tuple carries ``None`` slots safetensors cannot hold, and its
    meta_state is a bare int where the metadata tree must be all strings. The
    in-process state contract is load-bearing for the paged-cache handlers, so
    rather than changing the class this stand-in presents the same cache in
    wire form (arrays-only state, with the slot layout encoded as a flag
    string), and its ``from_state`` hands back a real ``PoolingCache``, so a
    restored snapshot is indistinguishable from the cache it was taken from.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @property
    def state(self) -> tuple[Any, ...]:
        import mlx.core as mx

        kept = tuple(a for a in self._inner.state if a is not None)
        # An all-empty state must still occupy its slot in the flattened file
        # or every later cache's arrays would shift onto the wrong class. The
        # placeholder is never consumed: the flags drive consumption.
        return kept or (mx.zeros((1,)),)

    @property
    def meta_state(self) -> tuple[str, str]:
        flags = "".join("0" if a is None else "1" for a in self._inner.state)
        return (str(int(self._inner.ratio)), flags)

    @classmethod
    def from_state(cls, state: Any, meta_state: tuple[str, str]) -> Any:
        from omlx.patches.deepseek_v4.cache_extras import PoolingCache

        ratio, flags = meta_state
        cache = PoolingCache(int(ratio))
        arrays = iter(state if isinstance(state, (list, tuple)) else [state])
        # The existing state setter replays any remainder rows through
        # accumulate_windows, so the restored cache continues exactly like the
        # live one it was copied from.
        cache.state = tuple(next(arrays) if f == "1" else None for f in flags)
        return cache


class EmptyLeafSnapshot:
    """Serialisation stand-in for a cache state with unserialisable leaves.

    safetensors can hold neither a zero-size array nor a ``None`` leaf, yet
    both are legitimate cache states: a sparse-attention branch below its
    engagement length keeps an untouched rotating member whose state slices are
    zero-size, and a recurrent slot may simply not be written yet. This
    stand-in stores only the substantive arrays and records the position,
    shape and dtype of every dropped leaf, so the restored cache is exactly
    what a deepcopy of the live one would have held.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @property
    def state(self) -> tuple[Any, ...]:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        kept = tuple(
            leaf
            for _key, leaf in tree_flatten(self._inner.state)
            if leaf is not None and leaf.size > 0
        )
        # Same slot-holding placeholder as PoolingCacheSnapshot: the layout
        # below never marks it for consumption.
        return kept or (mx.zeros((1,)),)

    @property
    def meta_state(self) -> tuple[Any, ...]:
        from mlx.utils import tree_flatten

        layout = []
        for key, leaf in tree_flatten(self._inner.state):
            if leaf is None:
                layout.append(f"{key}=none")
            elif leaf.size == 0:
                shape = "x".join(str(d) for d in leaf.shape)
                layout.append(f"{key}=empty:{shape}:{leaf.dtype}")
            else:
                layout.append(f"{key}=array")
        return (type(self._inner).__name__, tuple(layout), self._inner.meta_state)

    @classmethod
    def from_state(cls, state: Any, meta_state: Any) -> Any:
        import mlx.core as mx
        import mlx_lm.models.cache as cache_module
        from mlx.utils import tree_unflatten

        inner_name, layout, inner_meta = meta_state
        arrays = iter(state if isinstance(state, (list, tuple)) else [state])
        pairs = []
        for item in layout:
            key, _, kind = item.partition("=")
            if kind == "array":
                pairs.append((key, next(arrays)))
            elif kind == "none":
                pairs.append((key, None))
            else:
                _, shape_text, dtype_text = kind.split(":")
                shape = tuple(int(d) for d in shape_text.split("x") if d)
                dtype = getattr(mx, dtype_text.rsplit(".", 1)[-1])
                pairs.append((key, mx.zeros(shape, dtype=dtype)))
        inner_cls = getattr(cache_module, inner_name)
        return inner_cls.from_state(tree_unflatten(pairs), inner_meta)


def _has_unserialisable_leaves(entry: Any) -> bool:
    from mlx.utils import tree_flatten

    return any(
        leaf is None or leaf.size == 0 for _key, leaf in tree_flatten(entry.state)
    )


def _register_snapshot_classes() -> None:
    """Make the stand-ins resolvable where ``load_prompt_cache`` looks them up.

    Both the top-level loader and ``CacheList.from_state`` resolve class names
    against ``mlx_lm.models.cache`` globals, so the names written into a
    snapshot file must exist there when any rank loads it.
    """

    import mlx_lm.models.cache as cache_module

    for snapshot_class in (PoolingCacheSnapshot, EmptyLeafSnapshot):
        name = snapshot_class.__name__
        if getattr(cache_module, name, None) is not snapshot_class:
            setattr(cache_module, name, snapshot_class)


def _wrap_for_save(cache: list[Any]) -> list[Any]:
    """Swap serialisation-hostile entries for their wire stand-ins.

    Returns a parallel list; the live cache is never touched. Models whose
    states already serialise pass through unchanged.
    """

    from mlx_lm.models.cache import CacheList

    try:
        from omlx.patches.deepseek_v4.cache_extras import PoolingCache
    except ImportError:
        pooling_class: Any = None
    else:
        pooling_class = PoolingCache

    def wrap(entry: Any) -> Any:
        if pooling_class is not None and isinstance(entry, pooling_class):
            return PoolingCacheSnapshot(entry)
        if isinstance(entry, CacheList):
            members = [wrap(m) for m in entry.caches]
            if any(m is not o for m, o in zip(members, entry.caches)):
                return CacheList(*members)
            return entry
        if _has_unserialisable_leaves(entry):
            return EmptyLeafSnapshot(entry)
        return entry

    return [wrap(entry) for entry in cache]


@dataclass
class _Entry:
    tokens: tuple[int, ...]
    filename: str
    nbytes: int


def _digest(model: Any, tokens: tuple[int, ...]) -> str:
    hasher = hashlib.sha256()
    hasher.update(repr(model).encode("utf-8"))
    hasher.update(b"\x00")
    # Fixed-width little-endian keeps the digest stable across interpreters.
    hasher.update(struct.pack(f"<{len(tokens)}q", *tokens))
    return hasher.hexdigest()


def candidate_boundaries(prompt_len: int, step: int) -> tuple[int, ...]:
    """Prefix lengths a snapshot may exist at, longest first.

    Boundaries are the ``step`` multiples at or below ``prompt_len``: the exact
    positions prefill pauses at. Keeping them aligned means a restored prefix
    always leaves the next request's prefill on the same grid, so later
    snapshots keep landing on boundaries other requests can reuse. All ranks
    derive the same list from the same broadcast prompt length, so a one-hot
    vote over these indices lines up across ranks.
    """

    if prompt_len <= 0 or step <= 0:
        return ()
    return tuple(k * step for k in range(prompt_len // step, 0, -1))


def agreed_boundary(
    candidates: tuple[int, ...],
    summed_votes: list[int],
    world_size: int,
) -> int:
    """Longest boundary present on every rank, from summed one-hot votes.

    ``candidates`` is longest-first and ``summed_votes[i]`` is how many ranks
    reported ``candidates[i]``. A boundary is taken only when all of them did, so
    a snapshot missing on any single rank is never restored; the alternative is
    one rank reusing a prefix the others recompute, which desyncs the pipeline.
    """

    for boundary, count in zip(candidates, summed_votes):
        if int(count) == world_size:
            return int(boundary)
    return 0


class SSDPromptSnapshotStore:
    """A rank-local, deterministic, count-bounded store of cache snapshots."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        max_entries: int = _MAX_ENTRIES_DEFAULT,
    ) -> None:
        self.directory = Path(directory)
        self.max_entries = max(1, int(max_entries))
        self._lock = threading.RLock()
        # Access-ordered: most-recently-used at the end. The order is advanced
        # only by put/load, both driven by the identical request stream every
        # rank sees, so eviction is the same decision on every rank.
        self._index: OrderedDict[str, _Entry] = OrderedDict()
        self._nbytes = 0
        # A cache type ``save_prompt_cache`` rejects will be rejected on every
        # boundary, so the store disables itself after the first such failure
        # rather than paying a doomed write per request. Known-hostile types
        # get a wire stand-in instead (see ``_wrap_for_save``); this flag is
        # the backstop for a type nobody has taught the store about yet.
        # Restores stay correct either way: nothing was stored.
        self._serialisable = True
        _register_snapshot_classes()
        self.directory.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        with self._lock:
            return len(self._index)

    @property
    def nbytes(self) -> int:
        with self._lock:
            return self._nbytes

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.safetensors"

    def put(self, model: Any, tokens: list[int], cache: list[Any]) -> bool:
        """Persist ``cache`` under the prefix ``tokens``. Best effort.

        Returns True when the snapshot is now on disk and indexed. A write that
        fails leaves the store unchanged rather than half-recorded, so the index
        never claims a file that is not there.
        """

        from mlx_lm.models.cache import save_prompt_cache

        if not self._serialisable:
            return False
        token_tuple = tuple(int(t) for t in tokens)
        if not token_tuple:
            return False
        cache = _wrap_for_save(cache)
        key = _digest(model, token_tuple)
        target = self._path(key)
        temporary = None
        try:
            # A ``.safetensors`` suffix matters: ``mx.save_safetensors`` appends
            # it otherwise, and the atomic rename below would miss the real file.
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{key}.", suffix=".safetensors", dir=self.directory
            )
            os.close(descriptor)
            save_prompt_cache(temporary, cache)
            size = os.path.getsize(temporary)
            os.replace(temporary, target)
        except OSError:
            # A transient disk problem: drop this snapshot, keep the store live.
            if temporary is not None:
                with suppress(OSError):
                    os.unlink(temporary)
            return False
        except Exception:
            # The cache type itself cannot be serialised. Stop trying for this
            # model so a boundary is not paid for on every request.
            if temporary is not None:
                with suppress(OSError):
                    os.unlink(temporary)
            self._serialisable = False
            return False
        with self._lock:
            previous = self._index.pop(key, None)
            if previous is not None:
                self._nbytes -= previous.nbytes
            self._index[key] = _Entry(token_tuple, target.name, size)
            self._nbytes += size
            self._index.move_to_end(key)
            self._evict_locked()
        return True

    def present_boundaries(
        self, model: Any, tokens: list[int], step: int
    ) -> tuple[int, ...]:
        """Prefix lengths this rank can restore for ``tokens``, longest first.

        Only boundaries whose file is actually on disk are reported, so a failed
        write simply omits that boundary from this rank's vote.
        """

        token_tuple = tuple(int(t) for t in tokens)
        found: list[int] = []
        with self._lock:
            for boundary in candidate_boundaries(len(token_tuple), step):
                prefix = token_tuple[:boundary]
                key = _digest(model, prefix)
                entry = self._index.get(key)
                if entry is None or entry.tokens != prefix:
                    continue
                if self._path(key).is_file():
                    found.append(boundary)
        return tuple(found)

    def load(self, model: Any, tokens: list[int], boundary: int) -> list[Any] | None:
        """Restore the snapshot for ``tokens[:boundary]`` and mark it used."""

        from mlx_lm.models.cache import load_prompt_cache

        token_tuple = tuple(int(t) for t in tokens)
        prefix = token_tuple[:boundary]
        if not prefix:
            return None
        key = _digest(model, prefix)
        with self._lock:
            entry = self._index.get(key)
            if entry is None or entry.tokens != prefix:
                return None
            path = self._path(key)
            if not path.is_file():
                self._index.pop(key, None)
                self._nbytes -= entry.nbytes
                return None
        try:
            cache = load_prompt_cache(str(path))
        except Exception:
            return None
        with self._lock:
            if key in self._index:
                self._index.move_to_end(key)
        return cache

    def _evict_locked(self) -> None:
        while len(self._index) > self.max_entries:
            key, entry = self._index.popitem(last=False)
            self._nbytes -= entry.nbytes
            with suppress(OSError):
                self._path(key).unlink()
