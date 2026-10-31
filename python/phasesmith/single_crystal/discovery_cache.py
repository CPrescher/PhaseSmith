"""Bounded, temporary reuse of pure Fourier proposals inside one inverse.

No files, supplied models, validation labels or cross-call state are retained.
Keys contain the exact ordered float bytes and all numerical search settings.
"""

from collections import OrderedDict
from contextvars import ContextVar
from functools import wraps

__all__ = []

_ACTIVE = ContextVar("phasesmith_discovery_reuse", default=None)


class _Cache:
    def __init__(self, max_bytes=64 * 1024**2, max_entries=512):
        self.rows = OrderedDict()
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.bytes = self.peak_bytes = self.hits = self.misses = 0

    def get(self, key):
        if key not in self.rows:
            self.misses += 1
            return None
        self.hits += 1
        self.rows.move_to_end(key)
        vectors, scores, layout, _ = self.rows[key]
        return vectors.copy(), scores.copy(), dict(layout)

    def put(self, key, result):
        vectors, scores, layout = result
        size = len(key[-1]) + vectors.nbytes + scores.nbytes + 512
        if size > self.max_bytes:
            return
        if key in self.rows:
            self.bytes -= self.rows.pop(key)[-1]
        while self.rows and (
            self.bytes + size > self.max_bytes or len(self.rows) >= self.max_entries
        ):
            _, row = self.rows.popitem(last=False)
            self.bytes -= row[-1]
        v, s = vectors.copy(), scores.copy()
        v.flags.writeable = s.flags.writeable = False
        self.rows[key] = v, s, dict(layout), size
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)


def scope(function):
    """Create and discard a separate cache for every ensemble invocation."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _ACTIVE.set(_Cache())
        try:
            return function(*args, **kwargs)
        finally:
            _ACTIVE.reset(token)

    return wrapped


def fourier(function):
    """Reuse only identical ordered inputs and numerical settings."""

    @wraps(function)
    def wrapped(engine, points, options):
        cache = _ACTIVE.get()
        if cache is None:
            return function(engine, points, options)
        key = (
            type(engine),
            options.grid_size,
            options.min_length,
            options.max_length,
            options.fft_peaks,
            options.native_fourier_refinement,
            points.shape,
            points.dtype.str,
            points.tobytes(order="C"),
        )
        cached = cache.get(key)
        if cached is not None:
            return cached
        result = function(engine, points, options)
        cache.put(key, result)
        return result

    return wrapped


def statistics():
    """Deterministic bookkeeping, without candidate or validation information."""
    cache = _ACTIVE.get()
    if cache is None:
        return None
    return dict(
        hits=cache.hits,
        misses=cache.misses,
        entries=len(cache.rows),
        retained_bytes=cache.bytes,
        peak_bytes=cache.peak_bytes,
        max_bytes=cache.max_bytes,
        max_entries=cache.max_entries,
        scope="Temporary exact ordered-input reuse inside one inverse; no files",
    )
