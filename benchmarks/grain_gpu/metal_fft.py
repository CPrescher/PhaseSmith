"""Analysis-only Metal FFT/maxima replacement; production Rust remains unchanged.

CPU float64 prepares deposition coordinates/weights and exact radial intervals.
Metal accumulates each voxel in original contribution order, computes the float32
forward FFT, and detects 26-neighbor maxima. CPU sorts only compact peak records.
"""

from itertools import product
from time import perf_counter

import mlx.core as mx
import numpy as np

_DEPOSIT = mx.fast.metal_kernel(
    name="grain_deposit_ordered",
    input_names=["keys", "starts", "weights", "size"],
    output_names=["grid"],
    source="""
    uint t = thread_position_in_grid.x;
    if (t >= size[0]) return;
    float value = 0.0f;
    for (uint j = starts[t]; j < starts[t+1]; ++j) value = value + weights[j];
    grid[keys[t]] = value;
    """,
    compile_options={"math_mode": "safe"},
)

_MAXIMA = mx.fast.metal_kernel(
    name="grain_maxima",
    input_names=["field", "lower", "upper", "adjacent", "dims"],
    output_names=["indices", "count"],
    atomic_outputs=True,
    source="""
    int W = dims[0], Z = dims[1], NZ = dims[2];
    uint i = thread_position_in_grid.x;
    if (i >= W*W*Z) return;
    uint z = i % Z, xy = i / Z, y = xy % W, x = xy / W;
    if (z < lower[xy] || z >= upper[xy]) return;
    float s = field[i];
    if (!isfinite(s)) return;
    for (int dx=-1; dx<=1; ++dx) {
      int xx = adjacent[x*3+dx+1];
      if (xx < 0) continue;
      for (int dy=-1; dy<=1; ++dy) {
        int yy = adjacent[y*3+dy+1];
        if (yy < 0) continue;
        uint other = xx*W+yy;
        for (int dz=-1; dz<=1; ++dz) {
          uint zz = clamp(int(z)+dz, 0, NZ-1);
          if (zz < Z && zz >= lower[other] && zz < upper[other] && field[other*Z+zz] > s) return;
        }
      }
    }
    uint slot = atomic_fetch_add_explicit(&count[0], 1u, memory_order_relaxed);
    atomic_store_explicit(&indices[slot], i, memory_order_relaxed);
    """,
    compile_options={"math_mode": "safe"},
)


def contributions(q, n, step):
    """Stable voxel grouping preserves dx,dy,dz,point float32 addition order."""
    position = q / step
    lower = np.floor(position).astype(np.int64)
    fraction = position - lower
    ids, weights = [], []
    for off in product((0, 1), repeat=3):
        ix = (lower + off) % n
        w = np.where(off, fraction, 1 - fraction)
        ids.append(((ix[:, 0] * n + ix[:, 1]) * n + ix[:, 2]).astype(np.uint32))
        weights.append(((w[:, 0] * w[:, 1]) * w[:, 2]).astype(np.float32))
    ids, weights = np.concatenate(ids), np.concatenate(weights)
    order = np.argsort(ids, kind="stable")
    ids, weights = ids[order], weights[order]
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)].astype(np.uint32)
    return ids[starts[:-1]], starts, weights


def radial_intervals(n, step, minimum, maximum):
    """Binary search compares original float64 (fx²+fy²)+fz² at boundaries."""
    spacing = 1.0 / (n * step)
    freq = np.arange(n, dtype=np.float64)
    freq[n // 2 :] -= n
    freq *= spacing
    xy = freq[:, None] ** 2 + freq[None, :] ** 2
    zz = (np.arange(n // 2 + 1) * spacing) ** 2

    def bound(value, right):
        lo = np.zeros((n, n), dtype=np.int32)
        hi = np.full((n, n), len(zz), dtype=np.int32)
        while np.any(lo < hi):
            mid = (lo + hi) // 2
            v = xy + zz[np.minimum(mid, len(zz) - 1)]
            smaller = v <= value if right else v < value
            active = lo < hi
            lo = np.where(active & smaller, mid + 1, lo)
            hi = np.where(active & ~smaller, mid, hi)
        return lo.astype(np.uint32).reshape(-1)

    return bound(minimum * minimum, False), bound(maximum * maximum, True)


class MetalFFT:
    def __init__(self):
        if not mx.metal.is_available():
            raise RuntimeError("Metal GPU unavailable")
        self.records = []

    def windowed_maxima(self, q, n, step, minimum, maximum, limit, *, debug=False):
        start = perf_counter()
        q = np.array(q, dtype=np.float64, order="C", copy=True)
        if (
            q.ndim != 2
            or q.shape[1] != 3
            or not len(q)
            or not np.isfinite(q).all()
            or not isinstance(n, int)
            or n % 2
            or not 16 <= n <= 256
            or not np.isfinite(step)
            or step <= 0
            or np.max(np.abs(q)) / step >= n / 2 - 1
            or not np.isfinite([minimum, maximum]).all()
            or not 0 < minimum <= maximum
            or not isinstance(limit, int)
            or not 1 <= limit <= 1000
        ):
            raise ValueError("invalid bounded FFT input")
        if not np.isfinite(1 / (n * step)):
            raise ValueError("nonfinite direct-grid spacing")
        keys, starts, weights = contributions(q, n, step)
        lower, upper = radial_intervals(n, step, minimum, maximum)
        spacing = 1 / (n * step)
        freq = np.arange(n, dtype=np.float64)
        freq[n // 2 :] -= n
        freq *= spacing
        axes = np.flatnonzero(freq**2 <= maximum**2).astype(np.int32)
        zlen = int(np.sum((np.arange(n // 2 + 1) * spacing) ** 2 <= maximum**2))
        width = len(axes)
        inverse = np.full(n, -1, dtype=np.int32)
        inverse[axes] = np.arange(width)
        adjacent = inverse[(axes[:, None] + np.arange(-1, 2)) % n].reshape(-1)
        selection = (axes[:, None] * n + axes).reshape(-1)
        lower, upper = lower[selection], upper[selection]
        columns = np.unique(keys // n)
        compact_keys = (np.searchsorted(columns, keys // n) * n + keys % n).astype(np.uint32)
        column_map = np.full(n * n, len(columns), dtype=np.int32)
        column_map[columns] = np.arange(len(columns))
        prepared = perf_counter()
        grid = _DEPOSIT(
            inputs=[
                mx.array(compact_keys),
                mx.array(starts),
                mx.array(weights),
                mx.array([len(keys)], dtype=mx.uint32),
            ],
            grid=(len(keys), 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(len(columns), n)],
            output_dtypes=[mx.float32],
            init_value=0,
        )[0]
        # Explicit z -> y -> x order, matching the CPU transform axes.
        field = mx.fft.rfft(grid, axis=1)[:, :zlen]
        field = mx.concatenate([field, mx.zeros((1, zlen), dtype=mx.complex64)], axis=0)
        field = mx.take(field, mx.array(column_map), axis=0).reshape(n, n, zlen)
        field = mx.fft.fft(field, axis=1)
        field = mx.take(field, mx.array(axes), axis=1)
        field = mx.fft.fft(field, axis=0).real
        field = mx.take(field, mx.array(axes), axis=0)
        indices, count = _MAXIMA(
            inputs=[
                field,
                mx.array(lower),
                mx.array(upper),
                mx.array(adjacent),
                mx.array([width, zlen, n // 2 + 1], dtype=mx.int32),
            ],
            grid=(field.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(field.size,), (1,)],
            output_dtypes=[mx.uint32, mx.uint32],
            init_value=0,
        )
        # Materialization and GPU completion are part of every measured call.
        total = int(count.item())
        ids = np.array(indices[:total])
        values = np.array(mx.take(field.reshape(-1), mx.array(ids)))
        finished_gpu = perf_counter()
        original_ids = (axes[ids // (width * zlen)] * n + axes[(ids // zlen) % width]) * (
            n // 2 + 1
        ) + ids % zlen
        order = np.lexsort((original_ids, -values))[:limit]
        ids, values = ids[order], values[order]
        vectors = np.column_stack(
            (
                freq[axes[ids // (width * zlen)]],
                freq[axes[(ids // zlen) % width]],
                (ids % zlen) * spacing,
            )
        )
        self.records.append(
            dict(
                points=len(q),
                grid=n,
                prepare_seconds=prepared - start,
                gpu_submit_wait_compact_seconds=finished_gpu - prepared,
                total_seconds=perf_counter() - start,
                total_maxima=total,
            )
        )
        if debug:
            dense_grid = np.zeros((n * n, n), dtype=np.float32)
            dense_grid[columns] = np.array(grid)
            return vectors, values, total, dense_grid.reshape(n, n, n), np.array(field), axes
        return vectors, values, total
