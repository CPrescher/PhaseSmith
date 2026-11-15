"""Analysis-only hybrid row search: exact Rust geometry, Metal integer sorting.

CPU retains normalization, direction voting, float64 projection keys, precise
longitudinal ordering, phase selection, cache policy, and residual iterations.
Metal sorts and compacts batches of the initial full-population row projections.
"""

import ctypes
import json
import os
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

_COMPACT = mx.fast.metal_kernel(
    name="row_compact_segments",
    input_names=["keys", "dims"],
    output_names=["locations", "count"],
    atomic_outputs=True,
    source="""
    uint i = thread_position_in_grid.x;
    uint N=dims[0], total=dims[1], keep=dims[2];
    if (i>=total) return;
    uint j=i%N, base=i-j, key=keys[i];
    bool eligible=false;
    for(uint off=0;off<keep;++off) {
        if(j>=off && j-off+keep<=N && keys[base+j-off]==key && keys[base+j-off+keep-1]==key) {
            eligible=true; break;
        }
    }
    if(eligible) {
        uint slot=atomic_fetch_add_explicit(&count[0],1u,memory_order_relaxed);
        atomic_store_explicit(&locations[slot],i,memory_order_relaxed);
    }
    """,
    compile_options={"math_mode": "safe"},
)


_PROJECT = mx.fast.metal_kernel(
    name="row_project_screened",
    input_names=["points", "vectors", "axes", "dims", "epsilon"],
    output_names=["keys", "uncertain", "count"],
    atomic_outputs=True,
    source="""
    uint i=thread_position_in_grid.x;
    uint N=dims[0], B=dims[1], G=dims[2];
    if(i>=N*B) return;
    uint d=i/N, j=i%N, k=axes[d];
    uint aaxis=(k==0)?1:0, baxis=(k==2)?1:2;
    float t=points[j*3+k]/vectors[d*3+k];
    float a=points[j*3+aaxis]-vectors[d*3+aaxis]*t;
    float b=points[j*3+baxis]-vectors[d*3+baxis]*t;
    float e=epsilon[0], width=epsilon[1];
    float sa0=(a+0.0f*e)/width, sa1=(a+e)/width;
    float sb0=(b+0.0f*e)/width, sb1=(b+e)/width;
    int a0=int(floor(sa0)), a1=int(floor(sa1));
    int b0=int(floor(sb0)), b1=int(floor(sb1));
    int bias=int(dims[3]), side=int(dims[4]);
    atomic_store_explicit(&keys[(d*G)*N+j],uint((a0+bias)*side+b0+bias),memory_order_relaxed);
    atomic_store_explicit(&keys[(d*G+1)*N+j],uint((a1+bias)*side+b1+bias),memory_order_relaxed);
    if(G==4) {
        atomic_store_explicit(&keys[(d*G+2)*N+j],uint((a1+bias)*side+b0+bias),memory_order_relaxed);
        atomic_store_explicit(&keys[(d*G+3)*N+j],uint((a0+bias)*side+b1+bias),memory_order_relaxed);
    }
    if(abs(sa0-rint(sa0))<=0.002f || abs(sa1-rint(sa1))<=0.002f ||
       abs(sb0-rint(sb0))<=0.002f || abs(sb1-rint(sb1))<=0.002f) {
        uint slot=atomic_fetch_add_explicit(&count[0],1u,memory_order_relaxed);
        atomic_store_explicit(&uncertain[slot],i,memory_order_relaxed);
    }
    """,
    compile_options={"math_mode": "safe"},
)


def exact_projection(points, vectors, grids, epsilon):
    axes = np.argmax(abs(vectors), axis=1)
    other0 = np.where(axes == 0, 1, 0)
    other1 = np.where(axes == 2, 1, 2)
    di = np.arange(len(vectors))
    t = points[:, axes].T / vectors[di, axes, None]
    a = points[:, other0].T - vectors[di, other0, None] * t
    b = points[:, other1].T - vectors[di, other1, None] * t
    width = 2 * epsilon
    bias = int(np.ceil(2 / width)) + 2
    side = 2 * bias + 1
    keys = []
    for ox, oy in [(0, 0), (1, 1), (1, 0), (0, 1)][:grids]:
        aa = np.floor((a + ox * epsilon) / width).astype(np.int64)
        bb = np.floor((b + oy * epsilon) / width).astype(np.int64)
        keys.append(((aa + bias) * side + bb + bias).astype(np.uint32))
    return np.stack(keys, axis=1).reshape(len(vectors) * grids, len(points))


_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_double,
)


class MetalRows:
    def __init__(self, library=None, batch_key_elements=8_388_608, gpu_projection=False):
        if not mx.metal.is_available():
            raise RuntimeError("Metal unavailable")
        if library is None:
            library = Path(
                os.environ.get(
                    "PHASESMITH_ROW_GPU_LIBRARY",
                    "/Users/clemens/Codex/2026-11-12/phasesmith-row-gpu/bridge/target/release/"
                    "libphasesmith_row_gpu_experiment.dylib",
                )
            )
        self.lib = ctypes.CDLL(str(library))
        self.lib.row_gpu_run.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_void_p,
        ]
        self.lib.row_gpu_run.restype = ctypes.c_void_p
        self.lib.row_gpu_free.argtypes = [ctypes.c_void_p]
        self.callback = _CALLBACK(self._sort_callback)
        self.records = []
        self.error = None
        self.batch_key_elements = batch_key_elements
        self.lib.row_gpu_set_batch.argtypes = [ctypes.c_size_t]
        self.lib.row_gpu_set_projection.argtypes = [ctypes.c_bool]
        self.gpu_projection = gpu_projection
        self.projection_audits = []
        self.audit_projection = False

    def compact(self, keys, keep):
        if not isinstance(keys, mx.array):
            keys = mx.array(np.ascontiguousarray(keys, dtype=np.uint32))
        if keys.ndim != 2 or not keys.shape[1] or not 1 <= keep <= 100:
            raise ValueError("requires (rows,points) integer keys and keep in 1..100")
        n = keys.shape[1]
        gpu = mx.array(keys)
        order = mx.argsort(gpu, axis=1)
        sorted_keys = mx.take_along_axis(gpu, order, axis=1)
        locations, count = _COMPACT(
            inputs=[sorted_keys, mx.array([n, keys.size, keep], dtype=mx.uint32)],
            grid=(keys.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(keys.size,), (1,)],
            output_dtypes=[mx.uint32, mx.uint32],
            init_value=0,
        )
        size = int(count.item())
        loc = locations[:size]
        records = mx.stack(
            [loc // n, mx.take(sorted_keys.reshape(-1), loc), mx.take(order.reshape(-1), loc)],
            axis=1,
        )
        # Host-visible compact output plus all GPU synchronization is timed.
        return np.array(records, dtype=np.uint32)

    def project(self, p, v, grids, epsilon):
        if epsilon != 0.002:
            raise ValueError("screened GPU projections require projection_epsilon=0.002")
        n = len(p)
        batch = len(v)
        axes = np.argmax(abs(v), axis=1).astype(np.uint32)
        width = 2 * epsilon
        bias = int(np.ceil(2 / width)) + 2
        side = 2 * bias + 1
        keys, uncertain, count = _PROJECT(
            inputs=[
                mx.array(p.astype(np.float32)),
                mx.array(v.astype(np.float32)),
                mx.array(axes),
                mx.array([n, batch, grids, bias, side], dtype=mx.uint32),
                mx.array([epsilon, width], dtype=mx.float32),
            ],
            grid=(n * batch, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(batch * grids * n,), (batch * n,), (1,)],
            output_dtypes=[mx.uint32, mx.uint32, mx.uint32],
            init_value=0,
        )
        size = int(count.item())
        pairs = np.array(uncertain[:size])
        di = pairs // n
        pi = pairs % n
        if size:
            ax = axes[di]
            oa = np.where(ax == 0, 1, 0)
            ob = np.where(ax == 2, 1, 2)
            t = p[pi, ax] / v[di, ax]
            a = p[pi, oa] - v[di, oa] * t
            b = p[pi, ob] - v[di, ob] * t
            indices = []
            correct = []
            for g, (ox, oy) in enumerate([(0, 0), (1, 1), (1, 0), (0, 1)][:grids]):
                aa = np.floor((a + ox * epsilon) / width).astype(np.int64)
                bb = np.floor((b + oy * epsilon) / width).astype(np.int64)
                indices.append(((di * grids + g) * n + pi).astype(np.uint32))
                correct.append(((aa + bias) * side + bb + bias).astype(np.uint32))
            keys = mx.put_along_axis(
                keys, mx.array(np.concatenate(indices)), mx.array(np.concatenate(correct)), axis=0
            )
        keys = keys.reshape(batch * grids, n)
        if self.audit_projection:
            expected = exact_projection(p, v, grids, epsilon)
            actual = np.array(keys)
            mismatches = int(np.sum(expected != actual))
            self.projection_audits.append(
                dict(keys=keys.size, corrected_pairs=size, mismatches=mismatches)
            )
            if mismatches:
                raise RuntimeError(f"{mismatches} GPU projection keys differ after correction")
        return keys, size

    def _sort_callback(self, ptr, rows, n, keep, out, capacity, points, directions, grids, epsilon):
        start = perf_counter()
        try:
            if self.gpu_projection:
                p = np.ctypeslib.as_array(points, shape=(n * 3,)).reshape(n, 3)
                v = np.ctypeslib.as_array(directions, shape=(rows // grids * 3,)).reshape(
                    rows // grids, 3
                )
                keys, uncertain = self.project(p, v, grids, epsilon)
            else:
                keys = np.ctypeslib.as_array(ptr, shape=(rows * n,)).reshape(rows, n)
                uncertain = 0
            records = self.compact(keys, keep)
            if len(records) > capacity:
                raise RuntimeError("GPU output exceeds bounded capacity")
            target = np.ctypeslib.as_array(out, shape=(capacity * 3,))
            target[: records.size] = records.reshape(-1)
            self.records.append(
                dict(
                    rows=rows,
                    points=n,
                    retained=len(records),
                    corrected_pairs=uncertain,
                    seconds=perf_counter() - start,
                )
            )
            return len(records)
        except BaseException as error:
            self.error = error
            return np.iinfo(np.uintp).max

    def run(self, points, options_json, *, gpu=True):
        q = np.array(points, dtype=np.float64, order="C", copy=True)
        if (
            q.ndim != 2
            or q.shape[1] != 3
            or len(q) > 100000
            or not np.isfinite(q).all()
            or np.max(abs(q), initial=0) > 1000
        ):
            raise ValueError("requires finite (n,3), n<=100000 and |q|<=1000")
        self.error = None
        self.lib.row_gpu_set_batch(self.batch_key_elements)
        self.lib.row_gpu_set_projection(self.gpu_projection)
        cb = ctypes.cast(self.callback, ctypes.c_void_p) if gpu else None
        ptr = self.lib.row_gpu_run(
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(q), options_json.encode(), cb
        )
        try:
            text = ctypes.string_at(ptr).decode()
        finally:
            self.lib.row_gpu_free(ptr)
        if self.error:
            raise RuntimeError("GPU callback failed") from self.error
        parsed = json.loads(text)
        if "error" in parsed:
            raise ValueError(parsed["error"])
        return text
