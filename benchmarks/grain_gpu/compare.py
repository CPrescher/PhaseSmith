"""Paired complete searches with only FFT/maxima replaced by Metal.

Start with all three BLAS thread-limit variables set to 1. No runtime production
backend is registered. Cold first calls and warmed alternating repeats are saved.
"""

import argparse
import hashlib
import json
import platform
from contextlib import contextmanager
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np
from metal_fft import MetalFFT
from phasesmith import _core
from phasesmith.single_crystal import grains as G

ROOT = Path(__file__).resolve().parents[2]
NATIVE = _core.ExperimentalLatticeEngine


def synthetic():
    rng = np.random.default_rng(33)
    parts = []
    for edges in ([4.3, 5.1, 7.7], [5.2, 6.2, 8.3]):
        basis = np.diag(edges) @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
        h = np.unique(rng.integers(-6, 7, (1400, 3)), axis=0)
        h = h[np.any(h != 0, axis=1)]
        parts.append(h @ np.linalg.inv(basis).T + rng.normal(0, 0.0005, h.shape))
    return np.concatenate([*parts, rng.uniform(-1, 1, (600, 3))])


def fingerprint(obj):
    if isinstance(obj, np.ndarray):
        return dict(
            shape=list(obj.shape),
            dtype=obj.dtype.str,
            sha256=hashlib.sha256(obj.tobytes()).hexdigest(),
        )
    if is_dataclass(obj):
        return {f.name: fingerprint(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): fingerprint(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [fingerprint(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def compare(a, b):
    ca = [c for c in a.candidates if c.accepted]
    cb = [c for c in b.candidates if c.accepted]

    def indexed_union(cs):
        return (
            np.logical_or.reduce([c.indexed for c in cs]) if cs else np.zeros(len(a.training), bool)
        )

    details = []
    from scipy.optimize import linear_sum_assignment

    if ca and cb:
        cost = np.array([[-np.sum(x.indexed & y.indexed) for y in cb] for x in ca])
        ia, ib = linear_sum_assignment(cost)
        for i, j in zip(ia, ib, strict=True):
            x, y = ca[i], cb[j]
            change = y.direct_basis @ np.linalg.inv(x.direct_basis)
            integer = np.rint(change).astype(np.int64)
            lattice_error = float(np.max(np.abs(y.direct_basis - integer @ x.direct_basis)))
            details.append(
                dict(
                    cpu=int(i),
                    gpu=int(j),
                    indexed_exact=np.array_equal(x.indexed, y.indexed),
                    basis_exact=np.array_equal(x.direct_basis, y.direct_basis),
                    cell_max_abs=float(np.max(np.abs(x.cell - y.cell))),
                    basis_max_abs=float(np.max(np.abs(x.direct_basis - y.direct_basis))),
                    hkl_exact=np.array_equal(x.hkl, y.hkl),
                    residual_max_abs=float(np.max(np.abs(x.residual - y.residual))),
                    basis_change=integer.tolist(),
                    basis_change_det=round(np.linalg.det(integer)),
                    basis_change_max_abs_error=lattice_error,
                    lattice_equivalent_at_1e_minus_10=abs(round(np.linalg.det(integer))) == 1
                    and lattice_error <= 1e-10,
                    transformed_hkl_exact=np.array_equal(y.hkl, x.hkl @ integer.T),
                )
            )
    fa, fb = fingerprint(a), fingerprint(b)
    return dict(
        all_result_fields_bitwise_equal=fa == fb,
        changed_result_fields=[k for k in fa if fa[k] != fb[k]],
        candidates_cpu=len(a.candidates),
        candidates_gpu=len(b.candidates),
        accepted_cpu=len(ca),
        accepted_gpu=len(cb),
        acceptance_sequence_equal=[c.accepted for c in a.candidates]
        == [c.accepted for c in b.candidates],
        labels_exact=np.array_equal(a.labels, b.labels),
        label_disagreement_count=int(np.sum(a.labels != b.labels)),
        accepted_indexed_union_exact=np.array_equal(indexed_union(ca), indexed_union(cb)),
        accepted_matches=details,
    )


@contextmanager
def backend(kind, metal, records):
    class Engine:
        def __init__(self, workers):
            self.cpu = NATIVE(workers)

        def __getattr__(self, name):
            return getattr(self.cpu, name)

        def windowed_maxima(self, *args):
            t = perf_counter()
            result = (self.cpu if kind == "cpu" else metal).windowed_maxima(*args)
            records.append(dict(points=len(args[0]), grid=args[1], seconds=perf_counter() - t))
            return result

    _core.ExperimentalLatticeEngine = Engine
    try:
        yield
    finally:
        _core.ExperimentalLatticeEngine = NATIVE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--case", action="append")
    p.add_argument("--mode", choices=["single", "ensemble"], default="single")
    p.add_argument("--grid", type=int, default=192)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = {"synthetic-two-grain": synthetic()}
    for path in sorted((ROOT / "tests/fixtures/row-search").glob("*.npz")):
        cases[path.stem] = np.load(path)["points"]
    if args.case:
        cases = {k: v for k, v in cases.items() if k in args.case}
    report = dict(
        platform=platform.platform(),
        device=mx.device_info(),
        numpy=np.__version__,
        mlx=mx.__version__,
        mode=args.mode,
        grid=args.grid,
        workers=args.workers,
        timing_scope=(
            "Complete position-only find_grains or default 28-search ensemble, including "
            "deposit preparation, transfer, GPU completion, refinement, ownership and "
            "validation. Imports and input loading excluded. No detector-space "
            "fit/recovery supplied."
        ),
        cases={},
        cold_runs=[],
        warm_runs=[],
    )
    options = G.GrainSearchOptions(
        max_length=15, grid_size=args.grid, max_grains=4, workers=args.workers
    )
    search = G.find_grains if args.mode == "single" else G.find_grains_ensemble
    m = MetalFFT()
    previous = {}
    for name, q in cases.items():
        report["cases"][name] = dict(
            points=len(q),
            input_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
            source="fixed synthetic"
            if name.startswith("synthetic")
            else (
                "frozen measured training-coordinate fixture; new deterministic 80/20 "
                "split inside this population; exposed regression only"
            ),
            options=asdict(options),
        )
        np.savez_compressed(args.output / (name + "-input.npz"), points=q)
        for repeat in range(-1, args.repeats):
            results = {}
            row = {}
            for kind in ["cpu", "gpu"] if repeat % 2 else ["gpu", "cpu"]:
                stages = []
                m.records.clear()
                t = perf_counter()
                with backend(kind, m, stages):
                    result = search(q, options=options)
                elapsed = perf_counter() - t
                results[kind] = result
                rec = dict(
                    case=name,
                    repeat=repeat,
                    backend=kind,
                    seconds=elapsed,
                    fft_calls=len(stages),
                    fft_seconds=sum(x["seconds"] for x in stages),
                    fft_stages=stages,
                    gpu_stages=list(m.records),
                    accepted=sum(c.accepted for c in result.candidates),
                    repeat_bitwise_equal=None
                    if (name, kind) not in previous
                    else fingerprint(result) == previous[name, kind],
                )
                previous[name, kind] = fingerprint(result)
                (args.output / f"{name}-{repeat}-{kind}-fingerprint.json").write_text(
                    json.dumps(fingerprint(result), indent=2) + "\n"
                )
                row[kind] = rec
                print(
                    name,
                    repeat,
                    kind,
                    round(elapsed, 4),
                    "accepted",
                    rec["accepted"],
                    "fft",
                    round(rec["fft_seconds"], 4),
                    flush=True,
                )
            pair = dict(
                case=name,
                repeat=repeat,
                cpu=row["cpu"],
                gpu=row["gpu"],
                comparison=compare(results["cpu"], results["gpu"]),
            )
            report["cold_runs" if repeat < 0 else "warm_runs"].append(pair)
            (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            print("comparison", pair["comparison"], flush=True)
    summary = []
    for name in cases:
        rows = [r for r in report["warm_runs"] if r["case"] == name]
        cpu = float(np.median([r["cpu"]["seconds"] for r in rows]))
        gpu = float(np.median([r["gpu"]["seconds"] for r in rows]))
        summary.append(
            dict(
                case=name,
                cpu_median=cpu,
                gpu_median=gpu,
                cpu_over_gpu=cpu / gpu,
                cpu_fft_median=float(np.median([r["cpu"]["fft_seconds"] for r in rows])),
                gpu_fft_median=float(np.median([r["gpu"]["fft_seconds"] for r in rows])),
                comparisons=[r["comparison"] for r in rows],
            )
        )
    report["summary"] = summary
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
