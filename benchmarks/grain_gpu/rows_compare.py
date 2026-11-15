"""Paired CPU/Metal row grouping and full row-based grain finding.

Only the initial projection sorting/bucketing stage uses Metal. Geometry,
naive/smart direction voting and later FFT cell inference remain CPU-identical.
"""

import argparse
import hashlib
import json
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from compare import compare, fingerprint, synthetic
from metal_rows import MetalRows
from phasesmith import _core
from phasesmith.single_crystal.grains import GrainSearchOptions, find_grains
from phasesmith.single_crystal.rows import RowSearchOptions, group_rows

ROOT = Path(__file__).resolve().parents[2]
NATIVE = _core.experimental_row_search


def large_cloud(n, *, return_truth=False):
    if not 1 <= n <= 12000:
        raise ValueError("fixed generator supports 1..12000 points")
    rng = np.random.default_rng(64000 + n)
    parts = []
    truths = []
    for edges in ([4.3, 5.1, 7.7], [5.2, 6.2, 8.3]):
        basis = np.diag(edges) @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
        truths.append(basis.copy())
        all_h = np.empty((0, 3), dtype=int)
        target = 2 * n // 5
        while len(all_h) < target:
            h = np.unique(np.concatenate([all_h, rng.integers(-16, 17, (n, 3))]), axis=0)
            q = h @ np.linalg.inv(basis).T
            all_h = h[np.any(h != 0, axis=1) & (np.max(abs(q), axis=1) < 1.6)]
        h = all_h[rng.choice(len(all_h), target, replace=False)]
        parts.append(h @ np.linalg.inv(basis).T + rng.normal(0, 0.0005, h.shape))
    q = np.concatenate([*parts, rng.uniform(-1.6, 1.6, (n - sum(map(len, parts)), 3))])
    return (q, truths) if return_truth else q


def scientific(obj):
    """Exclude clocks only; preserve all arrays, budgets, counts and options."""
    if is_dataclass(obj):
        return {f.name: scientific(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {
            k: scientific(v) for k, v in obj.items() if not k.endswith("seconds") or k == "seconds"
        }
    if isinstance(obj, (tuple, list)):
        return [scientific(v) for v in obj]
    return fingerprint(obj)


@contextmanager
def backend(kind, metal, calls):
    def run(q, options):
        start = perf_counter()
        text = NATIVE(q, options) if kind == "cpu" else metal.run(q, options)
        r = json.loads(text)
        calls.append(
            dict(
                points=len(q),
                seconds=perf_counter() - start,
                stage1_seconds=r["stage1_seconds"],
                stage2_seconds=r["stage2_seconds"],
                compute_seconds=r["compute_seconds"],
                effective_smart=r["effective_smart"],
                groups=len(r["groups"]),
                complete=not r["budget_exhausted"] and not r["cancelled"],
            )
        )
        return text

    _core.experimental_row_search = run
    try:
        yield
    finally:
        _core.experimental_row_search = NATIVE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--case", action="append")
    p.add_argument("--mode", choices=["grouping", "full"], default="full")
    p.add_argument("--batch-elements", type=int, default=8_388_608)
    p.add_argument("--gpu-projection", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = {"synthetic-2674": synthetic()}
    for path in sorted((ROOT / "tests/fixtures/row-search").glob("*.npz")):
        cases[path.stem] = np.load(path)["points"]
    for n in [6000, 12000]:
        cases[f"synthetic-{n}"] = large_cloud(n)
    if args.case:
        cases = {k: v for k, v in cases.items() if k in args.case}
    m = MetalRows(batch_key_elements=args.batch_elements, gpu_projection=args.gpu_projection)
    report = dict(
        mode=args.mode,
        batch_elements=args.batch_elements,
        gpu_projection=args.gpu_projection,
        cases={},
        runs=[],
    )
    ro = RowSearchOptions()
    go = GrainSearchOptions(max_length=15, grid_size=192, max_grains=4)
    previous = {}
    for name, q in cases.items():
        report["cases"][name] = dict(points=len(q), sha256=hashlib.sha256(q.tobytes()).hexdigest())
        np.savez_compressed(args.output / (name + "-input.npz"), points=q)
        # Native snapshot baseline must match the installed Rust CPU kernel.
        opts = json.dumps(ro._native_options())
        assert scientific(json.loads(m.run(q, opts, gpu=False))) == scientific(
            json.loads(NATIVE(q, opts))
        )
        for repeat in range(-1, args.repeats):
            results = {}
            timings = {}
            for kind in ["cpu", "gpu"] if repeat % 2 else ["gpu", "cpu"]:
                calls = []
                m.records.clear()
                start = perf_counter()
                with backend(kind, m, calls):
                    result = (
                        group_rows(q, options=ro)
                        if args.mode == "grouping"
                        else find_grains(q, method="rows", options=go, row_options=ro)
                    )
                elapsed = perf_counter() - start
                snap = scientific(result)
                if (name, kind) in previous:
                    assert snap == previous[name, kind], ("repeat drift", name, kind)
                previous[name, kind] = snap
                results[kind] = result
                timings[kind] = dict(
                    seconds=elapsed,
                    calls=calls,
                    gpu_batches=len(m.records),
                    gpu_callback_seconds=sum(x["seconds"] for x in m.records),
                )
                (args.output / f"{name}-{repeat}-{kind}-fingerprint.json").write_text(
                    json.dumps(snap, indent=2) + "\n"
                )
                print(name, repeat, kind, round(elapsed, 4), "calls", calls, flush=True)
            comparison = dict(
                scientific_fields_bitwise_equal=scientific(results["cpu"])
                == scientific(results["gpu"])
            )
            if args.mode == "grouping":
                comparison["raw_ordered_groups_equal"] = all(
                    np.array_equal(x, y)
                    for x, y in zip(results["cpu"].groups, results["gpu"].groups, strict=True)
                )
                assert results["cpu"].complete and results["gpu"].complete
            else:
                comparison["grain_comparison"] = compare(results["cpu"], results["gpu"])
                comparison["raw_ordered_groups_equal"] = all(
                    np.array_equal(x, y)
                    for x, y in zip(
                        results["cpu"].row_grouping.groups,
                        results["gpu"].row_grouping.groups,
                        strict=True,
                    )
                )
            assert comparison["scientific_fields_bitwise_equal"], (name, repeat, comparison)
            row = dict(case=name, repeat=repeat, timing=timings, comparison=comparison)
            report["runs"].append(row)
            (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            print("EXACT SCIENTIFIC PARITY", name, repeat, flush=True)
    summary = []
    for name in cases:
        rows = [r for r in report["runs"] if r["case"] == name and r["repeat"] >= 0]
        cpu = float(np.median([r["timing"]["cpu"]["seconds"] for r in rows]))
        gpu = float(np.median([r["timing"]["gpu"]["seconds"] for r in rows]))
        summary.append(
            dict(
                case=name,
                cpu_seconds=cpu,
                gpu_seconds=gpu,
                cpu_over_gpu=cpu / gpu,
                all_exact=all(r["comparison"]["scientific_fields_bitwise_equal"] for r in rows),
                effective_smart=rows[0]["timing"]["cpu"]["calls"][0]["effective_smart"],
            )
        )
    report["summary"] = summary
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
