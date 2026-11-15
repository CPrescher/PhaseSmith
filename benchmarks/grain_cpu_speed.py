"""Paired complete CPU grain-search benchmark against saved first-party source.

Run with the three numerical thread limits set to 1 before Python startup.
Uses exactly the same native extension for both arms. Only timing fields are
excluded from bitwise result fingerprints. No search setting is shortened.
"""

import argparse
import hashlib
import json
import sys
import types
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from phasesmith.single_crystal import grains, ownership, recovery

_MODULES = {"ownership": ownership, "recovery": recovery, "grains": grains}


def fingerprint(value):
    if isinstance(value, np.ndarray):
        return dict(
            shape=value.shape,
            dtype=value.dtype.str,
            sha256=hashlib.sha256(value.tobytes()).hexdigest(),
        )
    if is_dataclass(value):
        return {f.name: fingerprint(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {
            k: fingerprint(v)
            for k, v in value.items()
            if not k.endswith("seconds") or k == "seconds"
        }
    if isinstance(value, (list, tuple)):
        return [fingerprint(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@contextmanager
def modules(selected):
    import phasesmith.single_crystal as package

    previous = {name: sys.modules["phasesmith.single_crystal." + name] for name in selected}
    for name, module in selected.items():
        sys.modules["phasesmith.single_crystal." + name] = module
        setattr(package, name, module)
    if "grain_remaining" in sys.modules:
        sys.modules["grain_remaining"].G = selected["grains"]
    try:
        yield selected["grains"]
    finally:
        for name, module in previous.items():
            sys.modules["phasesmith.single_crystal." + name] = module
            setattr(package, name, module)
        if "grain_remaining" in sys.modules:
            sys.modules["grain_remaining"].G = previous["grains"]


def load_baseline(root):
    result = {}
    # Relative imports must resolve the corresponding baseline options classes.
    with modules(_MODULES):
        for name in _MODULES:
            canonical = "phasesmith.single_crystal." + name
            mod = types.ModuleType(canonical)
            mod.__package__ = "phasesmith.single_crystal"
            mod.__file__ = str(root / "single_crystal" / f"{name}.py")
            sys.modules[canonical] = mod
            import phasesmith.single_crystal as package

            setattr(package, name, mod)
            exec(compile(Path(mod.__file__).read_text(), mod.__file__, "exec"), mod.__dict__)
            result[name] = mod
    return result


def cases():
    rng = np.random.default_rng(33)
    parts = []
    for edges in ([4.3, 5.1, 7.7], [5.2, 6.2, 8.3]):
        basis = np.diag(edges) @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
        h = np.unique(rng.integers(-6, 7, (1400, 3)), axis=0)
        h = h[np.any(h != 0, axis=1)]
        parts.append(h @ np.linalg.inv(basis).T + rng.normal(0, 0.0005, h.shape))
    result = {"synthetic-2674": np.concatenate([*parts, rng.uniform(-1, 1, (600, 3))])}
    for n in [6000, 12000]:
        rng = np.random.default_rng(64000 + n)
        parts = []
        for edges in ([4.3, 5.1, 7.7], [5.2, 6.2, 8.3]):
            basis = np.diag(edges) @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
            all_h = np.empty((0, 3), int)
            target = 2 * n // 5
            while len(all_h) < target:
                h = np.unique(np.concatenate([all_h, rng.integers(-16, 17, (n, 3))]), axis=0)
                q = h @ np.linalg.inv(basis).T
                all_h = h[np.any(h != 0, axis=1) & (np.max(abs(q), axis=1) < 1.6)]
            h = all_h[rng.choice(len(all_h), target, replace=False)]
            parts.append(h @ np.linalg.inv(basis).T + rng.normal(0, 0.0005, h.shape))
        result[f"synthetic-{n}"] = np.concatenate(
            [*parts, rng.uniform(-1.6, 1.6, (n - sum(map(len, parts)), 3))]
        )
    repo = Path(__file__).resolve().parents[1]
    for path in sorted((repo / "tests/fixtures/row-search").glob("*.npz")):
        with np.load(path) as z:
            result[path.stem] = z["points"].copy()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["fft", "rows", "ensemble", "detector"], default="fft")
    p.add_argument("--case", action="append")
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    before = load_baseline(args.baseline)
    inputs = cases() if args.mode != "detector" else {"detector": None}
    if args.case:
        inputs = {k: v for k, v in inputs.items() if k in args.case}
    report = dict(
        mode=args.mode,
        runs=[],
        input_hashes={
            k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in inputs.items() if v is not None
        },
    )
    for name, q in inputs.items():
        previous = {}
        for repeat in range(args.repeats) if args.mode == "detector" else range(-1, args.repeats):
            results = {}
            times = {}
            captures = {}
            for kind in ["before", "after"] if repeat % 2 else ["after", "before"]:
                selected = before if kind == "before" else _MODULES
                with modules(selected) as G:
                    options = G.GrainSearchOptions(max_length=15, grid_size=192, max_grains=4)
                    if args.mode == "detector":
                        from grain_remaining import control

                        sys.modules["grain_remaining"].G = G
                        original = G.find_grains_ensemble
                        actual = []

                        def run(*a, _original=original, _actual=actual, **kw):
                            start = perf_counter()
                            r = _original(*a, **kw)
                            _actual.append((r, perf_counter() - start))
                            return r

                        G.find_grains_ensemble = run
                        try:
                            start = perf_counter()
                            control()
                            seconds = perf_counter() - start
                        finally:
                            G.find_grains_ensemble = original
                        result = [x[0] for x in actual]
                        captures[kind] = [x[1] for x in actual]
                    else:
                        start = perf_counter()
                        result = (
                            G.find_grains_ensemble(q, options=options)
                            if args.mode == "ensemble"
                            else G.find_grains(q, options=options, method=args.mode)
                        )
                        seconds = perf_counter() - start
                snap = fingerprint(result)
                if kind in previous:
                    assert snap == previous[kind], (name, kind, "repeat drift")
                previous[kind] = snap
                results[kind] = snap
                times[kind] = seconds
                (args.output / f"{name}-{repeat}-{kind}-fingerprint.json").write_text(
                    json.dumps(snap, indent=2) + "\n"
                )
                print(args.mode, name, repeat, kind, round(seconds, 5), flush=True)
            assert results["before"] == results["after"], (
                args.mode,
                name,
                repeat,
                "result mismatch",
            )
            report["runs"].append(
                dict(
                    case=name, repeat=repeat, seconds=times, detector_subcalls=captures, exact=True
                )
            )
            (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = []
    for name in inputs:
        rows = [r for r in report["runs"] if r["case"] == name and r["repeat"] >= 0]
        old = float(np.median([r["seconds"]["before"] for r in rows]))
        new = float(np.median([r["seconds"]["after"] for r in rows]))
        summary.append(
            dict(
                case=name,
                before_seconds=old,
                after_seconds=new,
                speedup=old / new,
                wall_saved_fraction=1 - new / old,
            )
        )
    report["summary"] = summary
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
