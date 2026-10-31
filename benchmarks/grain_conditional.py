"""Conditional FFT benchmark with optional exact pre-change source/binary replay.

Only global-training q positions are read. No chemistry, supplied cells,
comparator groups or reserved coordinates select or rank the inverse.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import platform
import sys
import types
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from phasesmith import _core
from phasesmith.experimental import grains as G


def legacy(root):
    """Load an explicitly saved first-party version; never an external oracle."""
    path = root / "python/phasesmith/experimental/grains.py"
    tree = ast.parse(path.read_text())
    names = {"_conditioned_proposals", "_vectors", "_triplets", "_fourier_peaks"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(G.__dict__)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    binary = root / "core-before-native.abi3.so"
    name = "_phasesmith_conditional_before"
    parent = types.ModuleType(name)
    parent.__path__ = []
    sys.modules[name] = parent
    spec = importlib.util.spec_from_file_location(name + "._core", binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return namespace["_conditioned_proposals"], module.ExperimentalLatticeEngine(4)


def main(args):
    if args.points:
        path = Path(args.points)
        data = np.load(path)
        q = data["points"][data["training"]]
        source = dict(
            path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    else:
        rng = np.random.default_rng(842024)
        q = rng.uniform(-1.6, 1.6, (12000, 3))
        for start, edges in [(0, [4.7, 6.3, 8.1]), (300, [5.2, 7.1, 9.3])]:
            a = np.diag(edges) @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
            h = rng.integers(-5, 6, (300, 3))
            q[start : start + 300] = h @ np.linalg.inv(a).T + rng.normal(0, 0.0008, (300, 3))
        source = dict(kind="synthetic two minority lattices and diffuse contamination", seed=842024)
    options = G.GrainSearchOptions(grid_size=256, max_grains=40, min_indexed=12, workers=4)
    engine = _core.ExperimentalLatticeEngine(4)
    cond = G.GrainConditioningOptions()
    records = []
    old, old_engine = legacy(Path(args.legacy_dir)) if args.legacy_dir else (None, None)
    for repeat in range(args.repeats):
        before = None
        if old:
            start = perf_counter()
            prior, before = old(old_engine, q, options, cond, diverse=True)
            records.append(
                dict(
                    repeat=repeat,
                    mode="before-all-80",
                    seconds=perf_counter() - start,
                    proposals=len(prior),
                )
            )
        start = perf_counter()
        current, audit = G._conditioned_proposals(engine, q, options, cond, diverse=True)
        elapsed = perf_counter() - start
        exact = None
        if before is not None:
            exact = np.array_equal(np.asarray(prior), np.asarray(current))
            assert exact
            assert np.array_equal(before["seed_vectors"], audit["seed_vectors"])
        records.append(
            dict(
                repeat=repeat,
                mode="after-all-80",
                seconds=elapsed,
                proposals=len(current),
                exact_basis_replay=exact,
            )
        )
        start = perf_counter()
        ranked, rank_audit = G._conditioned_proposals(
            engine, q, options, replace(cond, max_seeds=12, rank_by_population=True), diverse=True
        )
        records.append(
            dict(
                repeat=repeat,
                mode="after-phase-ranked-12",
                seconds=perf_counter() - start,
                proposals=len(ranked),
                seed_ids=[x["seed"] for x in rank_audit["seeds"]],
                scope="Changed proposal coverage, not an equal-output acceleration",
            )
        )
    result = dict(
        source=source,
        training_points=len(q),
        platform=platform.platform(),
        workers=4,
        runs=records,
        timing_scope=(
            "Cold single-view conditional search only, including its global seed FFT. "
            "No final ownership/fitting/validation or imports. Same-80 replay is exact; "
            "ranked-12 coverage is a separate policy."
        ),
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points")
    parser.add_argument("--legacy-dir")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
