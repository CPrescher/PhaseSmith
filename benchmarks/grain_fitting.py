"""Measured selected-row traffic and complete proposal fits; first-party baseline."""

import argparse
import ast
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
from phasesmith import _core
from phasesmith.single_crystal import grains as G

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--legacy-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--repeats", type=int, default=5)
args = parser.parse_args()
legacy_dir = args.legacy_dir
if args.repeats < 1:
    parser.error("repeats must be positive")
pkg = types.ModuleType("_fitting_before")
pkg.__path__ = []
sys.modules["_fitting_before"] = pkg
spec = importlib.util.spec_from_file_location(
    "_fitting_before._core", legacy_dir / "core-before-ascent.abi3.so"
)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
p = legacy_dir / "source-before/python/phasesmith/experimental/grains.py"
node = next(
    n
    for n in ast.parse(p.read_text()).body
    if isinstance(n, ast.FunctionDef) and n.name == "_refine"
)
ns = G.__dict__.copy()
exec(compile(ast.Module(body=[node], type_ignores=[]), str(p), "exec"), ns)
old_refine = ns["_refine"]
engines = [old.ExperimentalLatticeEngine(4), _core.ExperimentalLatticeEngine(4)]
rng = np.random.default_rng(157183)
a = np.diag([4.7, 6.3, 8.1]) @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
h = rng.integers(-5, 6, (400, 3))
signal = h @ np.linalg.inv(a).T + rng.normal(0, 0.0008, h.shape)
background = rng.uniform(-1.6, 1.6, (12000, 3))
phase = background @ a[0]
background = background[abs(phase - np.rint(phase)) < 0.08][:953]
q = np.concatenate([signal, background])
bases = [a + rng.normal(0, 0.008, (3, 3)) for _ in range(80)]
reciprocal = [np.linalg.inv(b).T for b in bases]
options = G.GrainSearchOptions()
records = []
for repeat in range(args.repeats):
    for mode in [0, 1] if repeat % 2 == 0 else [1, 0]:
        engine = engines[mode]
        start = time.perf_counter()
        cloud = engine.prepare_fitting(q) if mode else None
        outputs = []
        for b, r in zip(bases, reciprocal, strict=True):
            if mode:
                result = cloud.selected(b, r, 0.015)
            else:
                hh, err = engine.score(q, b, r)
                ids = np.flatnonzero((err < 0.015) & np.any(hh != 0, axis=1))
                result = hh[ids], q[ids], ids, err[ids]
            outputs.append(result)
        records.append(
            dict(
                stage="selected_rows", repeat=repeat, mode=mode, seconds=time.perf_counter() - start
            )
        )
        if mode == 0:
            expected = outputs
        else:
            for result, oldresult in zip(outputs, expected, strict=True):
                for x, y in zip(result, oldresult, strict=True):
                    np.testing.assert_array_equal(x.view(np.uint8), y.view(np.uint8))
    for mode in [0, 1] if repeat % 2 == 0 else [1, 0]:
        start = time.perf_counter()
        cloud = engines[mode].prepare_fitting(q) if mode else None
        result = [
            G._refine(engines[mode], q, b, options, _cloud=cloud)
            if mode
            else old_refine(engines[mode], q, b, options)
            for b in bases
        ]
        records.append(
            dict(
                stage="complete_refinement",
                repeat=repeat,
                mode=mode,
                seconds=time.perf_counter() - start,
            )
        )
        if mode == 0:
            expected_fit = result
        else:
            for x, y in zip(result, expected_fit, strict=True):
                if x is None or y is None:
                    assert x is y
                else:
                    np.testing.assert_array_equal(x.view(np.uint8), y.view(np.uint8))
args.output.write_text(
    json.dumps(
        dict(points=len(q), bases=len(bases), exact_all_results=True, records=records), indent=2
    )
    + "\n"
)
for stage in ["selected_rows", "complete_refinement"]:
    print(
        stage,
        [
            (
                i,
                float(
                    np.median(
                        [r["seconds"] for r in records if r["stage"] == stage and r["mode"] == i]
                    )
                ),
            )
            for i in [0, 1]
        ],
    )
