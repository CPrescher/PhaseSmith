"""Trace exact training-only cell fits across repeated held-out perturbations.

Diagnostic only: retain first-step inputs without changing any numerical value.
The optional saved arrays identify the earliest differing stage on the host BLAS.
"""

import argparse
import inspect
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from phasesmith.single_crystal.cell_comparison import CellComparisonOptions, compare_cells
from phasesmith.single_crystal.prediction import predict_spots
from phasesmith.single_crystal.reciprocal import PanelGeometry

SOLVE = np.linalg.solve
RECORDS = []
CHAIN_STATE = {}


def profile(frame, event, result):
    global CHAIN_STATE
    if event == "return" and frame.f_code.co_name == "metric_chain":
        CHAIN_STATE = {
            "metric_" + name: value
            for name, value in frame.f_locals.items()
            if name in {"a", "g", "c", "ci", "d", "scale", "m", "phi", "dc"}
        }


def trace(a, b):
    frame = inspect.currentframe().f_back
    result = SOLVE(a, b)
    if frame.f_code.co_name == "_fit_setting":
        state = frame.f_locals
        # Retain references, not formatted/rounded copies; these arrays are not
        # mutated by the optimizer after its next step.
        arrays = {
            **CHAIN_STATE,
            **{
                k: state[k]
                for k in ("b0", "b", "jac", "r", "weights", "chain", "j", "rr", "normal")
            },
        }
        arrays.update(
            projected_basis=state["setting"].projected_basis,
            solve_matrix=a,
            solve_rhs=b,
            step=result,
        )
        RECORDS.append((state["setting"].symbol, state["_"], arrays))
    return result


def fixture():
    rng = np.random.default_rng(65)
    geo = PanelGeometry(
        np.array([[0.2, -0.08, -0.08], [0, 0.0001, 0], [0, 0, 0.0001], [1, 0, 0], [0, 0, 1.0]]), 0.5
    )
    basis = np.diag([5.0, 7.0, 10.0])
    h = rng.integers(-5, 6, size=(1000, 3)).astype(np.int64)
    prediction = predict_spots(h, np.linalg.inv(basis).T, geo, 0.0)
    good = prediction.flags == 0
    h = h[good]
    observed = prediction.values[good].copy()
    observed += rng.normal(0, [0.03, 0.03, 0.002], observed.shape)
    points = h @ np.linalg.inv(basis).T
    training = np.arange(len(h)) % 5 != 0
    basis[0, 0] *= 1.001
    basis[1, 2] = 0.003
    return points, basis, training, observed, geo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--trace", action="store_true", help="capture intermediate arrays")
    parser.add_argument("--check", action="store_true", help="require exact held-out independence")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.trace:
        np.linalg.solve = trace
        sys.setprofile(profile)
    report = []
    for repetition in range(args.repetitions):
        points, basis, training, observed, geometry = fixture()
        kwargs = dict(
            training=training, geometry=geometry, options=CellComparisonOptions(max_subcell_index=1)
        )
        RECORDS.clear()
        started = time.perf_counter()
        first = compare_cells(points, basis, observed=observed, **kwargs)
        first_records = list(RECORDS)
        changed = observed.copy()
        changed[~training, :2] += 5
        RECORDS.clear()
        second = compare_cells(points, basis, observed=changed, **kwargs)
        elapsed = time.perf_counter() - started
        second_records = list(RECORDS)
        differences = []
        if len(first_records) != len(second_records):
            differences.append(dict(trace_lengths=[len(first_records), len(second_records)]))
        for index, (left, right) in enumerate(zip(first_records, second_records, strict=False)):
            if left[:2] != right[:2]:
                differences.append(
                    dict(trace=index, left=left[:2], right=right[:2], scheduling=True)
                )
                break
            fields = [
                name for name in left[2] if left[2][name].tobytes() != right[2][name].tobytes()
            ]
            if fields:
                differences.append(
                    dict(trace=index, setting=left[0], iteration=left[1], fields=fields)
                )
                np.savez(args.output / f"first-{repetition}.npz", **left[2])
                np.savez(args.output / f"second-{repetition}.npz", **right[2])
                break
        exact = all(
            a.reciprocal_basis.tobytes() == b.reciprocal_basis.tobytes()
            and a.history.tobytes() == b.history.tobytes()
            and a.training_metrics == b.training_metrics
            and a.training_score == b.training_score
            and a.status == b.status
            for a, b in zip(first.settings, second.settings, strict=True)
        )
        report.append(
            dict(repetition=repetition, exact=exact, differences=differences, seconds=elapsed)
        )
        if args.trace and (differences or not exact):
            break
    sys.setprofile(None)
    try:
        from threadpoolctl import threadpool_info

        runtime = threadpool_info()
    except ImportError:
        runtime = []
    result = dict(
        numpy=np.__version__,
        build_config=np.show_config(mode="dicts"),
        numerical_libraries=runtime,
        comparisons=report,
        median_pair_seconds=statistics.median(row["seconds"] for row in report),
        tracing=args.trace,
    )
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.check and any(not row["exact"] or row["differences"] for row in report):
        raise SystemExit("cell comparison lost exact training independence")


if __name__ == "__main__":
    main()
