"""Diagnose serial/spawned positional fits without changing their arithmetic.

Run in separate processes with the numerical thread environment set before
Python starts. Requires the repository test fixtures and an installed wheel.
Small first-step arrays distinguish input, weighting and BLAS differences.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
_SOLVE = np.linalg.solve
_STEPS = 0


def _trace_solve(a, b):
    global _STEPS
    frame = inspect.currentframe().f_back
    if (
        os.environ.get("PHASESMITH_REPRO_OUTPUT")
        and frame.f_code.co_name == "_refine_crystal"
        and frame.f_locals.get("iteration") == 0
    ):
        state = frame.f_locals
        arrays = {name: state[name] for name in ("b", "j", "r", "normal", "gradient")}
        arrays.update(solve_matrix=a, solve_rhs=b)
        root = Path(os.environ["PHASESMITH_REPRO_OUTPUT"])
        prefix = root / f"step-{os.getpid()}-{_STEPS}"
        np.savez(prefix.with_suffix(".npz"), **arrays)
        prefix.with_suffix(".json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "initial_basis": state["b0"].tolist(),
                    "arrays": {
                        name: {
                            "shape": value.shape,
                            "strides": value.strides,
                            "address_mod_64": value.ctypes.data % 64,
                            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                        }
                        for name, value in arrays.items()
                    },
                },
                indent=2,
            )
            + "\n"
        )
        _STEPS += 1
    return _SOLVE(a, b)


np.linalg.solve = _trace_solve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--late-limits", action="store_true", help="simulate limits set after NumPy import"
    )
    parser.add_argument(
        "--check", action="store_true", help="require bitwise serial/spawned equality"
    )
    args = parser.parse_args()
    if args.late_limits:
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[name] = "1"
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["PHASESMITH_REPRO_OUTPUT"] = str(args.output.resolve())
    from phasesmith import _core
    from phasesmith.single_crystal import grains, recovery, search_execution
    from test_experimental_grains import positional_cloud

    q, observed, geometry, truth = positional_cloud()
    basis = np.linalg.inv(truth).T
    models = [basis.copy(), basis * 1.0003, basis * 0.9997]
    engine = _core.ExperimentalLatticeEngine(1)
    options = grains.GrainSearchOptions(workers=1)
    serial = recovery.fit_training_proposals(engine, q, observed, geometry, models, options)
    with search_execution.parallel_searches(2):
        parallel = recovery.fit_training_proposals(engine, q, observed, geometry, models, options)
    try:
        from threadpoolctl import threadpool_info

        libraries = threadpool_info()
    except ImportError:
        libraries = "Install optional threadpoolctl for runtime library details."
    record = dict(
        numpy=np.__version__,
        platform=platform.platform(),
        environment={name: os.environ.get(name) for name in search_execution._LIMITS},
        numerical_libraries=libraries,
        exact_audit=serial[1] == parallel[1],
        exact_models=all(
            a.tobytes() == b.tobytes() for a, b in zip(serial[0], parallel[0], strict=True)
        ),
        serial=serial[1],
        parallel=parallel[1],
    )
    (args.output / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if args.check and not (record["exact_audit"] and record["exact_models"]):
        raise SystemExit("serial/spawned positional reproducibility failed")


if __name__ == "__main__":
    main()
