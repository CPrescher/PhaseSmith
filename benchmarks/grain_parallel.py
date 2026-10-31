"""Strict two-component detector-space control with ordered process searches.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=python before Python startup.
Each member uses one native worker. Complete timing includes worker startup.
The original synthetic control's assertions and search budgets are unchanged.
"""

import argparse
import json
import platform
from pathlib import Path

from grain_remaining import control
from phasesmith.single_crystal import grains


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--serial-fits", action="store_true", help="keep searches parallel but fitting serial"
    )
    args = parser.parse_args()
    original = grains.find_grains_ensemble
    execution = []

    def find(*a, **kw):
        from phasesmith.single_crystal.search_execution import parallel_searches

        with parallel_searches(args.ensemble_workers, fit_tasks=not args.serial_fits):
            result = original(*a, **kw, ensemble_workers=args.ensemble_workers)
        execution.extend(d["parallel_search"] for d in result.diagnostics if "parallel_search" in d)
        return result

    grains.find_grains_ensemble = find
    try:
        result = control()
    finally:
        grains.find_grains_ensemble = original
    if args.ensemble_workers > 1:
        assert len(execution) == 2
        assert all(r["fitting_tasks_enabled"] == (not args.serial_fits) for r in execution)
        assert all(r["submitted_tasks"] == r["tasks"] for r in execution)
        assert all(r["native_workers_per_search"] == 1 for r in execution)
        assert all(j["native_workers"] == 1 for r in execution for j in r["jobs"])
        assert all(
            set(j["numerical_environment"].values()) == {"1"} for r in execution for j in r["jobs"]
        )
    args.output.write_text(
        json.dumps(
            dict(
                platform=platform.platform(),
                fitting_tasks_enabled=not args.serial_fits,
                result=result,
                execution=execution,
                scope=(
                    "Strict two-component control; ordinary and native ascent; "
                    "no inverse cells/HKLs"
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    print([(r["native"], r["accepted"]) for r in result["runs"]])


if __name__ == "__main__":
    main()
