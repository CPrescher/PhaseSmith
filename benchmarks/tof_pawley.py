"""Reproducible one-/three-bank TOF Pawley benchmarks with scientific gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from phasesmith import TofInstrument, _core
from phasesmith.pattern import TofPowderPattern
from phasesmith.refinement.pawley import PawleyOptions
from phasesmith.refinement.tof_pawley import (
    TofPawleyBank,
    TofPawleyInput,
    TofPawleyPhase,
    calculate,
    refine,
)


def run_case(bank_count, solver, repeats):
    families, samples = 256, 10001
    d = np.linspace(1.2, 3.8, families)
    d[1::2] = d[::2] + 0.001
    banks = []
    for index in range(bank_count):
        instrument = TofInstrument(
            0, 5000 - 300 * index, 0, 0, 0.18, 0.04, 0, 0, 1, 10, 0, 0, 0.3, 0, 0.4
        )
        x = 4500 + 16000 * np.linspace(0, 1, samples) ** 1.01
        banks.append(
            TofPawleyBank(
                f"bank{index}",
                TofPowderPattern(x),
                instrument,
                (
                    TofPawleyPhase(
                        "sample",
                        tuple(str(k) for k in range(families)),
                        d,
                        (1 + np.arange(families) % 7) * (index + 1),
                    ),
                ),
                "Synthetic nonuniform microsecond density; no amplitude corrections",
            )
        )
    options = PawleyOptions(solver=solver, max_elements=200_000_000)
    request = TofPawleyInput(tuple(banks))
    y = calculate(request, options).calculated_y
    request = TofPawleyInput(
        tuple(
            replace(
                b,
                pattern=TofPowderPattern(b.pattern.tof_us, observed_y=y[lo:hi]),
                phases=(replace(b.phases[0], intensities=b.phases[0].intensities * 0.8),),
            )
            for b, lo, hi in zip(
                banks, request.sample_offsets[:-1], request.sample_offsets[1:], strict=True
            )
        )
    )
    previous = None
    times = []
    for _ in range(repeats):
        start = perf_counter()
        fit = refine(request, options, max_iterations=50)
        times.append(perf_counter() - start)
        error = np.linalg.norm(fit.calculation.calculated_y - y) / np.linalg.norm(y)
        assert fit.termination_reason == "converged", fit.termination_reason
        assert error < 1e-6, error
        if previous is not None:
            np.testing.assert_array_equal(fit.calculation.calculated_y, previous[0])
            np.testing.assert_array_equal(fit.history, previous[1])
        previous = (fit.calculation.calculated_y.copy(), fit.history.copy())
    return dict(
        banks=bank_count,
        families_per_bank=families,
        samples_per_bank=samples,
        solver=solver,
        times_seconds=times,
        median_seconds=float(np.median(times)),
        relative_l2=float(error),
        termination=fit.termination_reason,
        profile_sha256=hashlib.sha256(fit.calculation.calculated_y.tobytes()).hexdigest(),
        history_sha256=hashlib.sha256(fit.history.tobytes()).hexdigest(),
        exact_repeats=True,
        jacobian_storage_elements=fit.calculation.jacobian_operator.storage_elements,
        diagnostics=dict(fit.diagnostics),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [run_case(1, "dense", args.repeats)]
    results.extend(run_case(banks, "matrix_free", args.repeats) for banks in (1, 3))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        native_binary_sha256=hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=results,
        peak_process_rss_bytes=peak if platform.system() == "Darwin" else peak * 1024,
        note="Process RSS includes the preceding dense case, Python and returned arrays.",
    )
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
