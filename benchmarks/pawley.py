"""Reproducible multi-peak Pawley fits; scientific gates precede timing claims."""

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
from phasesmith import ConstantWavelengthInstrument, PowderPattern, _core
from phasesmith.instrument import FcjGeometry
from phasesmith.radiation import WavelengthComponents
from phasesmith.refinement.pawley import (
    PawleyInput,
    PawleyOptions,
    PawleyPhase,
    build_parameter_set,
    calculate,
    refine,
)


def run_case(
    reflections, samples, axial, joint, repeats, solver="dense", spectrum=False, boundary=False
):
    x = np.linspace(10.0, 90.0, samples)
    positions = np.linspace(12.0, 88.0, reflections)
    # Adjacent pairs overlap heavily without being exactly coincident.
    positions[1::2] = positions[::2][: len(positions[1::2])] + 0.06
    areas = 1.0 + np.arange(reflections) % 7
    phases = tuple(
        PawleyPhase(
            str(k), tuple(str(v) for v in range(k, reflections, 2)), positions[k::2], areas[k::2]
        )
        for k in range(2)
    )
    instrument = ConstantWavelengthInstrument(1.54, 0.0001, 0.0, 0.001, 0.002, 0.001)
    if boundary:
        instrument = replace(instrument, x_deg=0.0, y_deg=0.0)
    truth = PawleyInput(
        PowderPattern(x, observed_y=np.zeros_like(x)),
        instrument,
        phases,
        axial_geometry=FcjGeometry(0.002, 0.002) if axial else None,
        fixed_spectrum=WavelengthComponents([1.54, 1.544], [1.0, 0.5]) if spectrum else None,
    )
    options = PawleyOptions(max_elements=150_000_000, solver=solver)
    y = calculate(truth, options).calculated_y
    request = replace(
        truth,
        pattern=PowderPattern(x, observed_y=y),
        parameters=None,
        phases=tuple(replace(p, intensities=p.intensities * 0.8) for p in phases),
        instrument=(
            replace(instrument, x_deg=0.002)
            if boundary
            else replace(instrument, w_deg2=0.0011)
            if joint
            else instrument
        ),
    )
    request = replace(
        request,
        parameters=build_parameter_set(
            request,
            profile_parameters=("x_deg", "y_deg") if boundary else ("w_deg2",) if joint else (),
        ),
    )
    timings, records, diagnostics = [], [], []
    previous = None
    for _ in range(repeats):
        start = perf_counter()
        fit = refine(request, options, max_iterations=30)
        timings.append(perf_counter() - start)
        if previous is not None:
            np.testing.assert_array_equal(fit.calculation.calculated_y, previous[0])
            np.testing.assert_array_equal(fit.history, previous[1])
        previous = (fit.calculation.calculated_y.copy(), fit.history.copy())
        diagnostics.append(dict(fit.diagnostics))
        error = np.linalg.norm(fit.calculation.calculated_y - y) / np.linalg.norm(y)
        assert error < 1e-6, (fit.termination_reason, error)
        assert fit.termination_reason == "converged", fit.termination_reason
        records.append(
            {
                "relative_l2": float(error),
                "rank": fit.rank,
                "accepted_steps": len(fit.history) - 1,
                "termination": fit.termination_reason,
                "profile_sha256": hashlib.sha256(
                    fit.calculation.calculated_y.tobytes()
                ).hexdigest(),
                "history_sha256": hashlib.sha256(fit.history.tobytes()).hexdigest(),
                "evaluations": fit.diagnostics["evaluations"],
                "linear_iterations": fit.diagnostics["linear_iterations"],
            }
        )
    assert all(r == records[0] for r in records), "repeatability failed"
    return dict(
        reflections=reflections,
        samples=samples,
        phases=2,
        fcj=axial,
        joint_width=joint,
        serial_native=True,
        solver=solver,
        fixed_spectrum=spectrum,
        lorentzian_boundary=boundary,
        jacobian_storage_elements=fit.calculation.jacobian_operator.storage_elements,
        times_seconds=timings,
        median_seconds=float(np.median(timings)),
        p95_seconds=float(np.quantile(timings, 0.95)),
        scientific=records[0],
        diagnostics=diagnostics,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--spectrum", action="store_true")
    parser.add_argument(
        "--boundary", action="store_true", help="Add a 256-peak zero-width face fit"
    )
    parser.add_argument("--solver", choices=("dense", "matrix_free"), default="dense")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = [(256, 10001, False, False), (256, 10001, True, True)]
    if args.large:
        cases.append((811, 23003, False, False))
    binary_hash = hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest()
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    results = [run_case(*c, args.repeats, args.solver, args.spectrum) for c in cases]
    if args.boundary:
        results.append(
            run_case(256, 10001, False, True, args.repeats, args.solver, args.spectrum, True)
        )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        build="maturin develop --release",
        native_binary_sha256=binary_hash,
        runner_sha256=runner_hash,
        peak_process_rss_bytes=peak if platform.system() == "Darwin" else peak * 1024,
        cases=results,
        note=(
            "Process RSS includes Python, reference data and returned derivatives "
            "for the selected solver."
        ),
    )
    with open(args.output, "x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
