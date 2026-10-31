"""Re-fit a plain-array declared scaling design without any research imports."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from phasesmith.experimental.empirical_scaling import (
    ScalingDesign,
    ScalingOptions,
    fit_empirical_scale,
    score_equivalents,
)


def run(case, output):
    with np.load(case, allow_pickle=False) as z:
        a = dict(z)
    design = ScalingDesign(
        *(a[k] for k in ("frames", "detector", "harmonics", "penalty", "lower", "upper"))
    )
    start = time.perf_counter()
    report, arrays = {}, {}
    for name in ["native", "ours"]:
        fit = fit_empirical_scale(
            design,
            a[name + "_intensity"],
            a["log_variance"],
            a["groups"],
            a["training"],
            options=ScalingOptions(strength=float(a["strength"]), iterations=150),
        )
        before = score_equivalents(
            design,
            a[name + "_intensity"],
            a["log_variance"],
            a["groups"],
            a["test"],
            np.zeros(design.parameters),
        )
        after = score_equivalents(
            design,
            a[name + "_intensity"],
            a["log_variance"],
            a["groups"],
            a["test"],
            fit.coefficients,
        )
        old = a[name + "_coefficients"]
        report[name] = dict(
            converged=fit.converged,
            coefficient_max_difference=float(abs(fit.coefficients - old).max()),
            scale_max_relative_difference=float(
                abs(fit.correction / a[name + "_correction"] - 1).max()
            ),
            test_rms_log_before=before["rms_log"],
            test_rms_log_after=after["rms_log"],
        )
        arrays[name + "_coefficients"] = fit.coefficients
        arrays[name + "_correction"] = fit.correction
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "scales.npz", **arrays)
    report["elapsed_seconds"] = time.perf_counter() - start
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.case, args.output)
