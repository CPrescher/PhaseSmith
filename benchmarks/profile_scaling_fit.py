"""Realistic native profile/scaling optimizer benchmark; no atomic targets.

Run the same script in preserved baseline and current wheel environments. The
optional external case provides original multi-exposure shoeboxes and a learned
31x31x23 bank; no raw frames or structure reference are read by this probe.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from phasesmith import _core
from phasesmith.experimental.empirical_scaling import (
    ScalingDesign,
    ScalingOptions,
    fit_empirical_scale,
)
from phasesmith.experimental.profile_refinement import ShapeFitOptions, refine_profiles
from phasesmith.experimental.profiles import _reference_observations
from phasesmith.experimental.reduction_io import load_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    record = dict(
        schema="phasesmith.native-profile-scaling-benchmark.v1",
        native=hasattr(_core, "experimental_profile_refine"),
        numpy=np.__version__,
        native_module_sha256=hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        optional_packages_available={
            name: importlib.util.find_spec(name) is not None
            for name in ("scipy", "h5py", "hdf5plugin")
        },
    )
    if args.case:
        obs, model = load_case(args.case)
        table = obs.summation
        selected = np.flatnonzero(
            (table.flags == 0)
            & (table.values[:, 0] > 50 * table.values[:, 1])
            & (table.values[:, 1] > 0)
        )[:64]
        subset = _reference_observations(obs, selected)
        model = replace(
            model,
            training_fingerprint=subset.fingerprint(),
            reference_ids=np.empty(0, np.int64),
            reference_centres=np.empty((0, 3)),
        )
        seconds, hashes = [], []
        for _ in range(3):
            start = time.perf_counter()
            fit = refine_profiles(
                subset, model, np.ones(len(selected), np.int64), options=ShapeFitOptions(workers=1)
            )
            seconds.append(time.perf_counter() - start)
            hashes.append(
                hashlib.sha256(
                    fit.parameters.tobytes() + fit.integration.values.tobytes()
                ).hexdigest()
            )
        assert len(set(hashes)) == 1
        record["profiles"] = dict(
            rows=len(selected),
            samples=len(subset.samples),
            shape=list(model.options.shape),
            workers=1,
            median_seconds=float(np.median(seconds)),
            repeats=seconds,
            deterministic=True,
            sha256=hashes[0],
            converged=int(fit.converged.sum()),
            failed=int(np.count_nonzero(fit.integration.flags & 16384)),
        )
    rng = np.random.default_rng(5812831)
    n, nf, nd, na = 4096, 141, 24, 8
    frame = rng.integers(0, nf + 1, n)
    frames = np.eye(nf + 1)[frame, :nf]
    detector = np.zeros((n, nd))
    nodes = rng.integers(0, nd, (n, 4))
    weights = rng.dirichlet(np.ones(4), n)
    for i in range(n):
        np.add.at(detector[i], nodes[i], weights[i])
    harmonic = rng.normal(scale=0.2, size=(n, na))
    p = nf + nd + na
    design = ScalingDesign(
        frames,
        detector,
        harmonic,
        np.eye(p) / 0.25,
        np.r_[np.full(nf, -np.log(2)), np.full(nd, -0.5), np.full(na, -1.0)],
        np.r_[np.full(nf, np.log(2)), np.full(nd + na, 1.0)],
    )
    labels = np.repeat(np.arange(n // 4), 4)
    teachers = labels % 5 != 0
    b = rng.normal(scale=0.015, size=p)
    f = frames @ b[:nf]
    d = 1 + detector @ b[nf : nf + nd]
    a = 1 + harmonic @ b[nf + nd :]
    intensity = np.exp(np.repeat(rng.uniform(6, 9, n // 4), 4) + f) * d * a
    intensity *= np.exp(rng.normal(scale=0.01, size=n))
    intensity[::53] *= 1.5
    intensity[-2:] = [-20, 0]
    teachers[-4:] = False
    seconds, hashes = [], []
    for _ in range(3):
        start = time.perf_counter()
        fit = fit_empirical_scale(
            design, intensity, np.full(n, 0.02**2), labels, teachers, options=ScalingOptions()
        )
        seconds.append(time.perf_counter() - start)
        hashes.append(hashlib.sha256(fit.coefficients.tobytes()).hexdigest())
    assert fit.converged and len(set(hashes)) == 1
    record["scaling"] = dict(
        rows=n,
        parameters=p,
        references=int(fit.reference_mask.sum()),
        median_seconds=float(np.median(seconds)),
        repeats=seconds,
        iterations=len(fit.history),
        deterministic=True,
        coefficients=fit.coefficients.tolist(),
        correction=fit.correction.tolist(),
    )
    record["loaded_optional_packages"] = [
        name for name in ("scipy", "h5py", "hdf5plugin") if name in sys.modules
    ]
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: value
                if key != "scaling"
                else {k: v for k, v in value.items() if k not in ("coefficients", "correction")}
                for key, value in record.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
