"""Many-peak finite-voxel value/derivative and empirical-pass timing control."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from phasesmith.experimental.empirical_scaling import ScalingDesign
from phasesmith.experimental.profiles import evaluate_profile, evaluate_profile_shape


def run(repeats=3):
    rng = np.random.default_rng(2910202618)
    z, y, x = np.mgrid[-4:5, -7:8, -7:8]
    p = np.exp(-0.5 * ((x / 2.1) ** 2 + (y / 2.3) ** 2 + (z / 1.7) ** 2))
    p /= p.sum()
    xx, yy, zz = np.meshgrid(
        np.arange(-10, 11), np.arange(-10, 11), np.arange(-5, 6), indexing="ij"
    )
    xyz = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    bounds = np.repeat(xyz, 2, axis=1) + np.tile([-0.5, 0.5], 3)
    centres = rng.uniform(-0.25, 0.25, (128, 3))
    timing = {"centre_only": [], "centre_and_width": []}
    for name, function in [
        ("centre_only", evaluate_profile),
        ("centre_and_width", evaluate_profile_shape),
    ]:
        for _ in range(repeats):
            start = time.perf_counter()
            checksum = 0.0
            for centre in centres:
                checksum += float(
                    function(
                        p,
                        bounds,
                        centre=centre,
                        planar_transform=np.array([[0.97, -0.08], [0.12, 1.03]]),
                    )[:, 0].sum()
                )
            timing[name].append(time.perf_counter() - start)
        assert abs(checksum - 128) < 1e-9
    n, nf, nd = 5389, 141, 24
    frames = np.eye(nf)[np.arange(n) % nf]
    detector = rng.uniform(0, 0.2, (n, nd))
    harmonic = rng.normal(0, 0.1, (n, 8))
    design = ScalingDesign(
        frames, detector, harmonic, np.eye(173), np.full(173, -0.5), np.full(173, 0.5)
    )
    b = rng.normal(0, 0.01, 173)
    start = time.perf_counter()
    for _ in range(20):
        result = design.evaluate(b)
    assert result[2].shape == (n, 173)
    timing["empirical_5389x173_seconds_per_pass"] = (time.perf_counter() - start) / 20
    timing["profile_workload"] = dict(
        peaks=128, samples_per_peak=len(bounds), voxels=int(p.size), repeats=repeats
    )
    timing["scope"] = (
        "same-mass baseline; five analytical partials replace four additional width mass passes"
    )
    return timing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(), indent=2) + "\n")
