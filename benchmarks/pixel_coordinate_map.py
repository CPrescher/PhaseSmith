"""Deterministic batch geometry adapter benchmark; no scalar-only timing."""

import json
from dataclasses import replace
from time import perf_counter

import numpy as np
from phasesmith.experimental.prediction import predict_spots
from phasesmith.experimental.reciprocal import PanelGeometry, reciprocal_vectors


class ShiftScale:
    def to_plane(self, xy):
        return xy * [1.1, 0.9] + [3, -2]

    def from_plane(self, xy):
        return (xy - [3, -2]) / [1.1, 0.9]

    def inverse_jacobian(self, xy):
        return np.broadcast_to(np.diag([1 / 1.1, 1 / 0.9]), (len(xy), 2, 2))


def main():
    rng = np.random.default_rng(193742)
    geometry = PanelGeometry.from_poni(
        distance=0.15,
        poni1=0.08,
        poni2=0.08,
        pixel1=75e-6,
        pixel2=75e-6,
        wavelength=0.2846,
        axis=[-1, 0, 0],
    )
    mapped = replace(geometry, pixel_map=ShiftScale())
    observations = np.c_[rng.uniform(0, 2100, (32000, 2)), rng.uniform(-35, 35, 32000)]
    hkl = rng.integers(-8, 9, (32000, 3), dtype=np.int64)
    basis = np.diag([1 / 5, 1 / 8, 1 / 18])
    results = {}
    for name, panel in [("affine", geometry), ("mapped", mapped)]:
        results[name] = {}
        for operation, function in [
            ("reciprocal", lambda panel=panel: reciprocal_vectors(observations, panel)),
            (
                "prediction",
                lambda panel=panel: predict_spots(hkl, basis, panel, observations[:, 2]),
            ),
        ]:
            function()
            elapsed = []
            for _ in range(5):
                start = perf_counter()
                function()
                elapsed.append(perf_counter() - start)
            results[name][operation + "_seconds_median"] = float(np.median(elapsed))
    print(
        json.dumps(
            dict(reflections=len(hkl), measurements=len(observations), results=results), indent=2
        )
    )


if __name__ == "__main__":
    main()
