"""Independent known-intensity stress test: curved ring versus a local plane.

The ring-template arm is a conditional known-background-shape control, not a
new production preprocessing mode or a claim that its template can be measured
without uncertainty on the real Fe7C3 scan.
"""

import json
from pathlib import Path

import numpy as np
from phasesmith.experimental.integration import IntegrationOptions
from phasesmith.experimental.prediction import ScanPredictions
from phasesmith.experimental.profiles import (
    ProfileModel,
    ProfileOptions,
    fit_profiles,
    prepare_profiles,
)
from phasesmith.experimental.profiles_reference import evaluate


def main():
    root = Path("runs/profile-bias-investigation-20261025/ring-background")
    root.mkdir(exist_ok=True)
    yy, xx = np.indices((61, 61))
    rad = np.hypot(xx - 30, yy - 30)
    ring = 40 * np.exp(-0.5 * ((rad - 20) / 1.2) ** 2)
    z, y, x = np.indices((5, 5, 5))
    p = np.exp(-0.5 * ((x - 2) ** 2 + (y - 2) ** 2 + (z - 2) ** 2))
    p[[0, -1]] = 0
    p[:, [0, -1]] = 0
    p[:, :, [0, -1]] = 0
    p /= p.sum()
    angle = np.column_stack([np.arange(7) - 3.5, np.arange(7) - 2.5])
    rng = np.random.default_rng(20261026)
    options = IntegrationOptions(
        pixel_radius=5,
        omega_half_width=3.2,
        background_model="plane",
        background_gap=2,
        background_margin=4,
    )
    profile_options = ProfileOptions(shape=(5, 5, 5), regions=(1, 1, 1), min_references=2, cutoff=0)
    result = {}
    for label, cy in [("on_ring", 49.83), ("ring_shoulder", 47.83)]:
        centre = np.array([[30.17, cy, 0.19]])
        truth = 1000.0
        P = []
        for lo, hi in angle:
            bounds = np.column_stack(
                [
                    xx.ravel() - 0.5,
                    xx.ravel() + 0.5,
                    yy.ravel() - 0.5,
                    yy.ravel() + 0.5,
                    np.full(xx.size, lo),
                    np.full(xx.size, hi),
                ]
            )
            P.append(evaluate(p, bounds, centre=centre[0])[:, 0].reshape(xx.shape))
        P = np.array(P)
        np.testing.assert_allclose(P.sum(), 1.0, atol=3e-15)
        mean = 82.0 + 0.05 * (xx - 30) + 0.02 * (yy - 30) + ring + truth * P
        prediction = ScanPredictions(np.array([[1, 0, 0]]), centre, np.array([3]), 1.0)
        table = {name: [] for name in ["plane", "known_ring_plus_plane"]}
        errors = {name: [] for name in table}
        flags = {name: [] for name in table}
        for _ in range(30):
            raw = rng.poisson(mean).astype(float)
            for name, offset in [("plane", np.zeros(xx.shape)), ("known_ring_plus_plane", ring)]:
                observations = prepare_profiles(
                    lambda i, raw=raw, offset=offset: raw[i] - offset,
                    angle,
                    prediction,
                    variance_loader=lambda i, mean=mean: mean[i],
                    options=options,
                )
                model = ProfileModel(
                    np.tile(p.ravel(), (2, 1)),
                    np.zeros(2, int),
                    np.zeros((2, 3)),
                    np.array([], int),
                    "independent known profile",
                    (61, 61),
                    (-3.5, 3.5),
                    profile_options,
                )
                fit = fit_profiles(observations, model, expected_noise=False)
                table[name].append(float(fit.values[0, 0]))
                errors[name].append(float(fit.values[0, 1]))
                flags[name].append(int(fit.flags[0]))
        result[label] = {
            name: dict(
                truth=truth,
                n=len(v),
                mean=float(np.mean(v)),
                bias=float(np.mean(v) - truth),
                mean_sigma=float(np.mean(errors[name])),
                flagged=int(np.count_nonzero(flags[name])),
            )
            for name, v in table.items()
        }
    record = dict(
        test="conditional known-ring-template plus local plane against a plane-only model",
        raw_noise=(
            "Poisson draws of the complete raw mean; both arms use identical raw-count variances"
        ),
        template_uncertainty=(
            "zero for this control because its shape is supplied by the generator; "
            "not a real-data validation"
        ),
        results=result,
    )
    (root / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
