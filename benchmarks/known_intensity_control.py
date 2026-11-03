"""Known signed/null intensity control, with detector noise and exact finite truth.

The detector-count mean is known. No external intensity defines the truth.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from phasesmith.experimental.integration import IntegrationOptions, ReflectionTable
from phasesmith.experimental.profiles import (
    SCATTERING_VARIATIONS,
    ProfileModel,
    ProfileObservations,
    ProfileOptions,
    fit_profiles,
)
from phasesmith.experimental.profiles_reference import evaluate


def stats(values, truth):
    report = []
    for level in np.unique(truth):
        selected = truth == level
        v = values[selected]
        delta = v[:, 0] - level
        pull = delta / v[:, 1]
        report.append(
            dict(
                truth=float(level),
                n=int(selected.sum()),
                mean=float(np.mean(v[:, 0])),
                bias=float(np.mean(delta)),
                mean_sigma=float(np.mean(v[:, 1])),
                negative=int(np.count_nonzero(v[:, 0] < 0)),
                mean_pull=float(np.mean(pull)),
                pull_std=float(np.std(pull, ddof=1)),
            )
        )
    return report


def main():
    root = Path("runs/profile-bias-investigation-20261025/known-intensity")
    root.mkdir(exist_ok=True)
    rng = np.random.default_rng(598731)
    levels = np.array([0.0, 30.0, 100.0, 300.0, 1000.0, 5000.0])
    truth = np.repeat(levels, 500)
    n = len(truth)
    phase = np.array(
        [[0.17, -0.23, 0.19], [-0.21, 0.13, -0.37], [0.33, 0.21, -0.07], [-0.37, -0.31, 0.29]]
    )
    z, y, x = np.indices((5, 5, 5))
    p = np.exp(-0.5 * ((x - 2) ** 2 + (y - 2) ** 2 + (z - 2) ** 2))
    p[[0, -1]] = 0
    p[:, [0, -1]] = 0
    p[:, :, [0, -1]] = 0
    p /= p.sum()
    angle_bounds = np.column_stack([np.arange(7) - 3.5, np.arange(7) - 2.5])
    yy, xx = np.mgrid[-3:4, -3:4]
    local = np.column_stack(
        [xx.ravel() - 0.5, xx.ravel() + 0.5, yy.ravel() - 0.5, yy.ravel() + 0.5]
    )
    templates = np.array(
        [
            [
                evaluate(p, np.column_stack([local, np.tile(b, (49, 1))]), centre=c)[:, 0]
                for b in angle_bounds
            ]
            for c in phase
        ]
    )
    np.testing.assert_allclose(templates.sum(axis=(1, 2)), 1.0, atol=3e-15)
    ids = np.tile(np.arange(n), 7)
    frames = np.repeat(np.arange(7), n)
    offsets = np.arange(len(ids) + 1) * 49
    centres = (
        np.column_stack([10 + 12 * np.arange(n), np.full(n, 20.0), np.zeros(n)])
        + phase[np.arange(n) % 4]
    )
    boxes = np.column_stack(
        [10 + 12 * ids - 3, np.full(len(ids), 17), 10 + 12 * ids + 4, np.full(len(ids), 24)]
    )
    P = templates[ids % 4, frames].reshape(-1)
    mean = 82.0 + np.repeat(truth[ids], 49) * P
    raw = rng.poisson(mean).astype(float)
    samples = np.column_stack([raw - 82.0, np.maximum(raw, 1.0), np.full(len(raw), 82.0)])
    t = ReflectionTable(
        np.column_stack([np.arange(n) + 1, np.zeros((n, 2), int)]),
        centres,
        np.zeros(n, np.int64),
        np.zeros((n, 6)),
        np.zeros(n, np.uint32),
        ids,
        frames,
        boxes,
        np.zeros((len(ids), 11)),
        np.zeros(len(ids), np.uint32),
        angle_bounds,
        tuple(str(i) for i in range(7)),
        IntegrationOptions(),
        "poisson_plus_read_noise_with_variance_floor",
        (1.0, 0.0, None),
    )
    o = ProfileObservations(
        t, samples, np.ones(len(raw), bool), offsets, np.zeros((len(ids), 9)), centres
    )
    opts = ProfileOptions(shape=(5, 5, 5), regions=(1, 1, 1), min_references=2, cutoff=0)
    m = ProfileModel(
        np.tile(p.ravel(), (2, 1)),
        np.zeros(2, int),
        np.zeros((2, 3)),
        np.array([], int),
        "known analytical profile",
        (50, int(12 * n + 30)),
        (-3.5, 3.5),
        opts,
    )
    fitted = {}
    fitted["native_known_profile"] = fit_profiles(o, m).values
    fitted["native_physical_variations"] = fit_profiles(
        o, m, variation_factors=SCATTERING_VARIATIONS, allow_shape_refinement=True
    ).values
    # Independent strong-reference shapes are frozen before weak measurement.
    trained = fit_profiles(
        o, m, variation_factors=SCATTERING_VARIATIONS, allow_shape_refinement=True
    )
    fractions = trained.coefficients[truth == 5000] / trained.values[truth == 5000, 0, None]
    fractions = np.median(fractions, axis=0)
    fractions /= fractions.sum()
    fitted["native_frozen_strong_shapes"] = fit_profiles(
        o, m, variation_factors=SCATTERING_VARIATIONS, fixed_fractions=np.tile(fractions, (n, 1))
    ).values
    narrow = replace(m, options=replace(opts, step=(0.75, 0.75, 0.75)))
    fitted["native_narrow_profile_control"] = fit_profiles(o, narrow).values
    b = samples.copy()
    b[:, 0] += 1.0
    b[:, 2] -= 1.0
    fitted["native_background_low_by_one_count"] = fit_profiles(replace(o, samples=b), m).values
    record = {name: stats(v, truth) for name, v in fitted.items()}
    (root / "native-results.json").write_text(json.dumps(record, indent=2) + "\n")
    # DIALS standalone fitting receives the same known profiles, pixels and background.
    # Array order is trial, exposure, pixel; no positive-intensity selection.
    np.savez_compressed(
        root / "oracle-input.npz",
        profiles=P.reshape(7, n, 49).transpose(1, 0, 2).reshape(n, -1),
        data=raw.reshape(7, n, 49).transpose(1, 0, 2).reshape(n, -1),
        background=np.full((n, 343), 82.0),
        truth=truth,
    )
    np.savez_compressed(root / "native-arrays.npz", truth=truth, **fitted)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
