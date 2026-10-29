"""Replay a measured many-reflection profile campaign and check exact repeatability."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from phasesmith import _core
from phasesmith.experimental.integration import IntegrationOptions, ReflectionTable
from phasesmith.experimental.profiles import (
    ProfileModel,
    ProfileObservations,
    ProfileOptions,
    fit_learned_profiles,
    fit_profiles,
    learn_profile_variations,
    learn_profiles,
)


def load_case(root):
    protocol = json.loads((root / "protocol.json").read_text())
    summary = json.loads((root / "native-summary.json").read_text())
    with np.load(root / "native.npz") as source:
        a = dict(source)
    with np.load(root / "observations.npz") as source:
        raw = dict(source)
    with np.load(root / "model.npz") as source:
        bank = dict(source)
    t = ReflectionTable(
        a["hkl"],
        a["predicted"],
        np.zeros(len(a["hkl"]), np.int64),
        a["summed"],
        a["sum_flags"],
        a["observation_ids"],
        a["frame_indices"],
        a["boxes"],
        a["frame_values"],
        a["frame_flags"],
        a["omega_bounds"],
        tuple(summary["frame_hashes"]),
        IntegrationOptions(**protocol["options"]),
        "poisson_plus_read_noise_with_variance_floor",
        (1.0, 0.0, 65535.0),
        raw["background_models"],
        a["angular_half_widths"],
    )
    obs = ProfileObservations(
        t,
        raw["samples"],
        raw["valid"],
        raw["offsets"],
        raw["covariance"],
        raw["centres"],
        raw["scales"],
        raw.get("planar_transforms"),
    )
    options = ProfileOptions(**protocol["profile_options"])
    model = ProfileModel(
        bank["profiles"],
        bank["counts"],
        bank["shifts"],
        bank["reference_ids"],
        obs.fingerprint(),
        tuple(protocol["geometry"]["shape"]),
        (float(a["omega_bounds"].min()), float(a["omega_bounds"].max())),
        options,
        bank.get("reference_centres"),
    )
    return obs, model, a


def run(root, output, repeats=3):
    obs, model, frozen = load_case(root)
    learning = []
    fitting = []
    shape_learning = []
    protocol = json.loads((root / "protocol.json").read_text())
    frozen_bank = dict(np.load(root / "model.npz"))
    for _ in range(repeats):
        start = time.perf_counter()
        learned = learn_profiles(
            obs, frozen["training"], detector_shape=model.detector_shape, options=model.options
        )
        learning.append(time.perf_counter() - start)
        for field in ("profiles", "shifts", "counts", "reference_ids"):
            assert getattr(learned, field).tobytes() == getattr(model, field).tobytes(), field
        if protocol.get("frozen_variations"):
            start = time.perf_counter()
            shapes = learn_profile_variations(obs, learned)
            shape_learning.append(time.perf_counter() - start)
            for field in ("fractions", "centres", "reference_ids", "model_ids"):
                assert (
                    getattr(shapes, field).tobytes() == frozen_bank["variation_" + field].tobytes()
                ), field
            start = time.perf_counter()
            result = fit_learned_profiles(obs, learned, shapes)
        else:
            start = time.perf_counter()
            result = fit_profiles(
                obs,
                learned,
                variation_factors=protocol.get("variation_factors"),
                allow_shape_refinement=protocol.get("variation_factors") is not None,
            )
        fitting.append(time.perf_counter() - start)
        assert result.values.tobytes() == frozen["profile"].tobytes()
        np.testing.assert_array_equal(result.flags, frozen["profile_flags"])
    report = dict(
        schema="phasesmith.3d-profile-performance.v1",
        case=str(root.resolve()),
        native_sha256=hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        reflections=len(obs.centres),
        samples=len(obs.samples),
        references=int(model.counts[0]),
        repetitions=repeats,
        shape_learning_seconds=shape_learning,
        learning_seconds=learning,
        fitting_seconds=fitting,
        bitwise_identical=True,
        baseline="first serial production implementation; no speedup claim",
        fingerprint=obs.fingerprint(),
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.case, args.output)
