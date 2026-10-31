"""Audit the full post-indexing native workflow on the same cached pixels as DIALS."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import phasesmith
from benchmark_3d_profiles import load_case
from phasesmith import _core
from phasesmith.experimental.integration import integrate_series
from phasesmith.experimental.prediction import ScanPredictions
from phasesmith.experimental.profiles import (
    fit_learned_profiles,
    learn_profile_variations,
    learn_profiles,
    prepare_profiles,
)


def main(root, output):
    original, model, frozen = load_case(root)
    protocol = json.loads((root / "protocol.json").read_text())
    case = Path(protocol["case"])
    basis = np.array(protocol["common_crystal_basis"])
    qmax = float(np.linalg.norm(frozen["hkl"] @ basis, axis=1).max())
    predictions = ScanPredictions(
        frozen["hkl"], frozen["predicted"], np.zeros(len(frozen["hkl"]), int), qmax
    )
    masks = {}

    def load(i):
        frame = np.load(case / "images" / f"frame_{i + 1:04d}.npy")
        masks[i] = frame == 0
        return frame

    total = time.perf_counter()
    start = total
    observed = prepare_profiles(
        load,
        original.summation.omega_bounds,
        predictions,
        mask_loader=lambda i: masks.pop(i),
        options=original.summation.options,
        saturation=65535,
        scales=original.scales,
        planar_transforms=original.planar_transforms,
        angular_half_widths=original.summation.angular_half_widths,
    )
    preparation = time.perf_counter() - start
    start = time.perf_counter()
    base = learn_profiles(
        observed, frozen["training"], detector_shape=model.detector_shape, options=model.options
    )
    learning = time.perf_counter() - start
    start = time.perf_counter()
    shapes = learn_profile_variations(observed, base)
    shape_learning = time.perf_counter() - start
    start = time.perf_counter()
    result = fit_learned_profiles(observed, base, shapes)
    fitting = time.perf_counter() - start
    elapsed = time.perf_counter() - total
    for name in ["samples", "valid", "covariance", "centres", "scales", "planar_transforms"]:
        assert getattr(observed, name).tobytes() == getattr(original, name).tobytes(), name
    for name in ("profiles", "shifts", "counts", "reference_ids", "reference_centres"):
        assert getattr(base, name).tobytes() == getattr(model, name).tobytes(), name
    with np.load(root / "model.npz") as saved:
        for name in ("fractions", "centres", "reference_ids", "model_ids"):
            assert getattr(shapes, name).tobytes() == saved["variation_" + name].tobytes(), name
    assert result.values.tobytes() == frozen["profile"].tobytes()
    assert result.flags.tobytes() == frozen["profile_flags"].tobytes()
    start = time.perf_counter()
    summed = integrate_series(
        load,
        original.summation.omega_bounds,
        predictions,
        mask_loader=lambda i: masks.pop(i),
        options=original.summation.options,
        saturation=65535,
        _profile_half_widths=original.summation.angular_half_widths,
    )
    summation_seconds = time.perf_counter() - start
    for name in ("values", "flags", "frame_values", "frame_flags", "background_models"):
        assert getattr(summed, name).tobytes() == getattr(original.summation, name).tobytes(), name
    report = dict(
        case=str(root.resolve()),
        input=(
            "same preserved NPY pixel arrays as DIALS; warm file cache; serial native integration"
        ),
        timer=(
            "post-indexing preparation, background collection, profile learning, "
            "strong-reference shape learning and target fitting; excludes imports, "
            "indexed-table loading, verification and export"
        ),
        preparation_seconds=preparation,
        base_learning_seconds=learning,
        shape_learning_seconds=shape_learning,
        fitting_seconds=fitting,
        total_seconds=elapsed,
        summation_only_seconds=summation_seconds,
        bitwise_identical=True,
        verification="samples, background, base model, strong shape teachers, target fits and sums",
        reflections=len(predictions.hkl),
        samples=len(observed.samples),
        package_path=phasesmith.__file__,
        native_path=_core.__file__,
        native_sha256=hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    main(a.case, a.output)
