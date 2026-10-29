"""Replay a portable case plus frozen choices without importing research scripts.

A recipe declares checksummed file paths, detector shape and ReductionConfig.
Writes fresh outputs only. It never runs an atomic refinement or chooses shape
models/regularization from structure agreement.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from phasesmith.experimental.reduction import ReductionConfig, reduce_observations
from phasesmith.experimental.reduction_io import load_case


def replay(recipe_path, output):
    recipe = json.loads(recipe_path.read_text())
    files = {}
    for name, record in recipe["files"].items():
        path = (recipe_path.parent / record["path"]).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"source checksum mismatch: {name}")
        files[name] = path
    observations, model = load_case(files["case"])
    with np.load(files["policy"], allow_pickle=False) as z:
        policy = dict(z)
    config = ReductionConfig.from_dict(recipe["reduction"])
    start = time.perf_counter()
    result = reduce_observations(
        observations,
        tuple(recipe["detector_shape"]),
        config,
        model=model,
        choices=policy["choices"],
        weak_mask=policy["weak_mask"],
        sensor_centroids=policy.get("sensor_centroids"),
        frozen_parameters=policy.get("frozen_parameters"),
        progress=lambda done, total: (
            print(f"{done}/{total}", flush=True) if done % 250 == 0 else None
        ),
    )
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "profiles.npz",
        values=result.profile.values,
        flags=result.profile.flags,
        parameters=result.parameters,
        model_ids=result.profile.model_ids,
    )
    report = dict(result.metadata, elapsed_seconds=time.perf_counter() - start)
    if "reference" in files:
        with np.load(files["reference"], allow_pickle=False) as z:
            reference = dict(z)
        a, b = result.profile.values[:, 0], reference["values"][:, 0]
        finite = np.isfinite(a) & np.isfinite(b)
        relative = abs(a[finite] - b[finite]) / np.maximum(abs(b[finite]), 1.0)
        report["comparison"] = dict(
            rows=len(a),
            finite_rows=int(finite.sum()),
            finite_masks_equal=bool(np.array_equal(np.isfinite(a), np.isfinite(b))),
            maximum_absolute_counts=float(abs(a[finite] - b[finite]).max()),
            relative_quantiles=np.quantile(relative, [0, 0.5, 0.9, 0.99, 1.0]).tolist(),
            maximum_sigma_difference=float(
                np.nanmax(abs(result.profile.values[:, 1] - reference["values"][:, 1]))
            ),
            flags_differ=int(np.count_nonzero(result.profile.flags != reference["flags"])),
        )
        # Numerical optimization can follow a nearby path after replacing FD
        # widths. State errors, never silently overwrite the frozen benchmark.
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report.get("comparison", {}), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.recipe, args.output)
