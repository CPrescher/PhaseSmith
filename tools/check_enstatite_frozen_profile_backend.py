"""Isolate native integration/scaling from optimizer paths with saved v9 shapes.

This diagnostic intentionally reads archived measurements and parameters. It is
separate from the fresh raw workflow and never starts a structure search.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from phasesmith.experimental.reciprocal import PanelGeometry
from phasesmith.experimental.reduction import ReductionConfig, reduce_observations
from phasesmith.experimental.reduction_io import load_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    observations, bank = load_case(args.baseline / "raw-observations-and-bank.npz")
    geometry = json.loads((args.baseline / "geometry.json").read_text())
    metadata = json.loads((args.baseline / "reduction.json").read_text())["metadata"]
    saved = np.load(args.baseline / "reduced-observations.npz", allow_pickle=False)
    references = np.zeros(len(observations.centres), bool)
    references[bank.reference_ids] = True
    result = reduce_observations(
        observations,
        bank.detector_shape,
        ReductionConfig.from_dict(metadata["config"]),
        geometry=PanelGeometry(geometry["geometry_vectors"], geometry["wavelength_angstrom"]),
        model=bank,
        reference_mask=references,
        choices=saved["choices"],
        frozen_parameters=saved["parameters"],
    )
    use = saved["use"]
    old = np.column_stack((saved["intensity"], saved["sigma"]))[use]
    new = result.intensity_sigma[use]
    relative = np.linalg.norm(new - old, axis=0) / np.linalg.norm(old, axis=0)
    raw_old = saved["raw_profiles"][use, :2]
    raw_new = result.profile.values[use, :2]
    raw_relative = np.linalg.norm(raw_new - raw_old, axis=0) / np.linalg.norm(raw_old, axis=0)
    standardized = abs(new[:, 0] - old[:, 0]) / old[:, 1]
    record = dict(
        schema="phasesmith.enstatite-frozen-profile-backend-control.v1",
        baseline=str(args.baseline),
        archived_arrays_read=True,
        atomic_reference_read=False,
        structure_search_executed=False,
        frozen_legacy_shape_parameters=True,
        observations=int(use.sum()),
        flags_identical=bool(np.array_equal(result.profile.flags, saved["flags"])),
        intensity_relative_l2=float(relative[0]),
        sigma_relative_l2=float(relative[1]),
        raw_profile_intensity_relative_l2=float(raw_relative[0]),
        raw_profile_sigma_relative_l2=float(raw_relative[1]),
        maximum_intensity_difference_in_baseline_sigma=float(standardized.max()),
        percentile95_intensity_difference_in_baseline_sigma=float(np.quantile(standardized, 0.95)),
        interpretation=(
            "Diagnostic only: fixed legacy shapes isolate native GLS, joint sigma and "
            "scaling from independent shape-optimizer trajectories."
        ),
    )
    record["passed_strict_reduction_targets"] = bool(
        max(relative) <= 1e-4
        and standardized.max() <= 0.1
        and np.quantile(standardized, 0.95) <= 0.01
        and record["flags_identical"]
    )
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(record, indent=2))
    return 0 if record["passed_strict_reduction_targets"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
