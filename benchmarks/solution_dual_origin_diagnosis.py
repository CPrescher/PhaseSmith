"""Audit a frozen failed phase map using only its peaks and supplied symmetry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from phasesmith import _core
from phasesmith.experimental.solution_dual_reference import peaks_reference
from phasesmith.io.space_groups import space_group_by_symbol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--space-group", required=True)
    parser.add_argument("--framework-element", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_result / "result.json"
    maps = args.source_result / "density-and-observations.npz"
    record = json.loads(source.read_text())
    if args.framework_element not in record["input_elements"]:
        parser.error("framework element must be among the original input element types")
    group = space_group_by_symbol(args.space_group).space_group
    operations = [
        {
            "rotation": op.rotation.tolist(),
            "translation": [[v.numerator, v.denominator] for v in op.translation],
        }
        for op in group.operations
    ]
    if operations != record["input_symmetry_operations"]:
        parser.error("supplied symmetry differs from the original operation set")
    with np.load(maps) as data:
        rho = data["density"]
        peaks = peaks_reference(rho, 32)
        indices = (np.floor(peaks * np.array(rho.shape) + 0.5).astype(int) % rho.shape).T
        heights = rho[tuple(indices)]
        prefix = peaks[heights >= 0.75 * heights[0]]
        model = _core._SymmetricSolutionModel(
            record["cell"],
            group._native,
            [args.framework_element] * len(prefix),
            prefix.ravel(),
            data["expanded_hkl"].ravel(),
            data["amplitudes"],
            "isotropic",
            record["options"]["symmetry_tolerance_angstrom"],
        )
    metadata = model.metadata()
    result = {
        "schema": "phasesmith.dual-space-origin-diagnosis.v1",
        "qualification": (
            "Post-solve diagnostic of a frozen failed density map, not solver input. "
            "The 75%-height prefix is data selected. No reference coordinates/formula were read."
        ),
        "source_result_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_maps_sha256": hashlib.sha256(maps.read_bytes()).hexdigest(),
        "original_input_sha256": record["source_sha256"],
        "original_rejections": record["rejected_trials"],
        "space_group": args.space_group,
        "cell": record["cell"],
        "framework_element": args.framework_element,
        "selected_peak_positions": prefix.tolist(),
        "symmetry_tolerance_angstrom": record["options"]["symmetry_tolerance_angstrom"],
        "recovered_maximum_matching_error_angstrom": metadata["maximum_matching_error"],
        "recovered_hand": metadata["hand"],
        "recovered_origin_shift": metadata["origin_shift"],
        "native_sha256": hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
