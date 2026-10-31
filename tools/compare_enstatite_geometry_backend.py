"""Post-run native geometry equivalence against the preserved MgSiO3 v8 result.

Only compares already saved candidates/arrays. No reference information is passed
back into either numerical worker. Targets come from the frozen v9 protocol.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from solution_corpus.audit import match_geometry  # noqa: E402


def compare(baseline, candidate, protocol):
    targets = protocol["geometry_equivalence"]

    def read(root, name):
        return json.loads((root / name).read_text())

    old_report, new_report = read(baseline, "report.json"), read(candidate, "report.json")
    gate = dict(
        baseline_structure_recovery=bool(old_report["raw_to_structure_search_passed"]),
        primary_structure_recovery=bool(new_report["raw_to_structure_search_passed"]),
        source_files_unchanged=bool(new_report["source_files_unchanged"]),
        same_source_manifest=read(baseline, "provenance.json")["source_hashes"]
        == read(candidate, "provenance.json")["source_hashes"],
    )
    identities = {}
    for filename in ("positions.npz", "indexed-positions.npz", "edge-exclusions.npz"):
        with (
            np.load(baseline / filename, allow_pickle=False) as old,
            np.load(candidate / filename, allow_pickle=False) as new,
        ):
            keys = (
                old.files
                if filename == "positions.npz"
                else (
                    ["hkl", "observed", "training", "eligible", "track_ids"]
                    if filename == "indexed-positions.npz"
                    else ["hkl", "flags"]
                )
            )
            identities[filename] = {
                key: bool(np.array_equal(old[key], new[key], equal_nan=old[key].dtype.kind in "fc"))
                for key in keys
            }
    gate["frozen_positions_partitions_and_edges"] = all(
        all(v.values()) for v in identities.values()
    )
    with (
        np.load(baseline / "indexed-positions.npz") as old,
        np.load(candidate / "indexed-positions.npz") as new,
    ):
        use = old["eligible"]
        delta = new["predicted"][use] - old["predicted"][use]
        geometry = dict(
            rows=int(use.sum()),
            maximum_xy_pixels=float(np.linalg.norm(delta[:, :2], axis=1).max()),
            maximum_omega_degrees=float(abs(delta[:, 2]).max()),
            native_metadata=read(candidate, "geometry.json"),
        )
    gate["prediction_xy"] = (
        geometry["maximum_xy_pixels"] <= targets["maximum_prediction_xy_difference_pixels"]
    )
    gate["prediction_omega"] = (
        geometry["maximum_omega_degrees"] <= targets["maximum_prediction_omega_difference_degrees"]
    )
    with (
        np.load(baseline / "reduced-observations.npz") as old,
        np.load(candidate / "reduced-observations.npz") as new,
    ):
        identical = {
            key: bool(np.array_equal(old[key], new[key])) for key in ("hkl", "use", "flags")
        }
        gate["exported_hkl_and_use_mask"] = identical["hkl"] and identical["use"]
        use = old["use"]
        intensity = new["intensity"][use] - old["intensity"][use]
        sigma = new["sigma"][use] - old["sigma"][use]
        standardized = abs(intensity) / old["sigma"][use]
        reduction = dict(
            observations=int(use.sum()),
            negative_observations=int((new["intensity"][use] < 0).sum()),
            identity=identical,
            intensity_relative_l2=float(
                np.linalg.norm(intensity) / np.linalg.norm(old["intensity"][use])
            ),
            sigma_relative_l2=float(np.linalg.norm(sigma) / np.linalg.norm(old["sigma"][use])),
            maximum_intensity_difference_in_baseline_sigma=float(standardized.max()),
            percentile95_intensity_difference_in_baseline_sigma=float(
                np.quantile(standardized, 0.95)
            ),
        )
    for key, target in (
        ("intensity_relative_l2", "intensity_relative_l2_difference"),
        ("sigma_relative_l2", "sigma_relative_l2_difference"),
        (
            "maximum_intensity_difference_in_baseline_sigma",
            "maximum_intensity_difference_in_baseline_sigma",
        ),
        (
            "percentile95_intensity_difference_in_baseline_sigma",
            "percentile95_intensity_difference_in_baseline_sigma",
        ),
    ):
        gate[key] = reduction[key] <= targets[target]
    old_search, new_search = read(baseline, "search.json"), read(candidate, "search.json")
    gate["r1_target"] = bool(new_search["target_met"])
    r1_difference = abs(old_search["amplitude_r1"] - new_search["amplitude_r1"])
    gate["amplitude_r1_difference"] = r1_difference <= targets["maximum_amplitude_r1_difference"]
    old_solution, new_solution = (
        read(baseline, "candidate/result.json"),
        read(candidate, "candidate/result.json"),
    )
    a, b = old_solution["candidates"][0], new_solution["candidates"][0]
    gate["composition"] = collections.Counter(a["species"]) == collections.Counter(b["species"])
    model = dict(b, cell=new_solution["cell"])
    coordinates = match_geometry(
        model, np.array(a["species"]), np.array(a["fractional_xyz"]), new_solution["cell"]
    )
    gate["candidate_coordinates"] = (
        coordinates["maximum_error_angstrom"]
        <= targets["maximum_candidate_coordinate_difference_angstrom"]
    )
    completion = read(candidate, "worker-complete.json")
    gate["worker_isolation"] = not any(
        completion[k] for k in ("reference_model_read", "source_hkls_read", "archived_arrays_read")
    )
    return dict(
        schema="phasesmith.enstatite-native-geometry-comparison.v1",
        baseline=str(baseline),
        output=str(candidate),
        targets=targets,
        passed=all(gate.values()),
        gates=gate,
        frozen_array_identity=identities,
        geometry=geometry,
        reduction=reduction,
        search=new_search,
        amplitude_r1_difference=r1_difference,
        composition=dict(sorted(collections.Counter(b["species"]).items())),
        candidate_coordinate_difference=coordinates,
        reference_assessment=new_report["reference_assessment"],
        completion=completion,
        protocol_provenance=read(candidate, "provenance.json"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "validation/enstatite-raw-workflow-v9.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate, json.loads(args.protocol.read_text()))
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                passed=report["passed"],
                gates=report["gates"],
                geometry=report["geometry"]["maximum_xy_pixels"],
                reduction=report["reduction"],
                r1=report["search"]["amplitude_r1"],
            ),
            indent=2,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
