"""Read-only development diagnostics and a multiplicity-reduction algebra probe.

This does not implement a reduced production refiner or measure a speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
from collections import Counter
from pathlib import Path

import gemmi
import numpy as np
from phasesmith import _core
from phasesmith.io.space_groups import space_group_by_symbol

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def family_probe(symbol, cell, sites, mode):
    operations = gemmi.find_spacegroup_by_name(symbol).operations()

    def orbit(h):
        return sorted(
            {tuple(sign * v for v in op.apply_to_hkl(h)) for op in operations for sign in (-1, 1)}
        )

    families = {}
    for h in itertools.product(range(-2, 3), repeat=3):
        if h != (0, 0, 0) and not operations.is_systematically_absent(h):
            mates = orbit(h)
            families[mates[0]] = mates
    hkl = sorted({h for mates in families.values() for h in mates})
    lookup = {h: i for i, h in enumerate(hkl)}
    groups = [np.array([lookup[h] for h in mates]) for mates in families.values()]
    representatives = np.array([indices[0] for indices in groups])
    multiplicity = np.array([len(indices) for indices in groups])
    observed = np.empty(len(hkl))
    rng = np.random.default_rng(301)
    for indices in groups:
        observed[indices] = rng.uniform(1, 20)
    species, points = [], []
    for element, point in sites:
        unique = []
        for op in operations:
            q = np.mod(op.apply_to_xyz(point), 1)
            if not any(np.linalg.norm((q - p + 0.5) % 1 - 0.5) < 1e-8 for p in unique):
                unique.append(q)
        species.extend([element] * len(unique))
        points.extend(unique)
    native = _core._SymmetricSolutionModel(
        cell,
        space_group_by_symbol(symbol).space_group._native,
        species,
        np.ascontiguousarray(points).ravel(),
        np.asarray(hkl, dtype=np.int64).ravel(),
        observed,
        mode,
        0.01,
        True,
    )
    parameters = np.asarray(native.initial_parameters(0.03))
    parameters += rng.normal(0, 0.005, len(parameters))
    state = native.evaluate(parameters)
    residual = np.asarray(state["residual"])
    jacobian = np.asarray(state["jacobian"]).reshape(len(parameters), len(hkl))
    # These derivatives already include the profiled amplitude-scale derivative.
    # Multiplicity weights must preserve that same global scale and normalization.
    for indices in groups:
        np.testing.assert_allclose(residual[indices], residual[indices[0]], atol=2e-12, rtol=1e-10)
        np.testing.assert_allclose(
            jacobian[:, indices],
            np.repeat(jacobian[:, indices[:1]], len(indices), axis=1),
            atol=2e-12,
            rtol=1e-10,
        )
    weighted_r = residual[representatives] * np.sqrt(multiplicity)
    weighted_j = jacobian[:, representatives] * np.sqrt(multiplicity)
    full_normal = jacobian @ jacobian.T
    reduced_normal = weighted_j @ weighted_j.T
    full_gradient = jacobian @ residual
    reduced_gradient = weighted_j @ weighted_r
    amplitudes = (residual * np.linalg.norm(observed) + observed) / state["amplitude_scale"]
    representative_a = amplitudes[representatives]
    representative_o = observed[representatives]
    reduced_scale = np.sum(multiplicity * representative_a * representative_o) / np.sum(
        multiplicity * representative_a**2
    )
    reduced_r1 = np.sum(
        multiplicity * np.abs(reduced_scale * representative_a - representative_o)
    ) / np.sum(multiplicity * representative_o)
    np.testing.assert_allclose(reduced_scale, state["amplitude_scale"], atol=2e-12, rtol=1e-10)
    np.testing.assert_allclose(reduced_r1, state["r1"], atol=2e-12, rtol=1e-10)
    np.testing.assert_allclose(weighted_r @ weighted_r, state["objective"], atol=2e-12, rtol=1e-10)
    np.testing.assert_allclose(reduced_gradient, full_gradient, atol=2e-12, rtol=1e-10)
    np.testing.assert_allclose(reduced_normal, full_normal, atol=2e-12, rtol=1e-10)
    return {
        "space_group": symbol,
        "mode": mode,
        "expanded_rows": len(hkl),
        "families": len(groups),
        "parameters": len(parameters),
        "scale_absolute_difference": abs(float(reduced_scale) - state["amplitude_scale"]),
        "r1_absolute_difference": abs(float(reduced_r1) - state["r1"]),
        "objective_absolute_difference": abs(float(weighted_r @ weighted_r) - state["objective"]),
        "gradient_maximum_absolute_difference": float(
            np.max(np.abs(reduced_gradient - full_gradient))
        ),
        "normal_maximum_absolute_difference": float(np.max(np.abs(reduced_normal - full_normal))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "benchmarks/structure-solution-phase-workspace-fused-macos.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = json.loads(args.report.read_text())
    if panel["split"] != "development" or panel["holdout_opened"]:
        raise ValueError("requires a closed-holdout development report")
    manifest = json.loads((ROOT / "validation/solution-corpus-v5.json").read_text())
    expected = {c["id"] for c in manifest["cases"] if c["split"] == "development"}
    rows = [r for r in panel["runs"] if r["method"] == "phasesmith"]
    if len(rows) != 35 or {r["id"] for r in rows} != expected:
        raise ValueError("requires all 35 development cases")
    stages = Counter()
    cases = []
    for row in rows:
        effort = row["effort"]
        phase = sum(s["seconds"] for s in effort["phase_slices"])
        final = sum(s["refinement_seconds"] for s in effort["phase_origins"])
        families = effort["phase_quality"][0]["allowed_families"]
        expanded = row["expanded_measured_rows_for_resource_cap"]
        for trace in effort["phase_origins"]:
            stages.update(
                {
                    k: trace[k]
                    for k in (
                        "origin_seconds",
                        "recycling_seconds",
                        "completion_seconds",
                        "screening_seconds",
                        "refinement_seconds",
                    )
                }
            )
        cases.append(
            {
                "id": row["id"],
                "correct": row["correct"],
                "total_count_correct": row["total_non_h_count_correct"],
                "element_counts_correct": row.get("reference_audit", {}).get("counts_match", False),
                "target_met": row["target_met"],
                "reference_exceeds_native_resource_cap": row[
                    "reference_exceeds_native_resource_cap"
                ],
                "families": families,
                "expanded_rows": expanded,
                "expansion_ratio": expanded / families,
                "phase_seconds": phase,
                "refinement_seconds": final,
            }
        )
    examples = [
        ("P 1", [6, 7, 8, 78, 96, 107], [("Si", [0.1, 0.2, 0.3]), ("O", [0.4, 0.6, 0.2])]),
        ("P -1", [6, 7, 8, 78, 96, 107], [("Si", [0.12, 0.23, 0.31]), ("O", [0, 0, 0])]),
        ("P m", [6, 7, 8, 90, 96, 90], [("Si", [0.41, 0, 0.12]), ("O", [0.15, 0.26, 0.33])]),
        (
            "P 31 2 1",
            [4.913, 4.913, 5.405, 90, 90, 120],
            [("Si", [0.4697, 0, 1 / 3]), ("O", [0.4133, 0.2672, 0.1188])],
        ),
        ("F m -3 m", [4.2, 4.2, 4.2, 90, 90, 90], [("Si", [0, 0, 0]), ("O", [0.5, 0.5, 0.5])]),
    ]
    checks = [
        family_probe(*example, mode)
        for example in examples
        for mode in ("isotropic", "anisotropic")
    ]
    result = {
        "schema": "phasesmith.next-routes-diagnostic.v1",
        "report": str(args.report.relative_to(ROOT) if args.report.is_absolute() else args.report),
        "report_sha256": sha(args.report),
        "probe_source_sha256": sha(__file__),
        "native_sha256": sha(_core.__file__),
        "holdout_outcomes_opened": False,
        "qualification": (
            "Saved development outcomes plus synthetic algebra checks; no new solves or speed "
            "measurements. The probe folds existing full Jacobians; it is not a reduced "
            "production evaluator."
        ),
        "stage_seconds": dict(stages),
        "phase_seconds": sum(c["phase_seconds"] for c in cases),
        "median_expansion_ratio": statistics.median(c["expansion_ratio"] for c in cases),
        "total_count_right_elements_wrong": sum(
            c["total_count_correct"] and not c["element_counts_correct"] for c in cases
        ),
        "incorrect_below_r1_target": [
            c["id"] for c in cases if not c["correct"] and c["target_met"]
        ],
        "correct_above_r1_target": [c["id"] for c in cases if c["correct"] and not c["target_met"]],
        "phase_stop_reasons": dict(
            Counter(s["stop_reason"] for r in rows for s in r["effort"]["phase_slices"])
        ),
        "cases": cases,
        "synthetic_multiplicity_checks": checks,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("cases", "synthetic_multiplicity_checks")
            },
            indent=2,
        )
    )
    print(f"Passed {len(checks)} synthetic objective/gradient/normal-matrix equivalence checks")


if __name__ == "__main__":
    main()
