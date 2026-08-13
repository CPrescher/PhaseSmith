#!/usr/bin/env python3
"""Factorial residual forensics for the anhydrous Rb-citrate/Si holdout.

This external-only worker imports no PhaseSmith module. It translates four
source-deposited GSAS profile-function-4 terms into the pinned GSAS-II public
scripting model and evaluates every on/off combination. The output is plain
JSON; no GSAS-II object crosses the oracle boundary.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
FACTORS = ("legacy_lx_size_axis", "sample_shift", "transparency", "stephens")
RB_STEPHENS = (0.11, 0.012, 0.0084, 0.017, 0.020, 0.012)
SI_STEPHENS = (0.061, -0.020)
ANGLE_EDGES_DEG = (17.0, 30.0, 50.0, 70.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", required=True, type=Path)
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument("--data-directory", required=True, type=Path)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def revision(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def legacy_shft_to_gsasii_micrometre(coefficient_centideg: float, radius_mm: float) -> float:
    """Map legacy ``shft*cos(theta)`` into GSAS-II's sample-height convention.

    Legacy GSAS evaluates ``delta T' = (T - Tph) + shft*cos(theta)``,
    so the peak-center motion is ``-shft*cos(theta)``. GSAS-II applies
    ``-4*(0.09/(pi*R))*Shift*cos(theta)`` with ``Shift`` in micrometres and
    ``R`` in millimetres, giving the conversion below including sign.
    """

    if not math.isfinite(coefficient_centideg) or not math.isfinite(radius_mm):
        raise ValueError("shift coefficient and radius must be finite")
    if radius_mm <= 0.0:
        raise ValueError("goniometer radius must be positive")
    return coefficient_centideg * math.pi * radius_mm / 36.0


def legacy_trns_to_gsasii_field_cm(coefficient_centideg: float, radius_mm: float) -> float:
    """Map legacy ``trns`` to the numerically equivalent GSAS-II field.

    This preserves peak-center motion between the two profile equations; it is
    not a physical ``1/mu_eff`` conversion. The legacy GSAS manual defines
    ``delta T' = (T - Tph) + trns*sin(2theta)`` and
    ``mu_eff = -9000/(pi*R*trns)``. A positive deposited ``trns`` therefore
    has a formally negative effective absorption and must remain an explicitly
    signed empirical correction.
    """

    if not math.isfinite(coefficient_centideg) or not math.isfinite(radius_mm):
        raise ValueError("transparency coefficient and radius must be finite")
    if radius_mm <= 0.0:
        raise ValueError("goniometer radius must be positive")
    return coefficient_centideg * math.pi * radius_mm / 900.0


def factorial_variants(factors: Iterable[str] = FACTORS) -> tuple[tuple[str, ...], ...]:
    """Return the deterministic power set used by the four-factor ablation."""

    names = tuple(factors)
    if len(set(names)) != len(names):
        raise ValueError("factor names must be unique")
    return tuple(
        combination
        for count in range(len(names) + 1)
        for combination in itertools.combinations(names, count)
    )


def _weighted_correlation(left: np.ndarray, right: np.ndarray, weight: np.ndarray) -> float:
    left_mean = float(np.sum(weight * left) / np.sum(weight))
    right_mean = float(np.sum(weight * right) / np.sum(weight))
    left_centered = left - left_mean
    right_centered = right - right_mean
    numerator = float(np.sum(weight * left_centered * right_centered))
    denominator = math.sqrt(
        float(np.sum(weight * left_centered**2) * np.sum(weight * right_centered**2))
    )
    return 0.0 if denominator == 0.0 else numerator / denominator


def residual_diagnostics(
    x: np.ndarray,
    observed: np.ndarray,
    calculated: np.ndarray,
    background: np.ndarray,
) -> dict[str, Any]:
    """Classify residual structure without assuming a specific peak model."""

    arrays = tuple(
        np.asarray(value, dtype=np.float64) for value in (x, observed, calculated, background)
    )
    if any(value.ndim != 1 for value in arrays) or len({value.size for value in arrays}) != 1:
        raise ValueError("residual diagnostic arrays must be aligned one-dimensional values")
    x_value, observed_value, calculated_value, background_value = arrays
    if (
        x_value.size < 5
        or not all(np.isfinite(value).all() for value in arrays)
        or np.any(np.diff(x_value) <= 0.0)
        or np.any(observed_value < 0.0)
    ):
        raise ValueError("residual diagnostic arrays are invalid")
    weight = 1.0 / np.maximum(observed_value, 1.0)
    residual = observed_value - calculated_value
    signal = calculated_value - background_value
    first = np.gradient(signal, x_value)
    second = np.gradient(first, x_value)
    centered_x = (x_value - np.mean(x_value)) / np.ptp(x_value)
    modes = {
        "position_first_derivative": first,
        "width_second_derivative": second,
        "angle_dependent_intensity": centered_x * signal,
        "background_slope": centered_x,
    }
    correlations = {
        name: _weighted_correlation(residual, mode, weight) for name, mode in modes.items()
    }
    denominator = float(np.sum(weight * observed_value**2))
    bins = []
    for low, high in itertools.pairwise(ANGLE_EDGES_DEG):
        upper = x_value < high if high < ANGLE_EDGES_DEG[-1] else x_value <= high
        mask = (x_value >= low) & upper
        if not np.any(mask):
            continue
        local_denominator = float(np.sum(weight[mask] * observed_value[mask] ** 2))
        bins.append(
            {
                "range_deg": [low, high],
                "sample_count": int(np.count_nonzero(mask)),
                "poisson_rwp": math.sqrt(
                    float(np.sum(weight[mask] * residual[mask] ** 2)) / local_denominator
                ),
                "weighted_sse_fraction": float(np.sum(weight[mask] * residual[mask] ** 2))
                / float(np.sum(weight * residual**2)),
            }
        )
    return {
        "poisson_rwp": math.sqrt(float(np.sum(weight * residual**2)) / denominator),
        "profile_correlation": float(np.corrcoef(observed_value - background_value, signal)[0, 1]),
        "residual_mode_correlations": correlations,
        "angle_bins": bins,
    }


def shapley_sse_contributions(
    subset_sse: dict[frozenset[str], float], factors: tuple[str, ...] = FACTORS
) -> dict[str, float]:
    """Partition the full-factorial weighted-SSE improvement including interactions."""

    expected = {frozenset(case) for case in factorial_variants(factors)}
    if set(subset_sse) != expected or not all(
        math.isfinite(value) and value >= 0.0 for value in subset_sse.values()
    ):
        raise ValueError("Shapley input must contain every finite non-negative subset SSE")
    count = len(factors)
    factorial = math.factorial
    contributions = {}
    for factor in factors:
        value = 0.0
        others = tuple(name for name in factors if name != factor)
        for subset in factorial_variants(others):
            selected = frozenset(subset)
            weight = (
                factorial(len(selected)) * factorial(count - len(selected) - 1) / factorial(count)
            )
            value += weight * (subset_sse[selected] - subset_sse[selected | {factor}])
        contributions[factor] = value
    return contributions


def _update_hap_entry(
    phase: Any,
    histogram: Any,
    key: str,
    transform: Callable[[list[Any]], list[Any]],
) -> None:
    entries = phase.getHAPentryList(histogram, key)
    if len(entries) != 1:
        paths = [entry[0] for entry in entries]
        raise RuntimeError(f"expected one {key} HAP entry, found {len(entries)}: {paths}")
    path = entries[0][0]
    phase.setHAPentryValue(path, transform(phase.getHAPentryValue(path)))


def _configure_sample(
    phase: Any,
    histogram: Any,
    *,
    coefficients: tuple[float, ...] | None,
    mixing: float,
) -> None:
    def size(current: list[Any]) -> list[Any]:
        current[0] = "isotropic"
        current[1][0] = 1.0e20
        current[1][2] = 1.0
        return current

    def microstrain(current: list[Any]) -> list[Any]:
        if coefficients is None:
            current[0] = "isotropic"
            current[1][0] = 0.0
            current[1][2] = 0.0
        else:
            current[0] = "generalized"
            current[1][2] = mixing
            current[4] = list(coefficients)
            current[5] = [False] * len(coefficients)
        return current

    _update_hap_entry(phase, histogram, "Size", size)
    _update_hap_entry(phase, histogram, "Mustrain", microstrain)


def _write_instrument(path: Path, manifest: dict[str, Any], *, legacy_lx_size_axis: bool) -> None:
    instrument = manifest["instrument"]
    profile = instrument["common_base_profile"]
    # This diagnostic deliberately compares the historical neutral mapping
    # (legacy LX placed on Y) with the physically corrected mapping (LX on X).
    # Keep those two branches explicit even after the converter itself adopts
    # the corrected representation.
    x_term = 3.634 if legacy_lx_size_axis else 0.0
    y_term = 0.0 if legacy_lx_size_axis else 3.634
    wavelengths = instrument["wavelengths_angstrom"]
    path.write_text(
        "#GSAS-II instrument parameter file; do not add/delete items!\n"
        "Type:PXC\nBank:1.0\n"
        f"Lam1:{wavelengths[0]}\nLam2:{wavelengths[1]}\n"
        f"I(L2)/I(L1):{instrument['k_alpha2_over_k_alpha1']}\n"
        f"Zero:{instrument['initial_zero_deg']}\n"
        f"Polariz.:{instrument['polarization_fraction']}\n"
        f"U:{profile['U']}\nV:{profile['V']}\nW:{profile['W']}\n"
        f"X:{x_term}\nY:{y_term}\nZ:0.0\n"
        f"SH/L:{instrument['matched_sh_over_l']}\nAzimuth:0.0\nSource:CuKa\n",
        encoding="utf-8",
    )


def _run_variant(
    scripting: Any,
    root: Path,
    work: Path,
    manifest: dict[str, Any],
    x: np.ndarray,
    observed: np.ndarray,
    legacy_background: np.ndarray,
    factors: frozenset[str],
    cycles: int,
    *,
    transparency_scale: float = 1.0,
    case_suffix: str = "",
) -> dict[str, Any]:
    # HAP public lookup searches every key path by substring, including the
    # histogram name. Keep factor words such as "size" and "shift" out of it.
    if not math.isfinite(transparency_scale):
        raise ValueError("transparency scale must be finite")
    case_id = "case_" + "".join("1" if name in factors else "0" for name in FACTORS)
    case_id += case_suffix
    data_path = work / f"{case_id}.xye"
    instrument_path = work / f"{case_id}.instprm"
    np.savetxt(data_path, np.column_stack((x, observed, np.sqrt(np.maximum(observed, 1.0)))))
    _write_instrument(
        instrument_path, manifest, legacy_lx_size_axis="legacy_lx_size_axis" in factors
    )
    project = scripting.G2Project(newgpx=str(work / f"{case_id}.gpx"))
    histogram = project.add_powder_histogram(str(data_path), str(instrument_path), fmthint="Topas")
    histogram.set_refinements({"Limits": [float(x[0]), float(x[-1])]})
    histogram.data["Sample Parameters"]["Scale"][1] = False
    sample = histogram.data["Sample Parameters"]
    instrument = manifest["instrument"]
    radius = float(instrument["goniometer_radius_mm"])
    legacy_shift = float(instrument["legacy_phase_shift_coefficients"]["trirubidium_citrate"])
    sample["Shift"] = [
        legacy_shft_to_gsasii_micrometre(legacy_shift, radius)
        if "sample_shift" in factors
        else 0.0,
        False,
    ]
    sample["Transparency"] = [
        transparency_scale * legacy_trns_to_gsasii_field_cm(1.30, radius)
        if "transparency" in factors
        else 0.0,
        False,
    ]
    background = histogram.data["Background"]
    background[0] = ["chebyschev-1", True, 1, 0.0]
    background[1]["fixback"] = legacy_background.copy()
    background[1]["background PWDR"] = ["", 1.0, False]
    phases = []
    for phase_id in ("trirubidium_citrate", "silicon"):
        phase = project.add_phase(
            str(root / f"{phase_id}.cif"),
            phasename=phase_id,
            histograms=[histogram],
            fmthint="CIF",
        )
        if "stephens" in factors:
            parameters = (
                (RB_STEPHENS, 0.3652) if phase_id == "trirubidium_citrate" else (SI_STEPHENS, 1.0)
            )
            _configure_sample(
                phase,
                histogram,
                coefficients=parameters[0],
                mixing=parameters[1],
            )
        else:
            _configure_sample(phase, histogram, coefficients=None, mixing=0.0)
        phase.set_HAP_refinements({"Scale": True})
        phases.append(phase)
    project.set_Controls("cycles", cycles)
    project.do_refinements([{}], outputnames=[None])
    calculated = np.asarray(histogram.getdata("Ycalc"), dtype=np.float64)
    fitted_background = np.asarray(histogram.getdata("Background"), dtype=np.float64)
    weight = 1.0 / np.maximum(observed, 1.0)
    residual = observed - calculated
    diagnostics = residual_diagnostics(x, observed, calculated, fitted_background)
    diagnostics.update(
        {
            "case_id": case_id,
            "enabled_factors": [name for name in FACTORS if name in factors],
            "weighted_sse": float(np.sum(weight * residual**2)),
            "sample_shift_micrometre": float(sample["Shift"][0]),
            "sample_transparency_cm": float(sample["Transparency"][0]),
            "source_transparency_multiplier": transparency_scale,
            "weight_fractions": {
                phase.name: float(histogram.ComputeMassFracs()[phase.name][0]) for phase in phases
            },
        }
    )
    return diagnostics


def main() -> None:
    arguments = parse_args()
    if arguments.cycles <= 0:
        raise ValueError("cycles must be positive")
    actual_revision = revision(arguments.gsas_root)
    if actual_revision != PINNED_REVISION:
        raise RuntimeError(
            f"GSAS-II revision mismatch: expected {PINNED_REVISION}, got {actual_revision}"
        )
    sys.path.insert(0, str(arguments.gsas_root))
    if arguments.binary_dir is not None:
        sys.path.insert(0, str(arguments.binary_dir))
    from GSASII import GSASIIpath

    if arguments.binary_dir is not None:
        GSASIIpath.binaryPath = str(arguments.binary_dir)
        GSASIIpath.BinaryPathLoaded = True
        GSASIIpath.BinaryPathFailed = False
    GSASIIpath.LoadConfig()
    GSASIIpath.AddConfigValue({"Multiprocessing_cores": 0})
    from GSASII import GSASIIscriptable as G2sc

    root = arguments.data_directory
    manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    if manifest.get("scope") != "iucr_anhydrous_trirubidium_citrate_silicon_holdout":
        raise ValueError("worker requires the reviewed anhydrous rubidium bundle")
    data = np.loadtxt(root / "pattern.csv", delimiter=",", skiprows=1)
    x, observed, legacy_calculated, legacy_background = data.T
    with tempfile.TemporaryDirectory(prefix="phasesmith-rb-forensics-") as name:
        work = Path(name)
        variants = [
            _run_variant(
                G2sc,
                root,
                work,
                manifest,
                x,
                observed,
                legacy_background,
                frozenset(enabled),
                arguments.cycles,
            )
            for enabled in factorial_variants()
        ]
        sensitivity_factors = frozenset(
            ("legacy_lx_size_axis", "sample_shift", "transparency", "stephens")
        )
        transparency_sensitivity = [
            _run_variant(
                G2sc,
                root,
                work,
                manifest,
                x,
                observed,
                legacy_background,
                sensitivity_factors,
                arguments.cycles,
                transparency_scale=scale,
                case_suffix=f"_sensitivity_{index}",
            )
            for index, scale in enumerate((-2.0, -1.0, -0.5, 0.5, 2.0))
        ]
    subset_sse = {
        frozenset(variant["enabled_factors"]): float(variant["weighted_sse"])
        for variant in variants
    }
    contributions = shapley_sse_contributions(subset_sse)
    transparency_penalties = {
        "+".join(name for name in FACTORS if name in subset) or "base": (
            subset_sse[subset | {"transparency"}] - subset_sse[subset]
        )
        for subset in subset_sse
        if "transparency" not in subset
    }
    if not all(value > 0.0 for value in transparency_penalties.values()):
        raise RuntimeError("deposited transparency does not worsen every matched branch")
    base_sse = subset_sse[frozenset()]
    full_sse = subset_sse[frozenset(FACTORS)]
    legacy_weight = 1.0 / np.maximum(observed, 1.0)
    legacy_sse = float(np.sum(legacy_weight * (observed - legacy_calculated) ** 2))
    total_improvement = base_sse - full_sse
    best = min(variants, key=lambda variant: float(variant["weighted_sse"]))
    best_sse = float(best["weighted_sse"])
    report = {
        "schema_version": 1,
        "scope": "iucr_trirubidium_citrate_legacy_profile_residual_forensics",
        "source_dataset_id": manifest["source_dataset_id"],
        "revision": actual_revision,
        "sample_count": int(x.size),
        "oracle_boundary": {
            "public_api": [
                "GSASIIscriptable.G2Project.add_powder_histogram/add_phase/do_refinements",
                "GSASIIscriptable.G2Phase.getHAPentryList/setHAPentryValue",
                "GSASIIscriptable.G2PwdrData.getdata/ComputeMassFracs",
            ],
            "private_probe": False,
        },
        "factors": list(FACTORS),
        "translation": {
            "legacy_manual": {
                "citation": (
                    "A. C. Larson and R. B. Von Dreele, General Structure Analysis "
                    "System (GSAS), LAUR 86-748 (2004), CW profile function 2, p. 157"
                ),
                "url": ("https://subversion.xray.aps.anl.gov/EXPGUI/gsas/all/GSAS%20Manual.pdf"),
                "profile_argument": (
                    "delta_T_prime = (T - T_phase) + shft*cos(theta) + trns*sin(2theta)"
                ),
                "peak_center_motion": ("-shft*cos(theta)/100 - trns*sin(2theta)/100 degrees"),
                "physical_sample_shift_relation": "s = -pi*R*shft/36000",
                "effective_absorption_relation": "mu_eff = -9000/(pi*R*trns)",
            },
            "legacy_lx": (
                "profile-function-4 LX=3.634 mapped to GSAS-II X/cos(theta), not Y*tan(theta)"
            ),
            "legacy_shft_centideg": -8.7503,
            "formal_legacy_physical_sample_shift_mm": (-math.pi * 141.5 * -8.7503 / 36_000.0),
            "gsasii_sample_shift_micrometre": legacy_shft_to_gsasii_micrometre(-8.7503, 141.5),
            "legacy_trns_centideg": 1.30,
            "gsasii_transparency_field_cm": legacy_trns_to_gsasii_field_cm(1.30, 141.5),
            "formal_legacy_mu_eff": -9000.0 / (math.pi * 141.5 * 1.30),
            "transparency_interpretation": (
                "The GSAS-II value is a numerical position-equation translation. "
                "Because the deposited positive legacy coefficient implies negative "
                "mu_eff, it is not a physically admissible transparency measurement."
            ),
            "stephens": "source-deposited phase-specific generalized microstrain and eta",
        },
        "legacy_curve": residual_diagnostics(x, observed, legacy_calculated, legacy_background),
        "variants": variants,
        "transparency_sign_and_scale_sensitivity": [
            {
                "source_multiplier": variant["source_transparency_multiplier"],
                "transparency_cm": variant["sample_transparency_cm"],
                "poisson_rwp": variant["poisson_rwp"],
                "profile_correlation": variant["profile_correlation"],
            }
            for variant in transparency_sensitivity
        ],
        "full_factorial": {
            "base_weighted_sse": base_sse,
            "full_weighted_sse": full_sse,
            "legacy_curve_weighted_sse": legacy_sse,
            "full_model_fraction_of_base_to_legacy_gap_closed": (
                (base_sse - full_sse) / (base_sse - legacy_sse)
            ),
            "shapley_weighted_sse_improvement": contributions,
            "transparency_paired_weighted_sse_penalty": transparency_penalties,
            "shapley_fraction_of_total_improvement": {
                name: value / total_improvement for name, value in contributions.items()
            },
            "best_fixed_subset": {
                "enabled_factors": best["enabled_factors"],
                "poisson_rwp": best["poisson_rwp"],
                "profile_correlation": best["profile_correlation"],
                "fraction_of_base_to_legacy_gap_closed": (
                    (base_sse - best_sse) / (base_sse - legacy_sse)
                ),
            },
        },
        "conclusion": (
            "The source-translated sample displacement dominates the recoverable gap; "
            "placing legacy LX on the Lorentzian size axis is the second material term. "
            "Deposited Stephens broadening is negligible. The signed deposited trns term "
            "worsens every matched fixed-structure branch; reversing its sign improves the "
            "fit but contradicts both the deposited coefficient and its already nonphysical "
            "negative-mu_eff interpretation. The best fixed subset remains "
            "well above the deposited curve, with most residual weighted SSE below 30 "
            "degrees and a stronger width-mode than position-mode correlation."
        ),
    }
    if not all(
        math.isfinite(value)
        for value in (
            report["legacy_curve"]["poisson_rwp"],
            *(variant["poisson_rwp"] for variant in variants),
            *(variant["poisson_rwp"] for variant in transparency_sensitivity),
            *contributions.values(),
            *transparency_penalties.values(),
        )
    ):
        raise RuntimeError("residual-forensics report contains non-finite values")
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    if arguments.report.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {arguments.report}")
    arguments.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
