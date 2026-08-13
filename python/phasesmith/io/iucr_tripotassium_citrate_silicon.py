"""Narrow conversion for the IUCr tripotassium-citrate/Si holdout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .iucr_silicon_standard import (
    _block,
    _normalized_silicon_cif,
    _powder_rows,
    _scalar,
)

IUCR_TRIPOTASSIUM_CITRATE_SILICON_PHASES = ("tripotassium_citrate", "silicon")

_LEGACY_WEIGHT_FRACTIONS = {
    "tripotassium_citrate": 0.9856,
    "silicon": 0.0144,
}


def convert_iucr_tripotassium_citrate_silicon_bundle(
    source: str | Path, destination: str | Path
) -> Path:
    """Convert the checksum-pinned KADU1578 pdCIF into a neutral holdout bundle.

    The deposited refinement uses 2,696 contiguous points from a 3,217-count
    scan. The bundle retains the observed counts, deposited legacy-GSAS curve
    and background, both structures, physical instrument metadata, and an
    explicit inventory of source terms that are not silently translated.
    """

    source_path = Path(source)
    destination_root = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.resolve() == destination_root.resolve():
        raise ValueError("source file and destination directory must differ")
    text = source_path.read_text(encoding="utf-8")
    markers = (
        "data_KADU1578_publ",
        "data_KADU1578_phase_1",
        "data_KADU1578_phase_2",
        "data_KADU1578_p_01",
        "'Bruker D2 Phaser'",
        "HAP1 1MASSFR    0.9856",
        "HAP2 1MASSFR    0.0144",
        "CRS1  OD  1A",
        "#10(S/L) =   0.0168 #11(H/L) =   0.0200",
    )
    if any(marker not in text for marker in markers):
        raise ValueError("IUCr CIF does not match the reviewed KADU1578 archive")

    powder_block = _block(text, "KADU1578_p_01")
    rows = _powder_rows(powder_block)
    sample_count = int(_scalar(powder_block, "_pd_meas_number_of_points"))
    x = np.linspace(
        _scalar(powder_block, "_pd_meas_2theta_range_min"),
        _scalar(powder_block, "_pd_meas_2theta_range_max"),
        sample_count,
    )
    selected = np.all(np.isfinite(rows), axis=1)
    if (
        sample_count != 3217
        or int(np.count_nonzero(selected)) != 2696
        or not np.array_equal(np.flatnonzero(selected), np.arange(520, 3216))
    ):
        raise ValueError("IUCr deposited refinement mask does not match the reviewed archive")
    weight, legacy_background, legacy_calculated, observed = rows[selected].T
    selected_x = x[selected]
    legacy_rwp = float(
        np.sqrt(
            np.sum(weight * np.square(observed - legacy_calculated))
            / np.sum(weight * np.square(observed))
        )
    )
    legacy_rp = float(np.sum(np.abs(observed - legacy_calculated)) / np.sum(observed))
    if (
        np.any(np.diff(selected_x) <= 0.0)
        or np.any(observed < 0.0)
        or abs(legacy_rwp - 0.0485) > 5.0e-5
        or abs(legacy_rp - 0.0381) > 5.0e-5
    ):
        raise ValueError("IUCr deposited powder result failed its numerical checksum")

    destination_root.mkdir(parents=True, exist_ok=False)
    np.savetxt(
        destination_root / "pattern.csv",
        np.column_stack((selected_x, observed, legacy_calculated, legacy_background)),
        delimiter=",",
        header="two_theta_deg,observed,legacy_calculated,legacy_background",
        comments="",
        fmt="%.12g",
    )
    (destination_root / "tripotassium_citrate.cif").write_text(
        _block(text, "KADU1578_phase_1"), encoding="utf-8"
    )
    (destination_root / "silicon.cif").write_text(
        _normalized_silicon_cif("KADU1578_phase_2", u_iso_angstrom2=0.01),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "scope": "iucr_anhydrous_tripotassium_citrate_silicon_holdout",
        "source_dataset_id": "iucr-tripotassium-citrate-si-standard",
        "source_block": "KADU1578_p_01",
        "instrument": {
            "device": "Bruker D2 Phaser",
            "geometry": "reflection, flat specimen",
            "wavelengths_angstrom": [1.540629, 1.544451],
            "k_alpha2_over_k_alpha1": 0.5,
            "polarization_fraction": 0.5,
            "source_over_radius": 0.0168,
            "detector_over_radius": 0.0200,
            "matched_sh_over_l": 0.0368,
            "standalone_zero_shift": "not deposited in the supplementary pdCIF",
            "legacy_phase_profile_functions": {
                "tripotassium_citrate": 4,
                "silicon": 2,
            },
        },
        "data_selection": {
            "source_sample_count": 3217,
            "refined_sample_count": 2696,
            "source_index_interval_inclusive": [520, 3215],
            "two_theta_min_deg": float(selected_x[0]),
            "two_theta_max_deg": float(selected_x[-1]),
        },
        "silicon_standard": {
            "material": "silicon internal standard",
            "fixed_lattice_a_angstrom": 5.43105,
            "deposited_weight_fraction": 0.0144,
            "source_does_not_identify_nist_srm": True,
            "candidate_calibration_windows_two_theta_deg": [
                [46.95, 47.78],
                [55.77, 56.63],
                [68.78, 69.68],
            ],
        },
        "legacy_gsas_reference": {
            "weight_fractions": _LEGACY_WEIGHT_FRACTIONS,
            "rwp": legacy_rwp,
            "rp": legacy_rp,
            "rexp": 0.0339,
            "profile_correlation": float(
                np.corrcoef(
                    observed - legacy_background,
                    legacy_calculated - legacy_background,
                )[0, 1]
            ),
        },
        "translation_diagnostics": {
            "normalized_silicon_setting": (
                "The deposited silicon structure is normalized to IT 227, F d -3 m :1, "
                "with Si at (1/8,1/8,1/8)."
            ),
            "source_only_terms": [
                "main-phase GSAS profile function 4 with Stephens anisotropic broadening",
                "second-order generalized spherical-harmonic texture (reported index 1.001)",
                "Suortti surface-roughness correction with coefficients 0.37 and 0.70",
                "phase-specific legacy profile functions",
            ],
            "axial_geometry_translation": (
                "The deposited unequal S/L=0.0168 and H/L=0.0200 ratios are retained. "
                "A future common-model comparison may compress them to SH/L=0.0368 "
                "only with the same disclosed equal-height convention in both programs."
            ),
            "review_rule": (
                "Source-only terms remain disclosed and unmodified; a common-model "
                "workflow must identify each approximation before claiming parity."
            ),
        },
    }
    manifest_path = destination_root / "experiment.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path
