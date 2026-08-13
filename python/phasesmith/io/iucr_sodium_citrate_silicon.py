"""Narrow conversion for the IUCr sodium-dihydrogen-citrate/Si holdout."""

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

IUCR_SODIUM_CITRATE_SILICON_PHASES = ("sodium_dihydrogen_citrate", "silicon")

_LEGACY_WEIGHT_FRACTIONS = {
    "sodium_dihydrogen_citrate": 0.8126,
    "silicon": 0.1874,
}


def convert_iucr_sodium_citrate_silicon_bundle(source: str | Path, destination: str | Path) -> Path:
    """Convert the checksum-pinned RAMM012A pdCIF into a neutral holdout bundle.

    The deposited refinement uses 4,452 contiguous points from a 4,701-count
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
        "data_RAMM012A_publ",
        "data_RAMM012A_phase_1",
        "data_RAMM012A_phase_2",
        "data_RAMM012A_p_01",
        "'Bruker D2 Phaser'",
        "HAP1 1MASSFR    0.8126",
        "HAP2 1MASSFR    0.1874",
        "CRS1  OD  1A",
        "#10(S/L) =   0.0182 #11(H/L) =   0.0005",
    )
    if any(marker not in text for marker in markers):
        raise ValueError("IUCr CIF does not match the reviewed RAMM012A archive")

    powder_block = _block(text, "RAMM012A_p_01")
    rows = _powder_rows(powder_block, expected_rows=4701)
    sample_count = int(_scalar(powder_block, "_pd_meas_number_of_points"))
    x = np.linspace(
        _scalar(powder_block, "_pd_meas_2theta_range_min"),
        _scalar(powder_block, "_pd_meas_2theta_range_max"),
        sample_count,
    )
    selected = np.all(np.isfinite(rows), axis=1)
    if (
        sample_count != 4701
        or int(np.count_nonzero(selected)) != 4452
        or not np.array_equal(np.flatnonzero(selected), np.arange(248, 4700))
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
        or abs(legacy_rwp - 0.0843) > 5.0e-5
        or abs(legacy_rp - 0.0632) > 5.0e-5
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
    (destination_root / "sodium_dihydrogen_citrate.cif").write_text(
        _block(text, "RAMM012A_phase_1"), encoding="utf-8"
    )
    (destination_root / "silicon.cif").write_text(
        _normalized_silicon_cif("RAMM012A_phase_2", u_iso_angstrom2=0.030352),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "scope": "iucr_sodium_dihydrogen_citrate_silicon_holdout",
        "source_dataset_id": "iucr-sodium-dihydrogen-citrate-si-standard",
        "source_block": "RAMM012A_p_01",
        "instrument": {
            "device": "Bruker D2 Phaser",
            "geometry": "reflection, flat specimen",
            "wavelengths_angstrom": [1.540629, 1.544451],
            "k_alpha2_over_k_alpha1": 0.5,
            "polarization_fraction": 0.5,
            "initial_zero_deg": -0.04480,
            "source_over_radius": 0.0182,
            "detector_over_radius": 0.0005,
            "silicon_profile": {
                "U": 2.336,
                "V": 0.0,
                "W": 3.777,
                "X": 2.718,
                "Y": 15.991,
            },
        },
        "data_selection": {
            "source_sample_count": 4701,
            "refined_sample_count": 4452,
            "source_index_interval_inclusive": [248, 4699],
            "two_theta_min_deg": float(selected_x[0]),
            "two_theta_max_deg": float(selected_x[-1]),
        },
        "silicon_standard": {
            "material": "NIST SRM 640b",
            "fixed_lattice_a_angstrom": 5.43105,
            "deposited_weight_fraction": 0.1874,
        },
        "legacy_gsas_reference": {
            "weight_fractions": _LEGACY_WEIGHT_FRACTIONS,
            "rwp": legacy_rwp,
            "rp": legacy_rp,
            "rexp": 0.0244,
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
                "main-phase generalized spherical-harmonic preferred orientation",
                "Suortti surface-roughness correction with coefficients 0.34 and 0.70",
                "phase-specific legacy profile functions",
            ],
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
