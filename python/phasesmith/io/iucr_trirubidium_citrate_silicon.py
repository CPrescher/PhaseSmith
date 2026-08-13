"""Narrow conversion for the IUCr anhydrous trirubidium-citrate/Si holdout."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .iucr_silicon_standard import (
    _block,
    _normalized_silicon_cif,
    _powder_rows,
    _scalar,
)

IUCR_TRIRUBIDIUM_CITRATE_SILICON_PHASES = ("trirubidium_citrate", "silicon")

_LEGACY_WEIGHT_FRACTIONS = {
    "trirubidium_citrate": 0.9785,
    "silicon": 0.0215,
}

_REFLECTION_COLUMNS = (
    "h",
    "k",
    "l",
    "phase_id",
    "wavelength_id",
    "f_squared_measured",
    "f_squared_calculated",
    "phase_deg",
    "d_spacing_angstrom",
    "i100_measured",
)


def _source_reflections(powder_block: str) -> np.ndarray:
    header = (
        "    _refln_index_h\n"
        "    _refln_index_k\n"
        "    _refln_index_l\n"
        "    _pd_refln_phase_id\n"
        "    _pd_refln_wavelength_id\n"
        "    _refln_observed_status\n"
        "    _refln_F_squared_meas\n"
        "    _refln_F_squared_calc\n"
        "    _refln_phase_calc\n"
        "    _refln_d_spacing\n"
        "    _gsas_i100_meas\n"
    )
    try:
        remainder = powder_block.split(header, 1)[1]
    except IndexError as error:
        raise ValueError("IUCr CIF does not contain the reviewed reflection loop") from error
    rows: list[list[float]] = []
    for line in remainder.splitlines():
        fields = line.split()
        if len(fields) != 11:
            break
        if fields[5] != "o":
            raise ValueError("IUCr reflection loop contains an unsupported status")
        rows.append(
            [
                *[float(value) for value in fields[:5]],
                *[float(value) for value in fields[6:]],
            ]
        )
    result = np.asarray(rows, dtype=np.float64)
    if (
        result.shape != (1197, len(_REFLECTION_COLUMNS))
        or not np.isfinite(result).all()
        or not np.all(result[:, 8] > 0.0)
        or not np.all(result[:, 6] >= 0.0)
        or set(result[:, 3]) != {1.0, 2.0}
        or set(result[:, 4]) != {1.0, 2.0}
    ):
        raise ValueError("IUCr reflection loop failed its reviewed numerical contract")
    return result


def convert_iucr_trirubidium_citrate_silicon_bundle(
    source: str | Path, destination: str | Path
) -> Path:
    """Convert the checksum-pinned RAMM077C pdCIF into a neutral bundle.

    The deposited refinement excludes the first 594 of 4,701 measured points
    because of beam spillover. The bundle retains that exact finite mask, the
    observed counts, deposited legacy-GSAS curve and background, both
    structures, the source-deposited reflection table, and an explicit
    inventory of source-only model terms.
    """

    source_path = Path(source)
    destination_root = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.resolve() == destination_root.resolve():
        raise ValueError("source file and destination directory must differ")
    text = source_path.read_text(encoding="utf-8")
    markers = (
        "data_RAMM077C_publ",
        "data_RAMM077C_phase_1",
        "data_RAMM077C_phase_2",
        "data_RAMM077C_p_01",
        "'Bruker D2 Phaser'",
        "97.850(10)",
        "2.15(5)",
        "_pd_instr_dist_src/spec     141.5",
        "#10(S/L) =   0.0097 #11(H/L) =   0.0097",
        "#8(shft) =  -8.7503",
        "The 5-17 \\% region was excluded because of the effects of beam spillover.",
    )
    if any(marker not in text for marker in markers):
        raise ValueError("IUCr CIF does not match the reviewed RAMM077C archive")

    powder_block = _block(text, "RAMM077C_p_01")
    sample_count = int(_scalar(powder_block, "_pd_meas_number_of_points"))
    rows = _powder_rows(powder_block, expected_rows=sample_count)
    source_reflections = _source_reflections(powder_block)
    x = np.linspace(
        _scalar(powder_block, "_pd_meas_2theta_range_min"),
        _scalar(powder_block, "_pd_meas_2theta_range_max"),
        sample_count,
    )
    selected = np.all(np.isfinite(rows), axis=1)
    if (
        sample_count != 4701
        or int(np.count_nonzero(selected)) != 4106
        or not np.array_equal(np.flatnonzero(selected), np.arange(594, 4700))
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
        or abs(legacy_rwp - 0.0246) > 5.0e-5
        or abs(legacy_rp - 0.0195) > 5.0e-5
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
    np.savetxt(
        destination_root / "source_reflections.csv",
        source_reflections,
        delimiter=",",
        header=",".join(_REFLECTION_COLUMNS),
        comments="",
        fmt="%.12g",
    )
    (destination_root / "trirubidium_citrate.cif").write_text(
        _block(text, "RAMM077C_phase_1"), encoding="utf-8"
    )
    (destination_root / "silicon.cif").write_text(
        _normalized_silicon_cif("RAMM077C_phase_2", u_iso_angstrom2=0.01),
        encoding="utf-8",
    )

    radius_mm = 141.5
    legacy_shift_centideg = -8.7503
    legacy_transparency_centideg = 1.30
    source_translated_displacement_mm = legacy_shift_centideg * math.pi * radius_mm / 36_000.0
    legacy_manual_physical_sample_shift_mm = -source_translated_displacement_mm
    translated_gsasii_transparency_field_cm = (
        legacy_transparency_centideg * math.pi * radius_mm / 900.0
    )
    # Legacy GSAS profile-function-4 LX is the Lorentzian size term. In the
    # current TCH convention that is X/cos(theta), not Y*tan(theta).
    common_profile = {"U": 0.0, "V": 0.0, "W": 5.109, "X": 3.634, "Y": 0.0}
    manifest = {
        "schema_version": 1,
        "scope": "iucr_anhydrous_trirubidium_citrate_silicon_holdout",
        "source_dataset_id": "iucr-trirubidium-citrate-si-standard",
        "source_block": "RAMM077C_p_01",
        "instrument": {
            "device": "Bruker D2 Phaser",
            "geometry": "reflection, flat specimen",
            "wavelengths_angstrom": [1.540593, 1.544451],
            "k_alpha2_over_k_alpha1": 0.5,
            "polarization_fraction": 0.5,
            "goniometer_radius_mm": radius_mm,
            "radius_is_source_deposited": True,
            "initial_zero_deg": 0.0,
            "source_translated_sample_displacement_mm": source_translated_displacement_mm,
            "legacy_manual_physical_sample_shift_mm": legacy_manual_physical_sample_shift_mm,
            "legacy_transparency_centideg": legacy_transparency_centideg,
            "translated_gsasii_transparency_field_cm": translated_gsasii_transparency_field_cm,
            "source_over_radius": 0.0097,
            "detector_over_radius": 0.0097,
            "matched_sh_over_l": 0.0194,
            "common_base_profile": common_profile,
            "legacy_phase_profile_functions": {
                "trirubidium_citrate": 4,
                "silicon": 4,
            },
            "legacy_phase_shift_coefficients": {
                "trirubidium_citrate": -8.7503,
                "silicon": -8.7503,
            },
        },
        "data_selection": {
            "source_sample_count": 4701,
            "refined_sample_count": 4106,
            "source_index_interval_inclusive": [594, 4699],
            "two_theta_min_deg": float(selected_x[0]),
            "two_theta_max_deg": float(selected_x[-1]),
            "excluded_source_region_two_theta_deg": [5.0, 17.0],
            "exclusion_reason": "beam spillover in the deposited refinement",
        },
        "source_reflections": {
            "file": "source_reflections.csv",
            "columns": list(_REFLECTION_COLUMNS),
            "row_count": int(source_reflections.shape[0]),
            "observed_status": "o",
            "phase_id_to_name": {
                "1": "trirubidium_citrate",
                "2": "silicon",
            },
            "provenance": (
                "Source-deposited pdCIF reflection loop; values are preserved without "
                "recalculation so structure-factor conversion can be audited independently."
            ),
        },
        "silicon_standard": {
            "material": "NIST SRM 640b",
            "fixed_lattice_a_angstrom": 5.43105,
            "deposited_weight_fraction": 0.0215,
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
            "rexp": 0.0225,
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
                "phase-specific legacy mixing coefficients",
                "phase-specific Stephens anisotropic broadening",
                "legacy signed transparency-position coefficient 1.30; numerically "
                "translated to the pinned GSAS-II equation but rejected by the "
                "fixed-structure factorial ablation",
            ],
            "common_profile_basis": (
                "Both deposited phase profiles share GU=0, GV=0, GW=5.109, GP=0, "
                "and LX=3.634. Legacy LX is Lorentzian size broadening, so the neutral "
                "base-profile record maps it to current X/cos(theta): "
                "U/V/W/X/Y=0/0/5.109/3.634/0."
            ),
            "geometry_contract": (
                "The source deposits equal source/specimen and specimen/detector radii "
                "of 141.5 mm and equal S/L=H/L=0.0097. The article says Si verifies "
                "the calibrated goniometer zero. Legacy shft=-8.7503 centidegrees maps "
                f"analytically to {source_translated_displacement_mm:.12g} mm in the current "
                "PhaseSmith/GSAS-II Bragg-Brentano parameter convention. The legacy manual's "
                f"physical sample-shift variable uses the opposite sign "
                f"({legacy_manual_physical_sample_shift_mm:.12g} mm); instrument zero "
                "remains zero."
            ),
            "correction_contract": (
                "The source declares no absorption-intensity or surface-roughness correction. "
                "In the legacy profile-position equation, trns=1.30 implies a formally "
                "negative effective absorption. Its numerically equivalent pinned-GSAS-II "
                f"field is {translated_gsasii_transparency_field_cm:.12g} cm, but this is "
                "not a physical 1/mu conversion and PhaseSmith does not apply it because the "
                "factorial oracle ablation worsens the fixed-structure reconstruction. An "
                "unpolarized Cu source fraction of 0.5 is an explicit holdout assumption."
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
