"""Narrow conversion for the IUCr dicesium-citrate/Si standard example."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

IUCR_SILICON_PHASES = (
    "dicesium_hydrogen_citrate",
    "cesium_dihydrogen_citrate",
    "silicon",
)

_SOURCE_BLOCKS = {
    "dicesium_hydrogen_citrate": "RAMM016C_phase_1",
    "cesium_dihydrogen_citrate": "RAMM016C_phase_2",
    "silicon": "RAMM016C_phase_3",
}
_LEGACY_WEIGHT_FRACTIONS = {
    "dicesium_hydrogen_citrate": 0.6003,
    "cesium_dihydrogen_citrate": 0.2700,
    "silicon": 0.1302,
}


def _scalar(text: str, name: str) -> float:
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s+([0-9.eE+-]+)", text)
    if match is None:
        raise ValueError(f"IUCr powder block does not contain {name}")
    return float(match.group(1))


def _block(text: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^data_{re.escape(name)}\s.*?(?=^data_|\Z)",
        text,
    )
    if match is None:
        raise ValueError(f"IUCr CIF does not contain data_{name}")
    return match.group(0).rstrip() + "\n"


def _powder_rows(powder_block: str, *, expected_rows: int = 3217) -> np.ndarray:
    header = (
        "    _pd_proc_ls_weight\n"
        "    _pd_proc_intensity_bkg_calc\n"
        "    _pd_calc_intensity_total\n"
        "    _pd_meas_counts_total\n"
    )
    try:
        remainder = powder_block.split(header, 1)[1]
    except IndexError as error:
        raise ValueError("IUCr CIF does not contain the reviewed powder loop") from error
    rows: list[list[float]] = []
    for line in remainder.splitlines():
        fields = line.split()
        if len(fields) != 4:
            break
        rows.append([np.nan if value == "." else float(value) for value in fields])
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (expected_rows, 4):
        raise ValueError(f"IUCr powder loop does not contain the expected {expected_rows} rows")
    return result


def _normalized_silicon_cif(
    source_block: str = "RAMM016C_phase_3", *, u_iso_angstrom2: float = 0.01
) -> str:
    if not source_block or "'" in source_block or "\n" in source_block:
        raise ValueError("normalized silicon source block must be a safe CIF label")
    if not np.isfinite(u_iso_angstrom2) or u_iso_angstrom2 < 0.0:
        raise ValueError("normalized silicon Uiso must be nonnegative and finite")
    return f"""data_silicon_nist_srm_640b
_audit_creation_method 'PhaseSmith normalization of IUCr {source_block}'
_chemical_name_common 'silicon internal standard NIST SRM 640b'
_chemical_formula_sum 'Si'
_chemical_formula_weight 28.09
_cell_formula_units_Z 8
_cell_length_a 5.43105
_cell_length_b 5.43105
_cell_length_c 5.43105
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 227
_symmetry_space_group_name_H-M 'F d -3 m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
Si1 Si 0.125 0.125 0.125 1.0 {u_iso_angstrom2:.12g}
"""


def convert_iucr_silicon_standard_bundle(source: str | Path, destination: str | Path) -> Path:
    """Convert the checksum-pinned IUCr CIF to a shared benchmark bundle.

    The source contains raw counts, the deposited legacy-GSAS calculation and
    background, and all three crystal structures in one multi-block CIF. Only
    the 2,820 points included by the deposited refinement are emitted.
    """

    source_path = Path(source)
    destination_root = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.resolve() == destination_root.resolve():
        raise ValueError("source file and destination directory must differ")
    text = source_path.read_text(encoding="utf-8")
    markers = (
        "data_RAMM016C_publ",
        "data_RAMM016C_phase_1",
        "data_RAMM016C_phase_2",
        "data_RAMM016C_phase_3",
        "data_RAMM016C_p_01",
        "'Bruker D2 Phaser'",
        "HAP3 1MASSFR    0.1302",
    )
    if any(marker not in text for marker in markers):
        raise ValueError("IUCr CIF does not match the reviewed RAMM016C archive")
    powder_block = _block(text, "RAMM016C_p_01")
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
        or int(np.count_nonzero(selected)) != 2820
        or not np.array_equal(np.flatnonzero(selected), np.arange(396, 3216))
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
        or abs(legacy_rwp - 0.0623) > 5.0e-5
        or abs(legacy_rp - 0.0497) > 5.0e-5
    ):
        raise ValueError("IUCr deposited powder result failed its numerical checksum")

    destination_root.mkdir(parents=True, exist_ok=False)
    pattern = np.column_stack((selected_x, observed, legacy_calculated, legacy_background))
    np.savetxt(
        destination_root / "pattern.csv",
        pattern,
        delimiter=",",
        header="two_theta_deg,observed,legacy_calculated,legacy_background",
        comments="",
        fmt="%.12g",
    )
    for phase_id, block_name in _SOURCE_BLOCKS.items():
        phase_text = (
            _normalized_silicon_cif() if phase_id == "silicon" else _block(text, block_name)
        )
        (destination_root / f"{phase_id}.cif").write_text(phase_text, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "scope": "iucr_dicesium_citrate_silicon_internal_standard",
        "source_dataset_id": "iucr-dicesium-citrate-si-standard",
        "source_block": "RAMM016C_p_01",
        "instrument": {
            "device": "Bruker D2 Phaser",
            "geometry": "reflection, flat specimen",
            "wavelengths_angstrom": [1.540629, 1.544451],
            "k_alpha2_over_k_alpha1": 0.5,
            "polarization_fraction": 0.5,
            "initial_profile": {"U": 2.336, "V": 0.0, "W": 3.777, "X": 2.718, "Y": 1.868},
            "initial_zero_deg": -0.04480,
            "sh_over_l": 0.03384,
        },
        "data_selection": {
            "source_sample_count": 3217,
            "refined_sample_count": 2820,
            "two_theta_min_deg": float(selected_x[0]),
            "two_theta_max_deg": float(selected_x[-1]),
            "background": "deposited legacy-GSAS background fixed in both new workflows",
        },
        "silicon_calibration": {
            "role": "absolute specimen-displacement standard",
            "fixed_lattice_a_angstrom": 5.43105,
            "goniometer_radius_mm": 141.5,
            "windows_two_theta_deg": [
                [46.95, 47.78],
                [55.77, 56.63],
                [68.78, 69.68],
            ],
            "excluded_silicon_reflections": {
                "111": "overlaps the dominant citrate peak near 28.28 degrees"
            },
            "refined_parameters": [
                "silicon scale",
                "constant residual",
                "sample displacement",
            ],
            "frozen_after_calibration": [
                "sample displacement",
                "silicon lattice",
            ],
            "fixed_during_calibration": {
                "instrument_zero_shift_deg": -0.04480,
                "reason": (
                    "separate zero and specimen displacement are too correlated in "
                    "three isolated Si windows"
                ),
            },
        },
        "legacy_gsas_reference": {
            "weight_fractions": _LEGACY_WEIGHT_FRACTIONS,
            "rwp": legacy_rwp,
            "rp": legacy_rp,
            "rexp": 0.0296,
            "profile_correlation": float(
                np.corrcoef(observed - legacy_background, legacy_calculated - legacy_background)[
                    0, 1
                ]
            ),
        },
        "translation": {
            "silicon_setting": (
                "The ambiguous deposited Hall symbol is normalized to IT 227, "
                "F d -3 m :1, retaining Si at (1/8,1/8,1/8)."
            ),
            "common_profile_subset": (
                "The deposited phase-specific legacy profiles are initialized from the "
                "histogram U/V/W/X/Y values and represented by common instrumental "
                "broadening plus per-phase isotropic size and microstrain."
            ),
            "omitted_legacy_terms": (
                "The main phase's Stephens anisotropic broadening and legacy profile-function "
                "differences are intentionally outside the GSAS-II/PhaseSmith common model."
            ),
        },
    }
    manifest_path = destination_root / "experiment.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path
