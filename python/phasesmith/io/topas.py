"""Narrow, auditable translations from deposited TOPAS text inputs.

This module intentionally does not claim to parse the TOPAS language.  The
Rowles converter recognizes one checksum-pinned deposited model and emits a
neutral bundle containing ordinary XY, CIF, GSAS-II instrument, and JSON
files.  Unsupported source terms remain visible in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

ROWLES_SAMPLES = ("1a", "1e")
ROWLES_WEIGHED_WEIGHT_FRACTIONS = {
    "1a": {"Al2O3": 0.0115, "ZnO": 0.0404, "CaF2": 0.9481},
    "1e": {"Al2O3": 0.5512, "ZnO": 0.1525, "CaF2": 0.2962},
}

_PATTERN_NAMES = {sample: f"{sample}_1000000_0-010_n001.xy" for sample in ROWLES_SAMPLES}
_INPUT_NAMES = {sample: f"robustness2_{sample}_4.INP" for sample in ROWLES_SAMPLES}
_REQUIRED_SOURCE_MARKERS = (
    '#include "row119.inc"',
    "LP_Factor( 0)",
    "Rp 250",
    "Rs 250",
    "start_X  21",
    "finish_X  HAL",
    "filament_length  12",
    "sample_length  15",
    "receiving_slit_length  12",
    "primary_soller_angle    2.5",
    "secondary_soller_angle  2.5",
    'phase_name "Corundum"',
    'phase_name "Fluorite"',
    'phase_name "Zincite"',
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(text: str, pattern: str, label: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"deposited TOPAS input does not contain {label}")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError(f"deposited TOPAS {label} is not finite")
    return value


def _line_components(text: str) -> list[tuple[float, float, float]]:
    matches = re.findall(
        r"^\s*la\s+([0-9.eE+-]+)\s+lo\s+([0-9.eE+-]+)\s+lh\s+([0-9.eE+-]+)",
        text,
        flags=re.MULTILINE,
    )
    components = [
        (float(area), float(wavelength), float(width)) for area, wavelength, width in matches
    ]
    if len(components) != 7 or any(not all(map(math.isfinite, row)) for row in components):
        raise ValueError("expected seven finite wavelength components in deposited TOPAS input")
    return components


def _weighted_component(rows: list[tuple[float, float, float]]) -> tuple[float, float]:
    weight = sum(row[0] for row in rows)
    return sum(row[0] * row[1] for row in rows) / weight, weight


def _phase_values(text: str) -> dict[str, float]:
    patterns = {
        "corundum_a": r"#prm\s+a_cor\s+=\s+randomValue\(([0-9.]+)",
        "corundum_c": r"#prm\s+c_cor\s+=\s+randomValue\(([0-9.]+)",
        "fluorite_a": r"#prm\s+a_flu\s+=\s+randomValue\(([0-9.]+)",
        "zincite_a": r"#prm\s+a_zin\s+=\s+randomValue\(([0-9.]+)",
        "zincite_c": r"#prm\s+c_zin\s+=\s+randomValue\(([0-9.]+)",
        "corundum_al_z": r"site Al[^\n]*?z\s+!z1_cor\s+([0-9.]+)",
        "corundum_o_x": r"site O\s+num_posns\s+18\s+x\s+!x2_cor\s+([0-9.]+)",
        "zincite_o_z": r"site O\s+num_posns\s+2[^\n]*?z\s+!z2_zincite\s+([0-9.]+)",
    }
    return {name: _number(text, pattern, name) for name, pattern in patterns.items()}


def _edge_values(text: str) -> tuple[float, float, float]:
    edge = _number(
        text,
        r"Absorption_Edge_Correction\(\s*2,,\s*([0-9.eE+-]+)",
        "absorption-edge wavelength",
    )
    sharpness = _number(text, r"prm\s+!a_erf\s+([0-9.eE+-]+)", "a_erf")
    floor_numerator = _number(text, r"prm\s+!edge_extra\s*=\s*([0-9.eE+-]+)\s*/", "edge_extra")
    floor_denominator = _number(
        text,
        r"prm\s+!edge_extra\s*=\s*[0-9.eE+-]+\s*/\s*([0-9.eE+-]+)",
        "edge_extra denominator",
    )
    return edge, sharpness, floor_numerator / floor_denominator


def _white_continuum_values(text: str) -> tuple[float, float]:
    amplitude_numerator = _number(
        text,
        r"prm\s+!a_white\s*=\s*([0-9.eE+-]+)\s*/\s*1000000",
        "a_white for 0.010 degree data",
    )
    decay_per_angstrom2 = _number(text, r"prm\s+!b_white\s+([0-9.eE+-]+)", "b_white")
    return amplitude_numerator / 1_000_000.0, decay_per_angstrom2


def _cif_files(values: dict[str, float]) -> dict[str, str]:
    return {
        "Al2O3.cif": f"""data_corundum_from_rowles_topas
_audit_creation_method 'PhaseSmith narrow translation of robustness2 TOPAS input'
_chemical_formula_sum 'Al2 O3'
_cell_formula_units_Z 6
_cell_length_a {values["corundum_a"]:.9g}
_cell_length_b {values["corundum_a"]:.9g}
_cell_length_c {values["corundum_c"]:.9g}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
_symmetry_space_group_name_H-M 'R -3 c :H'
_space_group_IT_number 167
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_B_iso_or_equiv
Al1 Al 0 0 {values["corundum_al_z"]:.9g} 1 0.32
O1 O {values["corundum_o_x"]:.9g} 0 0.25 1 0.33
""",
        "CaF2.cif": f"""data_fluorite_from_rowles_topas
_audit_creation_method 'PhaseSmith narrow translation of robustness2 TOPAS input'
_chemical_formula_sum 'Ca F2'
_cell_formula_units_Z 4
_cell_length_a {values["fluorite_a"]:.9g}
_cell_length_b {values["fluorite_a"]:.9g}
_cell_length_c {values["fluorite_a"]:.9g}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'F m -3 m'
_space_group_IT_number 225
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_B_iso_or_equiv
Ca1 Ca 0 0 0 1 0.41
F1 F 0.25 0.25 0.25 1 0.62
""",
        "ZnO.cif": f"""data_zincite_from_rowles_topas
_audit_creation_method 'PhaseSmith narrow translation of robustness2 TOPAS input'
_chemical_formula_sum 'Zn O'
_cell_formula_units_Z 2
_cell_length_a {values["zincite_a"]:.9g}
_cell_length_b {values["zincite_a"]:.9g}
_cell_length_c {values["zincite_c"]:.9g}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
_symmetry_space_group_name_H-M 'P 63 m c'
_space_group_IT_number 186
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_B_iso_or_equiv
Zn1 Zn 0.333333333 0.666666667 0 1 0.45
O1 O 0.333333333 0.666666667 {values["zincite_o_z"]:.9g} 1 0.73
""",
    }


def convert_rowles_topas_bundle(source: str | Path, destination: str | Path) -> Path:
    """Translate the pinned Rowles TOPAS inputs into a common neutral bundle."""

    source_root = Path(source)
    destination_root = Path(destination)
    if source_root.resolve() == destination_root.resolve():
        raise ValueError("source and destination directories must differ")
    sources = [source_root / "row119.inc"]
    input_texts: dict[str, str] = {}
    for sample in ROWLES_SAMPLES:
        input_path = source_root / _INPUT_NAMES[sample]
        pattern_path = source_root / _PATTERN_NAMES[sample]
        if not input_path.is_file() or not pattern_path.is_file():
            raise FileNotFoundError(f"Rowles source bundle is incomplete for sample {sample}")
        text = input_path.read_text(encoding="utf-8")
        missing = [marker for marker in _REQUIRED_SOURCE_MARKERS if marker not in text]
        if missing:
            raise ValueError(f"deposited TOPAS input for {sample} is missing {missing[0]!r}")
        declared_sample = re.search(r"macro sample \{\s*(1[ae])\s*\}", text)
        if declared_sample is None or declared_sample.group(1) != sample:
            raise ValueError(f"deposited TOPAS sample macro does not match {sample}")
        input_texts[sample] = text
        sources.extend((input_path, pattern_path))
    if not sources[0].is_file():
        raise FileNotFoundError("Rowles source bundle is missing row119.inc")

    reference = input_texts["1a"]
    values = _phase_values(reference)
    components = _line_components(reference)
    if (
        _phase_values(input_texts["1e"]) != values
        or _line_components(input_texts["1e"]) != components
        or _edge_values(input_texts["1e"]) != _edge_values(reference)
        or _white_continuum_values(input_texts["1e"]) != _white_continuum_values(reference)
    ):
        raise ValueError(
            "1a and 1e TOPAS inputs do not share the expected structure/instrument model"
        )
    alpha1, alpha1_weight = _weighted_component(components[1:3])
    alpha2, alpha2_weight = _weighted_component(components[3:5])
    ratio = alpha2_weight / alpha1_weight
    edge_angstrom, edge_sharpness, edge_floor = _edge_values(reference)
    white_amplitude, white_decay = _white_continuum_values(reference)

    destination_root.mkdir(parents=True, exist_ok=True)
    for sample in ROWLES_SAMPLES:
        shutil.copyfile(source_root / _PATTERN_NAMES[sample], destination_root / f"{sample}.xy")
    for name, cif in _cif_files(values).items():
        (destination_root / name).write_text(cif, encoding="utf-8")
    instrument = f"""#GSAS-II instrument parameter file generated from a neutral common model
Type:PXC
Bank:1.0
Lam1:{alpha1:.12g}
Lam2:{alpha2:.12g}
Zero:0.0
Polariz.:0.5
Azimuth:0.0
I(L2)/I(L1):{ratio:.12g}
U:2.0
V:-2.0
W:5.0
X:0.0
Y:0.0
Z:0.0
SH/L:0.002
Source:CuKa
"""
    (destination_root / "common.instprm").write_text(instrument, encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "scope": "curtin_rowles_qpa_topas_common_subset",
        "source": {
            "dataset_doi": "10.25917/5f44ad65411cc",
            "format": "TOPAS v6 input plus two-column XY",
            "files": {path.name: _sha256(path) for path in sources},
        },
        "patterns": {
            sample: {
                "file": f"{sample}.xy",
                "source_file": _PATTERN_NAMES[sample],
                "weighed_weight_fractions": ROWLES_WEIGHED_WEIGHT_FRACTIONS[sample],
            }
            for sample in ROWLES_SAMPLES
        },
        "common_model": {
            "range_two_theta_deg": [21.0, 150.0],
            "background": {"kind": "chebyshev", "terms": 7},
            "radiation": {
                "kind": "Cu K-alpha doublet",
                "lambda1_angstrom": alpha1,
                "lambda2_angstrom": alpha2,
                "lambda2_over_lambda1_intensity": ratio,
                "polarization_fraction": 0.5,
            },
            "profile_initializer": {
                "u_deg2": 2.0e-4,
                "v_deg2": -2.0e-4,
                "w_deg2": 5.0e-4,
                "x_deg": 0.0,
                "y_deg": 0.0,
                "fcj_sample_over_radius": 0.001,
                "fcj_detector_over_radius": 0.001,
            },
            "phases": ["Al2O3", "ZnO", "CaF2"],
        },
        "topas_source_model": {
            "emission_lines": [
                {
                    "area": area,
                    "wavelength_angstrom": wavelength,
                    "lorentzian_hwhm_milliangstrom": width,
                    "reference": index == 1,
                }
                for index, (area, wavelength, width) in enumerate(components)
            ],
            "reference_line_index": 1,
            "minimum_relative_height": 1.0e-5,
            "absorption_edge_filter": {
                "kind": "error_function_high_pass",
                "edge_angstrom": edge_angstrom,
                "sharpness_per_angstrom": edge_sharpness,
                "floor": edge_floor,
            },
            "angle_dependent_white_continuum": {
                "amplitude": white_amplitude,
                "gaussian_decay_per_angstrom2": white_decay,
                "bragg_angle_factor": "1/tan(theta)",
                "supported": False,
            },
        },
        "translation": {
            "translated": [
                "two-column observed patterns and Poisson count uncertainties",
                "three crystal structures, cells, coordinates, occupancies, and initial Biso",
                "seven-term polynomial background",
                "21-150 degree refinement range",
                "Cu K-alpha1/K-alpha2 groups as area-weighted doublet centroids",
            ],
            "approximated": [
                "TOPAS seven-line emission spectrum reduced to the common K-alpha doublet",
                "detailed axial convolution replaced by a neutral SH/L=0.002 initializer",
                "U/V/W values are neutral common-model initializers, not TOPAS parameters",
            ],
            "omitted": [
                "K-beta, continuum, and line-specific emission widths from the common workflow",
                "angle-dependent white-radiation term from the experimental source-spectrum slice",
                "Tube_Tails and detector equatorial convolution",
                "physical source/sample/slit lengths and Soller-limited axial convolution",
                "mixture absorption and absorption-edge correction from the common workflow",
                "TOPAS randomized multi-start and correlation-screening logic",
            ],
        },
    }
    manifest_path = destination_root / "experiment.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path
