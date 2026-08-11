"""Narrow translation for the pinned XRED anatase/rutile example."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np


def _number(text: str, field: str) -> float:
    match = re.search(rf"(?m)^{re.escape(field)}\s+([0-9.eE+-]+)", text)
    if match is None:
        raise ValueError(f"XRED anatase CIF does not contain {field}")
    return float(match.group(1))


def convert_xred_tio2_bundle(source: str | Path, destination: str | Path) -> Path:
    """Validate XRED inputs and emit an explicit common-model bundle.

    The COD anatase CIF declares only a nonstandard ``I 41/a m d S`` symbol.
    Its explicit operation set matches origin choice 1, which GSAS-II cannot
    infer from that symbol. The additional GSAS-facing CIF states the matched
    standard setting while retaining the deposited asymmetric coordinates.
    """

    source_root = Path(source)
    destination_root = Path(destination)
    if source_root.resolve() == destination_root.resolve():
        raise ValueError("source and destination directories must differ")
    required = {name: source_root / name for name in ("data.csv", "anatase.cif", "rutile.cif")}
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing XRED source files: {', '.join(missing)}")
    data = np.loadtxt(required["data.csv"], delimiter=",")
    if (
        data.ndim != 2
        or data.shape != (2501, 2)
        or not np.all(np.isfinite(data))
        or np.any(np.diff(data[:, 0]) <= 0.0)
        or np.any(data[:, 1] < 0.0)
    ):
        raise ValueError("pinned XRED TiO2 pattern has an unexpected shape or values")
    anatase = required["anatase.cif"].read_text(encoding="utf-8")
    markers = ("_cod_database_code               1010942", "'I 41/a m d S'", "Ti1 Ti4+", "O1 O2-")
    if any(marker not in anatase for marker in markers):
        raise ValueError("XRED anatase CIF does not match the reviewed COD setting")
    a = _number(anatase, "_cell_length_a")
    c = _number(anatase, "_cell_length_c")
    oxygen = re.search(r"(?m)^O1\s+O2-\s+\d+\s+\w+\s+\S+\s+\S+\s+([0-9.]+)", anatase)
    if oxygen is None:
        raise ValueError("XRED anatase CIF oxygen coordinate is missing")
    normalized = f"""data_anatase_origin_choice_1
_audit_creation_method 'PhaseSmith setting normalization of COD 1010942'
_chemical_formula_sum 'Ti O2'
_cell_formula_units_Z 4
_cell_length_a {a:.10g}
_cell_length_b {a:.10g}
_cell_length_c {c:.10g}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 141
_symmetry_space_group_name_H-M 'I 41/a m d :1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
Ti1 Ti 0 0 0 1 0.01
O1 O 0 0 {float(oxygen.group(1)):.10g} 1 0.01
"""
    destination_root.mkdir(parents=True, exist_ok=False)
    for name, path in required.items():
        shutil.copyfile(path, destination_root / name)
    (destination_root / "anatase-gsas.cif").write_text(normalized, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "scope": "xred_tio2_anatase_rutile_common_model",
        "source_dataset_id": "xred-tio2-anatase-rutile",
        "assumptions": {
            "radiation": "single Cu K-alpha1 approximation",
            "wavelength_angstrom": 1.54051,
            "polarization_fraction": 0.7,
            "background": (
                "eight-term Chebyshev residual; XRED states source background was removed"
            ),
            "composition_status": "unknown; fitted fractions are cross-program diagnostics only",
        },
        "setting_translation": {
            "source": "COD 1010942 explicit operations and I 41/a m d S",
            "matched_group": "I 41/a m d :1",
            "reason": "GSAS-II cannot infer the nonstandard S symbol from the deposited CIF",
        },
    }
    path = destination_root / "experiment.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
