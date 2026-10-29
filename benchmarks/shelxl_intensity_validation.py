"""Optional local SHELXL validation of an explicitly supplied atomic template.

Not imported by PhaseSmith or Dioptas. All signed rows are written to HKLF4;
no automatic OMIT, U restraints, symmetry choice or weight adjustment is added.
The caller supplies a template, executable and intensity/sigma convention.
"""

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np


def hkl_text(indices, intensity, sigma):
    h, i, s = np.asarray(indices), np.asarray(intensity), np.asarray(sigma)
    if (
        h.ndim != 2
        or h.shape[1] != 3
        or h.dtype.kind not in "iu"
        or i.shape != (len(h),)
        or s.shape != i.shape
        or not np.isfinite(i).all()
        or not np.isfinite(s).all()
        or np.any(s <= 0)
        or np.any(np.all(h == 0, axis=1))
    ):
        raise ValueError("HKLF4 requires integer nonzero hkl, finite signed I and positive sigma")

    def field(value):
        for digits in range(4, -1, -1):
            text = f"{value:8.{digits}f}"
            if len(text) == 8:
                return text
        raise ValueError("HKLF4 intensity range exceeded; choose a declared global normalization")

    lines = []
    for hh, value, error in zip(h, i, s, strict=True):
        header = "".join(f"{int(v):4d}" for v in hh)
        if len(header) != 12:
            raise ValueError("HKLF4 Miller-index field exceeded")
        error_text = field(error)
        if float(error_text) <= 0:
            raise ValueError(
                "sigma rounds to zero in HKLF4; choose an explicit global normalization"
            )
        lines.append(header + field(value) + error_text + "\n")
    return "".join(lines) + "   0   0   0  0.0000  0.0000\n"


def run(template, observations, executable, output, *, timeout=60.0):
    ins = Path(template).read_bytes()
    # Require a self-contained reflection/atomic template. Auxiliary file
    # directives and a non-HKLF4 input require explicit separate handling.
    text = ins.decode()
    if re.search(r"(?im)^\s*(?:\+|HKLF\s+(?!4\s*$))", text) or not re.search(
        r"(?im)^HKLF\s+4\s*$", text
    ):
        raise ValueError("validation requires a self-contained HKLF4 template")
    with np.load(observations, allow_pickle=False) as z:
        a = dict(z)
    contents = hkl_text(a["hkl"], a["intensity"], a["sigma"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "model.ins").write_bytes(ins)
    (output / "model.hkl").write_text(contents)
    start = time.perf_counter()
    result = subprocess.run(
        [str(Path(executable).resolve()), "model", "-t1"],
        cwd=output,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    (output / "stdout.log").write_text(result.stdout + "\n" + result.stderr)
    if (
        result.returncode
        or not (output / "model.fcf").exists()
        or not (output / "model.cif").exists()
    ):
        raise RuntimeError("SHELXL did not produce FCF/CIF; inspect stdout.log and model.lst")
    # Gemmi is an optional validation dependency, never a runtime requirement.
    import gemmi

    cif = gemmi.cif.read_file(str(output / "model.cif")).sole_block()
    fcf = gemmi.cif.read_file(str(output / "model.fcf")).sole_block()

    def number(value):
        return float(re.sub(r"\(\d+\)$", "", str(value)))

    fo2 = np.array([number(v) for v in fcf.find_values("_refln_F_squared_meas")])
    fc2 = np.array([number(v) for v in fcf.find_values("_refln_F_squared_calc")])
    if not len(fo2) or len(fc2) != len(fo2):
        raise ValueError("unsupported FCF columns")
    fo, fc = np.sqrt(np.maximum(fo2, 0)), np.sqrt(np.maximum(fc2, 0))
    atom_u = {}
    labels = list(cif.find_values("_atom_site_aniso_label"))
    u = np.array(
        [
            [number(v) for v in cif.find_values("_atom_site_aniso_U_" + suffix)]
            for suffix in ("11", "22", "33", "23", "13", "12")
        ]
    ).T
    # CIF Uij -> Cartesian U through the direct/reciprocal cell metric.
    cell = np.array(
        [
            number(cif.find_value("_cell_" + k))
            for k in (
                "length_a",
                "length_b",
                "length_c",
                "angle_alpha",
                "angle_beta",
                "angle_gamma",
            )
        ]
    )
    ca, cb, cg = np.cos(np.deg2rad(cell[3:]))
    sg = np.sin(np.deg2rad(cell[5]))
    cy = (ca - cb * cg) / sg
    direct = np.array(
        [
            [cell[0], cell[1] * cg, cell[2] * cb],
            [0.0, cell[1] * sg, cell[2] * cy],
            [0.0, 0.0, cell[2] * np.sqrt(1 - cb * cb - cy * cy)],
        ]
    )
    metric = direct @ np.diag(np.linalg.norm(np.linalg.inv(direct).T, axis=0))
    atom_cartesian = {}
    for label, row in zip(labels, u, strict=True):
        atom_u[label] = row.tolist()
        aa, bb, cc, bc, ac, ab = row
        tensor = metric @ np.array([[aa, ab, ac], [ab, bb, bc], [ac, bc, cc]]) @ metric.T
        atom_cartesian[label] = dict(
            Ueq=float(np.trace(tensor) / 3), eigenvalues=np.linalg.eigvalsh(tensor).tolist()
        )
    report = dict(
        input_observations=len(a["hkl"]),
        fcf_reflections=len(fo),
        R1_all_from_FCF=float(abs(fo - fc).sum() / fo.sum()),
        R1_gt_cif=number(cif.find_value("_refine_ls_R_factor_gt")),
        unique=int(number(cif.find_value("_refine_ls_number_reflns"))),
        parameters=int(number(cif.find_value("_refine_ls_number_parameters"))),
        anisotropic_U_cif_order_11_22_33_23_13_12=atom_u,
        U_convention="CIF crystallographic Uij plus explicit "
        "direct/reciprocal-metric Cartesian conversion",
        cartesian_U=atom_cartesian,
        nonpositive_U=[label for label, v in atom_cartesian.items() if min(v["eigenvalues"]) <= 0],
        input_template_sha256=hashlib.sha256(ins).hexdigest(),
        executable_sha256=hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
        observation_file_sha256=hashlib.sha256(Path(observations).read_bytes()).hexdigest(),
        elapsed_seconds=time.perf_counter() - start,
    )
    (output / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.template, args.observations, args.executable, args.output), indent=2))
