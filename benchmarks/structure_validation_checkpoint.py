"""Limited structural checkpoint with held-out reflection families, not structure solution."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from phasesmith.crystallography import UnitCell, calculate_structure_factor_values
from phasesmith.io.cif import read_cif
from phasesmith.scattering import XrayNonResonant
from scipy.optimize import least_squares


def main(root, dials):
    source = Path(
        "/Users/clemens/Dropbox/Science/_Archive/Fe7C3/X-ray diffraction/"
        "Final plots and tables/Lattice parameter and Atom position table/"
        "cif-files/fe7c3d_p01.cif"
    )
    reference = read_cif(source).structure
    protocol = json.loads((root / "protocol.json").read_text())
    basis = np.array(protocol["common_crystal_basis"])
    direct = np.linalg.inv(basis).T
    structure = replace(
        reference, cell=UnitCell(*np.linalg.norm(direct, axis=1)[[1, 0, 2]], 90.0, 90.0, 90.0)
    )
    a = dict(np.load(root / "native.npz"))
    q = dict(np.load(root / "matched.npz"))
    b = dict(np.load(dials / "arrays.npz"))
    i, j = q["pairs"].T
    hkl = a["hkl"][:, [1, 0, 2]]
    multi = np.ones(len(hkl), np.int64)
    calc = calculate_structure_factor_values(structure, hkl, multi, XrayNonResonant())
    fc = calc.f_squared
    s2 = calc.s_inverse_angstrom**2
    rng = np.random.default_rng(20261025)
    wrong = replace(
        structure,
        sites=tuple(
            replace(
                s,
                fractional_xyz=tuple((np.array(s.fractional_xyz) + rng.uniform(-0.1, 0.1, 3)) % 1),
            )
            for s in structure.sites
        ),
    )
    fe = replace(structure, sites=tuple(s for s in structure.sites if s.element_symbol == "Fe"))
    expected_models = {
        "archived_Fe7C3": fc,
        "perturbed_sites_control": calculate_structure_factor_values(
            wrong, hkl, multi, XrayNonResonant()
        ).f_squared,
        "Fe_only_control": calculate_structure_factor_values(
            fe, hkl, multi, XrayNonResonant()
        ).f_squared,
    }
    common = q["input_quality"] & q["strong"]
    training = common & a["training"][i]
    heldout = common & a["heldout"][i]
    # LP is independently verified in structure_intensity_audit.py.
    lp = b["lp"][j]
    common_sigma = a["summed"][i, 1] * lp
    sets = {
        "native_frozen_profile": a["profile"][i, :2],
        "dials_CCD": np.column_stack(
            [b["intensity.prf.value"][j], np.sqrt(np.maximum(b["intensity.prf.variance"][j], 0))]
        ),
    }
    results = {}
    lower = -min(s.u_iso_angstrom2 for s in structure.sites if s.u_iso_angstrom2 is not None) + 1e-6
    for name, values in sets.items():
        observed = values[:, 0] * lp
        reports = {}
        for label, full_expected in expected_models.items():
            expected = full_expected[i]

            def residual(p, expected=expected, observed=observed):
                return (
                    np.exp(p[0] - 16 * np.pi**2 * p[1] * s2[i[training]]) * expected[training]
                    - observed[training]
                ) / common_sigma[training]

            fit = least_squares(
                residual,
                [0.0, 0.0],
                bounds=([-20.0, lower], [20.0, 0.1]),
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-10,
            )
            predicted = (
                np.exp(fit.x[0] - 16 * np.pi**2 * fit.x[1] * s2[i[heldout]]) * expected[heldout]
            )
            Fo = np.sqrt(np.maximum(observed[heldout], 0))
            Fc = np.sqrt(np.maximum(predicted, 0))
            reports[label] = dict(
                training=int(training.sum()),
                validation=int(heldout.sum()),
                intensity_scale=float(np.exp(fit.x[0])),
                common_delta_U_angstrom2=float(fit.x[1]),
                parameter_at_bound=bool(abs(fit.x[1] - lower) < 1e-5 or abs(fit.x[1] - 0.1) < 1e-5),
                heldout_R1=float(abs(Fo - Fc).sum() / Fo.sum()),
                heldout_wR2=float(
                    np.sqrt(
                        np.sum(((observed[heldout] - predicted) / common_sigma[heldout]) ** 2)
                        / np.sum((observed[heldout] / common_sigma[heldout]) ** 2)
                    )
                ),
            )
        results[name] = reports
    record = dict(
        test=(
            "fixed archived coordinates; scale and common Uiso shift fitted "
            "on training families only"
        ),
        source_cif=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        structure_not_used_for_integration=True,
        not_ab_initio_solution=True,
        not_full_atomic_refinement=True,
        weighting="identical pre-profile aperture variances for both intensity sets",
        correction_scope=(
            "LP with probe-assumed polarisation; no DAC absorption or frame scaling; "
            "same policy for both sets"
        ),
        caveat=(
            "These limited-model R factors do not establish structure solution "
            "or corrected intensity accuracy."
        ),
        results=results,
    )
    (root / "structure-checkpoint.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    np.savez_compressed(
        root / "structure-checkpoint-arrays.npz",
        hkl=hkl,
        fc2=fc,
        s2=s2,
        pairs=q["pairs"],
        training=training,
        heldout=heldout,
        lp=lp,
        **sets,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case", type=Path)
    p.add_argument("--dials", type=Path, required=True)
    a = p.parse_args()
    main(a.case, a.dials)
