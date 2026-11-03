"""Post-integration structural diagnostics; archived coordinates never train intensities.

This is a fixed-model consistency audit, not an ab-initio structure solution.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from phasesmith.crystallography import UnitCell, calculate_structure_factor_values
from phasesmith.io.cif import read_cif
from phasesmith.scattering import XrayNonResonant


def main():
    baseline = Path("runs/learned-3d-profiles-20261025-final")
    output = Path("runs/profile-bias-investigation-20261025/structure-audit")
    output.mkdir(exist_ok=True)
    source = Path(
        "/Users/clemens/Dropbox/Science/_Archive/Fe7C3/X-ray diffraction/Final "
        "plots and tables/Lattice parameter and Atom position table/cif-"
        "files/fe7c3d_p01.cif"
    )
    original = read_cif(source).structure
    a = dict(np.load(baseline / "native.npz"))
    correspondence = dict(np.load(baseline / "matched.npz"))
    ni, di = correspondence["pairs"].T
    d = dict(np.load("runs/learned-3d-profiles-dials-20261025/arrays.npz"))
    protocol = json.loads((baseline / "protocol.json").read_text())
    meta = protocol["geometry"]
    vectors = np.array(meta["vectors"])
    origin, fast, slow, beam, axis = vectors
    beam /= np.linalg.norm(beam)
    axis /= np.linalg.norm(axis)
    ray = origin + a["predicted"][:, 0, None] * fast + a["predicted"][:, 1, None] * slow
    ray /= np.linalg.norm(ray, axis=1)[:, None]
    lorentz = np.abs(np.cross(ray, beam) @ axis)
    # Reproduce the external probe's stated beam assumption, not a measured beam polarisation.
    normal = np.array([0.0, 1.0, 0.0])
    efield = np.cross(beam, normal)
    efield /= np.linalg.norm(efield)
    polarisation = 0.999 * (1 - (ray @ efield) ** 2) + 0.001 * (1 - (ray @ normal) ** 2)
    lp = lorentz / polarisation
    np.testing.assert_allclose(lp[ni], d["lp"][di], rtol=3e-13, atol=3e-15)
    # Match the archived orthorhombic setting by cell axes; not by intensity agreement.
    direct = np.array(protocol["common_crystal_basis"])
    real = np.linalg.inv(direct).T
    lengths = np.linalg.norm(real, axis=1)[[1, 0, 2]]
    structure = replace(original, cell=UnitCell(*lengths, 90.0, 90.0, 90.0))
    hkl = a["hkl"][:, [1, 0, 2]]
    multiplicity = np.ones(len(hkl), np.int64)
    fc2 = calculate_structure_factor_values(
        structure, hkl, multiplicity, XrayNonResonant()
    ).f_squared
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
    wrong_fc = calculate_structure_factor_values(
        wrong, hkl, multiplicity, XrayNonResonant()
    ).f_squared
    fe = replace(structure, sites=tuple(s for s in structure.sites if s.element_symbol == "Fe"))
    fe_fc = calculate_structure_factor_values(fe, hkl, multiplicity, XrayNonResonant()).f_squared
    common = correspondence["input_quality"] & correspondence["strong"]
    training = common & a["training"][ni]
    heldout = common & a["heldout"][ni]
    sets = {
        "original_native": a["profile"][ni, :2],
        "dials": np.column_stack(
            [d["intensity.prf.value"][di], np.sqrt(np.maximum(d["intensity.prf.variance"][di], 0))]
        ),
    }
    for name, path in [
        ("physical_native", "runs/physical-3d-profiles-P1-20261025"),
        ("weighted_physical_native", "runs/physical-weighted-profiles-P1-20261025"),
    ]:
        p = Path(path) / "native.npz"
        if p.exists():
            sets[name] = np.load(p)["profile"][ni, :2]
    result = {}
    common_sigma = a["summed"][ni, 1] * lp[ni]
    for name, data in sets.items():
        measured = data[:, 0] * lp[ni]
        sigma = data[:, 1] * lp[ni]
        records = {}
        for label, calculated in [
            ("archived_Fe7C3", fc2),
            ("perturbed_sites_control", wrong_fc),
            ("Fe_only_control", fe_fc),
        ]:
            expected = calculated[ni]
            ok = training & np.isfinite(measured) & np.isfinite(sigma) & (sigma > 0)
            weight = 1 / common_sigma[ok] ** 2
            scale = float(
                np.sum(weight * expected[ok] * measured[ok]) / np.sum(weight * expected[ok] ** 2)
            )
            rows = heldout & np.isfinite(measured) & np.isfinite(sigma) & (sigma > 0)
            model = scale * expected[rows]
            observed = measured[rows]
            Fobs = np.sqrt(np.maximum(observed, 0))
            Fcalc = np.sqrt(np.maximum(model, 0))
            records[label] = dict(
                training=int(ok.sum()),
                validation=int(rows.sum()),
                training_scale=scale,
                heldout_R1=float(np.sum(abs(Fobs - Fcalc)) / np.sum(Fobs)),
                heldout_wR2=float(
                    np.sqrt(
                        np.sum(((observed - model) / common_sigma[rows]) ** 2)
                        / np.sum((observed / common_sigma[rows]) ** 2)
                    )
                ),
            )
        result[name] = records
    record = dict(
        source_cif=str(source),
        source_cif_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        test=(
            "fixed archived-coordinate prediction; one intensity scale fitted only on"
            " training reflections with identical pre-profile aperture-variance "
            "weights for every intensity set"
        ),
        not_structure_solution=True,
        not_full_refinement=True,
        integration_used_structure_prior=False,
        reindexing="native (h,k,l) -> archived (k,h,l), selected by cell lengths",
        atomic_model=(
            "archived fractional coordinates and Uiso; nonresonant Waasmaier-Kirfel "
            "factors; frozen measured cell lengths"
        ),
        corrections=(
            "independent Lorentz/polarisation calculation matches DIALS LP; .999 "
            "polarisation and normal (0,1,0) are probe defaults, not measured "
            "metadata; no absorption correction"
        ),
        fixed_population=(
            "same original common strong training and reserved reflections for every "
            "intensity set; original 40 reserved pairs"
        ),
        results=result,
        limitation=(
            "A correct structure may refine despite biased data. Missing DAC "
            "absorption, unverified polarisation/gain and incompleteness prevent "
            "treating these R factors as final structure-quality values."
        ),
    )
    (output / "audit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    np.savez_compressed(
        output / "arrays.npz",
        hkl=hkl,
        fc2=fc2,
        wrong_fc2=wrong_fc,
        fe_fc2=fe_fc,
        lp=lp,
        pairs=correspondence["pairs"],
        training=training,
        heldout=heldout,
        **sets,
    )


if __name__ == "__main__":
    main()
