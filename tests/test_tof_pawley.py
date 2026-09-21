"""Native TOF Pawley numerical, multi-bank ownership and atomic restart gates."""

import json
from dataclasses import replace

import numpy as np
import phasesmith
import pytest
from phasesmith import TofInstrument, UnitCell
from phasesmith.control import CancellationToken
from phasesmith.io.space_groups import space_group_by_number
from phasesmith.pattern import TofPowderPattern
from phasesmith.refinement import (
    LatticeParameterBounds,
    LatticeParameterization,
)
from phasesmith.refinement.core import ConstraintTransform
from phasesmith.refinement.pawley import PawleyOptions, parameter_key
from phasesmith.refinement.tof_multibank import TofSharedLatticePhase
from phasesmith.refinement.tof_pawley import (
    TofPawleyBackground,
    TofPawleyBank,
    TofPawleyInput,
    TofPawleyPhase,
    TofPawleyProject,
    build_parameter_set,
    calculate,
    refine,
)
from phasesmith.tof_pawley_reference import evaluate


def instrument(difc=5000.0):
    return TofInstrument(
        -0.7, difc, -0.2, 0.3, 0.18, 0.04, 0.0005, 0.001, 1.0, 10.0, 0.05, 0.2, 0.3, 0.05, 0.4
    )


def request(*, cell=False, points=501):
    lattice = UnitCell(4, 4, 4, 90, 90, 90)
    group = space_group_by_number(221).space_group
    par = LatticeParameterization(group, lattice)
    cells = (
        (
            TofSharedLatticePhase(
                "phase",
                par,
                LatticeParameterBounds.around(par, relative_length=0.02, angle_delta_deg=0.1),
                lattice,
            ),
        )
        if cell
        else ()
    )
    hkl = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0]], dtype=np.int64)
    d = 4 / np.sqrt(np.sum(hkl * hkl, axis=1))
    banks = []
    for b, difc in enumerate((5000, 4400)):
        x = np.linspace(0, 1, points) ** 1.015 * 14500 + 6500
        mask = np.ones(points, dtype=bool)
        mask[::23] = False
        banks.append(
            TofPawleyBank(
                f"bank{b}",
                TofPowderPattern(
                    x,
                    observed_y=np.zeros(points),
                    uncertainty=np.linspace(0.8, 1.2, points),
                    mask=mask,
                ),
                instrument(difc),
                (
                    TofPawleyPhase(
                        "phase",
                        ("100", "110", "111", "200"),
                        d,
                        np.array([120, 75, 0, 90]) * (b + 1),
                        hkl,
                    ),
                ),
                "Synthetic density per microsecond; no incident-spectrum correction",
                TofPawleyBackground((0.2, 0.01), (float(x[0]), float(x[-1]))),
            )
        )
    return TofPawleyInput(tuple(banks), cells)


def observed(r):
    truth = calculate(r).calculated_y
    banks = []
    for b, lo, hi in zip(r.banks, r.sample_offsets[:-1], r.sample_offsets[1:], strict=True):
        banks.append(
            replace(
                b,
                pattern=TofPowderPattern(
                    b.pattern.tof_us,
                    observed_y=truth[lo:hi],
                    uncertainty=b.pattern.uncertainty,
                    mask=b.pattern.mask,
                ),
                phases=(replace(b.phases[0], intensities=np.full(4, 50.0)),),
                background=replace(b.background, coefficients=(0.1, 0)),
            )
        )
    return replace(r, banks=tuple(banks), parameters=None)


def test_nonuniform_independent_reference_all_profile_cell_derivatives_and_products():
    r = request(cell=True, points=171)
    names = (
        "zero",
        "difa",
        "difb",
        "alpha",
        "beta0",
        "beta1",
        "betaq",
        "sigma0",
        "sigma1",
        "sigma2",
        "sigmaq",
        "x",
        "y",
        "z",
    )
    r = replace(
        r,
        parameters=build_parameter_set(
            r, profile_parameters={"bank0": names, "bank1": ("difc",)}, lattice=True
        ),
    )
    o = PawleyOptions(support_fwhm=20)
    c = calculate(r, o)
    y, j, residual = evaluate(r, options=o)
    np.testing.assert_allclose(c.calculated_y, y, rtol=3e-8, atol=2e-9)
    # The existing native TOF quadrature differs from converged 768-point
    # independent quadrature by 5.8e-7 in the normalized beta derivative here.
    # Scale each complete column, avoiding arbitrary pointwise relative errors at zeros.
    error = np.max(np.abs(c.jacobian - j), axis=0)
    scale = np.maximum(np.max(np.abs(j), axis=0), 1e-12)
    assert np.max(error / scale) < 2e-6
    np.testing.assert_allclose(c.weighted_residual * c.included, residual, rtol=3e-8, atol=2e-9)
    t = ConstraintTransform(r.parameters)
    z = t.pack()
    for k, key in enumerate(t.free_keys):
        if key.module not in ("pawley_profile", "pawley_lattice"):
            continue
        dz = np.zeros_like(z)
        dz[k] = 1e-6
        # Centered differences through native trial values check both component and cell chains.
        yp = calculate(
            replace(r, parameters=r.parameters.replace_values(t.unpack(z + dz))), o
        ).calculated_y
        ym = calculate(
            replace(r, parameters=r.parameters.replace_values(t.unpack(z - dz))), o
        ).calculated_y
        np.testing.assert_allclose((yp - ym) / 2e-6, c.jacobian[:, k], rtol=3e-4, atol=3e-5)
    product = calculate(r, replace(o, solver="matrix_free")).jacobian_operator
    rng = np.random.default_rng(934)
    v = rng.normal(size=len(z))
    u = rng.normal(size=len(y))
    np.testing.assert_allclose(product.jvp(v), c.jacobian @ v, rtol=2e-13, atol=2e-10)
    assert u @ product.jvp(v) == pytest.approx(v @ product.vjp(u), rel=2e-12)


@pytest.mark.parametrize("solver", ["dense", "matrix_free"])
def test_bank_local_areas_zero_release_and_single_bank_reduction(solver):
    r = observed(request())
    o = PawleyOptions(solver=solver)
    fit = refine(r, o)
    assert fit.termination_reason == "converged"
    np.testing.assert_allclose(fit.intensities, [120, 75, 0, 90, 240, 150, 0, 180], atol=3e-7)
    for i, b in enumerate(r.banks):
        single = refine(TofPawleyInput((b,)), o)
        np.testing.assert_allclose(
            single.calculation.calculated_y,
            fit.calculation.calculated_y[r.sample_offsets[i] : r.sample_offsets[i + 1]],
            atol=2e-9,
        )


def test_joint_cell_calibration_atomic_resume_and_persistence(tmp_path):
    r = observed(request(cell=True, points=801))
    r = replace(
        r, parameters=build_parameter_set(r, profile_parameters={"bank1": ("difc",)}, lattice=True)
    )
    values = {
        parameter_key("lattice", "phase", "a_angstrom"): 3.998,
        parameter_key("profile", "bank1", "difc"): 4401.0,
    }
    r = replace(r, parameters=r.parameters.replace_values(values))
    o = PawleyOptions(solver="matrix_free", support_fwhm=100)
    # Observations above used support 20, so align the truth with this numerical contract.
    truth = request(cell=True, points=801)
    y = calculate(truth, o).calculated_y
    r = replace(
        r,
        banks=tuple(
            replace(
                b,
                pattern=TofPowderPattern(
                    b.pattern.tof_us,
                    observed_y=y[lo:hi],
                    uncertainty=b.pattern.uncertainty,
                    mask=b.pattern.mask,
                ),
            )
            for b, lo, hi in zip(r.banks, r.sample_offsets[:-1], r.sample_offsets[1:], strict=True)
        ),
    )
    full = refine(r, o, max_iterations=100)
    assert full.termination_reason == "converged"
    assert full.parameters.values()[
        parameter_key("lattice", "phase", "a_angstrom")
    ] == pytest.approx(4, abs=2e-7)
    assert full.parameters.values()[parameter_key("profile", "bank1", "difc")] == pytest.approx(
        4400, abs=2e-4
    )
    token = CancellationToken()

    def progress(event):
        if event["accepted_iterations"] >= 2:
            token.request()

    stopped = refine(r, o, cancellation=token, progress=progress)
    assert stopped.termination_reason == "cancelled"
    p = TofPawleyProject(r, o, stopped.checkpoint)
    path = tmp_path / "joint.json"
    p.save(path)
    loaded = TofPawleyProject.load(path)
    resumed = loaded.refine(max_iterations=100)
    np.testing.assert_array_equal(full.history, resumed.history)
    np.testing.assert_array_equal(full.calculation.calculated_y, resumed.calculation.calculated_y)
    with pytest.raises(ValueError, match="exist"):
        p.save(path)
    wire = json.loads(path.read_text())
    wire["input"]["banks"][1]["normalization"] = "changed"
    path.write_text(json.dumps(wire))
    with pytest.raises(ValueError, match="digest"):
        TofPawleyProject.load(path)


def test_invalid_models_degeneracy_and_workspace():
    r = request(cell=True)
    with pytest.raises(ValueError, match="unanchored"):
        replace(
            r,
            parameters=build_parameter_set(
                r, profile_parameters={"bank0": ("difc",), "bank1": ("difc",)}, lattice=True
            ),
        )
    with pytest.raises(ValueError, match="normalization"):
        replace(r, banks=(replace(r.banks[0], normalization=""),))
    with pytest.raises(ValueError, match="duplicate"):
        replace(r, banks=(r.banks[0], r.banks[0]))
    with pytest.raises(ValueError, match="workspace"):
        calculate(r, PawleyOptions(max_elements=100))
    with pytest.raises(TypeError, match="real"):
        TofPawleyPhase("a", ("x",), [1 + 1j], [1])
    with pytest.raises(ValueError, match="positive"):
        replace(r, tail_log=-1)


def test_native_bundle_exchange_and_version6_migration(tmp_path):
    import subprocess
    from pathlib import Path

    from phasesmith.project_bundle import ProjectBundle

    p = TofPawleyProject(observed(request(cell=True)), PawleyOptions(solver="matrix_free"))
    p.refine(max_iterations=1)
    bundle = ProjectBundle.from_tof_pawley(p, analysis_id="joint")
    source = tmp_path / "source"
    bundle.save(source)
    wire = json.loads((source / "manifest.json").read_text())
    assert wire["format_version"] == 7
    assert bundle.tof_pawley_analyses == ("joint",)
    assert bundle.analysis_counts["tof_pawley"] == 1
    assert "tof_us" not in wire["tof_pawley_analyses"][0]["banks"][0]
    assert wire["project"]["phases"] == []
    loaded = ProjectBundle.load(source).tof_pawley("joint")
    native_path = tmp_path / "native"
    subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "phasesmith-persistence",
            "--example",
            "pawley_bundle_exchange",
            "--",
            str(source),
            str(native_path),
        ],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    native = ProjectBundle.load(native_path).tof_pawley("joint")
    np.testing.assert_array_equal(loaded.refine().history, native.refine().history)
    np.testing.assert_array_equal(loaded.calculate().calculated_y, native.calculate().calculated_y)
    changed = replace(
        p.input,
        banks=(
            replace(p.input.banks[0], instrument=replace(p.input.banks[0].instrument, zero_us=1)),
            p.input.banks[1],
        ),
    )
    with pytest.raises(ValueError, match="shared histogram"):
        bundle.with_tof_pawley("joint", TofPawleyProject(changed, p.options))
    wire["format_version"] = 6
    wire.pop("tof_pawley_analyses")
    (source / "manifest.json").write_text(json.dumps(wire))
    migrated = ProjectBundle.load(source)
    assert migrated.tof_pawley_analyses == ()
    migrated.save(tmp_path / "migrated")
    assert json.loads((tmp_path / "migrated/manifest.json").read_text())["format_version"] == 7


def test_tof_area_and_centroid_and_inclusive_support():
    import math

    inst = instrument()
    d = 2.0
    area = 7.0
    support = 12.0
    tail = 20.0
    p = phasesmith.tof_profile_parameters([d], inst)
    radius = support * p.total_fwhm_us[0]
    left = p.position_us[0] - radius - tail / p.alpha_per_us[0]
    right = p.position_us[0] + radius + tail / p.beta_per_us[0]

    def calculation(x):
        bank = TofPawleyBank(
            "bank",
            TofPowderPattern(x),
            inst,
            (TofPawleyPhase("phase", ("family",), [d], [area]),),
            "Synthetic microsecond density",
        )
        return calculate(
            TofPawleyInput((bank,), tail_log=tail), PawleyOptions(support_fwhm=support)
        ).calculated_y

    edges = np.array(
        [left - 1e-5, left, left + 1e-5, p.position_us[0], right - 1e-5, right, right + 1e-5]
    )
    y = calculation(edges)
    assert y[0] == y[-1] == 0
    assert y[1] >= 0 and y[-2] >= 0 and y[3] > 0
    x = np.linspace(left, right, 10001)
    y = calculation(x)
    integral = np.trapezoid(y, x)
    eta = p.eta[0]
    # Truncation removes the symmetric TCH base tails. The normalized truncated
    # exponential convolution preserves that base area and adds its own mean shift.
    base_area = (1 - eta) * math.erf(
        math.sqrt(4 * math.log(2)) * support
    ) + eta * 2 / math.pi * math.atan(2 * support)
    assert integral == pytest.approx(area * base_area, rel=3e-6)
    alpha, beta = p.alpha_per_us[0], p.beta_per_us[0]
    factor = 1 - tail / math.expm1(tail)
    expected = p.position_us[0] + factor * (alpha / beta - beta / alpha) / (alpha + beta)
    assert np.trapezoid(x * y, x) / integral == pytest.approx(expected, abs=2e-4)


@pytest.mark.parametrize("case_index", [1, 2, 3])
def test_pawley_tof_against_pinned_profile_oracle(case_index):
    from pathlib import Path

    from phasesmith.oracle import load_fixture

    fixture = load_fixture(Path(__file__).parents[1] / "oracle/fixtures/tof_v1")
    case = fixture.cases[case_index]
    p = case["parameters"]
    x = fixture.arrays[case["arrays"]["x"]]
    oracle = fixture.arrays[case["arrays"]["value"]]
    inst = TofInstrument(
        p["position_us"] - 1000,
        1000,
        0,
        0,
        p["alpha_per_us"],
        p["beta_per_us"],
        0,
        0,
        p["sigma2_us2"],
        0,
        0,
        0,
        0,
        0,
        p["gamma_us"],
    )
    bank = TofPawleyBank(
        "oracle",
        TofPowderPattern(x, observed_y=oracle),
        inst,
        (TofPawleyPhase("phase", ("family",), [1], [1]),),
        "Pinned oracle profile densities; one unit integrated family area",
    )
    r = TofPawleyInput((bank,))
    o = PawleyOptions(support_fwhm=10000)
    actual = calculate(r, o).calculated_y
    # Unchanged native TOF/GSAS-II profile convention tolerance.
    assert np.max(np.abs(actual - oracle)) / np.max(np.abs(oracle)) < 2.5e-4
    fit = refine(
        replace(
            r,
            banks=(replace(bank, phases=(replace(bank.phases[0], intensities=np.zeros(1)),)),),
            parameters=None,
        ),
        o,
    )
    assert fit.termination_reason == "converged"
    assert fit.intensities[0] == pytest.approx(1, abs=2.5e-4)


def test_mixed_uncertainties_require_explicit_unit_weights():
    r = request()
    bank = r.banks[1]
    mixed = replace(
        r,
        banks=(
            r.banks[0],
            replace(
                bank,
                pattern=TofPowderPattern(
                    bank.pattern.tof_us,
                    observed_y=bank.pattern.observed_y,
                    mask=bank.pattern.mask,
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="sigmas in every bank"):
        calculate(mixed)
    calculation = calculate(mixed, PawleyOptions(use_uncertainty=False))
    assert np.isfinite(calculation.calculated_y).all()


def test_distinct_reference_cell_and_initial_cell_roundtrip(tmp_path):
    r = request(cell=True)
    shared = r.shared_lattice[0]
    changed = replace(shared, initial_cell=UnitCell(3.999, 3.999, 3.999, 90, 90, 90))
    r = replace(r, shared_lattice=(changed,), parameters=None)
    project = TofPawleyProject(r)
    path = tmp_path / "reference.json"
    project.save(path)
    loaded = TofPawleyProject.load(path)
    assert loaded.input.shared_lattice[0].initial_cell.a_angstrom == 3.999
    np.testing.assert_array_equal(project.calculate().calculated_y, loaded.calculate().calculated_y)


def test_triclinic_shared_cell_six_derivative_chains():
    cell = UnitCell(4.1, 4.4, 4.7, 85, 92, 103)
    group = space_group_by_number(1).space_group
    par = LatticeParameterization(group, cell)
    shared = TofSharedLatticePhase(
        "phase",
        par,
        LatticeParameterBounds.around(par, relative_length=0.02, angle_delta_deg=1),
        cell,
    )
    r = replace(request(cell=True, points=151), shared_lattice=(shared,), parameters=None)
    r = replace(r, parameters=build_parameter_set(r, lattice=True))
    c = calculate(r)
    y, j, _ = evaluate(r)
    np.testing.assert_allclose(c.calculated_y, y, rtol=3e-8, atol=2e-9)
    t = ConstraintTransform(r.parameters)
    z = t.pack()
    selected = [i for i, key in enumerate(t.free_keys) if key.module == "pawley_lattice"]
    assert len(selected) == 6
    for k in selected:
        np.testing.assert_allclose(c.jacobian[:, k], j[:, k], rtol=3e-5, atol=3e-6)
        delta = np.zeros_like(z)
        delta[k] = 1e-6
        yp = calculate(replace(r, parameters=r.parameters.replace_values(t.unpack(z + delta))))
        ym = calculate(replace(r, parameters=r.parameters.replace_values(t.unpack(z - delta))))
        np.testing.assert_allclose(
            (yp.calculated_y - ym.calculated_y) / 2e-6,
            c.jacobian[:, k],
            rtol=3e-4,
            atol=3e-5,
        )


def test_saved_tof_projects_match_published_schemas(tmp_path):
    from pathlib import Path

    jsonschema = pytest.importorskip("jsonschema")
    from phasesmith.project_bundle import ProjectBundle
    from referencing import Registry, Resource

    root = Path(__file__).parents[1] / "schemas"
    registry = Registry()
    for path in root.glob("*.json"):
        schema = json.loads(path.read_text())
        if "$id" in schema:
            registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    project = TofPawleyProject(observed(request(cell=True)))
    project.refine(max_iterations=1)
    project.save(tmp_path / "standalone.json")
    ProjectBundle.from_tof_pawley(project).save(tmp_path / "bundle")
    for name, path in (
        ("tof-pawley-project-v1.schema.json", tmp_path / "standalone.json"),
        ("native-project-v7.schema.json", tmp_path / "bundle/manifest.json"),
    ):
        schema = json.loads((root / name).read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema, registry=registry).validate(
            json.loads(path.read_text())
        )


def test_unrelated_bank_difc_does_not_anchor_shared_cell():
    r = request(cell=True)
    other = replace(r.banks[1].phases[0], phase_id="unrelated")
    r = replace(r, banks=(r.banks[0], replace(r.banks[1], phases=(other,))), parameters=None)
    with pytest.raises(ValueError, match="unanchored"):
        replace(
            r,
            parameters=build_parameter_set(
                r, profile_parameters={"bank0": ("difc",)}, lattice=True
            ),
        )


@pytest.mark.parametrize("change", ["unknown", "version", "repeats", "case", "range"])
def test_tof_acceptance_manifest_rejects_ignored_controls(change):
    from pathlib import Path

    from phasesmith.validation.tof_pawley import validate_manifest

    manifest = json.loads(
        (Path(__file__).parents[1] / "validation/pawley-tof-acceptance-v1.json").read_text()
    )
    assert validate_manifest(manifest) is manifest
    if change == "unknown":
        manifest["ignored"] = True
    elif change == "version":
        manifest["version"] = 2
    elif change == "repeats":
        manifest["repeats"] = 1
    elif change == "case":
        manifest["cases"][0]["ignored"] = True
    else:
        manifest["cases"][0]["d_range"] = [1, 0]
    with pytest.raises(ValueError):
        validate_manifest(manifest)
