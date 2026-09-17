"""Independent numerical and application contracts for native CW Pawley."""

import json
from dataclasses import replace

import numpy as np
import pytest
from phasesmith import ConstantWavelengthInstrument, PowderPattern, UnitCell
from phasesmith.control import CancellationToken
from phasesmith.instrument import FcjGeometry
from phasesmith.pawley_reference import evaluate, exhaustive_linear_fit
from phasesmith.refinement import AffineConstraint, Bounds, ChebyshevBackground, ParameterSet
from phasesmith.refinement.core import ConstraintTransform
from phasesmith.refinement.pawley import (
    PawleyInput,
    PawleyOptions,
    PawleyPhase,
    PawleyProject,
    build_parameter_set,
    calculate,
    parameter_key,
    refine,
)
from phasesmith.symmetry import SpaceGroup, SymmetryOperation


def request(areas=(3.0, 7.0), positions=(40.0, 40.06), signed=False, axial=None, background=None):
    x = np.linspace(39, 41, 1001)
    phase = PawleyPhase("a", tuple(str(i) for i in range(len(areas))), positions, areas)
    return PawleyInput(
        PowderPattern(x, observed_y=np.zeros_like(x)),
        ConstantWavelengthInstrument(1.54, 0.0, 0.0, 0.001, 0.002, 0.0),
        (phase,),
        background,
        axial,
        signed,
    )


def with_observations(r, y):
    return replace(
        r,
        pattern=PowderPattern(
            r.pattern.x,
            observed_y=y,
            uncertainty=r.pattern.uncertainty,
            mask=r.pattern.mask,
            background=r.pattern.background,
        ),
    )


@pytest.mark.parametrize("axial", [None, FcjGeometry(0.005, 0.005)])
def test_random_reference_and_all_profile_derivatives(axial):
    rng = np.random.default_rng(131)
    for _ in range(3):
        r = request(
            rng.uniform(0.5, 8, 3),
            rng.uniform(39.7, 40.3, 3),
            axial=axial,
            background=ChebyshevBackground("bg", (1.0, -0.1), (39.0, 41.0)),
        )
        r = replace(
            r,
            parameters=build_parameter_set(
                r, profile_parameters=("u_deg2", "v_deg2", "w_deg2", "x_deg", "y_deg")
            ),
        )
        options = PawleyOptions(support_fwhm=1000.0)
        native = calculate(r, options)
        y, jac, _ = evaluate(r, options=options)
        np.testing.assert_allclose(native.calculated_y, y, rtol=2e-9, atol=2e-8)
        np.testing.assert_allclose(native.jacobian, jac, rtol=2e-8, atol=2e-7)
        t = ConstraintTransform(r.parameters)
        z = t.pack()
        for j in range(len(z)):
            step = 1e-5
            plus, minus = z.copy(), z.copy()
            plus[j] += step
            minus[j] -= step
            yp = evaluate(r, plus, options)[0]
            ym = evaluate(r, minus, options)[0]
            np.testing.assert_allclose((yp - ym) / (2 * step), jac[:, j], rtol=3e-5, atol=2e-5)
        u, v = rng.normal(size=len(y)), rng.normal(size=len(z))
        assert np.dot(u, jac @ v) == pytest.approx(np.dot(jac.T @ u, v), rel=2e-13)


def test_small_bounded_fit_matches_independent_face_enumeration():
    rng = np.random.default_rng(453)
    r = request(
        (0.0, 0.0, 0.0),
        (39.8, 40.0, 40.08),
        background=ChebyshevBackground("bg", (0.0, 0.0), (39.0, 41.0)),
    )
    a = calculate(r).jacobian
    # Scaled-free coordinates, including a deliberately negative unconstrained area.
    y = a @ np.array([3000.0, -500.0, 5000.0, 20.0, -5.0]) + rng.normal(0, 0.02, len(a))
    r = with_observations(r, y)
    result = refine(r)
    expected = exhaustive_linear_fit(a, y, (0, 1, 2))
    np.testing.assert_allclose(result.calculation.calculated_y, a @ expected, rtol=1e-8, atol=2e-7)
    assert result.termination_reason == "converged"
    assert result.covariance is None
    assert 1 in result.active_bounds


def test_joint_width_and_area_recovery():
    truth = request()
    y = calculate(truth).calculated_y
    r = request((1.0, 1.0))
    r = replace(r, instrument=replace(r.instrument, w_deg2=0.0015), parameters=None)
    r = with_observations(r, y)
    r = replace(r, parameters=build_parameter_set(r, profile_parameters=("w_deg2",)))
    fit = refine(r)
    assert fit.termination_reason == "converged"
    np.testing.assert_allclose(fit.intensities, [3, 7], rtol=1e-6)
    assert fit.parameters.spec(
        parameter_key("profile", "instrument", "w_deg2")
    ).value == pytest.approx(0.001, rel=1e-6)


def test_rank_ties_signed_intensities_and_no_mutation():
    r = request((3.0, 7.0), (40.0, 40.0))
    y = calculate(r).calculated_y
    r = with_observations(request((0.0, 0.0), (40.0, 40.0)), y)
    original = r.phases[0].intensities.copy()
    fit = refine(r)
    assert fit.rank == 1 and fit.covariance is None
    assert sum(fit.intensities) == pytest.approx(10, abs=1e-7)
    np.testing.assert_array_equal(r.phases[0].intensities, original)
    assert not fit.intensities.flags.writeable
    r = replace(
        r,
        constraints=(
            AffineConstraint(
                parameter_key("intensity", "a", "1"), parameter_key("intensity", "a", "0"), 2.0
            ),
        ),
    )
    fit = refine(r)
    np.testing.assert_allclose(fit.intensities, [10 / 3, 20 / 3], atol=1e-7)
    assert fit.covariance is not None
    signed = request((-3.0,), (40.0,), signed=True)
    target = calculate(signed).calculated_y
    signed = with_observations(request((0.0,), (40.0,), signed=True), target)
    assert refine(signed).intensities[0] == pytest.approx(-3.0, abs=1e-8)


def test_masks_covariance_and_unobserved():
    r = request((1.0, 11.0), (40.0, 90.0))
    y = calculate(request((5.0, 11.0), (40.0, 90.0))).calculated_y
    mask = np.ones(len(y), dtype=bool)
    mask[100] = False
    y = y.copy()
    y[100] = 1e12
    r = replace(
        r,
        pattern=PowderPattern(
            r.pattern.x, observed_y=y, mask=mask, uncertainty=np.full(len(y), 2.0)
        ),
    )
    fit = refine(r)
    np.testing.assert_allclose(fit.intensities, [5.0, 11.0], atol=1e-7)
    assert fit.observed_free_parameters == 1 and fit.calculation.inactive_columns == (1,)
    one = with_observations(
        request((1.0,), (40.0,)), calculate(request((5.0,), (40.0,))).calculated_y
    )
    one = replace(
        one,
        pattern=PowderPattern(
            one.pattern.x, observed_y=one.pattern.observed_y, uncertainty=np.full(len(y), 2.0)
        ),
    )
    fit = refine(one)
    expected = 1 / np.sum((fit.calculation.jacobian[:, 0] / 2) ** 2)
    assert fit.covariance[0, 0] == pytest.approx(expected, rel=1e-12)


def test_cell_constructor_geometry_chain_and_joint_recovery():
    group = SpaceGroup((SymmetryOperation(np.eye(3, dtype=np.int64), (0, 0, 0)),))
    cell = UnitCell(4.0, 5.0, 6.0, 90.0, 90.0, 90.0)
    phase = PawleyPhase.from_cell(
        "cell", cell, group, wavelength_angstrom=1.54, two_theta_range=(20.0, 35.0)
    )
    x = np.linspace(20.0, 35.0, 2501)
    r = PawleyInput(
        PowderPattern(x, observed_y=np.zeros_like(x)),
        ConstantWavelengthInstrument(1.54, 0.0, 0.0, 0.001, 0.002, 0.0),
        (phase,),
    )
    specs = [
        replace(s, refine=s.key.name == "a_angstrom") if s.key.module == "pawley_lattice" else s
        for s in r.parameters.specs
    ]
    r = replace(r, parameters=ParameterSet(specs))
    options = PawleyOptions(support_fwhm=1000.0)
    native = calculate(r, options)
    y, j, _ = evaluate(r, options=options)
    np.testing.assert_allclose(native.calculated_y, y, rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(native.jacobian, j, rtol=2e-8, atol=1e-6)
    t = ConstraintTransform(r.parameters)
    z = t.pack()
    plus, minus = z.copy(), z.copy()
    plus[-1] += 1e-7
    minus[-1] -= 1e-7
    np.testing.assert_allclose(
        (evaluate(r, plus, options)[0] - evaluate(r, minus, options)[0]) / 2e-7,
        j[:, -1],
        rtol=1e-5,
        atol=1e-3,
    )
    truth = replace(
        r,
        parameters=r.parameters.replace_values(
            {parameter_key("lattice", "cell", "a_angstrom"): 4.001}
        ),
    )
    r = with_observations(r, calculate(truth).calculated_y)
    fit = refine(r)
    assert fit.parameters.spec(
        parameter_key("lattice", "cell", "a_angstrom")
    ).value == pytest.approx(4.001, abs=2e-7)


def test_checkpoint_roundtrip_cancellation_and_stale_rejection(tmp_path):
    r = with_observations(request((0.0, 0.0)), calculate(request()).calculated_y)
    token = CancellationToken()
    events = []

    def progress(event):
        events.append(event)
        if event["kind"] == "step_accepted":
            token.request()

    first = refine(r, cancellation=token, progress=progress)
    assert first.termination_reason == "cancelled"
    assert events[-1]["kind"] == "termination"
    project = PawleyProject(r, checkpoint=first.checkpoint)
    path = tmp_path / "fit.pawley.json"
    project.save(path)
    with pytest.raises(FileExistsError):
        project.save(path)
    resumed = PawleyProject.load(path).refine()
    full = refine(r)
    np.testing.assert_array_equal(resumed.history, full.history)
    np.testing.assert_array_equal(resumed.intensities, full.intensities)
    with pytest.raises(ValueError, match="match"):
        refine(with_observations(r, r.pattern.observed_y + 1), checkpoint=first.checkpoint)
    content = json.loads(path.read_text())
    content["input"]["x_deg"][0] -= 0.1
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="digest"):
        PawleyProject.load(path)


def test_boundaries_invalid_and_finite_support():
    r = request((3.0,), (40.0,))
    with pytest.raises(ValueError, match="limit"):
        calculate(r, PawleyOptions(max_elements=10))
    with pytest.raises(ValueError):
        replace(r, phases=(replace(r.phases[0], intensities=[-1.0]),), parameters=None)
    with pytest.raises(ValueError):
        PawleyPhase("a", ("a",), [40.0], [np.nan])
    no_points = replace(
        r,
        pattern=PowderPattern(
            r.pattern.x, observed_y=np.zeros(1001), mask=np.zeros(1001, dtype=bool)
        ),
    )
    with pytest.raises(ValueError, match="included"):
        refine(no_points)
    y = calculate(r).calculated_y
    area = np.trapezoid(y, r.pattern.x)
    assert 2.99 < area < 3.0  # finite Lorentzian tail loss, not grid renormalization
    assert np.trapezoid((r.pattern.x - 40) * y, r.pattern.x) == pytest.approx(0.0, abs=1e-12)
    assert np.all(calculate(r, PawleyOptions(support_fwhm=1.0)).calculated_y[:100] == 0)


def test_dependent_bound_is_enforced():
    r = with_observations(
        request((0.0, 0.0), (40.0, 40.0)),
        calculate(request((10.0, 10.0), (40.0, 40.0))).calculated_y,
    )
    target = parameter_key("intensity", "a", "1")
    r = replace(
        r,
        parameters=ParameterSet(
            [replace(s, bounds=Bounds(0, 4)) if s.key == target else s for s in r.parameters.specs]
        ),
        constraints=(AffineConstraint(target, parameter_key("intensity", "a", "0"), 2),),
    )
    fit = refine(r)
    np.testing.assert_allclose(fit.intensities, [2, 4], atol=1e-7)


def test_exact_support_endpoints_and_fixed_only_request():
    from phasesmith.cw import cw_profile_parameters

    r = request((3.0,), (40.0,))
    width = cw_profile_parameters([40.0], r.instrument).total_fwhm_deg[0]
    left, right = 40.0 - width, 40.0 + width
    x = np.array([np.nextafter(left, -np.inf), left, 40.0, right, np.nextafter(right, np.inf)])
    r = replace(
        r,
        pattern=PowderPattern(x, observed_y=np.ones(5)),
        parameters=ParameterSet([replace(s, refine=False) for s in r.parameters.specs]),
    )
    y = calculate(r, PawleyOptions(support_fwhm=1.0)).calculated_y
    assert y[0] == y[-1] == 0 and y[1] > 0 and y[-2] > 0
    fit = refine(r, PawleyOptions(support_fwhm=1.0))
    assert fit.rank == 0 and fit.covariance is None


def test_extraction_from_existing_pinned_gsasii_profile_fixture():
    """This gates area conventions/profile composition, not GSAS-II solver parity."""
    from pathlib import Path

    from phasesmith.oracle.fixtures import load_fixture

    fixture = load_fixture(
        Path(__file__).resolve().parents[1] / "oracle/fixtures/cw_instrument_profile_v1"
    )
    case = next(c for c in fixture.cases if c["case_kind"] == "cw_overlapping_reflections")
    p = case["parameters"]
    x = fixture.arrays[case["arrays"]["x"]]
    y = fixture.arrays[case["arrays"]["ycalc"]]
    phase = PawleyPhase("oracle", ("0", "1", "2"), p["positions_deg"], np.zeros(3))
    r = PawleyInput(
        PowderPattern(x, observed_y=y),
        ConstantWavelengthInstrument(**dict(p["instrument"])),
        (phase,),
    )
    fit = refine(r, PawleyOptions(support_fwhm=100.0))
    # The source fixture's validated profile discretization error is below 6e-6.
    np.testing.assert_allclose(fit.intensities, p["integrated_intensities"], rtol=8e-6)
    assert np.max(np.abs(fit.calculation.calculated_y - y)) / np.max(y) < 6e-6


def test_joint_composed_lorentzian_boundary():
    x = np.linspace(20.0, 120.0, 5001)
    instrument = ConstantWavelengthInstrument(1.54, 0.001, 0.0, 0.002, 0.0, 0.0)
    phase = PawleyPhase("boundary", ("1", "2", "3"), [30.0, 60.0, 110.0], [5.0, 7.0, 3.0])
    truth = PawleyInput(PowderPattern(x, observed_y=np.zeros_like(x)), instrument, (phase,))
    y = calculate(truth).calculated_y
    r = replace(
        truth,
        instrument=replace(instrument, x_deg=0.02),
        parameters=None,
        pattern=PowderPattern(x, observed_y=y),
    )
    r = replace(r, parameters=build_parameter_set(r, profile_parameters=("x_deg", "y_deg")))
    fit = refine(r)
    assert fit.calculation.rwp < 1e-6
    assert fit.termination_reason == "converged"
    assert fit.active_width_bounds
    assert fit.covariance is None


@pytest.mark.parametrize("seed", [91, 142, 387, 444])
def test_multiple_bound_faces_match_independent_enumeration(seed):
    rng = np.random.default_rng(seed)
    r = request(np.zeros(6), np.linspace(39.8, 40.2, 6))
    design = calculate(r).jacobian
    target = design @ rng.uniform(-2000.0, 3000.0, 6) + rng.normal(0, 0.01, len(design))
    fit = refine(with_observations(r, target))
    oracle = exhaustive_linear_fit(design, target, tuple(range(6)))
    np.testing.assert_allclose(fit.calculation.calculated_y, design @ oracle, rtol=1e-7, atol=2e-7)
    assert fit.termination_reason == "converged"


def test_initialization_preserves_coupled_geometry_and_area_bounds():
    from phasesmith.refinement import LinearConstraint

    truth = request((2.0, 10.0), (39.9, 40.1))
    truth = replace(truth, instrument=replace(truth.instrument, w_deg2=0.002), parameters=None)
    r = with_observations(request((1.0, 10.0), (39.9, 40.1)), calculate(truth).calculated_y)
    first, second = (parameter_key("intensity", "a", str(i)) for i in range(2))
    width = parameter_key("profile", "instrument", "w_deg2")
    specs = build_parameter_set(r, profile_parameters=("w_deg2",)).specs
    specs = [replace(s, bounds=Bounds(0.0, 10.0)) if s.key == second else s for s in specs]
    r = replace(
        r,
        parameters=ParameterSet(specs),
        constraints=(LinearConstraint(second, ((first, 1.0), (width, -1000.0)), 10.0),),
    )
    fit = refine(r)
    assert fit.termination_reason == "converged"
    np.testing.assert_allclose(fit.intensities, [2.0, 10.0], rtol=1e-6)
    assert fit.parameters.spec(width).value == pytest.approx(0.002, rel=1e-6)


def test_cell_only_cif_and_domain_persistence(tmp_path):
    phase = PawleyPhase.from_cif(
        "cell",
        """data_cell
_cell_length_a 4
_cell_length_b 5
_cell_length_c 6
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
""",
        wavelength_angstrom=1.54,
        two_theta_range=(20.0, 40.0),
        initial_intensity=2.0,
    )
    assert len(phase.reflection_ids) > 0
    np.testing.assert_array_equal(phase.intensities, np.full(len(phase.reflection_ids), 2.0))
    r = PawleyInput(
        PowderPattern(np.linspace(20.0, 40.0, 501)),
        ConstantWavelengthInstrument(1.54, 0.0, 0.0, 0.001, 0.002, 0.0),
        (phase,),
    )
    project = PawleyProject(r)
    path = tmp_path / "cell.pawley.json"
    project.save(path)
    loaded = PawleyProject.load(path)
    assert loaded.input.phases[0].reflection_ids == phase.reflection_ids
    np.testing.assert_array_equal(loaded.calculate().calculated_y, project.calculate().calculated_y)


def test_joint_multiphase_cell_and_profile_recovery():
    group = SpaceGroup((SymmetryOperation(np.eye(3, dtype=np.int64), (0, 0, 0)),))
    phases = tuple(
        PawleyPhase.from_cell(
            name,
            UnitCell(a, b, c, 90, 90, 90),
            group,
            wavelength_angstrom=1.54,
            two_theta_range=(24.0, 34.0),
            initial_intensity=area,
        )
        for name, a, b, c, area in [("one", 4.0, 5.0, 6.0, 3.0), ("two", 4.3, 5.2, 6.2, 5.0)]
    )
    x = np.linspace(24.0, 34.0, 2001)
    r = PawleyInput(
        PowderPattern(x, observed_y=np.zeros_like(x)),
        ConstantWavelengthInstrument(1.54, 0, 0, 0.001, 0.002, 0),
        phases,
    )
    specs = build_parameter_set(r, profile_parameters=("w_deg2",)).specs
    r = replace(
        r,
        parameters=ParameterSet(
            [
                replace(s, refine=s.key.name == "a_angstrom")
                if s.key.module == "pawley_lattice"
                else s
                for s in specs
            ]
        ),
    )
    targets = {
        parameter_key("lattice", "one", "a_angstrom"): 4.001,
        parameter_key("lattice", "two", "a_angstrom"): 4.299,
        parameter_key("profile", "instrument", "w_deg2"): 0.0012,
    }
    truth = replace(r, parameters=r.parameters.replace_values(targets))
    options = PawleyOptions(support_fwhm=1000.0)
    r = with_observations(r, calculate(truth, options).calculated_y)
    fit = refine(r, options)
    assert fit.termination_reason == "converged"
    for key, expected in targets.items():
        assert fit.parameters.spec(key).value == pytest.approx(expected, rel=2e-7, abs=1e-9)
    np.testing.assert_allclose(
        fit.calculation.calculated_y, r.pattern.observed_y, rtol=2e-6, atol=1e-6
    )


def test_uncertainty_rescaling_and_tied_physical_covariance():
    r = request((2.0, 4.0), (39.8, 40.2))
    r = replace(
        r,
        constraints=(
            AffineConstraint(
                parameter_key("intensity", "a", "1"), parameter_key("intensity", "a", "0"), 2.0
            ),
        ),
    )
    r = with_observations(r, calculate(r).calculated_y)
    fits = []
    for sigma in [1.0, 3.0]:
        weighted = replace(
            r,
            pattern=PowderPattern(
                r.pattern.x,
                observed_y=r.pattern.observed_y,
                uncertainty=np.full(len(r.pattern.x), sigma),
            ),
        )
        fits.append(refine(weighted))
    np.testing.assert_allclose(fits[1].covariance, 9.0 * fits[0].covariance, rtol=2e-13)
    t = ConstraintTransform(r.parameters, r.constraints)
    chain = t.derivative_matrix()
    physical = chain @ fits[0].covariance @ chain.T
    assert physical[1, 1] == pytest.approx(4.0 * physical[0, 0], rel=2e-13)
    assert physical[0, 1] == pytest.approx(2.0 * physical[0, 0], rel=2e-13)
