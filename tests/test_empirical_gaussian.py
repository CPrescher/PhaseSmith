"""Empirical width conventions preserve profiles and remain explicit on resume."""

from dataclasses import replace

import numpy as np
import phasesmith as ps
import pytest
from phasesmith.empirical import _strain
from phasesmith.refinement import (
    ConstraintTransform,
    FixedConstraint,
    ParameterSet,
    RefinementLimits,
)
from phasesmith.refinement import rietveld as rv
from phasesmith.refinement.workflow import _stage_input
from test_rietveld_refinement import P1_CIF, component_experiment, experiment, selection


def request(*, doublet=False, composite=False, strains=(8e-4,)):
    x = np.linspace(15, 100, 2501)
    selected = selection(phase_scale=True, sample_physics=True, instrument_parameters=("u_deg2",))
    base = rv.RietveldInput.from_cif(
        ps.PowderPattern(x, observed_y=np.zeros_like(x)),
        component_experiment() if doublet else experiment(),
        P1_CIF,
        phase_id="alpha",
        selection=selected,
        physics=ps.IsotropicMicrostrainBroadening(strains[0]),
    )
    phases = []
    for i, strain in enumerate(strains):
        physics = ps.IsotropicMicrostrainBroadening(strain)
        if composite:
            physics = ps.CompositePhysicsProvider((ps.IsotropicSizeBroadening(150), physics))
        phases.append(replace(base.phases[0], phase_id=f"phase{i}", physics=physics))
    phases = tuple(phases)
    truth = rv.calculate(base.pattern, base.experiment, phases).y
    phases = tuple(replace(p, scale=0.85) for p in phases)
    return rv.RietveldInput(
        ps.PowderPattern(x, observed_y=truth, uncertainty=np.sqrt(np.maximum(truth, 1))),
        base.experiment,
        phases,
        (None,) * len(phases),
        rv.build_parameter_set(phases, (None,) * len(phases), selected, experiment=base.experiment),
        selection=selected,
    )


@pytest.mark.parametrize("doublet", [False, True])
@pytest.mark.parametrize("composite", [False, True])
def test_transfer_preserves_randomized_total_widths_and_full_profiles(doublet, composite):
    rng = np.random.default_rng(20260917)
    for _ in range(3):
        strains = (8e-4, float(rng.uniform(8e-4, 1.2e-3)))
        old = request(doublet=doublet, composite=composite, strains=strains)
        convention = ps.EmpiricalGaussianConvention("phase0", 2e-4)
        new = convention.apply(old)
        assert convention.apply(new) is new
        assert _strain(new.phases[0]).rms_microstrain == 2e-4
        c = (360 / np.pi) ** 2
        # Independent variance identity over the complete angular domain.
        theta = np.deg2rad(np.linspace(0.01, 89.99, 1001))
        for before, after in zip(old.phases, new.phases, strict=True):
            u0 = old.experiment.instrument.u_deg2 + c * _strain(before).rms_microstrain ** 2
            u1 = new.experiment.instrument.u_deg2 + c * _strain(after).rms_microstrain ** 2
            np.testing.assert_allclose(u0 * np.tan(theta) ** 2, u1 * np.tan(theta) ** 2, rtol=7e-16)
        y0 = rv.calculate(old.pattern, old.experiment, old.phases).y
        y1 = rv.calculate(new.pattern, new.experiment, new.phases).y
        np.testing.assert_allclose(y1, y0, rtol=3e-13, atol=3e-13)
        # Includes normalization and moments of the actual finite-support grid.
        for power in range(3):
            np.testing.assert_allclose(
                np.sum(y1 * old.pattern.x**power), np.sum(y0 * old.pattern.x**power), rtol=3e-14
            )
        key = rv.sample_parameter_key("phase0", "isotropic_microstrain.rms")
        assert key not in ConstraintTransform(new.parameters, new.constraints).free_keys


def test_reference_is_restored_after_stage_selection():
    new = ps.EmpiricalGaussianConvention("phase0", 2e-4).apply(request())
    off = _stage_input(new, replace(new.selection, sample_physics=False))
    assert off.constraints == ()
    on = _stage_input(off, new.selection)
    assert on.constraints == new.constraints
    assert on.empirical_gaussian == new.empirical_gaussian
    key = rv.sample_parameter_key("phase0", "isotropic_microstrain.rms")
    with pytest.raises(ValueError, match="conflicts"):
        replace(new, constraints=(FixedConstraint(key, 3e-4),))


@pytest.mark.parametrize("reference", [0, -1e-4, float("nan"), float("inf"), 1e-200, 1e200])
def test_invalid_anchor(reference):
    with pytest.raises(ValueError):
        ps.EmpiricalGaussianConvention("phase0", reference)


@pytest.mark.parametrize("reference", [True, "0.001", None])
def test_invalid_anchor_type(reference):
    with pytest.raises(TypeError):
        ps.EmpiricalGaussianConvention("phase0", reference)


def test_rejects_infeasible_missing_and_ambiguous_conventions():
    old = request(strains=(8e-4, 1e-4))
    convention = ps.EmpiricalGaussianConvention("phase0", 2e-4)
    with pytest.raises(ValueError, match="cannot use"):
        convention.apply(old)
    with pytest.raises(ValueError, match="missing"):
        ps.EmpiricalGaussianConvention("absent", 2e-4).apply(old)
    old = request()
    key = rv.instrument_parameter_key("u_deg2")
    with pytest.raises(ValueError, match="existing constraints"):
        convention.apply(
            replace(old, constraints=(FixedConstraint(key, old.parameters.spec(key).value),))
        )
    specs = [replace(s, scale=s.scale * 2) if s.key == key else s for s in old.parameters.specs]
    with pytest.raises(ValueError, match="custom U/strain"):
        convention.apply(replace(old, parameters=ParameterSet(specs)))
    with pytest.raises(ValueError, match="reanchor"):
        ps.EmpiricalGaussianConvention("phase0", 3e-4).apply(convention.apply(old))
    with pytest.raises(ValueError, match="one isotropic"):
        convention.apply(
            replace(
                old,
                phases=(replace(old.phases[0], physics=None),),
                parameters=rv.build_parameter_set(
                    (replace(old.phases[0], physics=None),),
                    (None,),
                    old.selection,
                    experiment=old.experiment,
                ),
            )
        )
    bad = convention.to_record() | {"schema": "unknown"}
    with pytest.raises(ValueError, match="schema"):
        ps.EmpiricalGaussianConvention.from_record(bad)


@pytest.mark.parametrize("python_path", [False, True])
def test_refinement_and_resume_preserve_convention(python_path):
    old = request()
    new = ps.EmpiricalGaussianConvention("phase0", 2e-4).apply(old)
    options = rv.RietveldOptions(
        limits=RefinementLimits(max_iterations=4), estimate_covariance=False
    )
    kwargs = {"logger": lambda event: None} if python_path else {}
    complete = rv.refine(new, options, **kwargs)
    partial = rv.refine(new, replace(options, limits=RefinementLimits(max_iterations=1)), **kwargs)
    resumed = rv.refine(new, options, checkpoint=partial.checkpoint, **kwargs)
    np.testing.assert_array_equal(complete.calculation.y, resumed.calculation.y)
    assert complete.history == resumed.history
    assert complete.checkpoint.empirical_gaussian == new.empirical_gaussian
    assert _strain(complete.phases[0]).rms_microstrain == 2e-4
    assert complete.metrics.rwp < 1e-7
    assert (
        ps.rietveld_result_record(complete)["empirical_gaussian"]
        == new.empirical_gaussian.to_record()
    )
    with pytest.raises(ValueError, match="convention changed"):
        rv.refine(old, options, checkpoint=partial.checkpoint)


@pytest.mark.parametrize("doublet", [False, True])
def test_project_roundtrip_and_resumed_fit(tmp_path, doublet):
    new = ps.EmpiricalGaussianConvention("phase0", 2e-4).apply(request(doublet=doublet))
    options = rv.RietveldOptions(
        limits=RefinementLimits(max_iterations=4), estimate_covariance=False
    )
    project = ps.RietveldProject(new, replace(options, limits=RefinementLimits(max_iterations=1)))
    project.refine()
    saved = project.save(tmp_path / "empirical")
    restored = ps.RietveldProject.load(saved)
    assert restored.input.empirical_gaussian == new.empirical_gaussian
    assert restored.checkpoint.empirical_gaussian == new.empirical_gaussian
    assert restored.input.constraints == new.constraints
    if not doublet:
        assert restored.checkpoint._native is not None
    np.testing.assert_array_equal(restored.calculate().y, project.calculate().y)
    restored.options = options
    result = restored.refine()
    complete = rv.refine(new, options)
    np.testing.assert_allclose(result.calculation.y, complete.calculation.y, rtol=3e-12, atol=3e-12)


def test_anchor_removes_width_null_direction_and_preserves_analytical_chains():
    from phasesmith.refinement.runtime import RefinementRuntime

    old = request()
    new = ps.EmpiricalGaussianConvention("phase0", 2e-4).apply(old)
    options = rv.RietveldOptions()

    def linearization(q):
        value = rv._RietveldLinearization.prepare(
            q,
            q.experiment,
            q.background,
            q.phases,
            q.lattice_domains,
            q.parameters,
            options,
            RefinementRuntime(RefinementLimits()),
        )
        value.calculate()
        return value.weighted_free_jacobian.T

    before = linearization(old)
    after = linearization(new)
    assert np.linalg.matrix_rank(before) == 2  # three columns: scale, U, RMS strain
    assert after.shape[1] == np.linalg.matrix_rank(after) == 2
    transform = ConstraintTransform(new.parameters, new.constraints)
    point = transform.pack()
    for column in range(len(point)):

        def evaluate(delta, column=column):
            trial = point.copy()
            trial[column] += delta
            values = transform.unpack(trial)
            exp, bg = rv._apply_profile_background_values(new.experiment, new.background, values)
            phases, _ = rv._apply_parameter_values(
                new.phases, new.lattice_domains, new.parameters, values
            )
            return rv.calculate(new.pattern, exp, phases, background=bg).y / new.pattern.uncertainty

        h = 1e-6
        fd = (evaluate(h) - evaluate(-h)) / (2 * h)
        assert np.linalg.norm(fd - after[:, column]) / np.linalg.norm(after[:, column]) < 2e-7


def test_readiness_and_fit_advice_disclose_empirical_interpretation():
    new = ps.EmpiricalGaussianConvention("phase0", 2e-4).apply(request())
    project = ps.RietveldProject(new)
    assert any(
        d.code == "profile.empirical_gaussian" for d in project.review_readiness().diagnostics
    )
    project.refine()
    assert any(a.code == "empirical_gaussian_convention" for a in project.fit_report().advice)
