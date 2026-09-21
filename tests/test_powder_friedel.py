"""Powder intensities average opposite reflections before profile accumulation."""

from dataclasses import replace

import numpy as np
import phasesmith as ps
import pytest
from phasesmith.crystallography_reference import reference_structure_factor_values
from test_structural_pattern import _perturb_phase, instrument, inversion_group, pattern, xray_phase


def anomalous_phase(seed=17):
    rng = np.random.default_rng(seed)
    base = xray_phase()
    st = replace(
        base.structure,
        space_group=ps.SpaceGroup([ps.SymmetryOperation.identity()]),
        sites=tuple(
            replace(s, fractional_xyz=tuple(rng.uniform(0.1, 0.9, 3))) for s in base.structure.sites
        ),
    )
    return replace(
        base, structure=st, scattering=ps.XrayFixedDispersion({"Si": -0.2 + 2.1j, "O": 0.04 + 0.3j})
    )


def factors(phase, *, powder=True, dense=False, hkl=None):
    return (ps.calculate_structure_factors if dense else ps.calculate_structure_factor_values)(
        phase.structure,
        phase.reflections.hkl if hkl is None else hkl,
        phase.reflections.multiplicity,
        phase.scattering,
        correction=phase.intensity_correction,
        scale=phase.scale,
        powder_average=powder,
    )


@pytest.mark.parametrize("seed", range(8))
def test_powder_values_match_independent_fourier_sum_and_opposite_hkl(seed):
    phase = anomalous_phase(seed)
    hkl = phase.reflections.hkl
    raw = factors(phase, powder=False)
    mate = factors(phase, powder=False, hkl=-hkl)
    result = factors(phase)
    np.testing.assert_array_equal(result.f, raw.f)
    np.testing.assert_allclose(
        result.integrated_intensity,
        0.5 * (raw.integrated_intensity + mate.integrated_intensity),
        rtol=3e-14,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        result.f_squared, 0.5 * (raw.f_squared + mate.f_squared), rtol=3e-14, atol=2e-12
    )
    assert np.max(np.abs(raw.f_squared - result.f_squared)) > 1e-3
    np.testing.assert_allclose(
        result.f_squared, factors(phase, hkl=-hkl).f_squared, rtol=3e-14, atol=2e-12
    )
    from phasesmith import scattering_reference

    amplitudes, _ = scattering_reference.xray_non_resonant(
        [s.type_symbol for s in phase.structure.sites], result.s_inverse_angstrom
    )
    amplitudes = amplitudes.astype(complex) + np.array([-0.2 + 2.1j, 0.04 + 0.3j])
    expected = reference_structure_factor_values(
        phase.structure,
        hkl,
        phase.reflections.multiplicity,
        amplitudes,
        np.ones(len(hkl)),
        scale=phase.scale,
        powder_average=True,
    )
    np.testing.assert_allclose(result.f_squared, expected.f_squared, rtol=3e-13, atol=2e-11)
    np.testing.assert_allclose(
        result.integrated_intensity, expected.integrated_intensity, rtol=3e-13, atol=2e-11
    )


def test_all_dense_parameter_derivatives_and_representative_complex_amplitude():
    phase = anomalous_phase()
    dense = factors(phase, dense=True)
    raw = factors(phase, powder=False, dense=True)
    np.testing.assert_array_equal(dense.d_f_d_parameters, raw.d_f_d_parameters)
    np.testing.assert_array_equal(dense.f, raw.f)
    for i in range(len(dense.parameter_names)):
        direction = np.zeros(len(dense.parameter_names))
        direction[i] = 1
        step = 1e-6
        plus = factors(_perturb_phase(phase, direction, step))
        minus = factors(_perturb_phase(phase, direction, -step))
        fd = (plus.integrated_intensity - minus.integrated_intensity) / (2 * step)
        np.testing.assert_allclose(
            dense.d_integrated_intensity_d_parameters[i], fd, rtol=2e-6, atol=3e-6
        )


@pytest.mark.parametrize("limit", ["real", "inversion", "zero_scale"])
def test_special_limits(limit):
    phase = anomalous_phase()
    if limit == "real":
        phase = replace(phase, scattering=ps.XrayNonResonant())
    elif limit == "inversion":
        phase = replace(phase, structure=replace(phase.structure, space_group=inversion_group()))
    else:
        phase = replace(phase, scale=0)
    result = factors(phase, dense=True)
    raw = factors(phase, powder=False, dense=True)
    if limit == "zero_scale":
        np.testing.assert_array_equal(result.integrated_intensity, 0)
        np.testing.assert_allclose(
            result.d_integrated_intensity_d_parameters[-1],
            factors(replace(phase, scale=1)).integrated_intensity,
            rtol=2e-14,
        )
    else:
        np.testing.assert_array_equal(result.integrated_intensity, raw.integrated_intensity)
        np.testing.assert_array_equal(
            result.d_integrated_intensity_d_parameters, raw.d_integrated_intensity_d_parameters
        )


@pytest.mark.parametrize("doublet", [False, True])
def test_fused_pattern_products_and_unmerged_pair_sum(doublet):
    phase = anomalous_phase()
    exp = ps.ConstantWavelengthExperiment.x_ray(instrument())
    if doublet:
        exp = ps.ConstantWavelengthExperiment.x_ray_components(
            instrument(), ps.WavelengthComponents.doublet(1.5406, 1.5444, 0.5)
        )
    prepared = ps.PreparedStructuralPattern(pattern(), exp, phase)
    actual = prepared.calculate()
    count = phase.reflections.reflection_count
    refl = phase.reflections
    split = ps.StructuralReflectionBatch(
        [f"{i}:{sign}" for sign in ("+", "-") for i in refl.reflection_ids],
        np.concatenate((refl.hkl, -refl.hkl)),
        np.concatenate((refl.multiplicity // 2, refl.multiplicity // 2)),
    )
    separate = ps.calculate_structural_pattern(pattern(), exp, replace(phase, reflections=split))
    np.testing.assert_allclose(actual.y, separate.y, rtol=5e-14, atol=5e-10)
    expected = factors(phase)
    np.testing.assert_allclose(
        actual.reflections.f_squared[:count], expected.f_squared, rtol=3e-14, atol=2e-11
    )
    direction = (
        np.random.default_rng(27).normal(
            size=len(ps.p1_parameter_names(phase.structure.to_isotropic_site_batch()))
        )
        * 0.01
    )
    forward = prepared.jvp(direction)
    plus = ps.calculate_structural_pattern(pattern(), exp, _perturb_phase(phase, direction, 1e-6))
    minus = ps.calculate_structural_pattern(pattern(), exp, _perturb_phase(phase, direction, -1e-6))
    np.testing.assert_allclose(forward.d_y, (plus.y - minus.y) / (2e-6), rtol=5e-6, atol=3e-6)
    weights = np.sin(np.linspace(0, 3, pattern().x.size))
    reverse = prepared.vjp(weights)
    np.testing.assert_allclose(
        forward.d_y @ weights, direction @ reverse.gradient, rtol=3e-12, atol=3e-9
    )


def test_invalid_powder_mode():
    phase = anomalous_phase()
    for function in (ps.calculate_structure_factors, ps.calculate_structure_factor_values):
        with pytest.raises(TypeError, match="powder_average"):
            function(
                phase.structure,
                phase.reflections.hkl,
                phase.reflections.multiplicity,
                phase.scattering,
                powder_average="yes",
            )


def test_custom_scattering_fallback_uses_powder_average_once():
    phase = anomalous_phase()

    class Provider:
        descriptor = phase.scattering.descriptor
        calls = 0

        def evaluate(self, context):
            self.calls += 1
            return phase.scattering.evaluate(context)

    provider = Provider()
    exp = ps.ConstantWavelengthExperiment.x_ray(instrument())
    custom = ps.PreparedStructuralPattern(pattern(), exp, replace(phase, scattering=provider))
    assert not custom.uses_native_fused_path
    actual = custom.calculate()
    expected = ps.calculate_structural_pattern(pattern(), exp, phase)
    assert provider.calls == 1
    np.testing.assert_allclose(actual.y, expected.y, rtol=3e-14, atol=3e-10)
    np.testing.assert_allclose(
        actual.reflections.f_squared, expected.reflections.f_squared, rtol=3e-14, atol=2e-11
    )


def test_noncentrosymmetric_anisotropic_displacement_and_cell_derivative():
    phase = anomalous_phase()
    site = replace(
        phase.structure.sites[0],
        u_iso_angstrom2=None,
        anisotropic_displacement=ps.AnisotropicDisplacement(
            (0.020, 0.013, 0.027, 0.002, 0.001, 0.003), "U_cif"
        ),
    )
    phase = replace(
        phase, structure=replace(phase.structure, sites=(site, phase.structure.sites[1]))
    )
    actual = factors(phase, dense=True)
    opposite = factors(phase, powder=False, hkl=-phase.reflections.hkl)
    raw = factors(phase, powder=False)
    np.testing.assert_allclose(
        actual.f_squared, 0.5 * (raw.f_squared + opposite.f_squared), rtol=3e-14, atol=3e-11
    )
    step = 1e-6

    def evaluate(delta):
        st = replace(
            phase.structure,
            cell=replace(phase.structure.cell, a_angstrom=phase.structure.cell.a_angstrom + delta),
        )
        return factors(replace(phase, structure=st)).integrated_intensity

    np.testing.assert_allclose(
        actual.d_integrated_intensity_d_parameters[0],
        (evaluate(step) - evaluate(-step)) / (2 * step),
        rtol=2e-6,
        atol=3e-6,
    )


@pytest.mark.parametrize("callbacks", [False, True])
def test_anomalous_refinement_resume_and_project_roundtrip(tmp_path, callbacks):
    from phasesmith.refinement import RefinementLimits
    from phasesmith.refinement import rietveld as rv
    from test_rietveld_optimization import doublet_request

    request = doublet_request()
    phases = tuple(
        replace(p, scattering=ps.XrayFixedDispersion({"Si": -0.2 + 2.1j, "O": 0.04 + 0.3j}))
        for p in request.phases
    )
    truth = rv.calculate(
        request.pattern, request.experiment, tuple(replace(p, scale=p.scale * 1.4) for p in phases)
    ).y
    request = replace(request, phases=phases, pattern=replace(request.pattern, observed_y=truth))
    options = rv.RietveldOptions(max_scaled_parameter_step=1)
    kwargs = {"logger": lambda event: None} if callbacks else {}
    complete = rv.refine(request, options, **kwargs)
    partial = rv.refine(
        request, replace(options, limits=RefinementLimits(max_iterations=1)), **kwargs
    )
    resumed = rv.refine(request, options, checkpoint=partial.checkpoint, **kwargs)
    np.testing.assert_array_equal(resumed.calculation.y, complete.calculation.y)
    assert resumed.history == complete.history
    assert complete.metrics.rwp < 1e-7
    project = ps.RietveldProject(request, options)
    saved = project.save(tmp_path / "anomalous")
    loaded = ps.RietveldProject.load(saved)
    np.testing.assert_array_equal(loaded.calculate().y, project.calculate().y)
