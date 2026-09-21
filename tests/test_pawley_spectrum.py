"""Fixed detected-area spectra: one area per family, native derivative and restart contracts."""

import json
from dataclasses import replace

import numpy as np
import pytest
from phasesmith import ConstantWavelengthInstrument, PowderPattern, UnitCell
from phasesmith.instrument import FcjGeometry
from phasesmith.pawley_reference import evaluate
from phasesmith.project_bundle import ProjectBundle
from phasesmith.radiation import RadiationProbe, WavelengthComponents
from phasesmith.refinement.core import ConstraintTransform
from phasesmith.refinement.pawley import (
    PawleyInput,
    PawleyOptions,
    PawleyPhase,
    PawleyProject,
    build_parameter_set,
    calculate,
    refine,
)
from phasesmith.symmetry import SpaceGroup, SymmetryOperation


def request(axial=None):
    spectrum = WavelengthComponents([1.54, 1.544], [1.0, 0.5])
    phase = PawleyPhase("phase", ("a", "b", "c"), [40.0, 40.06, 40.7], [3.0, 0.0, 7.0])
    return PawleyInput(
        PowderPattern(np.linspace(39, 42, 901), observed_y=np.zeros(901)),
        ConstantWavelengthInstrument(1.54, 0.0001, 0.0, 0.001, 0.002, 0.001),
        (phase,),
        axial_geometry=axial,
        fixed_spectrum=spectrum,
    )


@pytest.mark.parametrize("axial", [None, FcjGeometry(0.003, 0.002)])
def test_spectrum_reference_derivatives_products_and_one_component(axial):
    r = request(axial)
    r = replace(
        r,
        parameters=build_parameter_set(
            r, profile_parameters=("u_deg2", "v_deg2", "w_deg2", "x_deg", "y_deg")
        ),
    )
    options = PawleyOptions(support_fwhm=1000)
    native = calculate(r, options)
    y, j, _ = evaluate(r, options=options)
    np.testing.assert_allclose(native.calculated_y, y, rtol=2e-9, atol=2e-8)
    np.testing.assert_allclose(native.jacobian, j, rtol=2e-8, atol=2e-7)
    t = ConstraintTransform(r.parameters)
    z = t.pack()
    for k in range(len(z)):
        dz = np.zeros_like(z)
        dz[k] = 1e-5
        # The zero area's lower face has a one-sided scientific domain; the independent
        # equation is differentiable on its signed extension for this derivative check.
        if k == 1:
            continue
        np.testing.assert_allclose(
            (evaluate(r, z + dz, options)[0] - evaluate(r, z - dz, options)[0]) / 2e-5,
            j[:, k],
            rtol=3e-5,
            atol=3e-5,
        )
    product = calculate(r, replace(options, solver="matrix_free"))
    v = np.random.default_rng(41).normal(size=len(z))
    np.testing.assert_allclose(product.jacobian_operator.jvp(v), j @ v, rtol=2e-8, atol=2e-7)
    mono = replace(r, fixed_spectrum=None)
    one = replace(r, fixed_spectrum=WavelengthComponents([1.54], [2.0]))
    np.testing.assert_array_equal(calculate(mono).calculated_y, calculate(one).calculated_y)
    scaled = replace(r, fixed_spectrum=WavelengthComponents([1.54, 1.544], [2.0, 1.0]))
    np.testing.assert_array_equal(calculate(r).calculated_y, calculate(scaled).calculated_y)


def test_spectrum_fit_restart_bundle_and_digest(tmp_path):
    truth = request()
    y = calculate(truth).calculated_y
    r = replace(
        truth,
        pattern=PowderPattern(truth.pattern.x, observed_y=y),
        phases=(replace(truth.phases[0], intensities=np.ones(3)),),
        parameters=None,
    )
    fits = [refine(r, PawleyOptions(solver=s)) for s in ("dense", "matrix_free")]
    for f in fits:
        assert f.termination_reason == "converged"
        np.testing.assert_allclose(f.intensities, [3, 0, 7], atol=2e-8)
    p = PawleyProject(r, PawleyOptions(solver="matrix_free"))
    p.refine(max_iterations=1)
    path = tmp_path / "pawley.json"
    p.save(path)
    loaded = PawleyProject.load(path)
    np.testing.assert_array_equal(p.refine().history, loaded.refine().history)
    bundle = ProjectBundle.from_pawley(p, probe=RadiationProbe.X_RAY)
    bundle.save(tmp_path / "bundle")
    restored = ProjectBundle.load(tmp_path / "bundle").pawley("histogram")
    np.testing.assert_array_equal(restored.calculate().calculated_y, p.calculate().calculated_y)
    w = json.loads(path.read_text())
    w["input"]["fixed_spectrum"][1][1] = 0.4
    path.write_text(json.dumps(w))
    with pytest.raises(ValueError, match="digest"):
        PawleyProject.load(path)
    with pytest.raises(ValueError, match="reference wavelength"):
        replace(r, fixed_spectrum=WavelengthComponents([1.5], [1]))
    with pytest.raises(ValueError, match="Bragg"):
        calculate(replace(r, fixed_spectrum=WavelengthComponents([1.54, 8], [1, 1])))


def test_component_union_and_lattice_chain():
    cell = UnitCell(4, 4, 4, 90, 90, 90)
    group = SpaceGroup([SymmetryOperation.identity()])
    spectrum = WavelengthComponents([1.54, 1.6], [1, 0.5])
    kwargs = dict(wavelength_angstrom=1.54, two_theta_range=(38, 46), initial_intensity=2)
    phase = PawleyPhase.from_cell("p", cell, group, fixed_spectrum=spectrum, **kwargs)
    union = set()
    for wave in spectrum.wavelengths_angstrom:
        single = PawleyPhase.from_cell(
            "p", cell, group, **dict(kwargs, wavelength_angstrom=float(wave))
        )
        union.update(single.reflection_ids)
    assert set(phase.reflection_ids) == union
    r = PawleyInput(
        PowderPattern(np.linspace(37, 47, 301), observed_y=np.zeros(301)),
        ConstantWavelengthInstrument(1.54, 0, 0, 0.001, 0.002, 0),
        (phase,),
        fixed_spectrum=spectrum,
    )
    r = replace(r, parameters=build_parameter_set(r, lattice=True))
    options = PawleyOptions(support_fwhm=1000)
    y, j, _ = evaluate(r, options=options)
    c = calculate(r, options)
    np.testing.assert_allclose(c.calculated_y, y, rtol=2e-9, atol=2e-8)
    np.testing.assert_allclose(c.jacobian, j, rtol=2e-8, atol=1e-6)
    t = ConstraintTransform(r.parameters)
    z = t.pack()
    for k, key in enumerate(t.free_keys):
        if key.module != "pawley_lattice":
            continue
        dz = np.zeros_like(z)
        dz[k] = 1e-6
        np.testing.assert_allclose(
            (evaluate(r, z + dz, options)[0] - evaluate(r, z - dz, options)[0]) / 2e-6,
            j[:, k],
            rtol=3e-4,
            atol=2e-3,
        )
