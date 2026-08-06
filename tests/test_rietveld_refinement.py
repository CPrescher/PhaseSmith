from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import rietveld
from rietveld.refinement import rietveld as structural_refinement

P1_CIF = """
data_p1
_chemical_name_common 'P1 structural test'
_cell_length_a 4.1
_cell_length_b 5.2
_cell_length_c 6.3
_cell_angle_alpha 77
_cell_angle_beta 83
_cell_angle_gamma 72
_space_group_name_H-M_alt 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
Si1 Si 0.11 0.22 0.33 0.9 0.012
O1 O 0.41 0.52 0.63 1.0 0.018
"""


def experiment() -> rietveld.ConstantWavelengthExperiment:
    instrument = rietveld.ConstantWavelengthInstrument(
        1.5406, 2.0e-4, -1.0e-4, 2.0e-4, 1.5e-3, 3.0e-3
    )
    return rietveld.ConstantWavelengthExperiment.x_ray(instrument)


def selection(**changes: bool) -> structural_refinement.RietveldParameterSelection:
    return replace(
        structural_refinement.RietveldParameterSelection(
            phase_scale=False,
            lattice=False,
            coordinates=False,
            occupancy=False,
            u_iso=False,
        ),
        **changes,
    )


def request_from_cif(
    selected: structural_refinement.RietveldParameterSelection,
) -> structural_refinement.RietveldInput:
    x = np.linspace(15.0, 100.0, 8_501)
    initial = structural_refinement.RietveldInput.from_cif(
        rietveld.PowderPattern(x, observed_y=np.zeros_like(x)),
        experiment(),
        P1_CIF,
        phase_id="alpha",
        selection=selected,
    )
    calculated = structural_refinement.calculate(
        initial.pattern, initial.experiment, initial.phases
    )
    return replace(
        initial,
        pattern=rietveld.PowderPattern(x, observed_y=calculated.y),
    )


def test_cif_request_builds_guarded_structural_phase_and_typed_parameters() -> None:
    request = request_from_cif(
        selection(phase_scale=True, lattice=True, coordinates=True, occupancy=True, u_iso=True)
    )
    phase = request.phases[0]
    domain = request.lattice_domains[0]
    assert domain is not None
    assert phase.structure.source is not None
    assert phase.structure.source.backend == "gemmi"
    assert phase.reflections.reflection_count > 0
    assert len(request.parameters.specs) == 1 + 6 + 2 * 3 + 2 + 2
    labels = tuple(spec.key.label for spec in request.parameters.specs)
    assert labels[0] == "phase[alpha].scale"
    assert "site[alpha/Si1].occupancy" in labels
    assert "site[alpha/O1].u_iso_angstrom2" in labels


def test_combined_structural_calculation_sums_profiles_and_background_once() -> None:
    request = request_from_cif(selection())
    first = request.phases[0]
    second = replace(first, phase_id="beta", scale=0.4)
    background = np.full(request.pattern.x.size, 0.3)
    pattern = rietveld.PowderPattern(
        request.pattern.x, observed_y=request.pattern.observed_y, background=background
    )
    combined = structural_refinement.calculate(pattern, request.experiment, (first, second))
    individual = tuple(
        rietveld.calculate_structural_pattern(pattern, request.experiment, phase)
        for phase in (first, second)
    )
    np.testing.assert_allclose(
        combined.profile_y,
        individual[0].profile_y + individual[1].profile_y,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(combined.y, combined.profile_y + background, atol=0.0)


def test_special_position_coordinate_selection_uses_only_allowed_tangent_space() -> None:
    inversion = rietveld.SymmetryOperation(-np.eye(3, dtype=np.int64), (0, 0, 0))
    group = rietveld.SpaceGroup([rietveld.SymmetryOperation.identity(), inversion])
    structure = rietveld.CrystalStructure(
        "special",
        "Special position",
        rietveld.UnitCell(4.0, 5.0, 6.0, 90.0, 90.0, 90.0),
        group,
        (
            rietveld.AtomSite("origin", "X1", "Si", "Si", (0.0, 0.0, 0.0)),
            rietveld.AtomSite(
                "general",
                "X2",
                "O",
                "O",
                (0.13, 0.24, 0.35),
            ),
        ),
    )
    generated = rietveld.PreparedReflectionGenerator(group).generate(
        structure.cell,
        rietveld.CwTwoThetaRange(20.0, 80.0, experiment().radiation.wavelength_angstrom),
    )
    phase = rietveld.RietveldPhase(
        "alpha",
        "Special",
        structure,
        rietveld.StructuralReflectionBatch.from_generated(generated),
        rietveld.XrayNonResonant(),
        rietveld.NeutralIntegratedIntensityCorrection(),
    )
    parameters = structural_refinement.build_parameter_set(
        (phase,),
        (None,),
        selection(coordinates=True),
    )
    coordinate_keys = tuple(spec.key for spec in parameters.specs if spec.key.module == "site")
    assert all(key.owner_id != "alpha/origin" for key in coordinate_keys)
    assert tuple(key.name for key in coordinate_keys) == ("x", "y", "z")


def test_site_coordinate_model_handles_non_origin_fixed_point_exactly() -> None:
    operation = rietveld.SymmetryOperation(
        -np.eye(3, dtype=np.int64),
        (Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)),
    )
    group = rietveld.SpaceGroup([rietveld.SymmetryOperation.identity(), operation])
    structure = rietveld.CrystalStructure(
        "fixed",
        "Fixed point",
        rietveld.UnitCell(4.0, 5.0, 6.0, 90.0, 90.0, 90.0),
        group,
        (rietveld.AtomSite("center", "X1", "Si", "Si", (0.5, 0.5, 0.5)),),
    )
    model = structural_refinement._site_coordinate_model(
        "alpha", structure, structure.sites[0], 1.0e-10
    )
    assert model.special_position
    assert model.parameter_names == ()
    assert model.basis.shape == (3, 0)
