"""Public Python facade coverage for structural multi-bank neutron TOF."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import phasesmith
import pytest
from phasesmith.refinement import (
    Bounds,
    LatticeParameterBounds,
    LatticeParameterization,
    RefinementLimits,
    StructuralTofBank,
    StructuralTofCancellation,
    StructuralTofMultiBankInput,
    StructuralTofRefinementOptions,
    StructuralTofSelection,
    TofInstrumentParameterBound,
    refine_structural_tof_multibank,
)


def _structure() -> phasesmith.CrystalStructure:
    return phasesmith.CrystalStructure(
        "tof-phase",
        "TOF phase",
        phasesmith.UnitCell(4.7, 5.1, 6.2, 82.0, 87.0, 74.0),
        phasesmith.SpaceGroup([phasesmith.SymmetryOperation.identity()]),
        (
            phasesmith.AtomSite("si", "Si1", "Si", "Si", (0.17, 0.23, 0.31), 0.82, 0.012),
            phasesmith.AtomSite("o", "O1", "O", "O", (0.37, 0.11, 0.19), 0.55, 0.018),
        ),
    )


def _phase() -> phasesmith.RietveldPhase:
    return phasesmith.RietveldPhase(
        "phase",
        "TOF phase",
        _structure(),
        phasesmith.StructuralReflectionBatch(
            ("101", "211", "123"),
            [[1, 0, 1], [2, 1, 1], [1, 2, 3]],
            [2, 4, 2],
        ),
        phasesmith.NeutronNuclear(),
        phasesmith.NeutralIntegratedIntensityCorrection(),
        1.0,
    )


def _instrument(zero: float) -> phasesmith.TofInstrument:
    return phasesmith.TofInstrument(
        zero,
        5_000.0,
        0.2,
        0.0,
        0.2,
        0.03,
        0.001,
        0.0,
        25.0,
        4.0,
        0.1,
        0.0,
        1.0,
        0.1,
        0.5,
    )


def _request() -> tuple[StructuralTofMultiBankInput, tuple[float, ...]]:
    phase = _phase()
    grid = np.arange(1_000.0, 25_010.0, 10.0)
    specifications = (
        ("bank-1", 88.05, _instrument(2.3), _instrument(1.2), 1.55, 1.3),
        ("bank-2", 120.0, _instrument(-1.8), _instrument(-0.7), 0.72, 0.9),
    )
    banks = []
    for index, specification in enumerate(specifications):
        bank_id, angle, truth_instrument, initial_instrument, truth_scale, initial_scale = (
            specification
        )
        correction = phasesmith.TimeOfFlightNeutronLorentz(angle)
        structure_factors = phasesmith.calculate_structure_factors(
            phase.structure,
            phase.reflections.hkl,
            phase.reflections.multiplicity,
            phase.scattering,
            correction=correction,
            scale=truth_scale,
        )
        observed = phasesmith.accumulate_tof(
            grid,
            phase.structure.cell.d_spacings(phase.reflections.hkl).d_spacing_angstrom,
            structure_factors.integrated_intensity,
            truth_instrument,
        ).y
        mask = np.arange(grid.size) % (19 + index) != 0
        pattern = phasesmith.TofPowderPattern(
            grid,
            observed_y=observed,
            uncertainty=np.ones(grid.size),
            mask=mask,
        )
        banks.append(
            StructuralTofBank(
                bank_id,
                pattern,
                initial_instrument,
                phasesmith.TofBankGeometry(angle),
                correction,
                initial_scale,
                Bounds(0.2, 3.0),
                True,
                instrument_bounds=(TofInstrumentParameterBound("zero", -10.0, 10.0),),
            )
        )
    return (
        StructuralTofMultiBankInput(
            phase,
            tuple(banks),
            StructuralTofSelection(),
        ),
        (1.55, 2.3, 0.72, -1.8),
    )


def _options(iterations: int) -> StructuralTofRefinementOptions:
    return StructuralTofRefinementOptions(
        RefinementLimits(
            max_iterations=iterations,
            max_evaluations=200,
            max_consecutive_rejections=8,
        ),
        objective_tolerance=1.0e-12,
        parameter_tolerance=1.0e-9,
        initial_damping=1.0e-3,
        max_scaled_parameter_step=1.0,
    )


def test_python_structural_tof_recovers_local_scale_and_zero() -> None:
    request, truth = _request()
    result = refine_structural_tof_multibank(request, _options(20))

    assert result.termination_reason.value == "converged"
    assert result.objective < 1.0e-12
    np.testing.assert_allclose(
        [spec.value for spec in result.parameters.specs], truth, rtol=0.0, atol=2.0e-6
    )
    assert result.input.phase == request.phase
    assert result.input.banks[0].scale != request.banks[0].scale
    assert all(not bank.y.flags.writeable for bank in result.banks)
    assert all(not bank.integrated_intensity.flags.writeable for bank in result.banks)


def test_python_structural_tof_supports_one_bank_and_rejects_no_banks() -> None:
    request, truth = _request()
    single_bank = replace(request, banks=request.banks[:1])

    result = refine_structural_tof_multibank(single_bank, _options(20))

    assert len(result.banks) == 1
    assert result.termination_reason.value == "converged"
    assert result.objective < 1.0e-12
    np.testing.assert_allclose(
        [spec.value for spec in result.parameters.specs], truth[:2], rtol=0.0, atol=2.0e-6
    )
    with pytest.raises(ValueError, match="at least one StructuralTofBank"):
        replace(request, banks=())


def test_python_structural_tof_checkpoint_resumes_exactly() -> None:
    request, _ = _request()
    uninterrupted = refine_structural_tof_multibank(request, _options(20))
    cancellation = StructuralTofCancellation()

    def stop_after_two(event: dict[str, object]) -> None:
        if event["kind"] == "step_accepted" and event["accepted_iterations"] == 2:
            cancellation.request("structural TOF facade stop")

    stopped = refine_structural_tof_multibank(
        request,
        _options(20),
        cancellation=cancellation,
        progress=stop_after_two,
    )
    assert stopped.termination_reason.value == "cancelled"
    assert stopped.checkpoint.completed_iterations == 2
    resumed = refine_structural_tof_multibank(
        request,
        _options(20),
        checkpoint=stopped.checkpoint,
    )
    assert resumed.history == uninterrupted.history
    assert resumed.parameters == uninterrupted.parameters
    assert resumed.input == uninterrupted.input


def test_python_structural_tof_recovers_one_shared_cubic_cell() -> None:
    group = phasesmith.space_group_by_number(221).space_group
    initial_cell = phasesmith.UnitCell(3.98, 3.98, 3.98, 90.0, 90.0, 90.0)
    truth_cell = phasesmith.UnitCell(4.0, 4.0, 4.0, 90.0, 90.0, 90.0)
    structure = phasesmith.CrystalStructure(
        "cubic-tof",
        "Cubic TOF",
        initial_cell,
        group,
        (phasesmith.AtomSite("ni", "Ni1", "Ni", "Ni", (0.0, 0.0, 0.0)),),
    )
    reflections = phasesmith.StructuralReflectionBatch(
        ("100", "110", "111", "200"),
        [[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0]],
        [6, 12, 8, 6],
    )
    phase = phasesmith.RietveldPhase(
        "cubic",
        "Cubic",
        structure,
        reflections,
        phasesmith.NeutronNuclear(),
        phasesmith.NeutralIntegratedIntensityCorrection(),
    )
    truth_structure = replace(structure, cell=truth_cell)
    grid = np.arange(6_000.0, 22_005.0, 5.0)
    banks = []
    for bank_id, angle, instrument in (
        ("bank-a", 88.05, _instrument(0.4)),
        ("bank-b", 120.0, _instrument(-0.6)),
    ):
        correction = phasesmith.TimeOfFlightNeutronLorentz(angle)
        factors = phasesmith.calculate_structure_factors(
            truth_structure,
            reflections.hkl,
            reflections.multiplicity,
            phase.scattering,
            correction=correction,
        )
        observed = phasesmith.accumulate_tof(
            grid,
            truth_cell.d_spacings(reflections.hkl).d_spacing_angstrom,
            factors.integrated_intensity,
            instrument,
        ).y
        banks.append(
            StructuralTofBank(
                bank_id,
                phasesmith.TofPowderPattern(grid, observed_y=observed),
                instrument,
                phasesmith.TofBankGeometry(angle),
                correction,
                refine_scale=False,
            )
        )
    parameterization = LatticeParameterization(group, initial_cell)
    bounds = LatticeParameterBounds.around(
        parameterization, relative_length=0.02, angle_delta_deg=1.0
    )
    request = StructuralTofMultiBankInput(
        phase,
        tuple(banks),
        StructuralTofSelection(lattice=True),
        bounds,
    )
    result = refine_structural_tof_multibank(request, _options(20))

    assert result.termination_reason.value == "converged"
    assert result.parameters.specs[0].key.name == "a_angstrom"
    assert abs(result.input.phase.structure.cell.a_angstrom - 4.0) < 2.0e-7
    assert result.objective < 1.0e-12
