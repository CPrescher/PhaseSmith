"""Public Python facade coverage for joint multi-bank TOF geometry."""

from __future__ import annotations

import numpy as np
import phasesmith
from phasesmith.crystallography import UnitCell
from phasesmith.instrument import TofInstrument
from phasesmith.pattern import TofPowderPattern
from phasesmith.refinement import (
    LatticeParameterBounds,
    LatticeParameterization,
    TofBankInstrumentModel,
    TofInstrumentParameterBound,
    TofLeBailBank,
    TofLeBailCancellation,
    TofLeBailInput,
    TofLeBailOptions,
    TofLeBailPhase,
    TofMultiBankGeometryInput,
    TofMultiBankGeometryIteration,
    TofMultiBankGeometryOptions,
    TofSharedLatticePhase,
    refine_tof_multibank_geometry,
)
from phasesmith.tof import accumulate_tof


def _cell(a: float) -> UnitCell:
    return UnitCell(a, a, a, 90.0, 90.0, 90.0)


def _instrument(difc: float, zero: float) -> TofInstrument:
    return TofInstrument(
        zero,
        difc,
        -0.2,
        0.3,
        0.18,
        0.04,
        0.0005,
        0.001,
        1.0,
        10.0,
        0.05,
        0.2,
        0.3,
        0.05,
        0.4,
    )


def _phase(cell: UnitCell, intensities: list[float], scale: float) -> TofLeBailPhase:
    hkl = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0]])
    return TofLeBailPhase(
        "alpha",
        "shared alpha",
        ("100", "110", "111", "200"),
        hkl,
        cell.d_spacings(hkl).d_spacing_angstrom,
        intensities,
        scale=scale,
    )


def _request() -> tuple[TofMultiBankGeometryInput, list[TofInstrument]]:
    initial_cell = _cell(3.992)
    truth_cell = _cell(4.0)
    group = phasesmith.space_group_by_number(221).space_group
    parameterization = LatticeParameterization(group, initial_cell)
    lattice = TofSharedLatticePhase(
        "alpha",
        parameterization,
        LatticeParameterBounds.around(parameterization, relative_length=0.02, angle_delta_deg=1.0),
        initial_cell,
    )
    truth_instruments = [_instrument(5_000.0, -0.7), _instrument(4_400.0, 1.3)]
    initial_instruments = [_instrument(5_000.0, 1.1), _instrument(4_400.0, -0.2)]
    specs = (
        ("bank-1", np.arange(7_000.0, 18_505.0, 5.0), [120.0, 75.0, 210.0, 90.0], 1.0, 29),
        ("bank-2", np.arange(6_000.0, 16_505.0, 5.0), [55.0, 180.0, 95.0, 140.0], 1.4, 31),
    )
    banks: list[TofLeBailBank] = []
    models: list[TofBankInstrumentModel] = []
    for index, (bank_id, grid, intensities, scale, stride) in enumerate(specs):
        truth_phase = _phase(truth_cell, intensities, scale)
        observed = (
            scale
            * accumulate_tof(
                grid,
                truth_phase.d_spacing_angstrom,
                truth_phase.integrated_intensity,
                truth_instruments[index],
            ).y
        )
        mask = np.arange(grid.size) % stride != 0
        input_ = TofLeBailInput(
            TofPowderPattern(
                grid,
                observed_y=observed,
                uncertainty=np.ones(grid.size),
                mask=mask,
            ),
            initial_instruments[index],
            (_phase(initial_cell, intensities, scale),),
        )
        banks.append(TofLeBailBank(bank_id, input_))
        models.append(
            TofBankInstrumentModel(
                bank_id,
                (TofInstrumentParameterBound("zero", -5.0, 5.0),),
            )
        )
    return TofMultiBankGeometryInput(tuple(banks), (lattice,), tuple(models)), truth_instruments


def _options(cycles: int) -> TofMultiBankGeometryOptions:
    return TofMultiBankGeometryOptions(
        TofLeBailOptions(cycles=cycles),
        unresolved_correlation=0.5,
    )


def _assert_history_equal(
    actual: tuple[TofMultiBankGeometryIteration, ...],
    expected: tuple[TofMultiBankGeometryIteration, ...],
) -> None:
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        assert left.iteration == right.iteration
        assert left.metrics == right.metrics
        assert left.maximum_relative_intensity_change == right.maximum_relative_intensity_change
        assert left.maximum_absolute_background_change == right.maximum_absolute_background_change
        assert left.scaled_geometry_step_norm == right.scaled_geometry_step_norm
        assert left.lattice_parameter_changes == right.lattice_parameter_changes
        assert left.instrument_parameter_changes == right.instrument_parameter_changes
        for left_metrics, right_metrics in zip(left.bank_metrics, right.bank_metrics, strict=True):
            np.testing.assert_array_equal(left_metrics.included, right_metrics.included)
            np.testing.assert_array_equal(left_metrics.residual, right_metrics.residual)
            np.testing.assert_array_equal(
                left_metrics.weighted_residual, right_metrics.weighted_residual
            )
            assert left_metrics.rp == right_metrics.rp
            assert left_metrics.rwp == right_metrics.rwp
            assert left_metrics.chi_square == right_metrics.chi_square
            assert left_metrics.reduced_chi_square == right_metrics.reduced_chi_square


def test_python_joint_tof_geometry_recovers_cell_and_bank_zero_terms() -> None:
    request, truth_instruments = _request()
    result = refine_tof_multibank_geometry(request, _options(30))

    assert abs(result.lattice_phases[0].cell.a_angstrom - 4.0) < 2.0e-7
    for actual, truth in zip(result.instruments, truth_instruments, strict=True):
        assert abs(actual.instrument.zero_us - truth.zero_us) < 2.0e-7
    assert result.metrics.rwp < 3.0e-7
    assert result.diagnostics.parameter_count == 3
    assert result.diagnostics.jacobian_rank == 3
    assert any(
        pair.left.family == "lattice" and pair.right.family == "instrument"
        for pair in result.diagnostics.unresolved_correlations
    )
    assert all(not bank.y.flags.writeable for bank in result.banks)


def test_python_joint_tof_geometry_checkpoint_resumes_exactly() -> None:
    request, _ = _request()
    options = _options(12)
    uninterrupted = refine_tof_multibank_geometry(request, options)
    cancellation = TofLeBailCancellation()

    def stop_after_four(event: dict[str, object]) -> None:
        if event["kind"] == "iteration" and event["accepted_iterations"] == 4:
            cancellation.request("facade checkpoint stop")

    partial = refine_tof_multibank_geometry(
        request,
        options,
        cancellation=cancellation,
        progress=stop_after_four,
    )
    assert partial.termination_reason == "cancelled"
    assert partial.checkpoint.completed_iterations == 4

    resumed = refine_tof_multibank_geometry(request, options, checkpoint=partial.checkpoint)
    _assert_history_equal(resumed.history, uninterrupted.history)
    assert resumed.lattice_phases == uninterrupted.lattice_phases
    assert resumed.instruments == uninterrupted.instruments
    assert resumed.metrics == uninterrupted.metrics
    assert resumed.diagnostics == uninterrupted.diagnostics
    for actual, expected in zip(resumed.banks, uninterrupted.banks, strict=True):
        np.testing.assert_array_equal(actual.y, expected.y)
        np.testing.assert_array_equal(actual.integrated_intensity, expected.integrated_intensity)
