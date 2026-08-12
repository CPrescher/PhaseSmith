from pathlib import Path

import numpy as np
import phasesmith
import pytest
from phasesmith.refinement import (
    TofLeBailCancellation,
    TofLeBailCheckpoint,
    TofLeBailInput,
    TofLeBailOptions,
    TofLeBailPhase,
    refine_tof_lebail,
)


def _instrument() -> phasesmith.TofInstrument:
    return phasesmith.TofInstrument(
        0.0,
        5000.0,
        0.0,
        0.0,
        0.2,
        0.03,
        0.0,
        0.0,
        4.0,
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def test_native_tof_lebail_recovers_one_integrated_intensity() -> None:
    instrument = _instrument()
    tof_us = np.linspace(4500.0, 5500.0, 1001)
    observed = phasesmith.accumulate_tof(
        tof_us,
        np.array([1.0]),
        np.array([20.0]),
        instrument,
        execution=phasesmith.ExecutionPolicy(threads=1),
    ).y
    pattern = phasesmith.TofPowderPattern(
        tof_us,
        observed_y=observed,
        uncertainty=np.ones_like(observed),
    )
    phase = TofLeBailPhase(
        "phase",
        "Synthetic phase",
        ["1,0,0"],
        [[1, 0, 0]],
        [1.0],
        [1.0],
    )

    result = refine_tof_lebail(
        TofLeBailInput(pattern, instrument, (phase,)),
        TofLeBailOptions(cycles=2, execution=phasesmith.ExecutionPolicy(threads=1)),
    )

    np.testing.assert_allclose(result.integrated_intensity, [20.0], rtol=2e-15)
    np.testing.assert_allclose(result.y, observed, rtol=2e-15, atol=2e-18)
    assert result.reflection_keys == (("phase", "1,0,0"),)
    assert result.metrics.rwp < 1.0e-14
    assert len(result.history) == 2
    assert not result.integrated_intensity.flags.writeable


def test_public_calibration_reader_maps_powgen_type_three_records() -> None:
    data = phasesmith.read_gsas_tof_instrument(
        "INS  2 ICONS22581.63 0 4.41 0\n"
        "INS  2PRCF1     3 21 0.002\n"
        "INS  2PRCF11 0.257460 0.091563 0.017334 0\n"
        "INS  2PRCF12 10 203.581 0 10.651\n",
        bank=2,
    )

    assert data.bank == 2
    assert data.profile_function == 3
    assert data.instrument.difc_us_per_angstrom == pytest.approx(22581.63)
    assert data.instrument.zero_us == pytest.approx(4.41)
    assert data.instrument.sigma2_us2_per_angstrom4 == pytest.approx(203.581)


def test_from_files_builds_one_typed_powgen_style_request(tmp_path: Path) -> None:
    boundaries = np.linspace(4000.0, 6000.0, 21)
    pattern_path = tmp_path / "bank.gsa"
    pattern_path.write_text(
        "TOF example\nBANK 2 21 21 SLOG 1 2 3 4 5 FXYE\n"
        + "".join(f"{tof:.6f} 1.0 1.0\n" for tof in boundaries),
        encoding="utf-8",
    )
    instrument_path = tmp_path / "instrument.prm"
    instrument_path.write_text(
        "INS  2 ICONS5000 0 0 0\n"
        "INS  2PRCF1     3 21 0.002\n"
        "INS  2PRCF11 0.2 0.03 0 0\n"
        "INS  2PRCF12 4 0 0 0\n",
        encoding="utf-8",
    )
    cif_path = tmp_path / "phase.cif"
    cif_path.write_text(
        """data_phase
_chemical_name_common 'Cubic phase'
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P m -3 m'
""",
        encoding="utf-8",
    )

    request = TofLeBailInput.from_files(
        pattern_path,
        instrument_path,
        cif_path,
        bank=2,
        search_min_d_angstrom=0.75,
        search_max_d_angstrom=1.25,
        background_terms=0,
    )

    assert isinstance(request.pattern, phasesmith.TofPowderPattern)
    assert request.instrument.difc_us_per_angstrom == 5000.0
    assert request.phases[0].reflection_ids
    assert request.background is None


def test_tof_progress_checkpoint_resume_and_cancellation_are_public() -> None:
    instrument = _instrument()
    tof_us = np.linspace(4500.0, 5500.0, 1001)
    observed = phasesmith.accumulate_tof(
        tof_us,
        np.array([1.0]),
        np.array([20.0]),
        instrument,
        execution=phasesmith.ExecutionPolicy(threads=1),
    ).y
    request = TofLeBailInput(
        phasesmith.TofPowderPattern(tof_us, observed_y=observed),
        instrument,
        (
            TofLeBailPhase(
                "phase",
                "Synthetic phase",
                ["1,0,0"],
                [[1, 0, 0]],
                [1.0],
                [1.0],
            ),
        ),
    )
    execution = phasesmith.ExecutionPolicy(threads=1)
    events: list[dict[str, object]] = []
    partial = refine_tof_lebail(
        request,
        TofLeBailOptions(cycles=2, execution=execution),
        progress=events.append,
    )
    assert isinstance(partial.checkpoint, TofLeBailCheckpoint)
    assert partial.checkpoint.completed_iterations == 2
    assert partial.termination_reason == "max_iterations"
    assert [event["kind"] for event in events] == [
        "start",
        "iteration",
        "iteration",
        "termination",
    ]

    resumed = refine_tof_lebail(
        request,
        TofLeBailOptions(cycles=4, execution=execution),
        checkpoint=partial.checkpoint,
    )
    uninterrupted = refine_tof_lebail(
        request,
        TofLeBailOptions(cycles=4, execution=execution),
    )
    np.testing.assert_array_equal(resumed.y, uninterrupted.y)
    np.testing.assert_array_equal(
        resumed.integrated_intensity, uninterrupted.integrated_intensity
    )
    assert [item.iteration for item in resumed.history] == [1, 2, 3, 4]
    np.testing.assert_array_equal(
        [item.metrics.rwp for item in resumed.history],
        [item.metrics.rwp for item in uninterrupted.history],
    )

    cancellation = TofLeBailCancellation()
    assert cancellation.request("test cancellation")
    assert not cancellation.request("ignored second reason")
    cancelled = refine_tof_lebail(
        request,
        TofLeBailOptions(cycles=4, execution=execution),
        cancellation=cancellation,
    )
    assert cancelled.termination_reason == "cancelled"
    assert cancelled.checkpoint.completed_iterations == 0
    assert cancelled.history == ()
    assert cancellation.reason == "test cancellation"
