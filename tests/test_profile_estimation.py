from __future__ import annotations

import numpy as np
import phasesmith
from phasesmith.refinement import lebail


def _instrument(w_deg2: float) -> phasesmith.ConstantWavelengthInstrument:
    return phasesmith.ConstantWavelengthInstrument(0.4133, 0.0, 0.0, w_deg2, 0.0, 0.0)


def _phase(instrument: phasesmith.ConstantWavelengthInstrument) -> lebail.LeBailPhase:
    positions = np.array([8.0, 12.0, 17.0, 23.0, 30.0, 38.0])
    d_spacing = instrument.wavelength_angstrom / (2.0 * np.sin(np.deg2rad(positions / 2.0)))
    reflections = phasesmith.ReflectionBatch(
        [f"r{index}" for index in range(positions.size)],
        np.column_stack(
            (np.arange(1, positions.size + 1), np.ones(positions.size), np.zeros(positions.size))
        ).astype(np.int64),
        d_spacing,
        positions,
        [12.0, 7.0, 5.0, 9.0, 4.0, 6.0],
    )
    structure = phasesmith.CrystalStructure(
        "dominant-structure",
        "Dominant structure",
        phasesmith.UnitCell(4.0, 4.0, 4.0, 90.0, 90.0, 90.0),
        phasesmith.SpaceGroup.p1(),
    )
    return lebail.LeBailPhase(
        "dominant",
        "Dominant phase",
        reflections,
        structure=structure,
    )


def _input() -> lebail.LeBailInput:
    truth = _instrument(2.5e-4)
    phase = _phase(truth)
    x = np.linspace(5.0, 42.0, 7401)
    blank = phasesmith.PowderPattern(x, background=np.full(x.size, 0.1))
    calculated = phasesmith.calculate_pattern(blank, truth, (phase,))
    pattern = phasesmith.PowderPattern(
        x,
        observed_y=calculated.y,
        background=blank.background,
    )
    start = phasesmith.starting_profile_from_fwhm(truth.wavelength_angstrom, 0.06)
    return lebail.LeBailInput(pattern, start, (phase,))


def test_python_profile_estimation_delegates_to_native_engine() -> None:
    input_data = _input()
    result = phasesmith.estimate_effective_profile(
        input_data,
        phasesmith.ProfileEstimationOptions(
            mode=phasesmith.ProfileEstimationMode.W_ONLY,
            lebail=lebail.LeBailOptions(
                max_iterations=40,
                use_uncertainty=False,
                max_scaled_parameter_step=1.0,
            ),
        ),
    )

    assert (
        result.instrument.wavelength_angstrom.hex()
        == input_data.instrument.wavelength_angstrom.hex()
    )
    assert result.active_parameters == ("w_deg2",)
    assert abs(result.instrument.w_deg2 - 2.5e-4) < 3.0e-8
    assert result.rwp < 3.0e-5
    assert result.integrated_intensity.shape == (6,)
    assert result.calculated_y.shape == input_data.pattern.x.shape
    assert not result.integrated_intensity.flags.writeable
    assert "sample broadening" in result.warnings[0]


def test_starting_profile_and_option_validation() -> None:
    with np.testing.assert_raises(ValueError):
        phasesmith.starting_profile_from_fwhm(0.4133, 0.0)
    with np.testing.assert_raises(ValueError):
        phasesmith.ProfileEstimationOptions(maximum_absolute_correlation=1.0)
