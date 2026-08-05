from __future__ import annotations

import sys

import numpy as np
import pytest
import rietveld
from rietveld.integrations import dioptas
from rietveld.refinement import TerminationReason, lebail


def models() -> tuple[
    np.ndarray,
    rietveld.ConstantWavelengthInstrument,
    rietveld.Phase,
    np.ndarray,
]:
    x = np.linspace(38.0, 42.0, 2001)
    instrument = rietveld.ConstantWavelengthInstrument(
        1.5406,
        2.0e-4,
        -1.0e-4,
        2.0e-4,
        1.5e-3,
        3.0e-3,
    )
    positions = np.array([39.8, 40.2])
    d_spacing = instrument.wavelength_angstrom / (2.0 * np.sin(np.deg2rad(positions / 2.0)))
    phase = rietveld.Phase(
        "alpha",
        "Alpha",
        rietveld.ReflectionBatch(
            ["alpha-0", "alpha-1"],
            [[1, 0, 0], [1, 1, 0]],
            d_spacing,
            positions,
            [7.0, 4.0],
        ),
    )
    background = np.full(x.size, 0.2)
    truth = rietveld.calculate_pattern(
        rietveld.PowderPattern(x, background=background),
        instrument,
        (phase,),
    ).y
    return x, instrument, phase, truth


def test_module_imports_without_dioptas_and_normalizes_excluded_mask() -> None:
    assert "dioptas" not in sys.modules
    x, _, _, truth = models()
    excluded = (x > 40.5) & (x < 41.0)
    data = dioptas.DioptasPatternData(
        x,
        truth,
        background_y=np.full(x.size, 0.2),
        uncertainty_y=np.ones(x.size),
        excluded_mask=excluded,
        metadata={"source": "fake-dioptas"},
    )
    np.testing.assert_array_equal(data.included_mask, ~excluded)
    assert data.x_deg.dtype == np.float64
    assert data.x_unit == "degree_2theta"
    assert data.intensity_unit == "arbitrary_intensity"
    assert data.metadata["source"] == "fake-dioptas"
    with pytest.raises(ValueError, match="not both"):
        dioptas.DioptasPatternData(
            x,
            truth,
            included_mask=~excluded,
            excluded_mask=excluded,
        )


def test_calculation_returns_display_ready_components_labels_and_progress() -> None:
    x, instrument, phase, truth = models()
    events: list[rietveld.ProgressEvent] = []
    source = dioptas.DioptasPatternData(
        x,
        truth,
        background_y=np.full(x.size, 0.2),
    )
    result = dioptas.calculate(source, instrument, (phase,), progress=events.append)
    np.testing.assert_allclose(result.calculated_y, truth, atol=2.0e-15)
    np.testing.assert_allclose(result.difference_y, 0.0, atol=2.0e-15)
    np.testing.assert_array_equal(result.background_y, 0.2)
    assert result.phase_curves[0].label == "alpha"
    assert result.phase_curves[0].y.shape == x.shape
    assert result.diagnostics["method"] == "calculation"
    assert result.x_unit == "degree_2theta"
    assert result.intensity_unit == "arbitrary_intensity"
    assert [(event.completed, event.total) for event in events] == [(0, 1), (1, 1)]


class FakeConsumer:
    def __init__(self, data: dioptas.DioptasPatternData) -> None:
        self.data = data
        self.published: dioptas.DioptasDisplayResult | None = None

    def read_pattern(self) -> dioptas.DioptasPatternData:
        return self.data

    def publish_result(self, result: dioptas.DioptasDisplayResult) -> None:
        self.published = result


def test_fake_consumer_contract_and_calculation_cancellation() -> None:
    x, instrument, phase, truth = models()
    source = dioptas.DioptasPatternData(x, truth)
    consumer = FakeConsumer(source)
    result = dioptas.run_calculation(consumer, instrument, (phase,))
    assert consumer.published is result
    with pytest.raises(rietveld.OperationCancelled):
        dioptas.calculate(source, instrument, (phase,), cancellation=lambda: True)


def test_lebail_progress_can_request_cooperative_iteration_cancellation() -> None:
    x, instrument, truth_phase, truth = models()
    starting = rietveld.Phase(
        truth_phase.phase_id,
        truth_phase.name,
        rietveld.ReflectionBatch(
            list(truth_phase.reflections.reflection_ids),
            truth_phase.reflections.hkl,
            truth_phase.reflections.d_spacing_angstrom,
            truth_phase.reflections.two_theta_deg,
            np.ones(2),
        ),
    )
    source = dioptas.DioptasPatternData(x, truth)
    cancelled = False
    events = []

    def progress(event: rietveld.ProgressEvent) -> None:
        nonlocal cancelled
        events.append(event)
        cancelled = True

    display, result = dioptas.refine_lebail(
        source,
        instrument,
        (starting,),
        options=lebail.LeBailOptions(max_iterations=20),
        progress=progress,
        cancellation=lambda: cancelled,
    )
    assert result.termination_reason is TerminationReason.CANCELLED
    assert len(result.history) == len(events) == 1
    assert events[0].stage == "lebail_iteration"
    assert display.diagnostics["termination_reason"] == "cancelled"
    assert display.included_mask.all()


def test_cancellation_before_first_iteration_resumes_identically() -> None:
    x, instrument, truth_phase, truth = models()
    source = dioptas.DioptasPatternData(x, truth)
    options = lebail.LeBailOptions(max_iterations=20)
    input_data = lebail.LeBailInput(source.pattern, instrument, (truth_phase,))
    cancelled = lebail.refine(input_data, options, cancellation=lambda: True)
    assert cancelled.termination_reason is TerminationReason.CANCELLED
    assert cancelled.history == ()
    assert np.isinf(cancelled.checkpoint.previous_rwp)
    resumed = lebail.refine(input_data, options, checkpoint=cancelled.checkpoint)
    uninterrupted = lebail.refine(input_data, options)
    np.testing.assert_array_equal(
        resumed.calculation.y,
        uninterrupted.calculation.y,
    )
    assert resumed.history == uninterrupted.history


def test_display_result_rejects_mismatched_public_construction() -> None:
    x = np.arange(3.0)
    with pytest.raises(ValueError, match="calculated_y"):
        dioptas.DioptasDisplayResult(
            x,
            x,
            np.ones(2),
            x,
            x,
            np.ones(3, dtype=np.bool_),
            (),
            {},
        )
