from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread

import numpy as np
import phasesmith
from numpy.testing import assert_array_equal


def test_long_native_background_kernel_releases_python_gil() -> None:
    samples = np.sin(np.linspace(0.0, 100.0, 50_000, dtype=np.float64)) ** 2
    start = Event()
    watchdog_ran = Event()

    def watchdog() -> None:
        start.wait()
        watchdog_ran.set()

    thread = Thread(target=watchdog, name="phasesmith-gil-watchdog")
    thread.start()
    previous_interval = sys.getswitchinterval()
    try:
        # Prevent normal bytecode switching from satisfying the assertion. The
        # watchdog can run here only while the native call has detached.
        sys.setswitchinterval(10.0)
        start.set()
        phasesmith.smooth_bruckner(samples, smooth_points=20, iterations=50)
        ran_during_native_call = watchdog_ran.is_set()
    finally:
        sys.setswitchinterval(previous_interval)
        thread.join(timeout=1.0)

    assert ran_during_native_call
    assert not thread.is_alive()


def test_native_cw_accumulation_is_safe_for_concurrent_callers() -> None:
    x = np.linspace(10.0, 120.0, 20_001, dtype=np.float64)
    positions = np.linspace(12.0, 118.0, 400, dtype=np.float64)
    intensities = np.linspace(1.0, 10.0, positions.size, dtype=np.float64)
    instrument = phasesmith.ConstantWavelengthInstrument(
        wavelength_angstrom=1.5406,
        u_deg2=0.01,
        v_deg2=-0.001,
        w_deg2=0.02,
        x_deg=0.002,
        y_deg=0.001,
    )

    def calculate() -> phasesmith.AccumulationResult:
        return phasesmith.accumulate_cw(x, positions, intensities, instrument)

    expected = calculate()
    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = tuple(executor.map(lambda _: calculate(), range(8)))

    for result in actual:
        assert_array_equal(result.y, expected.y)
        assert_array_equal(result.derivatives.local.values, expected.derivatives.local.values)
        assert_array_equal(
            result.derivatives.global_jacobian,
            expected.derivatives.global_jacobian,
        )
