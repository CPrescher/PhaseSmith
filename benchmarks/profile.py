"""Reproducible Python-to-Rust profile benchmark with allocation reporting."""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from rietveld import (
    CompositePhysicsProvider,
    ConstantWavelengthInstrument,
    FcjGeometry,
    IsotropicMicrostrainBroadening,
    IsotropicSizeBroadening,
    MarchDollasePreferredOrientation,
    PhysicsContext,
    ReciprocalMetric,
    ReflectionGeometryBatch,
    WavelengthComponents,
    _core,
    accumulate,
    accumulate_cw,
    accumulate_cw_fcj,
    accumulate_cw_fcj_components,
    accumulate_tch,
    calculate_cw_pattern,
    cw_profile_parameters,
)


def parse_args() -> argparse.Namespace:
    """Parse benchmark controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--require-release", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile from a non-empty sample."""

    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(np.ceil(fraction * len(ordered))) - 1))
    return ordered[index]


def measure(
    operation: Callable[[], Any], *, warmups: int, repetitions: int
) -> tuple[Any, list[float]]:
    """Measure an operation after a fixed number of warmup calls."""

    for _ in range(warmups):
        operation()
    timings_ms = []
    result = None
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        result = operation()
        timings_ms.append((time.perf_counter_ns() - started) / 1e6)
    return result, timings_ms


def report_case(name: str, output_bytes: int, timings_ms: list[float]) -> None:
    """Print timing and allocation metrics for one derivative layout."""

    median_ms = statistics.median(timings_ms)
    print(
        f"case={name} output_mb={output_bytes / 1e6:.3f} "
        f"min_ms={min(timings_ms):.3f} median_ms={median_ms:.3f} "
        f"p95_ms={percentile(timings_ms, 0.95):.3f}"
    )


def main() -> None:
    """Run and report the end-to-end native accumulation benchmark."""

    arguments = parse_args()
    if arguments.warmups < 0 or arguments.repetitions <= 0:
        raise ValueError("warmups must be non-negative and repetitions must be positive")
    if arguments.require_release and _core.BUILD_MODE != "release":
        raise RuntimeError(
            "native extension is not optimized; run `maturin develop --release --uv` first"
        )

    x = np.linspace(10.0, 110.0, 5_001)
    count = 200
    index = np.arange(count)
    positions = 10.1 + index * 0.49
    intensities = 100.0 + index % 31
    fwhms = 0.03 + (index % 7) * 0.002
    etas = 0.2 + (index % 5) * 0.1
    gaussian_fwhms = 0.02 + (index % 7) * 0.002
    lorentzian_fwhms = 0.01 + (index % 5) * 0.002
    cw_instrument = ConstantWavelengthInstrument(
        wavelength_angstrom=1.5406,
        u_deg2=2.0e-4,
        v_deg2=-1.0e-4,
        w_deg2=1.2e-4,
        x_deg=1.5e-3,
        y_deg=3.0e-3,
    )
    fcj_geometry = FcjGeometry(sample_over_radius=0.012, detector_over_radius=0.012)
    doublet = WavelengthComponents.doublet(
        cw_instrument.wavelength_angstrom,
        1.54443,
        0.5,
    )
    d_spacings = cw_instrument.wavelength_angstrom / (2.0 * np.sin(np.deg2rad(positions / 2.0)))
    reflection_geometry = ReflectionGeometryBatch(
        np.column_stack(
            (
                np.arange(count, dtype=np.int64),
                np.ones(count, dtype=np.int64),
                np.arange(count, dtype=np.int64) % 5,
            )
        ),
        d_spacings,
        positions,
        intensities,
    )
    sample_physics = CompositePhysicsProvider(
        (IsotropicSizeBroadening(50.0), IsotropicMicrostrainBroadening(5.0e-4))
    )
    sample_orientation_physics = CompositePhysicsProvider(
        (
            IsotropicSizeBroadening(50.0),
            IsotropicMicrostrainBroadening(5.0e-4),
            MarchDollasePreferredOrientation(
                0.72, (0.0, 0.0, 1.0), ReciprocalMetric.orthogonal(4.0, 4.0, 4.0)
            ),
        )
    )
    sample_contribution = sample_physics.evaluate(
        PhysicsContext(reflection_geometry, cw_instrument)
    )

    support_fwhm = 20.0
    lower = np.searchsorted(x, positions - support_fwhm * fwhms, side="left")
    upper = np.searchsorted(x, positions + support_fwhm * fwhms, side="right")
    active_peak_samples = int(np.sum(upper - lower))
    tch_fwhms = (
        gaussian_fwhms**5
        + 2.69269 * gaussian_fwhms**4 * lorentzian_fwhms
        + 2.42843 * gaussian_fwhms**3 * lorentzian_fwhms**2
        + 4.47163 * gaussian_fwhms**2 * lorentzian_fwhms**3
        + 0.07842 * gaussian_fwhms * lorentzian_fwhms**4
        + lorentzian_fwhms**5
    ) ** 0.2
    tch_lower = np.searchsorted(x, positions - support_fwhm * tch_fwhms, side="left")
    tch_upper = np.searchsorted(x, positions + support_fwhm * tch_fwhms, side="right")
    tch_active_peak_samples = int(np.sum(tch_upper - tch_lower))
    cw_widths = cw_profile_parameters(positions, cw_instrument).total_fwhm_deg
    cw_lower = np.searchsorted(x, positions - support_fwhm * cw_widths, side="left")
    cw_upper = np.searchsorted(x, positions + support_fwhm * cw_widths, side="right")
    cw_active_peak_samples = int(np.sum(cw_upper - cw_lower))
    cw_parameters = cw_profile_parameters(positions, cw_instrument)
    sample_gaussian = 2.3548200450309493 * np.sqrt(
        cw_parameters.gaussian_variance_deg2 + sample_contribution.gaussian_variance_deg2
    )
    sample_lorentzian = cw_parameters.lorentzian_fwhm_deg + sample_contribution.lorentzian_fwhm_deg
    sample_widths = (
        sample_gaussian**5
        + 2.69269 * sample_gaussian**4 * sample_lorentzian
        + 2.42843 * sample_gaussian**3 * sample_lorentzian**2
        + 4.47163 * sample_gaussian**2 * sample_lorentzian**3
        + 0.07842 * sample_gaussian * sample_lorentzian**4
        + sample_lorentzian**5
    ) ** 0.2
    sample_lower = np.searchsorted(x, positions - support_fwhm * sample_widths, side="left")
    sample_upper = np.searchsorted(x, positions + support_fwhm * sample_widths, side="right")
    sample_active_peak_samples = int(np.sum(sample_upper - sample_lower))
    apparent_limit = np.rad2deg(
        np.arccos(
            np.cos(np.deg2rad(positions))
            * np.sqrt(
                1.0 + (fcj_geometry.sample_over_radius + fcj_geometry.detector_over_radius) ** 2
            )
        )
    )
    fcj_radius = support_fwhm * cw_widths
    fcj_lower = np.searchsorted(x, np.minimum(positions, apparent_limit) - fcj_radius, side="left")
    fcj_upper = np.searchsorted(x, np.maximum(positions, apparent_limit) + fcj_radius, side="right")
    fcj_active_peak_samples = int(np.sum(fcj_upper - fcj_lower))

    call_arguments = (x, positions, intensities, fwhms, etas, support_fwhm)
    values, values_timings = measure(
        lambda: _core.accumulate_values(*call_arguments),
        warmups=arguments.warmups,
        repetitions=arguments.repetitions,
    )
    support_result, support_timings = measure(
        lambda: accumulate(
            x,
            positions,
            intensities,
            fwhms,
            etas,
            support_fwhm=support_fwhm,
            jacobian_layout="support",
        ),
        warmups=arguments.warmups,
        repetitions=arguments.repetitions,
    )
    dense_result, dense_timings = measure(
        lambda: accumulate(
            x,
            positions,
            intensities,
            fwhms,
            etas,
            support_fwhm=support_fwhm,
            jacobian_layout="dense",
        ),
        warmups=arguments.warmups,
        repetitions=arguments.repetitions,
    )
    tch_result, tch_timings = measure(
        lambda: accumulate_tch(
            x,
            positions,
            intensities,
            gaussian_fwhms,
            lorentzian_fwhms,
            support_fwhm=support_fwhm,
        ),
        warmups=arguments.warmups,
        repetitions=arguments.repetitions,
    )
    cw_result, cw_timings = measure(
        lambda: accumulate_cw(
            x,
            positions,
            intensities,
            cw_instrument,
            support_fwhm=support_fwhm,
        ),
        warmups=arguments.warmups,
        repetitions=arguments.repetitions,
    )
    sample_result, sample_timings = measure(
        lambda: calculate_cw_pattern(
            x,
            reflection_geometry,
            cw_instrument,
            physics=sample_physics,
            support_fwhm=support_fwhm,
        ),
        warmups=arguments.warmups,
        repetitions=arguments.repetitions,
    )
    sample_orientation_result, sample_orientation_timings = measure(
        lambda: calculate_cw_pattern(
            x,
            reflection_geometry,
            cw_instrument,
            physics=sample_orientation_physics,
            support_fwhm=support_fwhm,
        ),
        warmups=arguments.warmups,
        repetitions=arguments.repetitions,
    )
    fcj_result, fcj_timings = measure(
        lambda: accumulate_cw_fcj(
            x,
            positions,
            intensities,
            cw_instrument,
            fcj_geometry,
            support_fwhm=support_fwhm,
        ),
        warmups=arguments.warmups,
        repetitions=arguments.repetitions,
    )
    doublet_result, doublet_timings = measure(
        lambda: accumulate_cw_fcj_components(
            x,
            positions,
            intensities,
            cw_instrument,
            doublet,
            fcj_geometry,
            support_fwhm=support_fwhm,
        ),
        warmups=arguments.warmups,
        repetitions=arguments.repetitions,
    )

    input_bytes = sum(array.nbytes for array in (x, positions, intensities, fwhms, etas))
    print(f"python={platform.python_version()} numpy={np.__version__}")
    print(f"platform={platform.platform()}")
    print(f"native_build_mode={_core.BUILD_MODE} native_module={_core.__file__}")
    print(
        f"peaks={count} samples={x.size} active_peak_samples={active_peak_samples} "
        f"tch_active_peak_samples={tch_active_peak_samples} "
        f"cw_active_peak_samples={cw_active_peak_samples} "
        f"sample_active_peak_samples={sample_active_peak_samples} "
        f"fcj_active_peak_samples={fcj_active_peak_samples} "
        f"wavelength_component_counts=1,2 support_fwhm={support_fwhm:g}"
    )
    print(f"input_mb={input_bytes / 1e6:.3f} repetitions={arguments.repetitions}")
    report_case("values_only", values.nbytes, values_timings)
    report_case(
        "support_jacobian",
        support_result.y.nbytes + support_result.derivatives.local.nbytes,
        support_timings,
    )
    report_case(
        "dense_jacobian",
        dense_result.y.nbytes + dense_result.jacobian.nbytes,
        dense_timings,
    )
    report_case(
        "tch_support_jacobian",
        tch_result.y.nbytes + tch_result.derivatives.local.nbytes,
        tch_timings,
    )
    report_case(
        "cw_local_and_global_jacobian",
        cw_result.y.nbytes
        + cw_result.derivatives.local.nbytes
        + cw_result.derivatives.global_jacobian.nbytes,
        cw_timings,
    )
    report_case(
        "cw_size_strain_provider_and_native_jacobian",
        sample_result.y.nbytes
        + sample_result.derivatives.local.nbytes
        + sample_result.derivatives.global_jacobian.nbytes,
        sample_timings,
    )
    report_case(
        "cw_size_strain_orientation_provider_and_native_jacobian",
        sample_orientation_result.y.nbytes
        + sample_orientation_result.derivatives.local.nbytes
        + sample_orientation_result.derivatives.global_jacobian.nbytes,
        sample_orientation_timings,
    )
    report_case(
        "cw_fcj_local_and_global_jacobian_order_48",
        fcj_result.y.nbytes
        + fcj_result.derivatives.local.nbytes
        + fcj_result.derivatives.global_jacobian.nbytes,
        fcj_timings,
    )
    report_case(
        "cw_fcj_doublet_local_and_global_jacobian_order_48",
        doublet_result.y.nbytes
        + doublet_result.derivatives.local.nbytes
        + doublet_result.derivatives.global_jacobian.nbytes,
        doublet_timings,
    )


if __name__ == "__main__":
    main()
