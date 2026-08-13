from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "oracle/scripts/benchmark_citrate_residual_forensics.py"
)
SPEC = importlib.util.spec_from_file_location("citrate_residual_forensics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_factorial_variants_are_complete_and_deterministic() -> None:
    variants = MODULE.factorial_variants()
    assert len(variants) == 16
    assert variants[0] == ()
    assert variants[-1] == MODULE.FACTORS
    assert len(set(variants)) == len(variants)


def test_legacy_shift_translation_matches_documented_gsasii_equation() -> None:
    radius = 141.5
    coefficient = -8.7503
    shift = MODULE.legacy_shft_to_gsasii_micrometre(coefficient, radius)
    constant = 0.09 / (np.pi * radius)
    assert -4.0 * constant * shift == pytest.approx(-coefficient / 100.0)
    legacy_physical_shift_mm = -np.pi * radius * coefficient / 36_000.0
    assert shift / 1_000.0 == pytest.approx(-legacy_physical_shift_mm)
    with pytest.raises(ValueError, match="positive"):
        MODULE.legacy_shft_to_gsasii_micrometre(coefficient, 0.0)


def test_legacy_transparency_translation_matches_documented_gsasii_equation() -> None:
    radius = 141.5
    coefficient = 1.30
    transparency = MODULE.legacy_trns_to_gsasii_field_cm(coefficient, radius)
    constant = 0.09 / (np.pi * radius)
    # Legacy delta_T_prime uses observed minus calculated position, so a
    # positive stored coefficient moves the peak center in the negative
    # direction. This is numerical equivalence, not a physical 1/mu mapping.
    assert -100.0 * constant * transparency == pytest.approx(-coefficient / 100.0)
    formal_mu_eff = -9000.0 / (np.pi * radius * coefficient)
    assert formal_mu_eff < 0.0


def test_shapley_partition_sums_to_full_factorial_improvement() -> None:
    factors = ("a", "b", "c")
    effects = {"a": 2.0, "b": 3.0, "c": 5.0}
    subset_sse = {
        frozenset(case): 20.0 - sum(effects[name] for name in case)
        for case in MODULE.factorial_variants(factors)
    }
    contributions = MODULE.shapley_sse_contributions(subset_sse, factors)
    assert contributions == pytest.approx(effects)
    assert sum(contributions.values()) == pytest.approx(
        subset_sse[frozenset()] - subset_sse[frozenset(factors)]
    )


def test_residual_diagnostics_distinguish_position_and_width_modes() -> None:
    x = np.linspace(17.0, 100.0, 4_106)
    signal = np.exp(-0.5 * ((x - 45.0) / 0.25) ** 2)
    background = np.full_like(x, 10.0)
    calculated = background + 1_000.0 * signal
    shifted = background + 1_000.0 * np.exp(-0.5 * ((x - 45.02) / 0.25) ** 2)
    widened = background + 1_000.0 * np.exp(-0.5 * ((x - 45.0) / 0.28) ** 2)
    shifted_result = MODULE.residual_diagnostics(x, shifted, calculated, background)
    widened_result = MODULE.residual_diagnostics(x, widened, calculated, background)
    shifted_modes = shifted_result["residual_mode_correlations"]
    widened_modes = widened_result["residual_mode_correlations"]
    assert abs(shifted_modes["position_first_derivative"]) > 0.9
    assert abs(widened_modes["width_second_derivative"]) > 0.7
    assert abs(widened_modes["width_second_derivative"]) > abs(
        widened_modes["position_first_derivative"]
    )


def test_residual_diagnostics_reject_invalid_axes() -> None:
    values = np.ones(5)
    with pytest.raises(ValueError, match="invalid"):
        MODULE.residual_diagnostics(values, values, values, values)
