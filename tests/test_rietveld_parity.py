from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from phasesmith import ExecutionPolicy, PowderPattern
from phasesmith.refinement.background import ChebyshevBackground
from phasesmith.validation import _rietveld_parity


@dataclass(frozen=True)
class FakePhase:
    phase_id: str
    scale: float


def test_exact_linear_profile_block_allows_the_empty_phase_active_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = np.linspace(10.0, 20.0, 7)
    observed = np.full(x.size, 4.25)
    pattern = PowderPattern(
        x,
        observed_y=observed,
        uncertainty=np.ones_like(observed),
        background=np.zeros_like(observed),
    )
    profiles = {
        "a": np.array([0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 0.0]),
        "b": np.array([0.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0]),
    }

    def calculate(*_args: object, **kwargs: object) -> SimpleNamespace:
        phase = kwargs.get("phases")
        if phase is None:
            phase = _args[2]
        selected = phase[0]
        return SimpleNamespace(profile_y=profiles[selected.phase_id] * selected.scale)

    monkeypatch.setattr(_rietveld_parity.rietveld, "calculate", calculate)
    phases, background = _rietveld_parity.solve_linear_profile_block(
        pattern,
        SimpleNamespace(),
        (FakePhase("a", 1.0), FakePhase("b", 1.0)),
        ChebyshevBackground("constant", (0.0,), (10.0, 20.0)),
        ExecutionPolicy(threads=1),
    )

    assert [phase.scale for phase in phases] == pytest.approx([0.0, 0.0], abs=1.0e-12)
    assert background.coefficients == pytest.approx((4.25,))


def test_exact_linear_profile_block_rejects_an_empty_phase_collection() -> None:
    x = np.linspace(10.0, 20.0, 7)
    pattern = PowderPattern(
        x,
        observed_y=np.ones(x.size),
        uncertainty=np.ones(x.size),
    )

    with pytest.raises(ValueError, match="at least one phase"):
        _rietveld_parity.solve_linear_profile_block(
            pattern,
            SimpleNamespace(),
            (),
            ChebyshevBackground("constant", (0.0,), (10.0, 20.0)),
            ExecutionPolicy(threads=1),
        )
