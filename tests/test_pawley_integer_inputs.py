"""Pawley family indices must not wrap during conversion to signed storage."""

import numpy as np
import pytest
from phasesmith.refinement.pawley import PawleyPhase
from phasesmith.refinement.tof_pawley import TofPawleyPhase


@pytest.mark.parametrize("phase_type", [PawleyPhase, TofPawleyPhase])
@pytest.mark.parametrize("value", [2**63, 2**64 - 1])
def test_unsigned_hkl_overflow_is_rejected(phase_type, value):
    hkl = np.array([[value, 0, 0]], dtype=np.uint64)
    with pytest.raises(ValueError, match="int64"):
        phase_type("phase", ("family",), [1.5], [1.0], hkl)


@pytest.mark.parametrize("phase_type", [PawleyPhase, TofPawleyPhase])
def test_representable_unsigned_hkl_keeps_its_value(phase_type):
    hkl = np.array([[1, 0, 0]], dtype=np.uint64)
    phase = phase_type("phase", ("family",), [1.5], [1.0], hkl)
    np.testing.assert_array_equal(phase.hkl, hkl)
    assert phase.hkl.dtype == np.int64
    assert not phase.hkl.flags.writeable
