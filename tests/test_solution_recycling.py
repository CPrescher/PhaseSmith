"""Independent full cycles, covariance and observation provenance."""

import itertools

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.io.space_groups import space_group_by_symbol
from phasesmith.single_crystal.reference.solution_recycling_reference import recycle_reference

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_solution_recycle_density"), reason="requires experimental-solution"
)


def inputs(symbol="P 21 m a"):
    shape = (16, 16, 16)
    group = space_group_by_symbol(symbol).space_group
    support = np.zeros(shape, dtype=np.uint8)
    support[0, 0, 0] = 1
    hkl = []
    for h in itertools.product(range(-4, 5), repeat=3):
        if (
            h == (0, 0, 0)
            or sum(v * v for v in h) > 16
            or group.systematic_absences(np.array([h], dtype=np.int64))[0]
        ):
            continue
        support[tuple(np.array(h) % shape)] = 2
        hkl.append(h)
    hkl = np.array(hkl, dtype=np.int64)
    # One complete unmeasured symmetry orbit; predictions remain support code 1.
    family = {
        tuple(sign * (np.array([2, 0, 0]) @ op.rotation))
        for op in group.operations
        for sign in (-1, 1)
    }
    missing = np.array([tuple(h) in family for h in hkl])
    for h in hkl[missing]:
        support[tuple(h % shape)] = 1
    hkl = hkl[~missing]
    amplitudes = np.array([2 + np.exp(-0.1 * sum(x * x for x in h)) for h in hkl])
    source = np.random.default_rng(471).normal(size=shape)
    return shape, group, source, hkl, amplitudes, support


def native(
    args, *, hand=1, shift=(0, 0, 0), cycles=10, source=None, max_seconds=10.0, cancellation=None
):
    shape, g, rho, h, a, s = args
    return _core._solution_recycle_density(
        (rho if source is None else source).ravel(),
        shape,
        g._native,
        h.ravel(),
        a,
        s.ravel(),
        hand,
        shift,
        cycles,
        max_seconds,
        cancellation,
    )


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm", "P 21 m a", "I4/mcm"])
@pytest.mark.parametrize(
    "hand,shift,cycles",
    [(1, (0, 0, 0), 1), (-1, (0.173, 0.287, 0.391), 3), (1, (0.137, 0.271, 0.419), 10)],
)
def test_full_cycles_against_numpy(symbol, hand, shift, cycles):
    args = inputs(symbol)
    shape, g, rho, h, a, s = args
    result = native(args, hand=hand, shift=shift, cycles=cycles)
    reference, fractions = recycle_reference(
        rho,
        h,
        a,
        s,
        [op.rotation for op in g.operations],
        [[float(v) for v in op.translation] for op in g.operations],
        hand=hand,
        shift=shift,
        cycles=cycles,
    )
    density = np.asarray(result["density"]).reshape(shape)
    np.testing.assert_allclose(density, reference, rtol=5e-9, atol=2e-11)
    np.testing.assert_allclose(result["negative_fractions"], fractions, atol=1 / np.prod(shape))
    f = np.prod(shape) * np.fft.ifftn(density)
    np.testing.assert_allclose(np.abs(f[tuple((h % shape).T)]), a, rtol=1e-11, atol=1e-11)
    assert np.max(np.abs(f[s == 0])) < 1e-10
    assert result["predicted_coefficients"] == np.count_nonzero(s == 1)
    assert result["cycles"] == cycles
    np.testing.assert_array_equal(
        result["density"], native(args, hand=hand, shift=shift, cycles=cycles)["density"]
    )
    for op in g.operations:
        rotated = h @ op.rotation
        lhs = f[tuple((rotated % shape).T)]
        rhs = f[tuple((h % shape).T)] * np.exp(
            -2j * np.pi * (h @ np.array([float(v) for v in op.translation]))
        )
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=2e-10)


def test_zero_map_fallback_and_invalid_boundaries():
    args = inputs("P-1")
    zero = native(args, source=np.zeros(args[0]), cycles=0)
    assert zero["phase_fallbacks"] > 0 and zero["cycles"] == 0
    assert np.isfinite(zero["density"]).all()
    for kw in ({"hand": 0}, {"shift": (np.nan, 0, 0)}, {"cycles": 51}):
        with pytest.raises(ValueError):
            native(args, **kw)
    rho = args[2].copy()
    rho.flat[0] = np.inf
    with pytest.raises(ValueError):
        native(args, source=rho)
    shape, g, rho, h, a, s = args
    bad = s.copy()
    bad[1, 0, 0] = 0
    with pytest.raises(ValueError):
        native((shape, g, rho, h, a, bad))
    wrong = a.copy()
    wrong[0] *= 2
    with pytest.raises(ValueError):
        native((shape, g, rho, h, wrong, s))


def test_noncommensurate_screw_restores_exact_symmetry():
    # A 1/3 screw translation cannot permute nodes on the even 16-node grid.
    shape, _, rho, _, _, _ = inputs()
    rho = rho - rho.mean() + 0.001  # Nonzero free F000 avoids an all-clipped first map.
    group = space_group_by_symbol("P 31 2 1").space_group
    family = {
        tuple(sign * (np.array(h) @ op.rotation))
        for h in itertools.product(range(-2, 3), repeat=3)
        for op in group.operations
        for sign in (-1, 1)
    }
    h = np.array(sorted(family - {(0, 0, 0)}), dtype=np.int64)
    h = h[~group.systematic_absences(h)]
    support = np.zeros(shape, dtype=np.uint8)
    support[0, 0, 0] = 1
    support[tuple((h % shape).T)] = 2
    # Reciprocal hexagonal invariant, not a Cartesian sphere in HKL coordinates.
    amplitude = 2 + np.exp(-0.1 * (h[:, 0] ** 2 + h[:, 0] * h[:, 1] + h[:, 1] ** 2 + h[:, 2] ** 2))
    result = native((shape, group, rho, h, amplitude, support), cycles=10)
    expected, _ = recycle_reference(
        rho,
        h,
        amplitude,
        support,
        [op.rotation for op in group.operations],
        [[float(v) for v in op.translation] for op in group.operations],
    )
    density = np.asarray(result["density"]).reshape(shape)
    np.testing.assert_allclose(density, expected, rtol=5e-9, atol=2e-11)
    f = np.prod(shape) * np.fft.ifftn(density)
    np.testing.assert_allclose(np.abs(f[tuple((h % shape).T)]), amplitude, atol=1e-10)
    for op in group.operations:
        np.testing.assert_allclose(
            f[tuple(((h @ op.rotation) % shape).T)],
            f[tuple((h % shape).T)] * np.exp(-2j * np.pi * (h @ np.array(op.translation, float))),
            atol=2e-10,
        )


def test_translated_inversion_fallback_respects_phase_line():
    from phasesmith.symmetry import SpaceGroup, SymmetryOperation

    shape, _, rho, h, amplitude, support = inputs("P-1")
    group = SpaceGroup(
        [
            SymmetryOperation.identity(),
            SymmetryOperation(-np.eye(3, dtype=np.int64), ("1/2", 0, 0)),
        ]
    )
    result = native((shape, group, rho, h, amplitude, support), source=np.zeros(shape), cycles=0)
    f = np.prod(shape) * np.fft.ifftn(np.asarray(result["density"]).reshape(shape))
    values = f[tuple((h % shape).T)]
    np.testing.assert_allclose(np.abs(values), amplitude, atol=1e-10)
    np.testing.assert_allclose(values.conj(), values * np.exp(-1j * np.pi * h[:, 0]), atol=1e-10)
    np.testing.assert_allclose(values[h[:, 0] % 2 != 0].real, 0, atol=1e-10)


def test_frame_composition_and_cooperative_stop():
    from phasesmith.single_crystal.solution import SolutionCancellation

    args = inputs("P 21 m a")
    shape, group, rho, h, amplitude, support = args
    # A band-limited map avoids unrepresented frequencies at the Nyquist boundary.
    f = np.prod(shape) * np.fft.ifftn(rho)
    f[support == 0] = 0
    rho = np.fft.fftn(f).real / np.prod(shape)
    shift = np.array([0.173, 0.287, 0.391])
    transformed = np.zeros(shape, complex)
    for index in np.argwhere(support > 0):
        k = np.where(index < np.array(shape) // 2, index, index - shape)
        transformed[tuple(index)] = f[tuple(-k % shape)] * np.exp(2j * np.pi * (k @ shift))
    moved = np.fft.fftn(transformed).real / np.prod(shape)
    result = native((shape, group, rho, h, amplitude, support), hand=-1, shift=shift)
    equivalent = native((shape, group, moved, h, amplitude, support))
    np.testing.assert_allclose(result["density"], equivalent["density"], rtol=5e-9, atol=2e-11)
    token = SolutionCancellation()
    token.cancel()
    for controls in ({"cancellation": token._native}, {"max_seconds": 0.0}):
        stopped = native(args, **controls)
        assert stopped["cycles"] == 0
        assert not stopped["negative_fractions"]
        assert np.isfinite(stopped["density"]).all()
