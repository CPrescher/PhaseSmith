from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import phasesmith as ps
import pytest
from phasesmith.powder_sticks import _merge_sticks

SCAN = ps.CwTwoThetaRange(2, 60, 0.71073)


def model(name):
    # Include a centring translation, partial occupancy, and a finite U tensor.
    group = ps.SpaceGroup(
        [
            ps.SymmetryOperation.identity(),
            ps.SymmetryOperation(np.eye(3, dtype=int), (Fraction(1, 2), Fraction(1, 2), 0)),
        ]
    )
    return ps.CrystalStructure(
        name,
        name,
        ps.UnitCell(4, 5, 6, 90, 90, 90),
        group,
        (
            ps.AtomSite(
                "si",
                "Si1",
                "Si",
                "Si",
                (0.2, 0.3, 0.4),
                0.8,
                0.0,
                ps.AnisotropicDisplacement((0.01, 0.02, 0.03, 0, 0, 0), "U_cif"),
            ),
        ),
    )


def scalar_reference(structure):
    refs = ps.PreparedReflectionGenerator(structure.space_group).generate(structure.cell, SCAN)
    values = ps.calculate_structure_factor_values(
        structure,
        refs.hkl,
        refs.multiplicity,
        ps.XrayNonResonant(),
        correction=ps.BraggBrentanoUnpolarizedLp(SCAN.wavelength_angstrom),
        execution=ps.ExecutionPolicy(threads=1),
    )
    angles = 2 * np.rad2deg(np.arcsin(SCAN.wavelength_angstrom / (2 * refs.d_spacing_angstrom)))
    x, y = [], []
    for angle, intensity in sorted(
        zip(angles, values.integrated_intensity, strict=True), key=lambda pair: pair[0]
    ):
        if not x or angle - x[-1] >= 1e-5:
            x.append(angle)
            y.append(intensity)
        else:
            y[-1] += intensity
    return x, y


@pytest.mark.parametrize(
    ("backend", "workers"), [("threads", None), ("threads", 2), ("processes", 2), ("threads", 1)]
)
def test_batch_matches_scalar_in_input_order(backend, workers):
    a = model("first")
    b = replace(a, structure_id="second", sites=(replace(a.sites[0], occupancy=0.4),))
    models = (b, a, b)
    result = ps.calculate_powder_sticks_batch(iter(models), SCAN, backend=backend, workers=workers)
    assert [r.structure_id for r in result] == ["second", "first", "second"]
    for pattern, structure in zip(result, models, strict=True):
        x, y = scalar_reference(structure)
        np.testing.assert_array_equal(pattern.two_theta_deg, x)
        # Scalar additions and NumPy's reduceat may group the final few bits differently.
        np.testing.assert_allclose(pattern.integrated_intensity, y, rtol=1e-14, atol=0)


def test_empty_batch_empty_range_and_cell_without_sites():
    assert ps.calculate_powder_sticks_batch([], SCAN) == ()
    (empty,) = ps.calculate_powder_sticks_batch([model("a")], ps.CwTwoThetaRange(0, 0.01, 1.0))
    for values in (empty.two_theta_deg, empty.integrated_intensity):
        assert values.shape == (0,) and values.dtype == np.float64
    with pytest.raises(ValueError, match=r"structures\[1\].*no atom sites"):
        ps.calculate_powder_sticks_batch([model("a"), replace(model("b"), sites=())], SCAN)


def test_merge_uses_lowest_angle_exclusive_tolerance_and_stable_ties():
    t = 1e-5
    x, y = _merge_sticks(np.array([2 * t, 0.0, t, 0.75 * t, 1.5 * t, 0.0]), np.arange(1.0, 7.0))
    np.testing.assert_array_equal(x, [0.0, t, 2 * t])
    np.testing.assert_array_equal(y, [12.0, 8.0, 1.0])


@pytest.mark.parametrize(
    "kwargs", [dict(workers=0), dict(workers=True), dict(workers=1.5), dict(backend="other")]
)
def test_invalid_execution_even_for_empty_batch(kwargs):
    with pytest.raises(ValueError):
        ps.calculate_powder_sticks_batch([], SCAN, **kwargs)


@pytest.mark.parametrize(
    "scan",
    [
        ps.CwTwoThetaRange(10, 1, 1),
        ps.CwTwoThetaRange(0, 180, 1),
        ps.CwTwoThetaRange(0, 10, np.nan),
    ],
)
def test_invalid_scan_even_for_empty_batch(scan):
    with pytest.raises(ValueError):
        ps.calculate_powder_sticks_batch([], scan)


def test_invalid_structure():
    with pytest.raises(TypeError, match=r"structures\[0\]"):
        ps.calculate_powder_sticks_batch([None], SCAN)


def test_default_caps_workers_by_cpu_budget_and_input_size(monkeypatch):
    import phasesmith.powder_sticks as sticks

    counts = []
    executor = sticks.ThreadPoolExecutor

    def record_pool(*, max_workers):
        counts.append(max_workers)
        return executor(max_workers=max_workers)

    monkeypatch.setattr(sticks, "ThreadPoolExecutor", record_pool)
    a = model("a")
    for cpu, size in ((12, 1), (12, 2), (12, 7), (3, 7)):
        monkeypatch.setattr(sticks.os, "process_cpu_count", lambda limit=cpu: limit, raising=False)
        ps.calculate_powder_sticks_batch([a] * size, SCAN)
    assert counts == [2, 4, 3]


@pytest.mark.parametrize("backend", ["threads", "processes"])
def test_worker_error_identifies_structure_and_next_batch_succeeds(backend):
    a = model("valid")
    bad = replace(
        a,
        structure_id="unknown-species",
        sites=(replace(a.sites[0], type_symbol="Xx", element_symbol="Xx"),),
    )
    with pytest.raises((ValueError, KeyError)) as caught:
        ps.calculate_powder_sticks_batch([a, bad], SCAN, workers=2, backend=backend)
    assert any("unknown-species" in note for note in caught.value.__notes__)
    assert len(ps.calculate_powder_sticks_batch([a, a], SCAN, workers=2, backend=backend)) == 2


@pytest.mark.parametrize(("backend", "workers"), [("threads", 1), ("threads", 2), ("processes", 2)])
def test_powder_sticks_arrays_are_read_only_and_processes_ignore_unpicklable_metadata(
    backend, workers
):
    a = replace(model("meta-a"), metadata={"unpicklable": lambda x: x})
    b = replace(model("meta-b"), metadata={"unpicklable": object()})
    result = ps.calculate_powder_sticks_batch([a, b], SCAN, workers=workers, backend=backend)
    assert [r.structure_id for r in result] == ["meta-a", "meta-b"]
    for item in result:
        assert not item.two_theta_deg.flags.writeable
        assert not item.integrated_intensity.flags.writeable


def test_batch_reuses_prepared_reflection_generator_for_shared_space_groups(monkeypatch):
    import phasesmith.powder_sticks as sticks

    calls = 0
    real_ctor = sticks.PreparedReflectionGenerator

    def counting_ctor(group):
        nonlocal calls
        calls += 1
        return real_ctor(group)

    monkeypatch.setattr(sticks, "PreparedReflectionGenerator", counting_ctor)
    a = model("one")
    b = replace(a, structure_id="two")
    c = replace(a, structure_id="three")
    result = ps.calculate_powder_sticks_batch([a, b, c], SCAN, workers=1)
    assert len(result) == 3
    assert calls == 1
