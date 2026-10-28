"""Independent equations, blind synthetic recovery, and external-HKL contracts."""

from __future__ import annotations

import itertools
import json
import subprocess
import sys

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.experimental.solution import (
    SolutionCancellation,
    SolutionOptions,
    solve_structure,
    write_solution,
)
from phasesmith.experimental.solution_reference import (
    amplitude_model_reference,
    charge_flip_reference,
)
from phasesmith.io.hkl import HklData, parse_hkl, read_hkl
from phasesmith.io.space_groups import space_group_by_symbol
from phasesmith.scattering_reference import xray_non_resonant

NATIVE = pytest.mark.skipif(
    not hasattr(_core, "_solve_structure"),
    reason="requires experimental-solution native feature",
)


def _quartz():
    """Synthetic quartz; positions generate observations and never enter solve()."""
    import gemmi

    cell = UnitCell(4.913, 4.913, 5.405, 90, 90, 120)
    gemmi_cell = gemmi.UnitCell(*cell.as_tuple())
    xyz, species = [], []
    for element, position in (("Si", (0.4697, 0, 1 / 3)), ("O", (0.4133, 0.2672, 0.1188))):
        for op in gemmi.find_spacegroup_by_name("P 31 2 1").operations():
            point = np.mod(op.apply_to_xyz(position), 1)
            if any(np.linalg.norm((point - other + 0.5) % 1 - 0.5) < 1e-8 for other in xyz):
                continue
            xyz.append(point)
            species.append(element)
    hkl = np.array(
        [
            h
            for h in itertools.product(range(-8, 9), repeat=3)
            if h > tuple(-v for v in h) and gemmi_cell.calculate_d(h) >= 0.77
        ],
        dtype=np.int64,
    )
    s = np.array([0.5 / gemmi_cell.calculate_d(tuple(h)) for h in hkl])
    factors, _ = xray_non_resonant(species, s)
    xyz = np.asarray(xyz)
    f = np.sum(
        factors
        * np.exp(-8 * np.pi**2 * 0.01 * s[:, None] ** 2)
        * np.exp(2j * np.pi * (hkl @ xyz.T)),
        axis=1,
    )
    data = HklData(hkl, np.abs(f) ** 2, np.maximum(np.abs(f) ** 2 * 0.01, 0.01))
    return cell, data, xyz, tuple(species)


def _coordinate_error(reference, ref_species, candidate, cell):
    """Match up to origin, inversion, and same-element permutations."""

    basis = cell.geometry().direct_basis
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
    best = np.inf
    for sign in (-1, 1):
        for anchor, element in zip(reference, ref_species, strict=True):
            if element != candidate.species[0]:
                continue
            shift = anchor - sign * candidate.fractional_xyz[0]
            model = (sign * candidate.fractional_xyz + shift) % 1
            maximum = 0.0
            for species in set(ref_species):
                expected = reference[np.array(ref_species) == species]
                found = model[np.array(candidate.species) == species]
                delta = (expected[:, None] - found[None, :] + 0.5) % 1 - 0.5
                distances = np.linalg.norm((delta[:, :, None, :] + shifts) @ basis.T, axis=-1).min(
                    axis=2
                )
                # Small exact subset assignment avoids importing optional SciPy
                # into tests that also verify its lazy-import boundary.
                assignments = {0: (0.0, ())}
                for row in distances:
                    next_assignments = {}
                    for mask, (cost, order) in assignments.items():
                        for column, value in enumerate(row):
                            if mask & (1 << column):
                                continue
                            key = mask | (1 << column)
                            proposal = (cost + value**2, (*order, column))
                            if key not in next_assignments or proposal < next_assignments[key]:
                                next_assignments[key] = proposal
                    assignments = next_assignments
                columns = assignments[(1 << len(distances)) - 1][1]
                maximum = max(maximum, distances[np.arange(len(distances)), columns].max())
            best = min(best, maximum)
    return best


def test_hkl_signed_records_fixed_width_and_provenance(tmp_path):
    text = "# integrated externally\n" + "".join(
        f"{h:4d}{k:4d}{ell:4d}{i:8.2f}{s:8.2f}{b:4d}\n"
        for h, k, ell, i, s, b in (
            (-1, 2, -3, -12.5, 1.2, 2),
            (1, -2, 3, 12345.6, 2.5, 1),
            (0, 0, 0, 0, 0, 0),
        )
    )
    path = tmp_path / "external.hkl"
    path.write_bytes(text.encode("ascii"))
    data = read_hkl(path)
    np.testing.assert_array_equal(data.intensity, [-12.5, 12345.6])
    np.testing.assert_array_equal(data.hkl, [[-1, 2, -3], [1, -2, 3]])
    np.testing.assert_array_equal(data.batch, [2, 1])
    assert data.source_sha256 == parse_hkl(text).source_sha256
    assert path.read_bytes() == text.encode("ascii")
    assert not data.hkl.flags.writeable and not data.intensity.flags.writeable


@pytest.mark.parametrize(
    "text",
    [
        "",
        "1 0 0 4 0",
        "1 0 0 nan 1",
        "1.0 0 0 4 1",
        "1 0 0 4 1 -1",
        "0 0 0 0 0\n1 0 0 4 1",
        "1 0 0 4 1 2 3",
        "1025 0 0 1 1",
    ],
)
def test_invalid_or_unsupported_hkl_records(text):
    with pytest.raises(ValueError):
        parse_hkl(text)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trials": True},
        {"cycles": 0},
        {"seed": -1},
        {"seed": 2**64},
        {"max_seconds": 0},
        {"r1_target": np.nan},
        {"weak_fraction": 0.6},
        {"grid_shape": (8, 8, 7)},
        {"polish_iterations": 201},
    ],
)
def test_invalid_solution_budgets(kwargs):
    with pytest.raises(ValueError):
        SolutionOptions(**kwargs)


@NATIVE
@pytest.mark.parametrize("weak_fraction", [0.0, 0.2])
def test_complete_native_fft_cycles_match_independent_numpy(weak_fraction):
    hkl = np.array(
        [h for h in itertools.product(range(-2, 3), repeat=3) if h != (0, 0, 0)], dtype=np.int64
    )
    amplitudes = np.array([1 + np.cos(np.dot(h, [0.3, 0.5, 0.7])) ** 2 for h in hkl])
    phases = hkl @ np.array([0.137, 0.271, 0.419])
    support = np.zeros((8, 8, 8), dtype=np.uint8)
    support[tuple((hkl % 8).T)] = 2
    support[0, 0, 0] = 1
    support[3, 0, 0] = support[5, 0, 0] = 1  # truly missing coefficients stay free
    before = support.copy()
    expected = charge_flip_reference(
        (8, 8, 8),
        hkl,
        amplitudes,
        phases,
        support,
        cycles=5,
        delta_sigma=1.1,
        weak_fraction=weak_fraction,
        cleanup_cycles=2,
    )
    actual = _core._solution_charge_flip(
        (8, 8, 8), hkl.ravel(), amplitudes, phases, support.ravel(), 5, 1.1, weak_fraction, 2
    )
    np.testing.assert_allclose(actual[0].reshape(8, 8, 8), expected[0], rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-12, atol=2e-14)
    assert actual[2] == expected[2]
    np.testing.assert_array_equal(support, before)
    coefficients = np.fft.ifftn(actual[0].reshape(8, 8, 8)) * 512
    np.testing.assert_allclose(
        np.abs(coefficients[tuple((hkl % 8).T)]), amplitudes, rtol=2e-12, atol=2e-14
    )
    np.testing.assert_allclose(coefficients[support == 0], 0, rtol=0, atol=2e-14)
    asymmetric = support.copy()
    asymmetric[3, 0, 0] = 0
    with pytest.raises(ValueError, match="Friedel symmetry"):
        _core._solution_charge_flip(
            (8, 8, 8), hkl.ravel(), amplitudes, phases, asymmetric.ravel(), 1, 1.1, weak_fraction, 0
        )


@NATIVE
def test_native_amplitude_model_matches_independent_complex_sums():
    cell = UnitCell(5, 6, 7, 90, 103, 90)
    hkl = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 1], [-2, 1, 3], [3, 1, 2]], dtype=np.int64
    )
    xyz = np.array([[0.13, 0.24, 0.35], [0.46, 0.51, 0.72], [0.72, 0.23, 0.61]])
    species = ["Si", "O", "O"]
    observed = np.array([10, 12, 9, 7, 8, 4], dtype=float)
    # Independent reciprocal metric derived directly from a Cartesian basis.
    beta = np.deg2rad(103)
    basis = np.array([[5, 0, 7 * np.cos(beta)], [0, 6, 0], [0, 0, 7 * np.sin(beta)]])
    s = np.linalg.norm(hkl @ np.linalg.inv(basis), axis=1) / 2
    factors, _ = xray_non_resonant(species, s)
    expected = amplitude_model_reference(hkl, observed, xyz, factors, s, 0.01)
    actual = _core._solution_model(
        list(cell.as_tuple()), hkl.ravel(), observed, species, xyz.ravel(), 0.01
    )
    for key in ("objective", "r1", "amplitude_scale", "residual", "jacobian"):
        np.testing.assert_allclose(
            np.asarray(actual[key]).ravel(),
            np.asarray(expected[key]).ravel(),
            rtol=2e-12,
            atol=3e-13,
        )


@NATIVE
@pytest.mark.parametrize("seed", [13, 29, 71])
def test_randomized_amplitude_jacobians_and_normalization(seed):
    rng = np.random.default_rng(seed)
    cell = UnitCell(*rng.uniform(4, 8, 3), 90, float(rng.uniform(95, 110)), 90)
    hkl = rng.integers(-3, 4, size=(12, 3), dtype=np.int64)
    hkl[np.all(hkl == 0, axis=1)] = [1, 0, 0]
    observed = rng.uniform(3, 20, len(hkl))
    species = ["Fe", "Si", "O"]
    xyz = rng.uniform(0.1, 0.9, (3, 3))
    beta = np.deg2rad(cell.beta_deg)
    a, b, c = cell.as_tuple()[:3]
    basis = np.array([[a, 0, c * np.cos(beta)], [0, b, 0], [0, 0, c * np.sin(beta)]])
    s = np.linalg.norm(hkl @ np.linalg.inv(basis), axis=1) / 2
    scattering, _ = xray_non_resonant(species, s)
    reference = amplitude_model_reference(hkl, observed, xyz, scattering, s, 0.02)
    native = _core._solution_model(
        list(cell.as_tuple()), hkl.ravel(), observed, species, xyz.ravel(), 0.02
    )
    for key in ("residual", "jacobian", "objective", "amplitude_scale", "r1"):
        np.testing.assert_allclose(
            np.ravel(native[key]), np.ravel(reference[key]), rtol=3e-12, atol=2e-13
        )
    # Scaling all measured amplitudes changes fitted k, not coordinates/objective.
    scaled = _core._solution_model(
        list(cell.as_tuple()), hkl.ravel(), observed * 17, species, xyz.ravel(), 0.02
    )
    np.testing.assert_allclose(native["residual"], scaled["residual"], rtol=2e-13, atol=2e-14)


@NATIVE
def test_blind_quartz_recovery_and_outputs(tmp_path):
    cell, data, expected, species = _quartz()
    result = solve_structure(
        data,
        cell=cell,
        composition={"Si": 3, "O": 6},
        space_group=space_group_by_symbol("P 31 2 1").space_group,
        options=SolutionOptions(trials=8, cycles=800, r1_target=1e-4, max_seconds=60),
    )
    print("quartz", result.termination, [c.r1 for c in result.candidates])
    assert result.best is not None
    assert result.best.r1 < 1e-4
    assert _coordinate_error(expected, species, result.best, cell) < 0.005
    assert result.target_met and result.termination == "r1_target"
    assert (
        result.best.as_structure(cell).space_group == space_group_by_symbol("P 31 2 1").space_group
    )
    assert len(result.best.as_structure(cell).sites) == 2
    path = write_solution(result, tmp_path / "solution")
    record = json.loads((path / "result.json").read_text())
    assert record["output_space_group"] == "supplied operations" and record["experimental"]
    with np.load(path / "density-and-observations.npz") as stored:
        np.testing.assert_array_equal(stored["input_intensity"], data.intensity)
    import gemmi

    block = gemmi.cif.read_file(str(path / "candidate-p1.cif")).sole_block()
    assert len(block.find_values("_atom_site_label")) == 9
    # Check both exported frames with the separate general structure-factor path.
    import phasesmith
    from phasesmith.io.cif import read_cif
    from phasesmith.scattering import XrayNonResonant

    structures = [result.best.as_structure(cell)] + [
        read_cif(path / name).structure for name in ("candidate.cif", "candidate-p1.cif")
    ]
    for structure in structures:
        values = phasesmith.calculate_structure_factor_values(
            structure,
            result.expanded_hkl,
            np.ones(len(result.expanded_hkl), dtype=np.int64),
            XrayNonResonant(),
        )
        predicted = result.best.amplitude_scale * np.sqrt(values.f_squared)
        exported_r1 = np.abs(predicted - result.amplitudes).sum() / result.amplitudes.sum()
        assert abs(exported_r1 - result.best.r1) < 2e-10
    assert not result.best.refinement.u_cartesian.flags.writeable
    with pytest.raises(FileExistsError):
        write_solution(result, path)


@NATIVE
def test_negative_means_retained_and_runtime_limit_is_truthful():
    cell = UnitCell(5, 5, 5, 90, 90, 90)
    data = parse_hkl("1 0 0 -4 2\n-1 0 0 8 1\n0 1 0 -1 1\n")
    result = solve_structure(
        data, cell=cell, composition={"Si": 1}, options=SolutionOptions(max_seconds=1e-9)
    )
    assert result.termination == "max_runtime" and not result.target_met
    assert result.best is None and result.nonpositive_families == 1
    np.testing.assert_array_equal(data.intensity, [-4, 8, -1])
    assert -1 in result.merged_intensity
    np.testing.assert_allclose(
        result.amplitudes, np.sqrt(np.maximum(result.merged_intensity, 0)), rtol=0, atol=0
    )


@NATIVE
def test_cooperative_cancellation_and_metric_rejection():
    cell, data, _, _ = _quartz()
    token = SolutionCancellation()
    assert token.cancel() and not token.cancel() and token.is_cancelled
    result = solve_structure(data, cell=cell, composition={"Si": 3, "O": 6}, cancellation=token)
    assert result.termination == "cancelled" and result.best is None
    with pytest.raises(ValueError, match="metric is incompatible"):
        solve_structure(
            parse_hkl("0 0 1 10 1"),
            cell=UnitCell(5, 6, 7, 90, 90, 90),
            composition={"Si": 1},
            space_group=space_group_by_symbol("P 4").space_group,
        )


@NATIVE
def test_cancellation_from_another_thread():
    import threading

    cell, data, _, _ = _quartz()
    token = SolutionCancellation()
    timer = threading.Timer(0.05, token.cancel)
    timer.start()
    try:
        result = solve_structure(
            data,
            cell=cell,
            composition={"Si": 3, "O": 6},
            cancellation=token,
            options=SolutionOptions(cycles=20000, r1_target=0, trials=128),
        )
    finally:
        timer.join()
    assert result.termination == "cancelled" and not result.target_met


@NATIVE
def test_cli_preserves_source_and_reports_unmet_target(tmp_path):
    path = tmp_path / "external.hkl"
    text = "1 0 0 16 1\n0 1 0 12 1\n0 0 1 9 1\n"
    path.write_text(text)
    output = tmp_path / "output"
    command = [
        sys.executable,
        "-m",
        "phasesmith.experimental.solution",
        str(path),
        "--cell",
        "5",
        "5",
        "5",
        "90",
        "90",
        "90",
        "--space-group",
        "P1",
        "--composition",
        "Si=2",
        "--output",
        str(output),
        "--cycles",
        "1",
        "--trials",
        "1",
        "--r1-target",
        "0",
    ]
    run = subprocess.run(command, capture_output=True, text=True, check=False)
    assert run.returncode == 3, run.stderr
    assert not json.loads(run.stdout)["r1_target_met"]
    assert path.read_text() == text
    again = subprocess.run(command, capture_output=True, text=True, check=False)
    assert again.returncode == 2 and "already exists" in again.stderr


@NATIVE
def test_legacy_p1_mode_remains_explicit():
    cell, data, _, _ = _quartz()
    result = solve_structure(
        data,
        cell=cell,
        composition={"Si": 3, "O": 6},
        space_group=space_group_by_symbol("P 31 2 1").space_group,
        options=SolutionOptions(trials=1, cycles=800, r1_target=1e-4, refinement="none"),
    )
    assert result.best.refinement is None
    assert result.best.as_structure(cell).space_group == space_group_by_symbol("P 1").space_group
