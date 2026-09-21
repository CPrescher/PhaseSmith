"""Manifest controls are executable policy, not descriptive result metadata."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from phasesmith.validation import pawley


def manifest():
    return json.loads(
        (Path(__file__).parents[1] / "validation/pawley-acceptance-v2.json").read_text()
    )


@pytest.mark.parametrize(
    "change", ["unknown", "version", "repeats", "stop", "profile", "background"]
)
def test_manifest_rejects_ignored_or_unsupported_controls(change):
    m = manifest()
    if change == "unknown":
        m["common"]["silently_ignored"] = True
    elif change == "version":
        m["schema_version"] = 1
    elif change == "repeats":
        m["common"]["repeats"] = 1
    elif change == "stop":
        m["common"]["accepted_terminations"] = ["max_runtime"]
    elif change == "profile":
        m["common"]["profile_parameters"] = ["unknown"]
    else:
        m["datasets"]["aps-sucrose-11bmb"]["background"]["ignored"] = 1
    with pytest.raises(ValueError):
        pawley.validate_manifest(m)


def test_manifest_controls_change_resolved_request(monkeypatch, tmp_path):
    m = pawley.validate_manifest(manifest())
    gate = m["datasets"]["aps-sucrose-11bmb"]
    path = tmp_path / gate["data_file"]
    path.write_text("test input")
    x = np.linspace(1.0, 24.0, 1001)
    data = SimpleNamespace(x=x, observed_y=2 + np.sin(x) ** 2, uncertainty=np.ones_like(x))
    monkeypatch.setattr(pawley, "verify_validation_dataset", lambda *args: [path])
    monkeypatch.setattr(pawley, "read_powder_data", lambda *args, **kwargs: data)
    first, options, _ = pawley.build_request("aps-sucrose-11bmb", tmp_path, gate, m["common"])
    changed = copy.deepcopy(m)
    common = changed["common"]
    common.update(
        profile_parameters=["w_deg2"],
        signed_intensities=True,
        initial_intensity=-1.0,
        use_uncertainty=False,
        tolerance=2e-9,
    )
    changed_gate = changed["datasets"]["aps-sucrose-11bmb"]
    changed_gate["range_deg"] = [5.0, 20.0]
    changed_gate["support_fwhm"] = 15.0
    changed_gate["background"]["width_deg"] = 0.4
    pawley.validate_manifest(changed)
    second, changed_options, _ = pawley.build_request(
        "aps-sucrose-11bmb", tmp_path, changed_gate, common
    )
    assert len(second.pattern.x) < len(first.pattern.x)
    assert second.signed_intensities and np.all(second.phases[0].intensities == -1)
    selected = [
        s.key.name for s in second.parameters.specs if s.refine and s.key.module == "pawley_profile"
    ]
    assert selected == ["w_deg2"]
    assert options.use_uncertainty and not changed_options.use_uncertainty
    assert changed_options.support_fwhm == 15 and changed_options.tolerance == 2e-9


def test_acceptance_thresholds_and_actual_repeatability():
    m = manifest()
    gate, common = m["datasets"]["aps-sucrose-11bmb"], m["common"]
    y = np.array([1.0, 2.0, 3.0])
    calc = y * 1.01
    rwp = np.linalg.norm(calc - y) / np.linalg.norm(y)
    calculation = SimpleNamespace(
        calculated_y=calc,
        background_y=np.zeros(3),
        included=np.ones(3, bool),
        rwp=rwp,
        chi_square=float(np.sum((calc - y) ** 2)),
    )
    fit = SimpleNamespace(
        calculation=calculation,
        intensities=np.ones(1),
        history=np.array([1.0, 0.0014]),
        termination_reason="converged",
    )
    request = SimpleNamespace(pattern=SimpleNamespace(observed_y=y, uncertainty=np.ones(3)))
    initial = SimpleNamespace(chi_square=1.0)
    checks, _ = pawley.check_fit(fit, initial, request, [fit], gate, common)
    assert all(checks.values())
    checks, _ = pawley.check_fit(
        fit, initial, request, [fit], {**gate, "maximum_rwp": 0.005}, common
    )
    assert not checks["rwp"]
    checks, _ = pawley.check_fit(
        fit,
        initial,
        request,
        [fit],
        gate,
        {**common, "minimum_relative_chi_square_improvement": 0.9999},
    )
    assert not checks["improvement"]
    repeat = copy.deepcopy(fit)
    repeat.calculation.calculated_y[0] += 0.001
    checks, _ = pawley.check_fit(fit, initial, request, [repeat], gate, common)
    assert not checks["repeatable"]
    repeat = copy.deepcopy(fit)
    repeat.termination_reason = "max_runtime"
    checks, _ = pawley.check_fit(fit, initial, request, [repeat], gate, common)
    assert not checks["converged"]


def test_cell_manifest_selection_and_bounds_are_executable(monkeypatch, tmp_path):
    m = json.loads(
        (Path(__file__).parents[1] / "validation/pawley-cell-acceptance-v3.json").read_text()
    )
    pawley.validate_manifest(m)
    name, gate = next(iter(m["datasets"].items()))
    path = tmp_path / gate["data_file"]
    path.write_text("test input")
    x = np.linspace(20.0, 125.5, 501)
    data = SimpleNamespace(x=x, observed_y=2 + np.sin(x) ** 2, uncertainty=np.ones_like(x))
    monkeypatch.setattr(pawley, "verify_validation_dataset", lambda *args: [path])
    monkeypatch.setattr(pawley, "read_powder_data", lambda *args, **kwargs: data)
    request, _, _ = pawley.build_request(name, tmp_path, gate, m["common"])
    selected = [
        s for s in request.parameters.specs if s.refine and s.key.module == "pawley_lattice"
    ]
    assert len(selected) == 1 and selected[0].key.name == "a_angstrom"
    assert selected[0].bounds.lower == pytest.approx(4.16 * 0.99)
    assert selected[0].bounds.upper == pytest.approx(4.16 * 1.01)
    changed = copy.deepcopy(m)
    changed["datasets"][name]["lattice"]["refine"] = ["alpha_deg"]
    with pytest.raises(ValueError, match="selection"):
        pawley.validate_manifest(changed)
    changed = copy.deepcopy(m)
    changed["datasets"][name]["lattice"]["accepted_ranges"]["a_angstrom"] = [4.2, 4.1]
    with pytest.raises(ValueError, match="interval"):
        pawley.validate_manifest(changed)


def test_matrix_free_manifest_preserves_science_and_validates_iterative_controls():
    m = json.loads(
        (Path(__file__).parents[1] / "validation/pawley-matrix-free-acceptance-v4.json").read_text()
    )
    pawley.validate_manifest(m)
    original = manifest()
    for key, value in original["common"].items():
        assert m["common"][key] == value
    for dataset, gate in original["datasets"].items():
        assert {k: v for k, v in m["datasets"][dataset].items() if k != "lattice"} == gate
    options = pawley._options(m["datasets"]["aps-sucrose-11bmb"], m["common"])
    assert options.solver == "matrix_free" and options.max_linear_iterations == 4000
    for key, value in [
        ("solver", "ignored"),
        ("max_linear_iterations", True),
        ("linear_tolerance", 0.0),
    ]:
        bad = copy.deepcopy(m)
        bad["common"][key] = value
        with pytest.raises(ValueError):
            pawley.validate_manifest(bad)


@pytest.mark.parametrize(
    "module,flag", [("tof_pawley", "--manifest"), ("pawley_spectrum", "--oracle-fixture")]
)
def test_installed_validation_reports_missing_external_asset(
    monkeypatch, tmp_path, capsys, module, flag
):
    import importlib
    import sys

    runner = importlib.import_module(f"phasesmith.validation.{module}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", [module, "--data-root", str(tmp_path), "--output", "report.json"]
    )
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert f"supply {flag} PATH" in capsys.readouterr().err
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("explicit", [False, True])
def test_tof_cli_resolves_manifest_from_caller_location(monkeypatch, tmp_path, explicit):
    import hashlib
    import sys

    from phasesmith.validation import tof_pawley

    source = Path(__file__).parents[1] / "validation/pawley-tof-acceptance-v1.json"
    target = tmp_path / (
        "external.json" if explicit else "validation/pawley-tof-acceptance-v1.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    monkeypatch.chdir(tmp_path)
    argv = ["tof_pawley", "--data-root", str(tmp_path), "--output", "report.json"]
    if explicit:
        argv.extend(["--manifest", str(target)])
    monkeypatch.setattr(sys, "argv", argv)
    seen = []

    def run_case(case, manifest, root):
        seen.append((case["id"], root))
        return {"passed": True}

    monkeypatch.setattr(tof_pawley, "run_case", run_case)
    tof_pawley.main()
    record = json.loads((tmp_path / "report.json").read_text())
    assert record["manifest_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert seen == [(case["id"], tmp_path) for case in record["manifest"]["cases"]]


@pytest.mark.parametrize("explicit", [False, True])
def test_spectrum_cli_resolves_external_fixture(monkeypatch, tmp_path, explicit):
    import sys

    from phasesmith.validation import pawley_spectrum

    fixture = tmp_path / ("external" if explicit else "oracle/fixtures/wavelength_components_v1")
    fixture.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    argv = ["pawley_spectrum", "--data-root", str(tmp_path), "--output", "report.json"]
    if explicit:
        argv.extend(["--oracle-fixture", str(fixture)])
    monkeypatch.setattr(sys, "argv", argv)
    seen = []

    def oracle(path):
        seen.append(path.resolve())
        return []

    monkeypatch.setattr(pawley_spectrum, "oracle_comparison", oracle)
    monkeypatch.setattr(pawley_spectrum, "measured_comparison", lambda root: [])
    pawley_spectrum.main()
    assert seen == [fixture]
    assert (tmp_path / "report.json").is_file()
