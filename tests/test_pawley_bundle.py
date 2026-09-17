"""Shared-bundle migration and explicit Python/native Pawley continuation."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from phasesmith import ConstantWavelengthInstrument, PowderPattern
from phasesmith.project_bundle import ProjectBundle
from phasesmith.radiation import RadiationProbe
from phasesmith.refinement.pawley import (
    PawleyInput,
    PawleyOptions,
    PawleyPhase,
    PawleyProject,
    calculate,
)


def project():
    x = np.linspace(39.0, 41.0, 301)
    instrument = ConstantWavelengthInstrument(1.54, 0.0, 0.0, 0.001, 0.002, 0.0)
    phase = PawleyPhase("cell-only", ("a", "b"), [40.0, 40.07], [3.0, 7.0])
    truth = PawleyInput(PowderPattern(x), instrument, (phase,))
    request = replace(
        truth,
        pattern=PowderPattern(x, observed_y=calculate(truth).calculated_y),
        phases=(replace(phase, intensities=np.zeros(2)),),
        parameters=None,
    )
    return PawleyProject(request, PawleyOptions(solver="matrix_free"))


def test_shared_pawley_bundle_and_explicit_native_exchange(tmp_path):
    p = project()
    p.refine(max_iterations=1)
    bundle = ProjectBundle.from_pawley(p, probe=RadiationProbe.NEUTRON, histogram_id="measured")
    source = tmp_path / "python.bundle"
    bundle.save(source)
    assert bundle.pawley_histograms == ("measured",)
    assert bundle.analysis_counts["pawley"] == 1
    wire = json.loads((source / "manifest.json").read_text())
    assert wire["format_version"] == 6
    assert wire["project"]["phases"] == []  # No fictitious atom-bearing phases.
    assert "x_deg" not in wire["pawley_analyses"][0]
    loaded = ProjectBundle.load(source)
    python_fit = loaded.pawley("measured").refine()
    native_path = tmp_path / "native.bundle"
    subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "phasesmith-persistence",
            "--example",
            "pawley_bundle_exchange",
            "--",
            str(source),
            str(native_path),
        ],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    native = ProjectBundle.load(native_path).pawley("measured")
    np.testing.assert_array_equal(
        native.calculate().calculated_y, python_fit.calculation.calculated_y
    )
    resumed = native.refine()
    np.testing.assert_array_equal(resumed.history, python_fit.history)
    np.testing.assert_array_equal(resumed.intensities, python_fit.intensities)
    updated = loaded.with_pawley("measured", native)
    updated.save(tmp_path / "updated.bundle")
    with pytest.raises(ValueError, match="exist"):
        updated.save(source)
    (source / "user-notes.txt").write_text("preserve me")
    updated.save(source, overwrite=True)
    assert (source / "user-notes.txt").read_text() == "preserve me"
    with pytest.raises(ValueError, match="unknown"):
        loaded.with_pawley("missing", p)
    with pytest.raises(TypeError, match="probe"):
        ProjectBundle.from_pawley(p, probe="neutron")


def test_bundle_stale_shared_state_corruption_and_legacy_migration(tmp_path):
    p = project()
    p.refine(max_iterations=1)
    bundle = ProjectBundle.from_pawley(p, probe=RadiationProbe.X_RAY)
    stale = replace(
        p.input,
        pattern=PowderPattern(p.input.pattern.x, observed_y=p.input.pattern.observed_y + 1.0),
    )
    with pytest.raises(ValueError, match="shared histogram"):
        bundle.with_pawley("histogram", PawleyProject(stale, p.options))
    path = tmp_path / "bundle"
    bundle.save(path)
    manifest = path / "manifest.json"
    original = json.loads(manifest.read_text())
    corrupted = json.loads(manifest.read_text())
    corrupted["pawley_analyses"][0]["checkpoint"]["free"][0] += 0.1
    manifest.write_text(json.dumps(corrupted))
    # Saved free states are validated against the full objective on resume.
    with pytest.raises(ValueError, match="objective"):
        ProjectBundle.load(path).pawley("histogram").refine()
    original["format_version"] = 999
    manifest.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="unsupported"):
        ProjectBundle.load(path)
    original["format_version"] = 5
    original.pop("pawley_analyses")
    manifest.write_text(json.dumps(original))
    legacy = ProjectBundle.load(path)
    assert legacy.pawley_histograms == ()
    legacy.save(tmp_path / "migrated")
    assert json.loads((tmp_path / "migrated/manifest.json").read_text())["format_version"] == 6
