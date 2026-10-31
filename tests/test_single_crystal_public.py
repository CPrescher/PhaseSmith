"""Supported namespace, installed capabilities and pre-0.8 migration contracts."""

import importlib
import inspect
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from phasesmith import _core, single_crystal
from phasesmith.single_crystal.search import ElementSearchConfig, search_structure_elements
from phasesmith.single_crystal.solution import (
    SolutionOptions,
    solve_structure_elements,
    upgrade_solution_record,
)

ROOT = Path(__file__).resolve().parents[1]


def test_supported_native_features_are_present_in_ordinary_builds():
    # No skip: CI and release wheels must actually contain these capabilities.
    for name in [
        "experimental_spots",
        "experimental_geometry_fit",
        "experimental_profile_refine",
        "experimental_empirical_fit",
        "_solve_structure",
        "_SharedOrbitPool",
        "NativeHdf5Frames",
        "NativeFramePreparation",
    ]:
        assert hasattr(_core, name), name


def test_retired_experimental_imports_are_absent():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("phasesmith.experimental.spots")
    import pickle

    from phasesmith.single_crystal.spots import SpotOptions

    assert isinstance(pickle.loads(pickle.dumps(SpotOptions())), SpotOptions)


def test_supported_solution_cli():
    completed = subprocess.run(
        [sys.executable, "-m", "phasesmith.single_crystal.solution", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--composition" in completed.stdout and "--output" in completed.stdout


def test_public_namespace_imports_without_optional_numeric_packages():
    subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/smoke_single_crystal.py",
            "--hdf5-fixture",
            ROOT / "crates/phasesmith-hdf5/tests/data/counts-bitshuffle-lz4.h5",
        ],
        check=True,
    )


def test_explicit_exports_and_shared_types():
    from phasesmith.crystallography import UnitCell
    from phasesmith.single_crystal import solution

    for name in single_crystal.__all__:
        module = getattr(single_crystal, name)
        assert isinstance(module.__all__, list)
        assert len(module.__all__) == len(set(module.__all__))
        assert all(hasattr(module, symbol) for symbol in module.__all__)
    assert solution.UnitCell is UnitCell
    assert "main" not in single_crystal.solution.__all__
    assert "_run_solution" not in single_crystal.solution.__all__


def test_named_configuration_matches_the_frozen_expansion_and_records_options():
    frozen = json.loads((ROOT / "validation/solution-holdout-freeze-v1.json").read_text())
    signature = inspect.signature(solve_structure_elements).parameters
    expected = {k: v for k, v in frozen["controls"].items() if k in signature}
    expected.update(symmetry_recycling=True, tangent_refinement=True)
    config = ElementSearchConfig.validated_v1()
    assert {k: v for k, v in config.controls if k in expected} == expected
    assert all(
        k in dict(config.controls)
        for k in signature
        if k
        not in {
            "observations",
            "cell",
            "elements",
            "options",
            "space_group",
            "maximum_atoms",
            "cancellation",
        }
    )
    assert config.to_record()["options"] == asdict(config.options)
    assert "shelxt" not in json.dumps(config.to_record())
    with pytest.raises(ValueError, match="frozen expansion"):
        ElementSearchConfig(config.name, config.options, config.controls[:-1])
    with pytest.raises(TypeError):
        search_structure_elements(
            None, cell=None, elements=["Fe"], configuration=config, phase_origin_selection=False
        )


def test_named_configuration_replays_explicit_search_bitwise():
    from test_solution_staged_refinement import synthetic

    cell, group, data, _, _ = synthetic("P-1")
    options = SolutionOptions(
        phasing="dual_space", trials=1, cycles=64, max_seconds=15, completion=False
    )
    config = ElementSearchConfig.validated_v1(options=options)
    direct = solve_structure_elements(
        data,
        cell=cell,
        space_group=group,
        elements=["Fe", "O"],
        maximum_atoms=16,
        options=options,
        **dict(config.controls),
    )
    named = search_structure_elements(
        data,
        cell=cell,
        space_group=group,
        elements=["Fe", "O"],
        maximum_atoms=16,
        configuration=config,
    )
    assert [c.to_record() for c in named.candidates] == [c.to_record() for c in direct.candidates]
    for a, b in zip(named.histories, direct.histories, strict=True):
        np.testing.assert_array_equal(a, b)
    assert named.termination == direct.termination
    assert named.to_record()["search_configuration"] == config.to_record()
    assert "search_configuration" not in named.to_record(legacy=True)
    assert named.to_record(legacy=True)["schema"] == "phasesmith.experimental-solution.v5"


@pytest.mark.parametrize("version", ["v4", "v5"])
def test_solution_summary_migration_is_copy_only_and_preserves_candidates(version):
    source = dict(
        schema="phasesmith.experimental-solution." + version,
        cell=[6, 7, 8, 90, 90, 90],
        candidates=[dict(r1=0.1, fractional_xyz=[[0.1, 0.2, 0.3]])],
        termination="search_budget",
        experimental=True,
    )
    if version == "v5":
        source["shared_search"] = {"elements": ["Fe"]}
    original = json.dumps(source)
    result = upgrade_solution_record(source)
    assert json.dumps(source) == original
    assert result["schema"] == "phasesmith.single-crystal-solution.v1"
    assert result["source_schema"].endswith(version)
    assert result["experimental"] is False
    assert result["candidates"] == source["candidates"]
    assert upgrade_solution_record(result) == result
    result["candidates"][0]["r1"] = 0
    assert source["candidates"][0]["r1"] == 0.1
    with pytest.raises(ValueError, match="unsupported"):
        upgrade_solution_record(dict(source, schema="unknown"))


def test_named_configuration_is_independent_of_future_solver_defaults(monkeypatch):
    from phasesmith.single_crystal import search

    original = search.ElementSearchConfig.validated_v1().to_record()
    signature = inspect.signature(search.solve_structure_elements)
    changed = signature.replace(
        parameters=[
            p.replace(default=999) if p.name == "beam_width" else p
            for p in signature.parameters.values()
        ]
    )

    def future_solver(*args, **kwargs):
        raise AssertionError("this test only constructs configuration metadata")

    future_solver.__signature__ = changed
    monkeypatch.setattr(search, "solve_structure_elements", future_solver)
    assert search.ElementSearchConfig.validated_v1().to_record() == original


@pytest.mark.parametrize("key,value", [("workers", True), ("cached_refinement", 1)])
def test_named_configuration_rejects_equal_but_wrong_scalar_types(key, value):
    config = ElementSearchConfig.validated_v1()
    controls = tuple((k, value if k == key else v) for k, v in config.controls)
    with pytest.raises(ValueError, match="frozen expansion"):
        ElementSearchConfig(config.name, config.options, controls)
