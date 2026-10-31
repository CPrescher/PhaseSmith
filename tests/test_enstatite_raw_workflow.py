"""Raw example I/O conventions, family isolation and joint-domain array adapters."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "enstatite_raw_workflow", ROOT / "benchmarks/enstatite_raw_workflow.py"
)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def test_raw_loader_preserves_zero_high_counts_reserved_mask_and_orientation():
    from phasesmith import _core

    if not hasattr(_core, "NativeHdf5Frames"):
        pytest.skip("bundled HDF5 feature build")
    path = ROOT / "crates/phasesmith-hdf5/tests/data/counts-bitshuffle-lz4.h5"
    data = np.arange(24, dtype=np.uint32).reshape(2, 3, 4)
    data[0, 0, :4] = [0, 70000, 2**32 - 2, 2**32 - 1]
    acquisition = dict(
        first_frame=0,
        last_frame_inclusive=1,
        vertical_flip=True,
        omega_at_original_frame_zero_degrees=-36.0,
        step_degrees=0.5,
        exposure_width_degrees=0.5,
        mask_minimum_reserved_uint32=2**32 - 2,
    )
    loader = workflow.RawFrames(path, acquisition, dict(dataset="detector/data", shape=[2, 3, 4]))
    try:
        first = loader.load(0)
        np.testing.assert_array_equal(first, data[0, ::-1])
        mask = loader.mask(0)
        assert not mask[2, 0] and not mask[2, 1] and mask[2, 2]
        assert first[2, 1] == 70000 and first[2, 0] == 0
        loader.load(1)
        loader.mask(1)
        assert not loader.pending_masks
        assert loader.loads == 2
        np.testing.assert_array_equal(loader.bounds, [[-36, -35.5], [-35.5, -35]])
    finally:
        loader.close()


def test_verified_digest_reuse_preserves_orientation_and_rejects_changed_sources():
    import hashlib

    from phasesmith.experimental.frame_validity import CombinedValidity, FrameValidity

    path = ROOT / "crates/phasesmith-hdf5/tests/data/counts-bitshuffle-lz4.h5"
    acquisition = dict(
        first_frame=0,
        last_frame_inclusive=1,
        vertical_flip=True,
        omega_at_original_frame_zero_degrees=-36.0,
        step_degrees=0.5,
        exposure_width_degrees=0.5,
        mask_minimum_reserved_uint32=2**32 - 2,
    )
    expected = dict(dataset="detector/data", shape=[2, 3, 4])
    with_loader = workflow.RawFrames(path, acquisition, expected)
    try:
        images = [with_loader.load(i) for i in range(2)]
        hashes = [with_loader.hashes[i] for i in range(2)]
    finally:
        with_loader.close()
    provider = FrameValidity(
        tuple(f"{path.name}#{i}" for i in range(2)),
        (3, 4),
        np.zeros((2, 3, 4), bool),
        1,
        provenance=dict(image_sha256=hashes),
    )
    assert provider.verify_frame(0, images[0]) == hashes[0]
    assert hashes[0] == hashlib.sha256(images[0].tobytes()).hexdigest()
    changed = images[0].copy()
    changed[0, 0] += 1
    with pytest.raises(ValueError, match="hash differs"):
        provider.verify_frame(0, changed)
    combined = CombinedValidity((provider, provider))
    for validity in [provider, combined]:
        loader = workflow.RawFrames(
            path, acquisition, expected, validity, execution=dict(reuse_verified_digest=True)
        )
        try:
            for i in range(2):
                assert loader.load(i).tobytes() == images[i].tobytes()
                assert loader.hashes[i] == hashes[i]
                loader.mask(i)
            assert loader.digest_reuses == 2
        finally:
            loader.close()


def test_poni_conversion_has_explicit_pixel_centres_and_axis(tmp_path):
    path = tmp_path / "calibration.poni"
    path.write_text(
        "Detector: Eiger2CdTe_4M\nDistance: 0.15\nPoni1: 0.08\nPoni2: 0.07\n"
        "Rot1: 0\nRot2: 0\nRot3: 0\nWavelength: 2.846e-11\n"
    )
    geometry = workflow.poni_geometry(path, 75e-6, 75e-6, [-1, 0, 0])
    np.testing.assert_allclose(
        geometry.vectors[0], [37.5e-6 - 0.08, 37.5e-6 - 0.07, 0.15], atol=1e-16
    )
    np.testing.assert_array_equal(geometry.vectors[4], [-1, 0, 0])
    assert geometry.wavelength == 0.2846


def test_equivalent_families_stay_together_under_reordering_and_friedel_inversion():
    hkl = np.array([[1, 2, 3], [-1, -2, -3], [-1, 2, -3], [3, 1, 4]])
    rotations = np.array([np.eye(3, dtype=int), np.diag([-1, 1, -1])])
    canonical, groups, folds = workflow.family_partition(hkl, rotations)
    np.testing.assert_array_equal(canonical[:3], np.repeat(canonical[:1], 3, axis=0))
    assert len(set(groups[:3])) == 1 and len(set(folds[:3])) == 1
    again = workflow.family_partition(hkl[::-1], rotations[::-1])
    np.testing.assert_array_equal(canonical, again[0][::-1])
    np.testing.assert_array_equal(folds, again[2][::-1])


def test_cell_prior_aligns_axes_without_coordinates_or_intensities():
    direct = np.diag([5.0, 8.0, 18.0])
    candidate = SimpleNamespace(direct_basis=direct, indexed=np.ones(25, bool))
    index, aligned, cell = workflow.aligned_setting(
        [candidate],
        [8, 5, 18, 90, 90, 90],
        dict(
            minimum_indexed_training=10,
            max_cell_angle_deviation_degrees=0.5,
            cell_metric_relative_tolerance=0.02,
        ),
        np.arange(25) < 20,
    )
    assert index == 0 and np.linalg.det(aligned) > 0
    np.testing.assert_allclose(cell, [8, 5, 18, 90, 90, 90], atol=1e-12)


def test_worker_rejects_reference_fields_before_loading_data():
    with pytest.raises(ValueError, match="unexpected/reference"):
        workflow.validate_worker_input(dict(reference_counts={"Mg": 16}))


def test_flagged_half_pixel_edge_centres_are_explicitly_excluded_from_mesh():
    centres = np.array([[-0.2, 2], [0.0, 2], [9.0, 2], [9.4, 2]])
    flags = np.array([1, 0, 0, 1], np.uint32)
    np.testing.assert_array_equal(
        workflow.interior_target_rows(centres, flags, (8, 10)), [False, True, True, False]
    )
    with pytest.raises(ValueError, match="unflagged reflection center"):
        workflow.interior_target_rows(centres, np.zeros(4, np.uint32), (8, 10))


def test_dac_cone_masks_incident_and_exit_rays_in_sample_frame_with_closed_boundary():
    from phasesmith.experimental.reciprocal import PanelGeometry

    geometry = PanelGeometry([[0, 0, 1], [0.001, 0, 0], [0, 0.001, 0], [0, 0, 1], [0, 1, 0]], 1.0)
    recipe = dict(axis_at_omega_zero=[0, 0, 2], half_opening_degrees=35.0)
    centres = np.array([[1, 0, 0], [1, 0, 35], [1, 0, 35.001], [1, 0, -35]])
    visible, entry, exit_ = workflow.dac_aperture_visibility(centres, geometry, recipe)
    np.testing.assert_array_equal(visible, [True, True, False, False])
    np.testing.assert_allclose(entry, [0, 35, 35.001, 35], atol=2e-13)
    delta = np.rad2deg(np.arctan(0.001))
    np.testing.assert_allclose(exit_, [delta, 35 - delta, 35.001 - delta, 35 + delta], atol=1e-11)
    boundary = np.array([[1000.0, 0, 0], [1000.01, 0, 0]])
    recipe["half_opening_degrees"] = 45.0
    np.testing.assert_array_equal(
        workflow.dac_aperture_visibility(boundary, geometry, recipe)[0], [True, False]
    )
    with pytest.raises(ValueError, match="invalid declared DAC"):
        workflow.dac_aperture_visibility(
            centres, geometry, dict(recipe, axis_at_omega_zero=[0, 0, 0])
        )


def test_search_retains_signed_warning_rows_but_preserves_other_flags_and_visibility():
    flags = np.array([0, 131072, 2, 131074, 16384, 0, 0], np.uint32)
    before = flags.copy()
    values = np.array([[-2, 1], [100, 3], [20, 2], [80, 2], [0, 1], [10, 1], [0, 0]])
    visible = np.array([True, True, True, True, True, False, True])
    np.testing.assert_array_equal(
        workflow.search_observation_mask(flags, values, visible, ["MISMATCH"]),
        [True, True, False, False, False, False, False],
    )
    np.testing.assert_array_equal(flags, before)
    with pytest.raises(ValueError, match="warning policy"):
        workflow.search_observation_mask(flags, values, visible, ["MASKED"])
    with pytest.raises(ValueError, match="measurement arrays"):
        workflow.search_observation_mask(flags, values, np.ones(7), ["MISMATCH"])


def test_selected_shoeboxes_preserve_original_joint_background_and_signed_counts():
    from phasesmith.experimental.profiles import select_observations
    from test_experimental_profiles import observations

    original = observations(covariance=True)
    from dataclasses import replace

    signed = original.summation.values.copy()
    signed[-1, 0] = -20.0
    original = replace(original, summation=replace(original.summation, values=signed))
    ids = np.array([1, 4, 5], np.int64)
    subset = select_observations(original, ids)
    np.testing.assert_array_equal(subset.summation.values, original.summation.values[ids])
    assert subset.summation.values[-1, 0] < 0
    kept_groups = np.flatnonzero(np.isin(original.summation.observation_ids, ids))
    np.testing.assert_array_equal(subset.covariance, original.covariance[kept_groups])
    np.testing.assert_array_equal(
        subset.summation.frame_values, original.summation.frame_values[kept_groups]
    )
    for name in ("centres", "scales", "planar_transforms"):
        np.testing.assert_array_equal(getattr(subset, name), getattr(original, name)[ids])
    for bad in ([1, 1], [5, 1], [-1], [len(original.centres)], []):
        with pytest.raises(ValueError):
            select_observations(original, np.array(bad, dtype=np.int64))
    for bad in (np.array([1.0]), np.array([True]), np.array([[1]])):
        with pytest.raises(ValueError):
            select_observations(original, bad)
    with pytest.raises(TypeError):
        select_observations(None, ids)


def test_joint_profile_collection_retains_explicit_domain_ids_without_changing_counts():
    from phasesmith import _core

    if not hasattr(_core, "experimental_integrate_reduce"):
        pytest.skip("experimental-spots native build required")
    from phasesmith.experimental.integration import IntegrationOptions
    from phasesmith.experimental.prediction import ScanPredictions
    from phasesmith.experimental.profiles import prepare_profiles

    predictions = ScanPredictions(
        np.array([[1, 0, 0], [1, 0, 0]]),
        np.array([[10, 10, 1.5], [20, 20, 1.5]]),
        np.array([1, 1]),
        1.0,
    )
    image = np.ones((32, 32)) * 10
    image[10, 10] = image[20, 20] = 100
    bounds = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    options = IntegrationOptions(
        pixel_radius=1, omega_half_width=1.0, background_margin=2, background_model="plane"
    )
    base = prepare_profiles(lambda _: image, bounds, predictions, options=options)
    joint = prepare_profiles(
        lambda _: image, bounds, predictions, options=options, grain_ids=np.array([0, 1], np.int64)
    )
    np.testing.assert_array_equal(joint.summation.grain_ids, [0, 1])
    np.testing.assert_array_equal(base.summation.values, joint.summation.values)
    np.testing.assert_array_equal(base.samples, joint.samples)
