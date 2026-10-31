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


def test_raw_loader_preserves_zero_high_counts_reserved_mask_and_orientation(tmp_path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("hdf5plugin")
    path = tmp_path / "raw.h5"
    data = np.arange(24, dtype=np.uint32).reshape(2, 3, 4)
    data[0, 0, :3] = [0, 70000, 2**32 - 2]
    with h5py.File(path, "w") as f:
        f.create_dataset("detector/data", data=data)
        f.create_dataset("detector/detector_information/pixel_size/xsize", data=75e-6)
        f.create_dataset("detector/detector_information/pixel_size/ysize", data=75e-6)
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


def test_selected_shoeboxes_preserve_original_joint_background_and_signed_counts():
    from phasesmith.experimental.profiles import select_observations
    from test_experimental_profiles import observations

    original = observations()
    from dataclasses import replace

    signed = original.summation.values.copy()
    signed[-1, 0] = -20.0
    original = replace(original, summation=replace(original.summation, values=signed))
    ids = np.array([1, 4, 5], np.int64)
    subset = select_observations(original, ids)
    np.testing.assert_array_equal(subset.summation.values, original.summation.values[ids])
    assert subset.summation.values[-1, 0] < 0
    for name in ("centres", "scales", "planar_transforms"):
        np.testing.assert_array_equal(getattr(subset, name), getattr(original, name)[ids])
    for bad in ([1, 1], [5, 1], [-1], [len(original.centres)], []):
        with pytest.raises(ValueError):
            select_observations(original, np.array(bad, dtype=np.int64))


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
