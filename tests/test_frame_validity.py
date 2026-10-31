"""Original-count shadow inference, exposure geometry and masked measurement support."""

import json

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.frame_validity import (
    CombinedValidity,
    DacApertureMask,
    FrameValidity,
    SeatShadowOptions,
    SeatShadowUnavailable,
    _dot_range,
    estimate_seat_shadows,
    flag_invalid_tracks,
)
from phasesmith.experimental.reciprocal import PanelGeometry


def test_snapshot_ownership_native_orientation_tiles_and_precise_fixed_mask(tmp_path):
    coarse = np.zeros((3, 2, 3), bool)
    coarse[1, 0, 1] = True
    fixed = np.zeros((3, 5), bool)
    fixed[2, 4] = True
    provenance = {"source": ["original"]}
    plan = FrameValidity(
        ("a", "b", "c"), fixed.shape, coarse, 2, fixed_mask=fixed, provenance=provenance
    )
    coarse[:] = True
    fixed[:] = True
    provenance["source"][0] = "changed"
    plan.provenance["source"][0] = "changed again"
    assert plan.provenance == {"source": ["original"]}
    assert not plan.excluded.flags.writeable and not plan.fixed_mask.flags.writeable
    expected = np.zeros((3, 5), bool)
    expected[:2, 2:4] = True
    expected[2, 4] = True
    np.testing.assert_array_equal(plan.mask(1), expected)
    mask = plan.mask(1)
    mask[:] = False
    np.testing.assert_array_equal(plan.mask(1), expected)
    np.testing.assert_array_equal(
        plan.rows(0, 15), np.array([plan.mask(i).ravel() for i in range(3)]).T
    )
    np.testing.assert_array_equal(
        plan.boxes_valid(1, np.array([[0, 0, 2, 2], [1, 0, 3, 2], [4, 2, 5, 3], [-1, 0, 1, 1]])),
        [True, False, False, False],
    )
    np.testing.assert_array_equal(
        plan.valid_points(np.array([1, 1, 0, -1]), np.array([[2, 0], [1, 0], [4, 2], [0, 0]])),
        [False, True, False, False],
    )
    path = tmp_path / "mask.npz"
    plan.save(path)
    loaded = FrameValidity.load(path, frame_ids=("a", "b", "c"), shape=(3, 5))
    assert loaded.fingerprint() == plan.fingerprint()
    with pytest.raises(FileExistsError):
        plan.save(path)
    with pytest.raises(ValueError, match="ordered frames"):
        FrameValidity.load(path, frame_ids=("b", "a", "c"))


@pytest.mark.parametrize("kind", ["pixels", "ids", "semantics"])
def test_archive_corruption_and_identity_changes_are_rejected(tmp_path, kind):
    path = tmp_path / "a.npz"
    plan = FrameValidity(("a",), (2, 2), np.zeros((1, 2, 2), bool))
    plan.save(path)
    with np.load(path, allow_pickle=False) as z:
        arrays = dict(z)
    if kind == "pixels":
        arrays["excluded"][0, 0, 0] = True
    else:
        meta = json.loads(arrays["metadata"].tobytes())
        meta["frame_ids" if kind == "ids" else "semantics"] = (
            ["different"] if kind == "ids" else "zero means unavailable"
        )
        arrays["metadata"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez_compressed(tmp_path / "bad.npz", **arrays)
    with pytest.raises(ValueError, match="mismatch"):
        FrameValidity.load(tmp_path / "bad.npz")


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_block_rejected(bad):
    with pytest.raises(ValueError):
        FrameValidity(("a",), (2, 2), np.zeros((1, 2, 2), bool), block=bad)


def scene():
    images = np.full((20, 192, 256), 80, np.uint16)
    images[:5, :, 128:] = 15
    images[15:19, :, :80] = 15
    images[0, 90:93, 180:183] = 5000
    images[19] = 10
    return images


def test_moving_seat_shadows_keep_spots_counts_and_uniform_dimming_distinct():
    pytest.importorskip("scipy")
    images = scene()
    before = images.copy()
    plan = estimate_seat_shadows(
        lambda i: images[i], len(images), options=SeatShadowOptions(7, 11, guard_pixels=16)
    )
    assert plan.mask(0)[95, 180] and not plan.mask(0)[95, 40]
    assert not plan.mask(10).any()
    assert plan.mask(17)[95, 30] and not plan.mask(17)[95, 220]
    assert not plan.mask(19).any()
    assert plan.provenance["globally_low_signal_frames"] == [19]
    np.testing.assert_array_equal(images, before)


def test_partial_shadow_on_uniformly_dim_frame_keeps_illuminated_region():
    pytest.importorskip("scipy")
    images = np.full((20, 192, 256), 80, np.uint16)
    images[-1] = 20
    images[-1, :, :80] = 3
    images[-1, 90:93, 30:33] = images[-1, 90:93, 210:213] = 5000
    plan = estimate_seat_shadows(
        lambda i: images[i], 20, options=SeatShadowOptions(7, 11, guard_pixels=16)
    )
    assert plan.mask(19)[40:150, :64].all()
    assert not plan.mask(19)[40:150, 128:].any()


def test_wholly_dark_unknown_frame_is_not_silently_a_valid_zero_or_an_inferred_shadow():
    pytest.importorskip("scipy")
    images = np.full((5, 32, 32), 80, np.uint16)
    images[0] = 0
    with pytest.raises(SeatShadowUnavailable, match="unknown visibility"):
        estimate_seat_shadows(lambda i: images[i], 5)
    # Explicitly known unavailable pixels can be represented without guessing
    # why the exposure was dark. Their exclusion survives in the snapshot.
    plan = estimate_seat_shadows(
        lambda i: images[i], 5, mask_loader=lambda i: np.full((32, 32), i == 0)
    )
    assert plan.mask(0).all() and not plan.mask(1).any()
    assert plan.provenance["known_unavailable_frames"] == [0]


def test_mask_estimation_source_order_reused_buffers_workers_and_callbacks_are_identical():
    pytest.importorskip("scipy")
    images = scene()
    masks = []
    for workers in [1, 4]:
        buffer = np.empty_like(images[0])
        seen, progress = [], []

        def load(i, seen=seen, buffer=buffer):
            seen.append(i)
            buffer[:] = images[i]
            return buffer

        plan = estimate_seat_shadows(
            load,
            20,
            options=SeatShadowOptions(7, 11),
            workers=workers,
            progress=lambda i, n, progress=progress: progress.append((i, n)),
        )
        assert seen == list(range(20)) and progress == [(i, 20) for i in range(21)]
        masks.append(plan)
    assert masks[0].fingerprint() == masks[1].fingerprint()
    masks[0].verify_frame(0, images[0])
    modified = images[0].copy()
    modified[0, 0] += 1
    with pytest.raises(ValueError, match="source frame hash"):
        masks[0].verify_frame(0, modified)


def test_invalid_pixels_partial_blocks_and_missing_illumination_are_explicit():
    pytest.importorskip("scipy")
    image = np.full((193, 257), 80.0)
    image[0, 0] = np.nan
    fixed = np.zeros(image.shape, bool)
    fixed[10, 10] = True
    plan = estimate_seat_shadows(lambda _: image, 5, fixed_mask=fixed)
    assert plan.mask(0)[10, 10] and plan.mask(0).shape == image.shape
    assert not plan.excluded.any()
    with pytest.raises(SeatShadowUnavailable, match="40 counts"):
        estimate_seat_shadows(lambda _: np.ones((32, 32)), 5)
    with pytest.raises(SeatShadowUnavailable, match="five frames"):
        estimate_seat_shadows(lambda _: image, 4)
    with pytest.raises(ValueError, match="nonnegative"):
        estimate_seat_shadows(lambda _: np.full((32, 32), -1.0), 5)
    with pytest.raises(ValueError, match="threshold"):
        SeatShadowOptions(threshold=True)


def test_optional_scipy_error_is_actionable_and_snapshot_access_needs_no_scipy(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        if name == "scipy":
            raise ModuleNotFoundError("simulated optional dependency absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scipy)
    plan = FrameValidity(("a",), (2, 2), np.zeros((1, 2, 2), bool))
    assert not plan.mask(0).any()
    with pytest.raises(ImportError, match=r"phasesmith\[masking\]"):
        estimate_seat_shadows(lambda _: pytest.fail("must not read frames"), 5)


def geometry(origin=(0.01, 0, 1), pixel=1e-5):
    return PanelGeometry([origin, [pixel, 0, 0], [0, pixel, 0], [0, 0, 1], [0, 1, 0]], 1.0)


def test_geometric_masks_use_whole_exposure_not_midpoint_and_reverse_scan_is_equal():
    good = DacApertureMask(
        geometry(),
        np.array([[34.0, 35.0]]),
        (1, 1),
        axis_at_omega_zero=np.array([0, 0, 1]),
        half_opening_degrees=35,
    )
    assert not good.mask(0).any()  # closed incident-angle boundary
    bad = DacApertureMask(
        geometry(),
        np.array([[34.0, 36.0]]),
        (1, 1),
        axis_at_omega_zero=np.array([0, 0, 1]),
        half_opening_degrees=35,
    )
    assert bad.mask(0).all()  # midpoint at 35 would pass; end does not
    reverse = DacApertureMask(
        geometry(),
        np.array([[36.0, 34.0]]),
        (1, 1),
        axis_at_omega_zero=np.array([0, 0, 1]),
        half_opening_degrees=35,
    )
    np.testing.assert_array_equal(bad.mask(0), reverse.mask(0))


def test_geometric_mask_checks_pixel_corners_and_tiling_is_unobservable():
    provider = DacApertureMask(
        geometry((0.08, 0, 1), 0.02),
        np.array([[0.0, 0.01]]),
        (1, 1),
        axis_at_omega_zero=np.array([0, 0, 1]),
        half_opening_degrees=5,
    )
    assert np.rad2deg(np.arctan(0.08)) < 5
    assert provider.mask(0).all()  # outer pixel corner is beyond aperture
    kw = dict(axis_at_omega_zero=np.array([0, 0, 1]), half_opening_degrees=10)
    a = DacApertureMask(
        geometry((-0.02, -0.01, 1), 0.005), np.array([[5.0, 6.0]]), (11, 17), tile_rows=1, **kw
    )
    b = DacApertureMask(a.geometry, a.omega_bounds, a.shape, tile_rows=8, **kw)
    assert a.fingerprint() == b.fingerprint()
    np.testing.assert_array_equal(a.mask(0), b.mask(0))


def test_geometric_block_masks_are_conservative_and_include_partial_edge_blocks():
    kw = dict(axis_at_omega_zero=np.array([0, 0, 1]), half_opening_degrees=15)
    geo = geometry((-0.05, -0.04, 0.15), 0.005)
    fine = DacApertureMask(geo, np.array([[-5.0, 5.0]]), (23, 31), **kw)
    coarse = DacApertureMask(geo, fine.omega_bounds, fine.shape, pixel_block=4, **kw)
    a, b = fine.mask(0), coarse.mask(0)
    assert a.any() and not a.all()
    assert not np.any(a & ~b)
    assert b.sum() >= a.sum()
    assert fine.fingerprint() != coarse.fingerprint()
    boxes = np.array([[0, 0, 4, 4], [12, 8, 16, 12], [28, 20, 31, 23]])
    np.testing.assert_array_equal(
        coarse.boxes_valid(0, boxes),
        [not b[y0:y1, x0:x1].any() for x0, y0, x1, y1 in boxes],
    )


def test_angular_extrema_match_independent_rodrigues_sampling_including_stationary_points():
    rng = np.random.default_rng(35061)
    rays = rng.normal(size=(7, 3))
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    normal = rng.normal(size=3)
    normal /= np.linalg.norm(normal)
    for lo, hi in [(-0.5, 1.7), (-7.0, 8.0), (5.1, 5.9)]:
        low, high = _dot_range(rays, axis, normal, lo, hi)
        angles = np.linspace(lo, hi, 20001)
        rotated = (
            np.cos(angles)[:, None] * normal
            + np.sin(angles)[:, None] * np.cross(axis, normal)
            + (1 - np.cos(angles))[:, None] * (axis @ normal) * axis
        )
        dots = rays @ rotated.T
        assert np.all(low <= dots.min(axis=1) + 2e-15)
        assert np.all(high >= dots.max(axis=1) - 2e-15)
        np.testing.assert_allclose(low, dots.min(axis=1), atol=1e-7, rtol=0)
        np.testing.assert_allclose(high, dots.max(axis=1), atol=1e-7, rtol=0)


def test_mask_composition_and_exposure_identity():
    bounds = np.array([[0.0, 1.0], [1.0, 2.0]])
    a = FrameValidity(("a", "b"), (4, 4), np.zeros((2, 2, 2), bool), 2, omega_bounds=bounds)
    coarse = a.excluded.copy()
    coarse[0, 0, 0] = True
    b = FrameValidity(a.frame_ids, a.shape, coarse, 2)
    both = CombinedValidity((a, b))
    np.testing.assert_array_equal(both.mask(0), b.mask(0))
    assert not both.boxes_valid(0, np.array([[0, 0, 1, 1]]))[0]
    with pytest.raises(ValueError, match="angles"):
        CombinedValidity(
            (a, FrameValidity(a.frame_ids, a.shape, coarse, 2, omega_bounds=bounds + 1))
        )


def test_track_neighbor_mask_is_flagged_without_deleting_counts_or_crossing_scan_gaps():
    from phasesmith.experimental.rotation import PositionTable

    bounds = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [5.0, 6.0]])
    table = PositionTable(
        np.array([[1, 1, 3, 3], [4, 4, 6, 6]]),
        np.array([[1, 2], [2, 3]]),
        np.array([[2, 2, 1.5, 0, 1], [5, 5, 2.5, 0, 1]]),
        np.zeros(2, np.uint32),
        np.array([[0, 1, 2, 2, 1.5], [1, 2, 5, 5, 2.5]]),
        bounds,
        ("a", "b", "c", "d"),
    )
    masks = np.zeros((4, 8, 8), bool)
    masks[0, 1, 1] = True
    masks[3, 5, 5] = True
    validity = FrameValidity(table.frame_hashes, (8, 8), masks, omega_bounds=bounds)
    result = flag_invalid_tracks(table, validity)
    np.testing.assert_array_equal(result.flags, [2, 0])
    np.testing.assert_array_equal(result.values, table.values)
    np.testing.assert_array_equal(result.observations, table.observations)


@pytest.mark.skipif(not hasattr(_core, "ExperimentalArPLSEngine"), reason="native omega build")
def test_shared_pixel_tiles_exclude_shadow_samples_from_temporal_background_fits():
    from phasesmith.experimental.omega_background import arpls_masked_rows

    moving = np.zeros((36, 2, 3), bool)
    moving[12:20, 0, 1] = True
    validity = FrameValidity(tuple(map(str, range(36))), (2, 3), moving)
    mask = validity.rows(0, 6)
    rows = np.random.default_rng(28).normal(80, 1, (6, 36))
    first = arpls_masked_rows(rows, mask)
    changed = rows.copy()
    changed[mask] = 1e12
    second = arpls_masked_rows(changed, mask)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    assert np.isnan(first[0][mask]).all()
    assert np.isfinite(first[0][0]).all()


@pytest.mark.skipif(
    not hasattr(_core, "experimental_profile_fit"), reason="experimental profiles build"
)
def test_masked_rocking_tail_is_not_a_valid_zero_or_a_profile_teacher():
    from phasesmith.experimental.integration import IntegrationOptions
    from phasesmith.experimental.prediction import ScanPredictions
    from phasesmith.experimental.profiles import (
        ProfileOptions,
        fit_profiles,
        learn_profiles,
        prepare_profiles,
    )

    bounds = np.c_[np.arange(5.0), np.arange(1.0, 6.0)]
    hkl = np.c_[np.arange(1, 7), np.zeros((6, 2), int)]
    centres = np.c_[np.arange(6) * 12 + 10, np.full(6, 16.0), np.full(6, 2.5)]
    predictions = ScanPredictions(hkl, centres, np.full(6, 2), 1.0)
    yy, xx = np.indices((32, 84))
    images = []
    for f in range(5):
        image = np.full((32, 84), 80.0)
        for x, y, _ in centres:
            image += (
                5000
                * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2))
                * np.exp(-0.5 * ((f - 2) / 0.7) ** 2)
            )
        images.append(image)
    masks = np.zeros((5, 32, 84), bool)
    masks[1, :, 7:14] = True
    validity = FrameValidity(tuple(map(str, range(5))), (32, 84), masks, omega_bounds=bounds)
    obs = prepare_profiles(
        lambda i: images[i],
        bounds,
        predictions,
        mask_loader=validity.mask,
        options=IntegrationOptions(
            pixel_radius=3, omega_half_width=2.4, background_model="plane", background_margin=3
        ),
    )
    options = ProfileOptions(
        shape=(7, 7, 5),
        step=(1.0, 1.0, 1.0),
        regions=(1, 1, 1),
        min_references=2,
        reference_snr=20.0,
        cutoff=0.0,
        iterations=16,
    )
    bank = learn_profiles(obs, np.ones(6, bool), detector_shape=(32, 84), options=options)
    assert 0 not in bank.reference_ids
    fit = fit_profiles(obs, bank)
    assert fit.flags[0] & 2
    assert np.isfinite(fit.values[1:, 0]).all()
    # Pixel values in a shadow cannot teach the plane or enter an amplitude.
    changed = [a.copy() for a in images]
    changed[1][:, 7:14] = 1e8
    other = prepare_profiles(
        lambda i: changed[i],
        bounds,
        predictions,
        mask_loader=validity.mask,
        options=obs.summation.options,
    )
    bank2 = learn_profiles(other, np.ones(6, bool), detector_shape=(32, 84), options=options)
    np.testing.assert_array_equal(bank.profiles, bank2.profiles)
    np.testing.assert_array_equal(fit.values, fit_profiles(other, bank2).values)
