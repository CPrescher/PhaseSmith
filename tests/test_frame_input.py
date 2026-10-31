import hashlib

import numpy as np
import pytest
from phasesmith.experimental.frame_input import preparation_plan
from phasesmith.experimental.frame_validity import CombinedValidity, DacApertureMask, FrameValidity
from phasesmith.experimental.reciprocal import PanelGeometry


@pytest.mark.parametrize("flip", [False, True])
def test_native_count_hash_and_composed_masks_are_exact_at_all_worker_counts(flip):
    rng = np.random.default_rng(612046)
    raw = rng.integers(0, 100_000, (11, 9, 13), dtype=np.uint32)
    raw[:, 1, :4] = [0, 70000, 2**32 - 2, 2**32 - 1]
    oriented = raw[:, ::-1] if flip else raw
    hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in oriented]
    ids = tuple(f"frame-{i}" for i in range(11))
    bounds = np.column_stack([np.arange(11) - 5, np.arange(11) - 4]).astype(float)
    fixed = np.zeros((9, 13), bool)
    fixed[2, 3] = True
    inferred = FrameValidity(
        ids,
        (9, 13),
        rng.random((11, 3, 4)) < 0.1,
        4,
        fixed_mask=fixed,
        omega_bounds=bounds,
        provenance=dict(image_sha256=hashes),
    )
    geo = PanelGeometry([[0.01, 0, 1], [0.001, 0, 0], [0, 0.001, 0], [0, 0, 1], [0, 1, 0]], 1.0)
    dac = DacApertureMask(
        geo,
        bounds,
        (9, 13),
        axis_at_omega_zero=[0, 0, 1],
        half_opening_degrees=35.0,
        pixel_block=3,
        frame_ids=ids,
        fixed_mask=fixed,
    )
    validity = CombinedValidity((inferred, dac, inferred))
    plan = preparation_plan((9, 13), 11, flip, 2**32 - 2, validity)
    assert plan.source_hash_providers == 2
    for workers in [1, 2, 4, 8, 10]:
        rows = plan.prepare(list(enumerate(raw)), workers, 1024**2)
        assert [row[0] for row in rows] == list(range(11))
        for index, image, mask, digest, error in rows:
            assert error is None
            assert image.dtype == np.uint32
            assert image.tobytes() == oriented[index].tobytes()
            assert digest == hashes[index]
            assert (
                mask.tobytes() == (validity.mask(index) | (oriented[index] >= 2**32 - 2)).tobytes()
            )
        assert raw[:, 1, :4].tolist() == [[0, 70000, 2**32 - 2, 2**32 - 1]] * 11
    changed = raw.copy()
    changed[3, 0, 0] += 1
    rows = plan.prepare(list(enumerate(changed)), 4, 1024**2)
    assert rows[3][1:4] == (None, None, None)
    assert "hash differs" in rows[3][4]
    assert all(row[4] is None for row in rows[:3] + rows[4:])
    with pytest.raises(ValueError, match="budget"):
        plan.prepare(list(enumerate(raw)), 4, 1)
    with pytest.raises(ValueError, match="workers"):
        plan.prepare(list(enumerate(raw)), 0, 1024**2)


def test_native_no_validity_and_source_shape_rejection():
    data = np.arange(12, dtype=np.uint32).reshape(3, 4)
    plan = preparation_plan((3, 4), 1, False, 2**32 - 2)
    row = plan.prepare([(0, data)], 2, 1024**2)[0]
    assert row[3] == hashlib.sha256(data.tobytes()).hexdigest()
    assert not row[2].any()
    with pytest.raises(ValueError, match="shape"):
        plan.prepare([(0, data.reshape(4, 3))], 1, 1024**2)
    row = plan.prepare([(1, data)], 1, 1024**2)[0]
    assert "shape/index" in row[4]
