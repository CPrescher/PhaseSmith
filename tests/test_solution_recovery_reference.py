"""Reference algebra for tensor-aware common-data assessment."""

import gemmi
import numpy as np
import pytest
from phasesmith.single_crystal.reference.solution_recovery_reference import (
    align_stored_frame,
    consensus_limit,
    model_score,
    pullback_tensors,
    symmorphic_intensity_group,
)

from benchmarks.unknown_group_corpus import symmetry_instructions, symmetry_prior_from_instructions


def test_tensor_pullback_keeps_off_diagonal_information():
    u = np.array([[0.02, 0.003, 0.004], [0.003, 0.03, 0.005], [0.004, 0.005, 0.04]])
    rotations = np.array([np.eye(3), np.diag([1, -1, -1])])
    full = np.array([q @ u @ q.T for q in rotations])
    np.testing.assert_allclose(pullback_tensors(full, rotations, [0, 1]), u, rtol=0, atol=1e-15)
    with pytest.raises(ValueError):
        pullback_tensors(full, rotations, [0, 2])
    full[1, 0, 0] = np.nan
    with pytest.raises(ValueError):
        pullback_tensors(full, rotations, [0, 1])


def test_more_parameters_need_a_real_objective_improvement():
    assert model_score(0.01, 500, 60) > model_score(0.01, 500, 20)
    assert model_score(0.001, 500, 60) < model_score(0.01, 500, 20)
    with pytest.raises(ValueError):
        model_score(0.001, 60, 60)


@pytest.mark.parametrize(
    "number, expected",
    [(1, True), (4, False), (7, False), (139, True), (146, True), (167, False), (225, True)],
)
def test_exact_screw_glide_and_known_centring_have_different_contracts(number, expected):
    group = gemmi.find_spacegroup_by_number(number)

    def record(op):
        return {
            "rotation": [[v // 24 for v in r] for r in op.rot],
            "translation": [[v, 24] for v in op.tran],
        }

    assert symmorphic_intensity_group([record(op) for op in group.operations()]) == expected
    prior = symmetry_prior_from_instructions(
        symmetry_instructions(
            {
                "space_group": group.xhm(),
                "cell": [8, 8, 8, 90, 90, 90],
                "wavelength": 0.71,
                "elements": ["Si", "O"],
            }
        )
    )
    processing = [
        {"rotation": r, "translation": t}
        for r in prior["laue_rotations"]
        for t in prior["centring_translations"]
    ]
    assert symmorphic_intensity_group(processing)


@pytest.mark.parametrize("hand", [-1, 1])
def test_saved_frame_is_a_coordinate_proposal_with_explicit_handedness(hand):
    xyz = np.array([[0.123, 0.247, 0.137], [0.323, 0.431, 0.251]])
    shift = np.array([0.213, 0.167, 0.371])
    source = np.mod(hand * (xyz - shift), 1)
    np.testing.assert_allclose(align_stored_frame(source, hand, shift), xyz, rtol=0, atol=1e-15)
    with pytest.raises(ValueError):
        align_stored_frame(source, 0, shift)


def test_target_can_only_tighten_consensus_stopping():
    assert consensus_limit(0.06) == 0.1
    assert consensus_limit(0.06, native_policy=True) == 0.06
    assert consensus_limit(0.2, native_policy=True) == 0.1
