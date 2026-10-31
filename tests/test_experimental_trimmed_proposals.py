"""Proposal-only robustness; reserved validation and tight-core gates stay fixed."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.grains import (
    GrainOwnershipOptions,
    GrainProposalRefinementOptions,
    GrainSearchOptions,
    _polish_proposals,
    _trimmed_proposal_fits,
)

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "ExperimentalLatticeEngine"), reason="experimental-spots feature not built"
)


@pytest.mark.parametrize(
    "fractions", [[], (0.2,), (1.0,), (np.nan,), (True,), (None,), (0.8, 0.8), (0.9, 0.8, 0.7, 0.6)]
)
def test_trim_fractions_validate(fractions):
    with pytest.raises(ValueError):
        GrainProposalRefinementOptions(trim_fractions=fractions)


def cloud():
    from phasesmith.experimental.prediction import predict_spots
    from phasesmith.experimental.reciprocal import PanelGeometry, reciprocal_vectors

    rng = np.random.default_rng(643)
    geometry = PanelGeometry.from_poni(
        distance=0.2,
        poni1=0.08,
        poni2=0.08,
        pixel1=8e-5,
        pixel2=8e-5,
        rot1=0.01,
        rot2=-0.03,
        rot3=0.05,
        wavelength=0.4,
        axis=[-0.97, 0.12, 0.2],
    )
    a = np.array([[4.7, 0, 0], [0.9, 6.3, 0], [-0.4, 0.2, 8.1]])
    a = a @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
    h = rng.integers(-5, 6, (300, 3))
    pred = predict_spots(h, np.linalg.inv(a).T, geometry, rng.uniform(-200, -30, len(h)))
    keep = pred.flags == 0
    obs = pred.values[keep] + rng.normal(size=(keep.sum(), 3)) * [0.12, 0.12, 0.025]
    return reciprocal_vectors(obs, geometry), obs, geometry, a, h[keep]


def test_trim_fits_are_training_only_bounded_deterministic_and_pool_order_independent():
    q, obs, g, a, h = cloud()
    # Physical-coordinate outliers remain within the initial broad fit gates.
    altered = obs.copy()
    altered[:8] += [4.0, -3.0, 0.6]
    variants, failures = _trimmed_proposal_fits(altered, h, a, g, (0.9, 0.8, 0.7))
    assert not failures and len(variants) == 4
    for fraction, rows, fit in variants:
        assert len(rows) == max(12, int(np.ceil(fraction * len(obs))))
        assert fit.fit_mask.sum() == len(rows)
        assert not hasattr(fit, "heldout_rms")
    repeat, _ = _trimmed_proposal_fits(altered, h, a, g, (0.9, 0.8, 0.7))
    for (_, rows, fit), (_, other, fitted) in zip(variants, repeat, strict=True):
        np.testing.assert_array_equal(rows, other)
        np.testing.assert_array_equal(fit.reciprocal_basis, fitted.reciprocal_basis)
    true = np.linalg.inv(a).T
    errors = [np.linalg.norm(fit.reciprocal_basis - true) for _, _, fit in variants]
    assert min(errors[1:]) < 0.4 * errors[0]
    options = GrainSearchOptions()
    polishing = GrainProposalRefinementOptions(trim_fractions=(0.9, 0.8, 0.7))
    engine = _core.ExperimentalLatticeEngine(1)
    models = [a.copy(), a * 1.001]
    first, audit = _polish_proposals(engine, q, obs, g, models, options, polishing, None)
    second, reverse_audit = _polish_proposals(
        engine, q, obs, g, list(reversed(models)), options, polishing, None
    )
    for candidate in first[audit["legacy_proposed"] :]:
        assert any(
            np.max(abs(candidate - other)) < 1e-10
            for other in second[reverse_audit["legacy_proposed"] :]
        )
    assert first and all(x["trimmed_status"] == "proposals" for x in audit["candidates"])
    legacy, _ = _polish_proposals(
        engine, q, obs, g, models, options, GrainProposalRefinementOptions(), None
    )
    assert audit["legacy_proposed"] == len(legacy)
    for old, new in zip(legacy, first, strict=False):
        np.testing.assert_array_equal(old, new)
    assert options.min_indexed == 20
    assert polishing.min_points == 18
    assert GrainOwnershipOptions().preserve_input_models is False
    with pytest.raises(ValueError):
        GrainOwnershipOptions(preserve_input_models=1)


def test_one_failed_trim_preserves_the_full_fit_and_other_variants(monkeypatch):
    import phasesmith.experimental.prediction as prediction

    _, obs, g, a, h = cloud()
    original = prediction.fit_crystal_proposal
    calls = []

    def fit(*args, **kw):
        calls.append(len(args[0]))
        if len(calls) == 3:
            raise ValueError("controlled rank failure")
        return original(*args, **kw)

    monkeypatch.setattr(prediction, "fit_crystal_proposal", fit)
    results, failures = _trimmed_proposal_fits(obs, h, a, g, (0.9, 0.8, 0.7))
    assert [x[0] for x in results] == [1.0, 0.9, 0.7]
    assert failures == [dict(retained_fraction=0.8, reason="controlled rank failure")]


def test_preserved_eligible_input_survives_a_bad_geometry_variant(monkeypatch):
    from dataclasses import replace

    import phasesmith.experimental.ownership as ownership

    q, obs, g, a, _ = cloud()
    options = GrainSearchOptions()
    charts, valid = ownership.observation_charts(obs, g)

    def damaging_fit(engine, points, observed, geometry, basis, search, settings, seed, **kw):
        result = basis if kw.get("all_training") else basis * 1.2
        return result, dict(status="controlled", training_rows=list(range(len(points))))

    monkeypatch.setattr(ownership, "_fit_one", damaging_fit)
    plain = GrainOwnershipOptions(alias_rounds=0, refinement_cycles=1)
    engine = _core.ExperimentalLatticeEngine(1)
    lost, _, _ = ownership.refine_ownership_pool(
        engine,
        q,
        obs,
        g,
        [a],
        charts,
        valid,
        options,
        plain,
        None,
        _preserve_reciprocal_branch=False,
    )
    assert lost == []
    kept, _, audit = ownership.refine_ownership_pool(
        engine,
        q,
        obs,
        g,
        [a],
        charts,
        valid,
        options,
        replace(plain, preserve_input_models=True),
        None,
        _preserve_reciprocal_branch=False,
    )
    assert len(kept) == 1
    np.testing.assert_array_equal(kept[0], a)
    assert audit["preserved_input_proposals"]["input_ids"] == [0]
    assert audit["candidate_selection"][1]["tight_training"] >= 20
    assert options.min_indexed == 20


def test_trim_only_is_explicit_and_never_silently_disables_polishing():
    with pytest.raises(ValueError):
        GrainProposalRefinementOptions(retain_split_fit=False)
    with pytest.raises(ValueError):
        GrainProposalRefinementOptions(retain_split_fit=1)
    q, obs, g, a, _ = cloud()
    engine = _core.ExperimentalLatticeEngine(1)
    proposed, audit = _polish_proposals(
        engine,
        q,
        obs,
        g,
        [a],
        GrainSearchOptions(),
        GrainProposalRefinementOptions(trim_fractions=(0.9, 0.8, 0.7), retain_split_fit=False),
        None,
    )
    assert proposed and audit["legacy_proposed"] == 0
    assert audit["trimmed_proposed"] == len(proposed)
