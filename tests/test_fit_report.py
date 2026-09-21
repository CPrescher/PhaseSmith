from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import phasesmith as ps
import pytest
from phasesmith.fit_report_reference import residual_diagnostics_reference
from phasesmith.refinement import rietveld
from test_rietveld_workflow import shifted_request


@pytest.mark.parametrize("seed", range(8))
def test_native_residual_evidence_matches_independent_numpy(seed):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.uniform(0.01, 2, 2001))
    residual = rng.normal(0, 3, x.size)
    weighted = residual / rng.uniform(0.1, 5, x.size)
    included = rng.random(x.size) > 0.2
    got = ps.diagnose_residuals(
        x, residual, weighted_residual=weighted, included=included, region_count=23
    )
    reference = residual_diagnostics_reference(x, residual, weighted, included, 23)
    assert got.included_count == reference["count"]
    assert got.adjacent_pair_count == reference["pairs"]
    for key in ("chi_square", "mean", "weighted_rms", "durbin_watson"):
        assert getattr(got, key) == pytest.approx(reference[key], rel=3e-14, abs=1e-13)
    np.testing.assert_array_equal([r.count for r in got.regions], reference["region_counts"])
    np.testing.assert_allclose(
        [r.chi_square for r in got.regions], reference["region_chi_square"], rtol=3e-13, atol=1e-10
    )
    assert sum(r.chi_square_fraction for r in got.regions) == pytest.approx(1, abs=2e-14)


def test_mask_gaps_region_edges_and_exact_fit():
    report = ps.diagnose_residuals(
        [0, 1, 2, 3, 4], [1, 99, -1, 2, 3], included=[True, False, True, True, True], region_count=2
    )
    assert [r.count for r in report.regions] == [1, 3]
    assert report.adjacent_pair_count == 2
    assert report.durbin_watson == pytest.approx(10 / 15, abs=1e-15)
    assert report.worst_regions(1)[0].lower == 2
    perfect = ps.diagnose_residuals([0, 1], [0, 0], region_count=3)
    assert perfect.durbin_watson is None
    assert all(r.chi_square_fraction == 0 for r in perfect.regions)
    isolated = ps.diagnose_residuals([0, 1, 2], [1, 0, -1], included=[True, False, True])
    assert isolated.durbin_watson is None
    assert isolated.adjacent_pair_count == 0


@pytest.mark.parametrize(
    "arguments",
    [
        {"x": [0, 0]},
        {"x": [1, 0]},
        {"x": [0, np.inf]},
        {"residual": [1, np.nan]},
        {"residual": [1]},
        {"residual": [True, False]},
        {"residual": [1 + 2j, 0j]},
        {"weighted_residual": [1e308, 1e308]},
        {"included": [1, 0]},
        {"included": [True]},
        {"included": [False, False]},
        {"region_count": 0},
        {"region_count": 4097},
        {"region_count": True},
    ],
)
def test_invalid_diagnostic_inputs_are_rejected(arguments):
    with pytest.raises(ValueError):
        ps.diagnose_residuals(**{"x": [0, 1], "residual": [1, 2], **arguments})


def test_report_preserves_actual_weights_and_abstains_from_model_attribution():
    request = shifted_request()
    project = ps.RietveldProject(request)
    with pytest.raises(ValueError, match="no refinement result"):
        project.fit_report()
    result = rietveld.refine(request, rietveld.RietveldOptions(estimate_covariance=False))
    before = result.calculation.y.copy()
    report = ps.build_fit_report(result, request.pattern)
    project.last_result = result
    assert project.fit_report() == report
    assert report.residuals.chi_square == pytest.approx(result.metrics.chi_square, rel=1e-14)
    assert report.attribution_available is False
    assert "No cause" in report.attribution_unavailable_reason
    assert "uncertainty_unavailable" in [v.code for v in report.advice]
    assert "review_termination" not in [v.code for v in report.advice]
    np.testing.assert_array_equal(result.calculation.y, before)
    record = json.loads(json.dumps(report.to_record(), allow_nan=False))
    assert record["schema"] == "phasesmith.fit-report.v1"
    wrong = ps.PowderPattern(request.pattern.x, observed_y=request.pattern.observed_y + 1)
    with pytest.raises(ValueError, match="observations"):
        ps.build_fit_report(result, wrong)
    wrong_mask = ps.PowderPattern(
        request.pattern.x,
        observed_y=request.pattern.observed_y,
        mask=np.arange(request.pattern.x.size) != 0,
    )
    with pytest.raises(ValueError, match="mask"):
        ps.build_fit_report(result, wrong_mask)
    unavailable = replace(result, jacobian_rank=None, covariance=None)
    assert "identifiability_unassessed" in [
        a.code for a in ps.build_fit_report(unavailable, request.pattern).advice
    ]


def test_reporting_uses_fit_weighting_even_if_uncertainties_were_disabled():
    request = shifted_request()
    request = replace(
        request,
        pattern=ps.PowderPattern(
            request.pattern.x,
            observed_y=request.pattern.observed_y,
            uncertainty=np.full(request.pattern.x.size, 7.0),
        ),
    )
    result = rietveld.refine(request, rietveld.RietveldOptions(use_uncertainty=False))
    report = ps.build_fit_report(result, request.pattern)
    assert report.residuals.chi_square == pytest.approx(result.metrics.chi_square, rel=1e-14)
    assert report.residuals.weighted_rms == pytest.approx(
        np.sqrt(np.mean(result.metrics.residual**2)), rel=1e-14
    )
