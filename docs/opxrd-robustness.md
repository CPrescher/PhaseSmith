# opXRD robustness campaign

## Claim boundary

The [opXRD paper](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aidi.202500044)
and [Zenodo deposit](https://zenodo.org/records/14279434) provide a valuable
heterogeneous experimental corpus. They do not provide a uniform set of
instrument files, certified structures, or reference Rietveld results. This
campaign therefore makes two narrower claims:

1. PhaseSmith handles or explicitly rejects representative real-data shapes,
   grids, intensity domains, source descriptions, and label depths.
2. Three single-phase, fully labelled cases support a same-input structural
   capability comparison with pinned GSAS-II.

Neither claim treats opXRD patterns as independent scientific replicates or as
bulk Rietveld ground truth.

## Immutable selection

`validation/opxrd-robustness-v1.json` pins the 615 MB non-time-series archive by
size and SHA-256, then pins every selected member separately. The 14 examples
span all nine archive directories and cover:

- full, partial, and effectively unlabeled phase records;
- laboratory, synchrotron, monochromatic, and doublet sources;
- 1,200 to 49,496 samples, including a 0.001 degree grid;
- negative observations, low dynamic range, strong background, and thin-film
  geometry;
- uniform, slightly nonuniform, and concatenated/nonmonotonic axes.

The 70,000-pattern in-situ archive is deliberately excluded because its
trajectories are strongly correlated. The grouping unit for interpretation is
the contributor/source directory, never the individual pattern.

Fetch the registered archive explicitly; it is not redistributed:

```bash
python -c "from phasesmith.validation import fetch_validation_dataset; \
fetch_validation_dataset('opxrd-robustness-v1', \
'validation/data/opxrd-robustness-v1')"
```

## Fixed procedure

The runner first validates the full archive and member digests, decodes the
nested JSON strings, and validates aligned finite vectors. It does not sort,
clip, shift, or resample source data.

For strictly increasing grids it computes an independent 64-bin, 20th-percentile
piecewise-linear baseline. It also attempts PhaseSmith's 0.1 degree,
50-iteration Smooth Bruckner estimator. The latter is evaluated twice and must
be bitwise deterministic. A nonuniform grid rejection is recorded as a valid
boundary outcome while the independent metrics remain available. A
nonmonotonic grid stops all axis-dependent calculation and records the number
of offending steps.

Peak detection operates in physical 2theta units after baseline subtraction:
local maxima must be at least 0.1 degrees apart and exceed both six times a
robust first-difference noise estimate and 1% of the robust intensity range.
Where cell, space group, and wavelength exist, unweighted predicted reflection
positions are compared with detected maxima. This is a coverage diagnostic,
not an intensity or indexing oracle.

The structural subset uses exactly one labelled phase and writes plain XYE,
CIF, and JSON files. Expanded archived atoms are represented in P1 to avoid
inventing an asymmetric unit. Cell and atoms remain fixed. Missing displacement
values use fixed `Uiso = 0.005 A^2`; missing geometry uses Bragg--Brentano
polarization 0.5 and `S/L = H/L = 0.001`; a present second wavelength uses a
fixed 0.5 intensity ratio.

Position alignment deliberately precedes width refinement. PhaseSmith scans
zero shift from -1 to +1 degrees in 0.025 degree steps. At every candidate it
exactly resolves phase scale and the eight-term Chebyshev background, selects
the minimum Poisson Rwp, and then performs a bounded zero-only local polish.
Zero is fixed before U/V/W and isotropic size/microstrain are released, so
profile widths cannot initially mask non-overlapping peaks. The GSAS-II worker
starts from the selected common zero, polishes zero alone, fixes it, and only
then releases widths. Wavelength components found in reverse order are sorted
shortest to longest and the correction is reported.

Two diagnostic warnings remain separate from residual acceptance: absolute
zero shift above 0.2 degrees and maximum instrument Gaussian FWHM above 0.5
degrees over the measured range. These are campaign warning thresholds, not
universal instrument limits. They prevent a lower Rwp from silently being
interpreted as a more credible physical fit.

## Complementary metrics

For observation vector `y` and calculated vector `c`, the report retains:

- relative L2: `||c-y||2 / ||y||2`, dominated by large residuals;
- normalized L1: `sum(|c-y|) / sum(|y|)`, less dominated by outliers;
- cosine similarity, sensitive mainly to vector direction;
- Pearson correlation, insensitive to affine offset and scale;
- Poisson Rwp using `sigma = sqrt(max(y, 1))`, only when every observation is
  nonnegative.

Poisson Rwp is omitted, with a reason, for negative-count patterns. Shifting or
clipping those values would manufacture a weighting model. Background-only
metrics compare each baseline with the raw observation; they measure how much
signal the estimator assigns to background, not background accuracy.

## Reviewed result (2026-08-31)

All archive, member, accounting, determinism, and metric-applicability checks
pass. Of 14 cases, 12 receive independent metrics. PhaseSmith's physical-width
background evaluates on seven; five nonuniform grids are explicitly rejected.
Two USC patterns contain two backward axis steps each and are rejected before
axis-dependent calculations. All three negative-observation cases suppress
Poisson Rwp. Seven records require wavelength-component reordering.

The three structural common-model comparisons are diagnostic:

| case | PhaseSmith / GSAS-II Poisson Rwp | relative L2 | normalized L1 | Pearson |
| --- | ---: | ---: | ---: | ---: |
| NbS2 synchrotron | 0.1389 / 0.0947 | 0.1741 / 0.1037 | 0.0942 / 0.0699 | 0.7407 / 0.9155 |
| ZrC laboratory | 0.2934 / 0.2914 | 0.4963 / 0.4946 | 0.1664 / 0.1633 | 0.8261 / 0.8339 |
| Li2TeC2 doublet | 0.1467 / 0.1449 | 0.1660 / 0.1645 | 0.1188 / 0.1177 | 0.9757 / 0.9762 |

Position-first refinement removes the former ZrC optimizer gap and preserves
the close Li2TeC2 total-profile agreement. Their PhaseSmith zero shifts are
0.0184 and 0.0545 degrees and their maximum instrument Gaussian FWHM values are
0.115 and 0.044 degrees, so both PhaseSmith fits pass the campaign plausibility
diagnostic.

NbS2 remains a useful warning rather than a parity pass. The scan improves
PhaseSmith Poisson Rwp from 0.1691 before alignment to 0.1555 after zero-only
alignment and 0.1389 after width/sample refinement. Starting from the common
scan result, the PhaseSmith and GSAS-II local polishes converge to essentially
the same large offset, +0.5209/+0.5211 degrees, confirming an axis/calibration
mismatch in the sparse source record.
GSAS-II's lower final residual also coincides with a 1.65 degree maximum
instrument Gaussian FWHM, versus 0.029 degrees in PhaseSmith. Its residual is
therefore not evidence of a more credible physical profile. The case remains
in the corpus specifically to keep that failure mode visible.

GSAS-II crosses the 0.5 degree width-warning threshold in all three cases, so
the close ZrC and Li2TeC2 residuals support calculation/workflow compatibility,
not refined instrument-parameter equivalence. Differences in signal-only
correlation also show that the two background implementations can decompose
baseline and peaks differently even when total-profile metrics are close.

Reviewed records are
`validation/results/2026-08-31-opxrd-robustness.json` and
`validation/results/2026-08-31-opxrd-gsasii.json`.

## Reproduction

```bash
python benchmarks/opxrd_robustness.py \
  --structural \
  --json-output validation/results/local-opxrd.json

python benchmarks/compare_gsasii_opxrd.py \
  --gsas-python /path/to/gsasii-python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/GSASII-bin \
  --json-output validation/results/local-opxrd-gsasii.json
```

The GSAS-II worker requires revision
`c0bc79b259cdf0065480b5fbd57674ddf12c4a23`, runs in its own interpreter,
imports no PhaseSmith module, and returns only plain JSON.
