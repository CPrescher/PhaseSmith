# GSAS-II validation oracle

GSAS-II is an optional, pinned black-box oracle. The exact revision is recorded
in `PINNED_GSASII.json`; it is not installed, vendored, or imported by the normal
Rietveld Engine package.

The public adapter in `python/rietveld/oracle/gsasii.py` extracts copies of `X`,
`Ycalc`, `Background`, and phase reflection lists using `G2PwdrData.getdata()`
and `G2PwdrData.reflections()`. The private adapter `_pinned_probe.py` exposes
only a small allowlist and verifies the checkout's exact Git revision before
reading internal data.

The external benchmark worker in `scripts/benchmark_cw_profile.py` uses the
same revision gate. It is driven by `benchmarks/compare_gsasii.py`, imports no
Rietveld Engine module, and reports only a numerically verified symmetric CW
profile-and-derivative comparison. See `../docs/gsasii-performance.md` for its
scope and limitations.

To prepare an oracle checkout:

```shell
git clone https://github.com/AdvancedPhotonSource/GSAS-II.git /path/to/GSAS-II
git -C /path/to/GSAS-II checkout c0bc79b259cdf0065480b5fbd57674ddf12c4a23
```

Use GSAS-II's own supported environment and installation procedure. Do not add
it to this project's Python environment.

## Fixture generation

`fixtures/schema.json` defines fixture format version 1. The committed
`symmetric_pseudo_voigt_v1` fixture contains narrow and broad Gaussian-dominant,
mixed, and Lorentzian-dominant cases plus one overlapping two-peak pattern. The
pinned private `GSASIIpwd.getPsVoigt` and `getdPsVoigt` probes provide values and
component-width/position derivatives. The generator converts centidegree
density, Gaussian variance, Lorentzian FWHM, and the probe's reversed position
derivative sign into the public degree/FWHM convention documented in
`docs/tch-profile.md`. `minimal_cw_histogram_v1` is generated through the public
scripting API and contains `X`, `Ycalc`, background, and a documented 15-column
powder reflection list for a deterministic synthetic phase. Manifests record
array hashes, units, generator hash, exact GSAS-II revision/tag, Python/NumPy
versions, and platform.

`cw_instrument_profile_v1` uses the same pinned private profile probes after
deriving Gaussian variance and Lorentzian FWHM from documented U/V/W/X/Y
equations. It covers low, middle, and high angle plus an overlapping reflection
batch. Its stored derivatives include the angle dependence of component widths
in the reflection-position derivative and dense U/V/W/X/Y rows.

`fcj_profile_v1` uses the pinned `getFCJVoigt3` and `getdFCJVoigt3` probes at
low, middle, and high angle plus the zero-asymmetry limit. The oracle exposes a
single `SH/L` parameter, recorded with the published equal-height mapping
`sample_over_radius = detector_over_radius = SH/L / 2`. The production model
evaluates the independently derived, quadrature-converged published integral;
fixture-local tolerances record the observable discretization difference in the
pinned compiled GSAS-II routine rather than tuning the production quadrature to
that approximation.

`wavelength_components_v1` composes two independently probed FCJ profiles at
Bragg-law K-alpha1/K-alpha2 positions, using normalized 1:0.5 integrated
intensities. Low, middle, and high-angle cases verify component positions,
weighted values, integrated area, centroid, and third moment. Together with the
single-component FCJ fixture, it covers the optional-doublet and explicitly
monochromatic paths.

`sample_physics_v1` configures isotropic size, isotropic microstrain, and
March--Dollase preferred orientation through the public generic HAP entry API.
It stores the complete public reflection list and normalized private profile
probes for low, middle, and high-angle reflections. The manifest records the
explicit conversion from GSAS-II micrometre/ppm/centidegree conventions to the
public nanometre/RMS-strain/degree conventions. Tests compare every reflection
width and orientation factor before comparing selected values, areas, and
moments.

`multiphase_v1` uses the public scripting API to configure two phases with
different HAP scales and stores public `X`, total `Ycalc`, background, and both
complete reflection lists. A controlled private two-reflection composition
isolates scale multiplication from structure-factor calculation and validates
the fused total, phase-separated diagnostics, component widths, and analytical
phase-scale rows.

`neutron_cw_v1` uses public scripting histogram type `PNC` and stores public
`X`, total `Ycalc`, background, and the complete neutron reflection list.
Pinned private probes cover symmetric and FCJ profiles for low-, middle-, and
high-angle reflections, including widths, areas, centroids, and third moments.

`tof_v1` uses public scripting histogram type `PNT` and stores calculation-bin
centers `X`, total `Ycalc`, background, and the complete 18-column TOF
reflection list. Pinned private `getEpsVoigt`/`getdEpsVoigt` probes cover three
d-spacings and store values, five direct derivatives, integrated area,
centroid, and third moment. The manifest records the GSAS-II-compatible
`sig-q * d` convention explicitly.

`lebail_v1` creates a deterministic synthetic `PNC` observation through public
scripting, enables public Le Bail mode, and stores observed/calculated arrays,
background, weights, the complete reflection list, and public residual trends.
The pinned internal probe is limited to installing deterministic observations
(because `getdata` returns a copy) and requesting GSAS-II's new-Le-Bail
initialization. The manifest records the degree-density conversion
`integrated_intensity = 0.01 * Fobs^2 * intensity_correction`.

Generation is deliberately separate from the normal package: the script imports
GSAS-II and NumPy, but never imports `rietveld`. Run it with GSAS-II's Python:

```shell
/path/to/gsas/python oracle/scripts/generate_symmetric_profile.py \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory

/path/to/gsas/python oracle/scripts/generate_powder_histogram.py \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory

/path/to/gsas/python oracle/scripts/generate_cw_instrument_profile.py \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory

/path/to/gsas/python oracle/scripts/generate_fcj_profile.py \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory

/path/to/gsas/python oracle/scripts/generate_wavelength_components.py \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory

/path/to/gsas/python oracle/scripts/generate_sample_physics.py \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory

/path/to/gsas/python oracle/scripts/generate_multiphase.py \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory

/path/to/gsas/python oracle/scripts/generate_neutron_cw.py \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory

/path/to/gsas/python oracle/scripts/generate_tof.py \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory

/path/to/gsas/python oracle/scripts/generate_lebail.py \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory
```

The generator refuses to replace `data.npz` or `manifest.json`. Regeneration
requires the explicit `--force` flag, after which both metadata and numerical
diffs must be reviewed. Normal tests load fixtures through
`rietveld.oracle.load_fixture`, which validates the pin, archive hash, member
names, array hashes, dtypes, shapes, and finiteness before returning data.

The powder generator creates and discards its GPX file in a temporary directory;
GSAS-II project objects never enter the fixture or normal test environment.

## Live oracle workflow

Normal CI validates committed fixtures and deliberately does not install or
import GSAS-II. The opt-in `Pinned GSAS-II oracle` workflow runs on a controlled
runner labeled `gsasii-oracle`. Repository variables `GSASII_PYTHON`,
`RIETVELD_GSASII_ROOT`, and `RIETVELD_GSASII_BINARY_DIR` must identify the
compatible interpreter, exact pinned checkout, and binary directory. The
workflow builds this package in an isolated environment and runs only tests
marked `external_oracle`. Missing configuration, an incorrect revision, absent
binaries, or a numerical mismatch is a hard failure; the live tests never
skip. The dispatch option `regenerate_symmetric_fixture` additionally runs the
external-only generator into the runner temporary directory and validates the
new archive; it never overwrites the committed golden fixture. Locally, the
equivalent comparison command is:

```shell
RIETVELD_GSASII_ROOT=/path/to/pinned/GSAS-II \
RIETVELD_GSASII_BINARY_DIR=/path/to/compatible/GSASII-bin/platform-directory \
/path/to/gsas/python -m pytest -q -m external_oracle tests/test_external_oracle.py
```

Run the performance comparison from the release-built normal environment while
pointing it at the separate oracle interpreter:

```shell
uv run python benchmarks/compare_gsasii.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory
```

GSAS-II is separately licensed and must be cited as requested by its authors.
No GSAS-II source is copied into this repository.
