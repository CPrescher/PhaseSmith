# GSAS-II validation oracle

GSAS-II is an optional, pinned black-box oracle. The exact revision is recorded
in `PINNED_GSASII.json`; it is not installed, vendored, or imported by the normal
PhaseSmith package.

The public adapter in `python/phasesmith/oracle/gsasii.py` extracts copies of `X`,
`Ycalc`, `Background`, and phase reflection lists using `G2PwdrData.getdata()`
and `G2PwdrData.reflections()`. The private adapter `_pinned_probe.py` exposes
only a small allowlist and verifies the checkout's exact Git revision before
reading internal data.

The external benchmark workers in `scripts/benchmark_cw_profile.py` and
`scripts/benchmark_structural_pattern.py` use the same revision gate. The
QARR worker in `scripts/benchmark_qarr.py` additionally runs a complete,
staged 1g or 1h real-data workflow and exposes explicit FCJ, sample-broadening, and
displacement ablations. Its optional trace-mean anisotropic ablation is a small,
revision-gated internal probe because that representation change is absent from
the public scripting API. The default paired comparison applies that same
trace-mean representation in PhaseSmith, matches the ten-term Chebyshev,
isotropic-size, Lorentzian-microstrain, and staged refinement parameterization,
and gates both 1g and the independent 1h holdout at 0.005 absolute for phase
fractions and Rwp and 0.002 for profile correlation.
`scripts/benchmark_real_lebail.py` runs the sucrose and Echidna
constant-wavelength cases through temporary GSAS-II Le Bail
projects; its exact-revision-gated `newLeBail` initialization is recorded as a
private probe in each report because the public API does not expose it. The
sucrose driver supplies the identical fixed Smooth Bruckner array to both
methods, while the worker explicitly overrides the legacy `.prm` import with
the matched symmetric U/V/W/X/Y and SH/L=0 starting state.
`scripts/benchmark_powgen_tof.py` runs the real POWGEN LaB6 bank,
exports its 18-column Le Bail reflection list and selected profile probes, and
neutralizes sample broadening so the peak-only reconstruction matches the
fixed-instrument PhaseSmith scope. GSAS-II uses a 16-term Chebyshev background;
PhaseSmith uses a fixed Smooth Bruckner baseline plus a 16-term Chebyshev
residual. Their Rwp and profile-correlation deltas are gated separately from
the same-reflection reconstruction. These workers are driven by the
corresponding scripts in `benchmarks/`, import no PhaseSmith
module, and report numerically verified profile-only and controlled
structure-to-profile comparisons. See `../docs/gsasii-performance.md` for their
scope and limitations.

`scripts/benchmark_nickel_tof_multibank.py` creates one pinned GSAS-II project
with LANL nickel banks 2--4 linked to one Fm-3m phase. It exports plain arrays
after refining one shared cubic cell and the three bank-local Zero terms. The
paired driver gates exact reflection positions, variance and alpha/beta chains,
and reconstruction of every GSAS-II peak-only bank from the same extracted
intensities. The reviewed minimum reconstruction correlation is 0.99999898 and
the maximum relative L2 difference is 0.00181021. PhaseSmith and GSAS-II return
cells 3.523612 and 3.523867 A, a 0.000255 A difference. Their Le Bail Rwp values
are reported but not cross-gated because the intensity redistribution and
background decompositions are different. GSAS-II selects 4,430 points per bank
at the nominal limits while PhaseSmith's inclusive explicit-center convention
selects 4,431; both counts are asserted. Temporary one-bank RAW views prevent
the pinned public scripting loader from reusing the first dataset.

`scripts/benchmark_nickel_tof_structural.py` uses the same bank-authentic views
and public scripting APIs to refine one Fm-3m Ni phase against banks 2--4.
Shared cell and Uiso, bank-local HAP scale and Zero, and bank-local twelve-term
backgrounds form 44 live variables. The worker exports only finite plain arrays
and JSON. The paired driver requires both native and oracle profiles to pass
Rwp/correlation gates and compares cell and Uiso directly; it records but does
not cross-gate Rwp across different background and optimizer contracts.

`scripts/benchmark_nist_srm660c.py` reads one bounded pdCIF member directly
from the checksum-pinned NIST archive and fits the same 17-parameter physical
empirical subset as PhaseSmith. NIST's millimetre specimen displacement is
converted explicitly to the micrometre `Shift` unit used by GSAS-II. The
SH/L=0.002 matched case passes at 20.495% PhaseSmith Rwp versus 20.864% GSAS-II
Rwp. A separate SH/L=0.02 expected-failure stress case gives 18.912% versus
17.024% and tracks the observable continuous-versus-discretized FCJ difference.
Neither comparison replaces the NIST reference. Negative Gaussian variance and
negative microstrain states found in unconstrained GSAS-II probes are excluded
explicitly.

The live real-data matrix covers sucrose, Echidna, QARR 1g, QARR 1h, POWGEN,
and the paired PbSO4 X-ray/neutron workflow. NIST SRM 660c is deliberately different:
the certified NIST values and released calculated profiles remain the primary
oracle, so a GSAS-II refit cannot redefine that case's truth.

`scripts/calibrate_rowles_fpa.py` is a separate exact-revision-gated diagnostic.
At the pinned revision GSAS-II exposes its NIST fundamental-parameters
calibration only through `GSASIIfpaGUI`, so the private access is contained in
that worker and only plain JSON `U/V/W/X/Y/SH/L` coefficients leave it. The
worker consumes the neutral Rowles manifest, generates isolated physical peaks,
and uses GSAS-II's empirical peak fitter to compress them. The paired benchmark
then compares fixed FPA-derived coefficients with the normal empirical GSAS-II
instrument refinement on the same real patterns. It is neither a runtime
dependency nor a TOPAS-equivalence claim.

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

The pre-release PhaseSmith rename on 2026-08-07 changed only generator
docstrings and temporary-directory prefixes. The recorded generator hashes were
updated accordingly; archive and member hashes were unchanged, and no fixture
arrays were regenerated.

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

`stephens_orthorhombic_v1` configures GSAS-II generalized microstrain for a
Pnma phase with six orthorhombic coefficients and a nontrivial mixing value.
The oracle-only adapter converts pure terms by `1e-12/(8 ln 2)` and GSAS-II's
factor-three mixed basis by `3e-12/(8 ln 2)`. Across all 395 public reflection
rows, PhaseSmith matches pinned Gaussian variances within `2.3e-15`
centidegree² and Lorentzian FWHMs within `3.4e-16` centidegree. Three private
profile probes differ by at most `2.40e-6` of the oracle peak maximum.

`benchmark_citrate_residual_forensics.py` is a real-data, public-API-only
diagnostic rather than a golden fixture generator. It evaluates all 16 on/off
combinations of the anhydrous-rubidium legacy `LX`, `shft`, `trns`, and
Stephens terms, refitting only two phase scales and one residual-background
constant. Its report includes angle-resolved residuals and a Shapley partition
of weighted-SSE improvement, eight paired transparency penalties, and an
explicit transparency sign/scale sensitivity. Legacy `shft` and `trns` signs
are derived from the cited GSAS profile-argument equation; the report separately
labels the opposite legacy physical-height sign and the nonphysical effective
absorption implied by the deposited positive `trns`. It imports no PhaseSmith
module and emits only plain JSON.

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
GSAS-II and NumPy, but never imports `phasesmith`. Run it with GSAS-II's Python:

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

/path/to/gsas/python oracle/scripts/generate_stephens_orthorhombic.py \
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

Run the rubidium residual-forensics worker on a separately converted neutral
bundle with:

```shell
/path/to/gsas/python oracle/scripts/benchmark_citrate_residual_forensics.py \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --data-directory /path/to/converted/rubidium-bundle \
  --report /path/to/new-report.json
```

The generator refuses to replace `data.npz` or `manifest.json`. Regeneration
requires the explicit `--force` flag, after which both metadata and numerical
diffs must be reviewed. Normal tests load fixtures through
`phasesmith.oracle.load_fixture`, which validates the pin, archive hash, member
names, array hashes, dtypes, shapes, and finiteness before returning data.

The powder generator creates and discards its GPX file in a temporary directory;
GSAS-II project objects never enter the fixture or normal test environment.

## Live oracle workflow

Normal CI validates committed fixtures and deliberately does not install or
import GSAS-II. The opt-in `Pinned GSAS-II oracle` workflow runs on a controlled
runner labeled `gsasii-oracle`. Repository variables `GSASII_PYTHON`,
`PHASESMITH_GSASII_ROOT`, and `PHASESMITH_GSASII_BINARY_DIR` must identify the
compatible interpreter, exact pinned checkout, and binary directory. The
workflow builds this package in an isolated environment and runs only tests
marked `external_oracle`. Missing configuration, an incorrect revision, absent
binaries, or a numerical mismatch is a hard failure; the live tests never
skip. The dispatch option `regenerate_symmetric_fixture` additionally runs the
external-only generator into the runner temporary directory and validates the
new archive; it never overwrites the committed golden fixture. Locally, the
equivalent comparison command is:

```shell
PHASESMITH_GSASII_ROOT=/path/to/pinned/GSAS-II \
PHASESMITH_GSASII_BINARY_DIR=/path/to/compatible/GSASII-bin/platform-directory \
/path/to/gsas/python -m pytest -q -m external_oracle \
  tests/test_external_oracle.py tests/test_real_data_oracle.py
```

Run the performance comparison from the release-built normal environment while
pointing it at the separate oracle interpreter:

```shell
uv run python benchmarks/compare_gsasii.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory

uv run python benchmarks/compare_gsasii_structural.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory

uv run python benchmarks/compare_gsasii_qarr.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --sample 1g \
  --data-directory validation/data/iucr-qarr-1g

uv run python benchmarks/compare_gsasii_nist_srm660c.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --specimen 100a \
  --data-directory validation/data/nist-srm660c-lab6-xray

uv run python benchmarks/compare_gsasii_real_lebail.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --case aps-sucrose-11bmb \
  --data-directory validation/data/aps-sucrose-11bmb

uv run python benchmarks/compare_gsasii_powgen_tof.py \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --data-directory validation/data/powgen-lab6-tof-calibration

uv run python benchmarks/compare_gsasii_nickel_tof_multibank.py \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --data-directory validation/data/lanl-nickel-tof

uv run python benchmarks/compare_gsasii_rowles_qpa.py \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --data-directory validation/data/curtin-rowles-qpa-topas \
  --sample all

uv run python benchmarks/compare_gsasii_rowles_fpa.py \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --data-directory validation/data/curtin-rowles-qpa-topas \
  --sample all
```

The Rowles worker consumes only the neutral bundle produced from the
checksum-pinned deposited TOPAS/XY files. It imports NumPy and GSAS-II in the
separate oracle interpreter and never imports `phasesmith`. The report labels
the matched common parameterization, exports optional plain calculation and
reflection arrays for diagnostics, and enumerates every TOPAS optics term
omitted from the shared input subset. The accepted cross gate requires phase
fractions and Rwp values to agree within 0.005 absolute; GSAS-II remains an
external black-box oracle rather than a runtime dependency.

The FPA diagnostic additionally retains the deposited Rowles instrument
geometry in the neutral manifest. Its reviewed result is negative: the
compressed profile fits the synthetic physical target at 6.952% Rwp, but
worsens real-pattern Rwp from 9.085% to 13.928% (`1a`) and from 8.195% to
11.710% (`1e`) relative to empirical GSAS-II calibration.

GSAS-II is separately licensed and must be cited as requested by its authors.
No GSAS-II source is copied into this repository.
