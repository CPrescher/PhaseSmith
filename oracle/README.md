# GSAS-II validation oracle

GSAS-II is an optional, pinned black-box oracle. The exact revision is recorded
in `PINNED_GSASII.json`; it is not installed, vendored, or imported by the normal
Rietveld Engine package.

The public adapter in `python/rietveld/oracle/gsasii.py` extracts copies of `X`,
`Ycalc`, `Background`, and phase reflection lists using `G2PwdrData.getdata()`
and `G2PwdrData.reflections()`. The private adapter `_pinned_probe.py` exposes
only a small allowlist and verifies the checkout's exact Git revision before
reading internal data.

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
```

The generator refuses to replace `data.npz` or `manifest.json`. Regeneration
requires the explicit `--force` flag, after which both metadata and numerical
diffs must be reviewed. Normal tests load fixtures through
`rietveld.oracle.load_fixture`, which validates the pin, archive hash, member
names, array hashes, dtypes, shapes, and finiteness before returning data.

The powder generator creates and discards its GPX file in a temporary directory;
GSAS-II project objects never enter the fixture or normal test environment.

GSAS-II is separately licensed and must be cited as requested by its authors.
No GSAS-II source is copied into this repository.
