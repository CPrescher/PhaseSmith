# Pawley oracle coverage

Follow-up: [the fixed-profile diagnosis](../docs/pawley-profile-diagnosis.md) isolates the discrepancy
to the pinned FCJ numerical evaluator and finite cutoff policy. Production
quadrature, width/area conversion and the Pawley optimizer pass the isolation
controls; “different physical models” is not established.

The external oracle remains pinned by `PINNED_GSASII.json` to
`c0bc79b259cdf0065480b5fbd57674ddf12c4a23` (#5838). No GSAS-II implementation
code was copied, translated, or added as a dependency.

`tests/test_pawley.py::test_extraction_from_existing_pinned_gsasii_profile_fixture`
uses the existing hash-validated `cw_instrument_profile_v1` overlapping-reflection
fixture as observed data, starts all three Pawley areas at zero, and checks
recovered areas and calculated signal. The profile-error bound is the existing
6e-6 fixture convention; the corresponding area tolerance is 8e-6 relative.
The fixture has not been regenerated. This checks intensity units and the native
objective against external numerical output. It does **not** compare two Pawley
optimizers or unresolved intensity assignments.

A read-only audit of the exact pinned `GSASIIscriptable.py` found no dedicated
Pawley constructor/reflection-initialization method. The public scripting
manual exposes `General/doPawley` through generic dictionary inspection, while
the data-organization manual documents `Pawley ref` as phase-owned state:

- [GSASIIscriptable public manual](https://gsas-ii-scripting.readthedocs.io/en/latest/GSASIIscriptable.html)
- [GSAS-II data organization](https://gsas-ii-scripting.readthedocs.io/en/latest/objvarorg.html)

Those online pages track newer revisions and are documentation context, not a
replacement pin. The live optimizer comparison is now implemented by
`scripts/generate_pawley.py` and `phasesmith.validation.pawley_oracle`.
The initializer lives in `_pinned_probe.initialize_pawley`; exact revision and
record-shape checks precede every private write. Normal PhaseSmith imports do
not load GSAS-II. Public scripting performs every profile calculation and fit.

The committed `fixtures/pawley_optimizer_v1` contains two 24,001-sample P1
triclinic cases, each with 53 families: fixed cell, and six refined cell
parameters from a perturbed cell. Both initialize F-squared at half of the
known synthetic values. The data use unit weights and fixed profile/background;
negative F-squared is unpenalized in GSAS-II and the comparison explicitly selects
signed native areas. A fixed RNG seed stabilizes otherwise unused preliminary
observed-F-squared records. Provenance records the exact revision, generator,
probe/helper hashes, Python/NumPy versions and compiled oracle binary hashes.

The conversion is `area = 0.01 * F_obs_squared * intensity_correction`. The
correction already includes multiplicity and scale: applying them again is an
error. The factor 0.01 translates the GSAS-II centidegree density convention to
an area multiplying the native per-degree profile. Formal GSAS-II `SH/L=0`
uses the pinned minimum 0.002; the native comparison uses equal physical ratios
0.001/0.001 and an explicit 10,000-FWHM support. These are separate profile
implementations, not an assertion of pointwise identity.

`validation/results/pawley-20260917-live-oracle-comparison.json` records fixed
states before fits, fitted profiles, cell lengths/angles, 42 isolated areas and
five overlapping groups. Both native cases converge. Fixed-state relative L2
is about 3.69e-4; fitted profile error is 3.64e-4 (fixed cell) and 1.99e-4
(refined cell). Isolated area errors stay below 8.9e-5; the largest group-sum
error is 5.40e-4. Individual overlapping areas differ by as much as 1.16%,
which is retained as a diagnostic rather than treated as an independently
measurable split. Relative cell-length differences are below 2.0e-6 and angle
differences below 2.1e-5 degrees.

The executable comparison uses explicit engineering agreement gates (1e-3
profile L2, 2e-4 isolated areas, 1e-3 group sums, 3e-6 relative lengths and
5e-5-degree angles). Its separate 1e-5 fixed-profile equivalence check **fails**.
The comparison is complete; exact GSAS-II optimizer/profile equivalence is not
claimed. This is the original plan's documented oracle-model exception, backed
by the independent reference/derivative tests and unchanged measured-data gates.

Explicit generation, in the separate pinned oracle environment:

```sh
python oracle/scripts/generate_pawley.py --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/matching/binaries --output /new/fixture/directory
```

Generation refuses an existing output directory. Comparison requires no GSAS-II:

```sh
PYTHONPATH=python python -m phasesmith.validation.pawley_oracle \
  --fixture oracle/fixtures/pawley_optimizer_v1 --output /new/comparison.json
```

The independently designed least-squares objective follows Pawley's published
method: G. S. Pawley (1981), J. Appl. Cryst. 14, 357–361,
[doi:10.1107/S0021889881009618](https://doi.org/10.1107/S0021889881009618).
