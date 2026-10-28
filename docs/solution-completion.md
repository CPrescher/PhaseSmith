# Partial-framework difference-Fourier completion

This experimental search uses only the candidate, corrected observations, supplied
cell/operations and full-cell composition. No reference or external-solver
coordinates enter any search. It follows the standard partial-model difference-
Fourier principle; see [McCusker et al. 1999, section 6](https://doi.org/10.1107/S0021889898009856).
Special-position projection is independently derived from affine fixed-point
constraints, related to [Grosse-Kunstleve & Adams 2002](https://doi.org/10.1107/S0108767301016658).
No external implementation code is used.

## Scope and equations

When a complete candidate fails supplied-symmetry matching or the requested fit
target, remove the lowest-atomic-number element, preserving every stronger atom.
Require this partial framework to admit complete same-element symmetry orbits
at the existing matching tolerance. Origin/hand recovery and exact projection use
the existing independent native symmetry implementation. This first bounded
completion supports **one missing species in one orbit**, not arbitrary fragment
assembly, occupancy refinement, chemical restraints or space-group discovery.
The number of images must equal that species' full-cell atom count exactly.

A partial framework can have more origin/site ambiguity than the full structure.
Use the existing bounded origin grid to retain up to eight different projected
framework settings; do not choose among them from the strong-atom matching
residual alone. Conservatively deduplicate settings when a translation satisfies
`R delta-delta in Z^3` for every supplied rotation and aligns all framework
positions (including continuous free-origin directions). This is a sufficient
normalizer condition; additional centred-lattice equivalences can remain.
Translations that preserve only the partial atom arrangement do not justify
merging distinct setting hypotheses. The missing-species data rank the completed
models across all retained settings. This hypothesis enumeration is bounded,
not an exhaustive proof of every compatible origin for arbitrary inputs.


For the projected framework, compute non-resonant complex structure factors
`Fp(h)=sum_j f_j(s) exp(-8 pi² U s²) exp(2 pi i h.x_j)` using the candidate's
common isotropic U, bounded to the existing physical interval. Profile the
amplitude scale `k=(Fo.|Fp|)/(|Fp|.|Fp|)`. Difference coefficients are
`D(h)=(Fo(h)/k-|Fp(h)|)*Fp(h)/|Fp(h)|`. Coefficients with
`|Fp| <= 1e-10 max(|Fp|)` receive zero rather than an invented phase. Missing
indices, absences and 000 are zero; measured positive/negative Friedel mates must
agree. The periodic map is `rho_diff=FFT(D).real/N`, in arbitrary model units.
There is no resolution extension and no clipping of signed difference values.
Source intensities remain unchanged; Fo retains the earlier derived convention.

Take the 64 strongest positive periodic maxima, with existing interpolation and
deterministic ties. Include the old missing-species positions as additional seeds
in the recovered group frame; they are proposals only. For a nearby operation,
solve `(R-I) delta=-(R x+t-n-x)` in the Cartesian minimum-norm metric. A projected
point is accepted only if the affine constraint is satisfied to 1e-8 Å and its
displacement from the original seed is at most the caller’s matching
tolerance (default 0.15 Å). Up to three strictly multiplicity-reducing projections generate
intersections of fixed sets. Orbit expansion uses 1e-8 fractional equality.
Keep only the required multiplicity, reject cross-element overlaps below 0.5 Å,
and deduplicate equivalent orbit representatives within 0.02 Å. These thresholds
are explicit search choices, not chemical bond-length information.

Score complete trial structures against the same unweighted profiled amplitude
objective, using all measured families after expansion. For a new orbit:
`F=Fp+f exp(-8 pi² U s²) sum_g exp(2 pi i h.(R_g x+t_g))`.
Values and all three fixed-topology coordinate derivatives
`dF/dx_l=2 pi i f exp(-8 pi² U s²) sum_g (h.R_g)_l exp(...)`
are accumulated in the same reflection/image pass. The objective Jacobian includes
the derivative of its profiled scale. Finite-difference checks use only directions
compatible with the fixed site stabilizer; crossing special-position topology is
not a differentiable operation.

For each retained framework setting, retain at most 128 distinct trial orbits,
rank by this common objective, and polish the best eight using the existing symmetry-constrained isotropic or
positive-definite anisotropic model. The completion must improve the original
candidate's objective; otherwise retain the original candidate. Exact composition,
non-overlap and multiplicity are checked before and after polishing. The model-
to-density hand/origin transformation is composed explicitly when exporting.
Up to eight frameworks therefore permit at most 1,024 orbit scores and 64
complete refinements per original trial. All stages obey the caller's original
global deadline/cancellation token, with bounded loops. No extra search restarts are hidden outside the supplied budget.

An explicit `completion=False` switch preserves the earlier search behavior.
Completion is experimental and enabled by default for symmetry-refined searches;
`refinement="none"` skips it. Success/failure and actual work counts are recorded.
A fit target remains a numerical condition, not a structure-validity certificate.

## Validation boundary

Independent NumPy complex sums, difference coefficients/FFT, affine projection
and centered differences verify the new equations. Existing ADP derivative tests
remain unchanged. Real recovery is assessed against reference geometry only after
saving each blind result. The reused thiourea/BiSeI/K[BrF4] cases are development
evidence. GSAS-II does not cover this structure-solution mechanism; its powder
fixtures are not regenerated. A realistic multi-reflection/multi-image orbit
benchmark covers the new scoring loop.


## Use

The optional native build remains:

```sh
maturin develop --release --features experimental-solution
```

`SolutionOptions(completion=True)` is the experimental default; use
`completion=False` or CLI `--no-completion` to disable this step. The existing
benchmark scripts explicitly default to the old path. Enable the new comparison
with a new output folder:

```sh
python benchmarks/solution_dual_comparison.py --completion \
  --output runs/completion-comparison
```

`candidate.completion` is an immutable typed audit record. It includes framework
species/counts, missing species/count, retained/selected framework settings,
seeds, affine projections, distinct orbit scores, constrained refinements and
before/after objectives. `applied` is separate from the stop status so a valid
retained result is visible even if a later budget/cancellation stop intervened.
Experimental JSON schema v4 adds this provenance. It does not change the default
powder wheel or introduce any external solver dependency.

## Development result

The first dual-space K[BrF4] trial retained four different Br/K settings.
Choosing only the first had produced a symmetry-consistent but wrong F orbit
with R1 0.0500; evaluating the alternatives instead finds R1 **0.0222664**.
A post-solve coordinate audit places every atom within **0.00591 Å** of the
published structure, with **0.00482 Å** RMS error. This is independent native
search from the same 214 corrected observations, supplied cell/group/composition
and candidate; no SHELXT solution or archived coordinates enter completion.
The native residual differs from SHELXL because scattering, extinction and the
objective/weighting treatment differ. These reused cases remain development
evidence, not untouched validation holdouts or a general success-rate claim.


The final serial comparison uses the same seed 0, eight-start/1,200-cycle limits,
60 s budget and 0.06 R1 target, with completion enabled uniformly:

| Case | Charge flipping + completion | Dual space + completion | Post-solve geometry |
| --- | ---: | ---: | --- |
| Thiourea | R1 0.049409, 3.99 s | R1 0.049409, 3.64 s | Max non-H error 0.01037 Å; unchanged |
| BiSeI | R1 0.052285, 0.47 s | R1 0.052285, 4.01 s | Max error 0.00252 Å; unchanged |
| K[BrF4] | R1 0.0222664, 3.47 s | R1 0.0222664, 4.70 s | All 24 atoms; max/RMS error 0.00591/0.00482 Å |

Both native strategies now complete K[BrF4] from their first phasing trial. The
successful dual-space attempt inspected four framework settings, 320 combined
map/old-candidate seeds, 7,680 affine proposals, 18 distinct orbit hypotheses and
16 complete constrained refinements. All counts are exposed in the frozen
`benchmarks/structure-solution-completion-macos.json`; the previous failed
uncompleted comparison is retained and referenced by checksum. Every blind result
was saved before the reference-coordinate audit. Times are individual host/run
measurements, not general speed claims.

The new 16-image orbit value/Jacobian benchmark has 16 fixed atoms and 1060 measured
reflections. Ten Criterion samples give a slope interval of 133.34–134.05 µs.
The full suite passes: **1,234 Python tests** (11 skipped, 34 deselected) and
**387 Rust tests/doctests** (34 ignored), with strict workspace Clippy and formatting.
A fresh default wheel passes import/crystallography checks and omits the optional
search. Tests cover independent complex sums, scale derivatives and centered
finite differences in four symmetry settings; signed-map normalization/Friedel
rules; mirror/glide/intersection projections; periodic boundaries and resource
rejections; retained origin alternatives; synthetic weak-orbit completion and
its independent geometry check; budget/cancellation stops; and equivalence of
asymmetric and full-cell CIF/structure-factor exports. Parameter uncertainty,
weighted F² refinement and general multiple-missing-orbit assembly remain outside
this slice.
