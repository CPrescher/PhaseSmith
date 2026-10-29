# Supplied-symmetry density recycling before weak-peak assignment

This independently implements the published phase-averaging/positivity step in
Sheldrick (2015), section 4,
[doi:10.1107/S2053273314026370](https://journals.iucr.org/a/issues/2015/01/00/sc5086/index.html).
The cell, exact symmetry and allowed element types remain supplied; counts and
reference coordinates are not inputs. Each retained strong-framework setting
gets its own recycling attempt. A wrong heavy-substructure origin must not erase
competing settings.

## Equations and conventions

Use `rho=FFT(F).real/N`, `F=N IFFT(rho)`. For aligned coordinates
`y=hand*x+s`, `F_y(h)=exp(2 pi i h.s) F_x(hand*h)`.
For operation `(R,t)`, `F(hR)=exp(-2 pi i h.t) F(h)`.
Within each symmetry/Friedel orbit average unit phasors after removing these
translation factors; conjugate the Friedel contributions. Duplicate stabilizer
operations remain in the average, enforcing centrosymmetric phase restrictions.
Normalize the mean; a degenerate zero mean uses a deterministic unit phase on any required centrosymmetric phase line (one
when unrestricted). Restore measured Fo amplitudes. Unmeasured allowed
coefficients use their current mean orbit magnitude and remain predictions, not
observations. Systematic absences/out-of-support coefficients stay zero. F000
is free and unmeasured. All orbit/summation orders are deterministic.

Each of ten cycles projects phases/amplitudes, synthesizes the map and clips
negative grid density to zero, then transforms back. A final reciprocal
projection/synthesis restores exact measured amplitudes and supplied symmetry,
including for fractional translations not commensurate with the grid. Therefore
the returned band-limited map may contain negative truncation ripples. Peak
extraction uses positive maxima. There is no resolution extension, extra measured
data, atom-count mask or invented absolute electron count.

The reciprocal support must be closed under the supplied operations and Friedel
inversion, without Nyquist aliases. Measured families must be complete and have
consistent amplitudes. Shapes retain the existing 2,097,152-node bound; projection
plans have a 64 MiB cap. Cooperative cancellation/deadlines apply between bounded
cycles. This discontinuous search projection has no claimed optimization
Jacobian. Existing structure-factor/refinement values and analytical derivatives,
Gaussian finite support and spherical integration kernels are unchanged.

Independent NumPy complete cycles, translated/hand-inverted origins, symmetry
phase residuals, measured-amplitude restoration, missing-coefficient provenance,
invalid boundaries and deterministic repeated runs validate the projection.
GSAS-II has no corresponding structure-phasing oracle.

## Search integration and assignment safeguards

`solve_structure_elements(..., symmetry_recycling=True, recycling_cycles=10)`
enables this experimental path for either charge flipping or count-free
dual-space phasing. The default is **off**, pending wider validation and because
this extra work can slow an already easy solve. The original exact space group,
cell, allowed neutral non-H types and all existing resource bounds still apply.
The cycle count must be an integer in [1,50]. The private equation-test boundary
also accepts zero cycles for a single reciprocal projection.

After the strong-prefix origin hypotheses have undergone the original constrained
framework fits, recycle the source map separately in every retained setting.
Keep at most 32 positive maxima above 10% of the strongest sampled maximum.
Construct the existing measured-amplitude ball-integral map using the recycled
phases. For a source-to-recycled frame `(h0,s0)` and a source-to-current-model frame
`(hp,sp)`, the recycled-to-current transform is
`h=hp*h0, s=sp-h*s0 (mod 1)`. This composed transform is used both for peak seeds
and for the inverse density-integral queries. The public source density remains
in its original frame, with the existing explicit model transform and both CIF
exports. Intermediate maps cannot silently change that provenance.

If the integrated-density guide is enabled, try one joint weak-orbit candidate
per retained framework. The top 24 recycled maxima feed the existing bounded
fixed-set projection generator (at most 64 proposals). Strong-site integral/Z
ratios supply a relative calibration. For each orbit, keep only mass/Z estimates
between 0.6 times the smallest and 1.6 times the largest allowed atomic number;
choose the nearest label in log ratio. Greedily add at most eight disjoint whole
orbits in the existing amplitude-objective order, respecting the atom cap and
0.5 angstrom minimum pair separation. These are proposal bounds, not inferred
stoichiometric constraints or a complete enumeration of count combinations.

Settle weak labels jointly at fixed coordinates with at most four sweeps. For
site j and each allowed element e, use the existing independent-NumPy-verified
shared structure-factor kernel:

`A_e(h)=|F_other(h)+f_e(h) D(h) sum_{x in orbit j} exp(2 pi i h.x)|`;
`k_e=(A_e . Fo)/(A_e . A_e)`;
`J_e=||k_e A_e-Fo||^2/||Fo||^2`.

Hold the strong anchor labels fixed; change a weak label only for a strict
objective improvement greater than 1e-10. Coordinates, tensors, counts and
absolute electron density are not estimated by these label sweeps. The candidate
then receives the existing short isotropic screen and complete requested-mode
refinement. Surviving batches compete with the original frameworks under the
unchanged penalized score and bounded beam. Ordinary whole-orbit birth, delete
and relabel rounds continue; their difference-map proposals are supplemented by
up to 16 recycled maxima in the correctly composed frame. Disabling the density
guide disables the joint mass-based batch but retains recycled peak seeds.

A low-R1 batch is provisional until a complete ordinary edit/shortlist round
finds no improvement of the preferred complete candidate in the same lineage
within the existing 1e-7 score threshold. Cancellation or expiry cannot validate
a partial round. This prevents acceptance based solely on a favorable batch R1.
An initial mass-only prototype reached R1 0.05764 with wrong thiourea C/N counts
(C8 N4 S4); that failed experiment remains under `runs/symmetry-recycling-probe/`
and is not a successful benchmark. Joint diffraction label checks and the full
round safeguard correct that failure without changing the count/geometry gates.

Actual per-setting origin/hand, completed cycles, predicted-coefficient count,
peak count, batch orbit count, alternative-label scores/changes and batch-fit
outcome are exported in typed `RecyclingAudit` records. Integrated-map preparation
counts include recycled maps; serial-edit score counters retain their earlier
meaning, with batch label work reported separately. If a projection rejects an
incompatible reciprocal support, the reason is retained and that lineage uses
the previous map path. Deadlines and cancellation retain bounded partial results.

## Paired measured comparison

Three serial alternating-order pairs per case use identical element-only inputs,
exact symmetry, dual-space phasing, seed 0, an 800-cycle ceiling, early handoffs,
cached refinement, 60-second total limit, R1 target 0.06, atom/beam bounds and
complete staged shortlists. Only `symmetry_recycling` changes. All solves and
reports are saved before reference formulas or coordinates are read. These are
reused development cases, with no claim of general solve coverage.

| Case | Previous median | Recycling median | Relative speed | Inferred full-cell counts |
|---|---:|---:|---:|---|
| K[BrF4] | 0.521 s | 0.767 s | 0.68x (47% slower) | Br4 F16 K4 |
| BiSeI | 0.356 s | 0.252 s | 1.41x faster | Bi4 I4 Se4 |
| Thiourea, non-H | 4.516 s | 1.499 s | 3.01x faster | C4 N8 S4 |

All 18 solves recover the correct counts. Final R1 remains 0.0222664, 0.0522851
and 0.0494090, with maximum element-matched coordinate errors 0.005909, 0.002516
and 0.010372 angstrom. Both CIF frames reproduce R1 within 2e-10 through the
general structure-factor API (36 exports checked). The paired quality gates are
unchanged: R1 increase <=1e-7, maximum reference error <0.02 angstrom, and paired
maximum-error increase <=1e-4 angstrom.

Screening/final-fit counts fall from 42/18 to 27/15 for K[BrF4], 23/10 to 14/7
for BiSeI, and 156/34 to 50/14 for thiourea. The added per-origin projection,
integration and joint-assignment work outweigh those savings in the easiest
K[BrF4] solve. BiSeI uses one origin setting, thiourea two and K[BrF4] four;
all get ten cycles per setting. Thiourea needs one post-batch edit round rather
than four sequential rounds. There is no case-name dispatch or supplied count
shortcut. The mixed result is why recycling remains explicitly opt-in.

The reproducible controller is
`benchmarks/solution_symmetry_recycling_comparison.py`. The frozen report is
`benchmarks/structure-solution-symmetry-recycling-macos.json`, with native/source
and input hashes, individual timings, actual effort and post-solve quality
checks. Raw outputs are under `runs/symmetry-recycling-final/`.

One additional serial charge-flipping pair per case also passes all count,
geometry, R1 and both-CIF checks. Previous/recycled times are 1.522/1.642 s for
K[BrF4], 0.546/0.439 s for BiSeI and 5.344/2.240 s for thiourea. This is a transfer
check rather than a repeated timing estimate. Its frozen report is
`benchmarks/structure-solution-symmetry-recycling-charge-flipping-macos.json`;
raw outputs are under `runs/symmetry-recycling-charge-flipping-final/`.

The realistic native benchmark runs ten cycles plus preparation/final projection
on a 32-cubed, 2,108-observation, eight-atom mirror model. Median time is 6.25 ms
(20 Criterion samples). It includes an independently initialized four-atom
charge-flipping source map. This isolates the projection cost; total search gains
and losses also include assignment, larger candidates and refinement. Results
and source hashes are frozen in
`benchmarks/structure-solution-symmetry-recycling-kernel-macos.json`.

Validation: 1,412 Python tests passed (11 external-data skips, 34 deselections),
392 Rust tests passed (34 ignored), strict workspace Clippy, Cargo formatting and
Python Ruff checks passed. Full NumPy cycle comparisons use 5e-9 relative /
2e-11 absolute tolerances; measured amplitudes and symmetry phase covariance have
separate 1e-10 to 2e-10 absolute checks. Deterministic repeated maps and
end-to-end P1, inversion and mirror count/coordinate recovery are also checked
with recycling enabled and disabled. No global tolerance was loosened. The
measured comparisons additionally validate both exported density/model frames.

A separately built default-feature wheel passes native cell/symmetry checks with
SciPy blocked and exposes none of the experimental solver/recycling symbols.
No new dependencies, release or push are included in this change.
