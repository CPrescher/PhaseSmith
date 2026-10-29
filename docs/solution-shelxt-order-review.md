# SHELXT order-of-operations review, 29 October 2026

This targeted follow-up inspected the existing external SHELXT 2018/2 audit
(`/Users/clemens/Codex/shelxt-binary-audit-2026-10-28/`), especially `_cgref_`,
`_sfcgls_`, `_spgnon_`, `_spgcen_` and the P1 solver export. Executable SHA-256 is
`4d712cb17f8952e1f59f476af7c61ad4ab5261e594c75441c7f81dcefe7b5560`.
Approximate exports cover 24 of 25 targeted entry points, not the complete
program: the large Fortran `_MAIN__` and outlined numerical helpers remain
unresolved. Decompiled names/types are inferred. No pseudocode or mechanically
translated implementation enters PhaseSmith.

## Evidence and order

The published algorithm provides the reliable overall schedule:
P1 dual-space phasing; origin/space-group selection; ten cycles of
symmetry-constrained density modification; peak integration/element assignment;
isotropic conjugate-gradient least-squares refinement; optional absolute-structure
and final assembly work. See Sheldrick (2015), sections 3–5 and Fig. 1,
[doi:10.1107/S2053273314026370](https://journals.iucr.org/a/issues/2015/01/00/sc5086/index.html).

The binary exports add narrower observations, not a recovered global schedule:

- `_cgref_.c`, lines 1550–1580, fills contiguous work arrays and calls `_sxis_`
  three times before its later iterative work. The exact array roles are not
  established by the decompiler.
- `_cgref_.c`, lines 2196–2212, contains runtime calls associated with the Intel
  OpenMP serialized/forked branches and an outlined helper. This establishes
  parallel machinery, not the exact equations inside the missing helper.
- `_sfcgls_.c`, around lines 1898–1935, evaluates an exponential outside a nested
  packed-arithmetic block; later sections update contiguous floating-point arrays.
  This is consistent with factoring common attenuation/scattering work, but does
  not establish the physical meaning of each inferred parameter or loop axis.
- Packed float arithmetic is visible in several kernels. It is insufficient
  evidence to change PhaseSmith's numerical precision or copy their tolerances.

## Applied change and remaining priorities

The current performance change independently factors fixed reciprocal/image
geometry out of repeated refinement evaluations, and hoists the isotropic
attenuation and derivative once per site/reflection. Its equations follow
ordinary structure-factor algebra and orthogonal symmetry, as written in
[solution-refinement-geometry-cache.md](solution-refinement-geometry-cache.md).
The decomposition was derived directly; the binary observations only help
prioritize this work. Values and analytical derivatives remain fused and f64.
All count/label alternatives, original beam/shortlist ranking and final
anisotropic fits are retained. The explicit direct kernel is the numerical and
performance ablation.

The next algorithmic candidate is supplied-symmetry density recycling *before*
weak-orbit assembly. It could expose several missing orbits together instead of
paying for sequential birth/refinement rounds. It must retain ambiguous heavy
framework origins, missing-data provenance and competing C/N assignments.
The published conjugate-gradient normal-equation method is another candidate
for much larger parameter sets; the current small dense solves and their
convergence/round-off behavior require a separate measured comparison before
replacement. Neither follow-up is claimed as implemented here.

Current isotropic screening before shortlisted anisotropic refinement is already
consistent with the useful cheap-before-expensive principle. Stopping at the
first low-R1 candidate is not safe: the recorded thiourea C/N failure is retained
in the earlier handoff report. Chemistry-specific carbon/bond heuristics and
parallel/f32 kernels are not substituted for the general mineral-compatible
search or its correctness checks.
