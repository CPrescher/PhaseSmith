# Structural TOF readiness and correction contract

PhaseSmith's completed TOF workflow is a facility-neutral profile and Le Bail
implementation. It can refine shared lattice geometry and bank-local profile
calibrations, but those capabilities alone do not make a structural Rietveld
model. Structural TOF intensities need an explicit observation convention in
addition to a peak shape.

This document records the Unit 38 capability review, the implemented single-bank
calculation primitive, and the remaining contract before PhaseSmith claims a
complete structural TOF refinement workflow.

## What is already reusable

The following production components do not depend on constant wavelength and
can be reused directly:

- reciprocal-metric d-spacings and analytical cell derivatives;
- symmetry expansion, multiplicity, nuclear structure factors, coordinate,
  occupancy, and displacement-parameter derivatives;
- constant bound coherent neutron scattering lengths for species that are not
  marked energy dependent by the pinned table;
- the 15-coefficient TOF calibration/profile model and its fused value,
  intensity, position, width, and coefficient derivatives;
- deterministic finite support, background composition, masks, uncertainties,
  runtime cancellation, checkpoints, and atomic multi-bank objectives; and
- stable shared reflection and parameter identities.

The current structural pattern engine itself is not reusable unchanged. Its
public input, position correction, sample-width contributions, wavelength
installation, and peak accumulation are explicitly in degrees `2theta` and use
a `ConstantWavelengthInstrument`.

## The missing physical contract

For a conventional focused one-dimensional TOF bank, each reflection has

```text
lambda_h = 2 d_h sin(theta_bank)
tof_h = Zero + DIFC d_h + DIFA d_h^2 + DIFB / d_h
```

where `2 theta_bank` is fixed bank geometry. `BNKPAR` is therefore not optional
metadata for structural work even though profile-only Le Bail extraction does
not need it.

The calculated integrated intensity must separate three choices:

```text
I_h = scale * multiplicity_h * |F_h|^2
      * C_geometry(d_h, theta_bank)
      * C_spectrum(lambda_h)
      * C_detector(lambda_h)
      * C_sample(h, lambda_h)
```

The factors after `|F|^2` depend on what upstream reduction already applied.
The POWGEN example discussed by Jacobs et al. was background corrected and
normalized by a vanadium measurement to account for detector efficiency and
remove the wavelength-dependent incident distribution. A raw or differently
reduced beamline dataset cannot safely inherit the same assumptions.

PhaseSmith will therefore keep two explicit baseline modes:

1. `Neutral` means the supplied observations have already been reduced to a
   convention in which no additional integrated-intensity correction is
   requested.
2. A named TOF neutron powder Lorentz model implements the conventional
   one-dimensional Rietveld factor

   ```text
   C_TOF = d_h^4 sin(theta_bank)
   d C_TOF / d(q_h^2) = -2 sin(theta_bank) / (q_h^2)^3
   ```

   with `q_h^2 = 1 / d_h^2`. It is selected explicitly and never inferred from
   a facility name or file extension.

Incident-spectrum, detector-efficiency, absorption, extinction, and other
sample corrections remain separate typed factors. A tabulated wavelength
factor must state its interpolation and extrapolation rules and provide the
derivative needed by the fused structural pass. PhaseSmith must not hide such a
factor in a phase scale or a profile coefficient.

Recent work on TOF Lorentz factors also shows why the observable must be named:
the appropriate factor changes with the reduced quantity and its binning. The
legacy one-dimensional Bragg-Rietveld convention is not a universal correction
for event data, total scattering, or a two-dimensional angle/wavelength
observable.

## Facility-neutral bank metadata

The application boundary now provides `TofBankGeometry`, independent of any
legacy file format. Its required field is finite `0 < two_theta_deg < 180`. The
bounded GSAS adapter populates it from the independently documented second
`BNKPAR` field. The core model never consumes legacy records or dictionaries.
Files without `BNKPAR` remain valid for profile-only use and return no geometry;
structural TOF must reject that omission. Flight paths and other source metadata
are not required by the calibrated `tof(d)` equation and are not guessed.

The existing `TofInstrument` remains the profile/calibration record. Keeping
geometry separate avoids pretending that fitted `DIFC` uniquely determines a
physical flight path or bank angle.

## Implemented single-bank structural calculation

Unit 38b provides the Rust `phasesmith-engine` structural TOF primitive. One
request combines a unit cell, exact space group, canonical reflections,
asymmetric sites, constant bound-coherent neutron species, explicit correction,
typed bank geometry, and bank-local `TofInstrument` on a strictly increasing
microsecond bin-center grid. It evaluates

```text
q_h^2 = h^T G* h
d_h = 1 / sqrt(q_h^2)
I_h = scale * multiplicity_h * correction_h * |F_h|^2
Y_i = sum_h I_h P_TOF(tof_i; d_h, instrument)
```

in production Rust. The local profile block supplies `dY/dI_h` and `dY/dd_h`.
The structural chain adds

```text
dd_h/dp_cell = -0.5 d_h^3 d(q_h^2)/dp_cell
dY_i/dp = sum_h [(dY_i/dI_h)(dI_h/dp) + (dY_i/dd_h)(dd_h/dp)]
```

with the second term zero for non-cell structural parameters. Dense, JVP, and
VJP interfaces use the established P1 structural order: six cell parameters;
three fractional-coordinate values per site; one occupancy per site; one
isotropic displacement value per site; and phase scale. The accumulation also
retains the 15 bank-instrument derivative rows for later guarded composition.

Finite support is inclusive at both ends. Centered-difference tests exclude
samples whose membership can change at a perturbed support boundary. The cell
rows use a local `2e-4` relative test tolerance because they chain through the
finite-quadrature asymmetric-profile d-spacing derivative; direct structural
rows remain at `3e-5`. JVP and VJP are checked against the dense result and by
the adjoint identity. The independent NumPy reference composes symmetry,
neutron intensity, reciprocal d-spacing, and high-order TOF quadrature without
calling the new Rust primitive. A Criterion case covers 128 reflections, 16
sites, and 14,501 TOF samples for value, dense, and JVP paths. On the 2026-08-12
review machine, the new baseline medians were 167.96 ms for values, 173.40 ms
for the dense structural Jacobian, and 167.72 ms for one JVP. There is no
earlier structural-TOF baseline against which to report a regression.

The structural kernel deliberately accepts only `Neutral` or
`TimeOfFlightNeutronLorentz`, and the Lorentz angle must bitwise match the typed
bank geometry. It does not infer incident-spectrum, detector, absorption,
extinction, or texture corrections. Observation preprocessing is explicit:
raw TOF counts can be divided by a separately supplied incident-spectrum
calibration before constructing the structural request.

The facility-neutral `TofIncidentSpectrum` implements the calibrated
Maxwellian-plus-Chebyshev function

```text
x = 2/t - 1
I_inc(t) = P1 + P2 t^-5 exp(-P3/t^2)
           + sum(Pj T_(j-3)(x), j=4..12)
```

where `t` is in milliseconds while the public API and inclusive validity
interval are in microseconds. Value and `dI_inc/d(tof_us)` are evaluated
together. The bounded GSAS adapter maps explicit `I ITYP 4` plus `ICOFF1..3`
records into this model, treats an absent/type-0 record as no calibration, and
rejects unsupported nonzero functions. Normalizing observations also requires
dividing their one-sigma uncertainties by the same positive intensity. The
model is not tied to POWGEN or LANL; other adapters may construct the same typed
calibration from their native metadata.

## Implemented multi-bank objective

The first Unit 38c slice composes the single-bank primitive into
`PreparedStructuralTofMultiBankObjective`. It requires one or more unique bank
IDs and evaluates one summed objective atomically. The established `MultiBank`
type names are retained for API stability, but a one-bank request is valid:

```text
Phi = 1/2 sum_bank sum_included [(Y_calc - Y_obs) / sigma]^2
```

The uncertainty denominator is omitted when uncertainty weighting is disabled
or a bank has no uncertainty array. Masks contribute exactly zero. Shared
setting-aware lattice, symmetry-allowed coordinates, occupancies, and isotropic
displacement parameters use the existing physical-to-native structural
transform. Each bank independently owns its scale, selected bounded instrument
coefficients, fixed plus optional Chebyshev background, geometry, correction,
mask, and uncertainty.

The prepared objective exposes values, JVP, VJP, gradient, and
`J^T W J + damping I` products. Structural reverse products are summed into
shared rows; the native structure-factor scale row is redirected to the
corresponding bank-local scale. Instrument products select rows from the
15-coefficient fused accumulation, and background products use the exact
Chebyshev basis. Deterministic tests compare the complete joint JVP and
objective gradient with centered differences, verify JVP/VJP adjoint identity,
and check the normal operator explicitly.

This slice is deliberately guarded to one fixed-topology neutron phase. The
phase record must carry unit placeholder scale, neutral placeholder correction,
neutral CW-contribution arrays, and no CW sample-physics or dynamic reflection
domain. Every bank then chooses its real scale and correction explicitly. The
prepared contract freezes observation arrays, geometry, correction choice,
parameter bounds, background identity/domain, support, weighting, and execution
policy while allowing only the declared numerical parameters to change.

## Implemented bounded solver

`refine_structural_tof_multibank` completes Unit 38c without alternating bank
fits. At every attempted iteration it evaluates the current atomic objective
and obtains the full scaled normal matrix by applying the objective's analytical
normal product to each scaled coordinate basis vector. Positive Levenberg
damping is added in scaled coordinates. The guarded LU solve falls back to SVD;
the result is capped by `max_scaled_parameter_step`, clipped to each physical
bound, and half-step backtracked. A trial is accepted only if the summed
masked/uncertainty-weighted objective over all banks decreases.

`StructuralTofMultiBankRefinementOptions` owns only controls used by this dense
solver: hard runtime limits, minimum accepted iterations, objective and scaled
step tolerances, damping factors, step cap, and backtrack count. It deliberately
does not expose the conjugate-gradient controls of the separate joint CW
solver. The dense solve is appropriate for this first guarded structural
contract; future large-site work may add an explicitly reviewed matrix-free
solver without changing parameter ownership.

Each accepted trial atomically checkpoints the complete original request,
accepted physical input, stable parameter set, objective, next damping, and
accepted history. Cancellation, wall/evaluation/iteration limits, and repeated
rejection return the last accepted state. Resume revalidates the exact frozen
request and parameter contract. Tests cover two-bank scale/Zero recovery, exact
partial/resumed equivalence, corrupted checkpoint rejection, cancellation after
an accepted checkpoint, and evaluation exhaustion while the normal matrix is
being assembled.

## Public Python facade

`phasesmith.refinement.tof_structural` exposes the completed native objective
and solver without a second Python numerical implementation. A
`StructuralTofMultiBankInput` requires one built-in neutron phase with neutral
placeholder correction, unit placeholder scale, fixed reflection topology,
and no CW sample-physics provider. Each `StructuralTofBank` separately owns its
observed microsecond pattern, calibrated instrument, `TofBankGeometry`, neutral
or matching-angle `TimeOfFlightNeutronLorentz` correction, scale and bounds,
optional residual Chebyshev background, and selected instrument bounds.

For a one-bank file workflow,
`StructuralTofMultiBankInput.from_files(...)` composes the bounded reduced-data
reader, one GSAS calibration bank, required detector geometry, a CIF structure,
and generated reflection topology. The caller must explicitly choose whether
the observations are already normalized or require the calibration's type-4
incident spectrum, and whether the bank uses a neutral or TOF-neutron Lorentz
correction. `correction="already_applied"` represents an upstream Lorentz or
equivalent correction and maps to a neutral in-engine factor, while `neutral`
means that no integrated-intensity correction is required. The caller must also
declare sample corrections as `none` or `already_applied`; named absorption,
extinction, or texture models are rejected until they have reviewed TOF
equations and analytical derivatives. Missing geometry or a requested but
absent incident spectrum is an error. Facility names, filenames, and
profile-function codes never select intensity physics implicitly; non-GSAS
callers can continue to construct the same typed request directly.

Every composed request retains a `StructuralTofRequestProvenance` record with
the source name, byte size, and SHA-256 of the pattern, instrument calibration,
and CIF. The reduction record is either a separately supplied
`reduction_path` or, when reduction metadata is embedded in the reduced pattern
header, the same checksum-pinned pattern bytes. The record also retains the
selected bank, normalization, integrated-intensity and sample-correction
declarations, whether a fixed background was supplied, and the SHA-256 of the
applied fixed-background float64 array. Source hashing obeys the same
caller-selected byte limits as the pattern, calibration, and CIF readers.
This is application-layer audit metadata and never enters the numerical Rust
kernel. Python checkpoints retain the provenance record and reject continuation
when it differs from the request, while the native checkpoint continues to
guard the exact numerical contract.

`tof_range_us=(lower, upper)` optionally selects an inclusive observed interval
before incident normalization and reflection generation. This is required when
a raw bank contains samples outside a calibrated type-4 spectrum's validity
domain. The explicit range is retained in provenance; facility names do not
choose it. A supplied full-bank fixed background is sliced with the same
indices, while an already selected background must match the selected samples.
`fixed_background_domain="input"` divides the supplied background together
with raw observations during type-4 normalization. The alternative
`"normalized"` installs a background already expressed on the normalized
intensity scale after observation/uncertainty normalization. The declaration
and the hash of the applied background are both retained in provenance, so an
already-normalized estimate cannot be silently divided twice.

`StructuralTofMultiBankInput.combine_file_banks(...)` joins one-bank requests
only after checking their exact common structure and numerical controls,
one-bank provenance, and unique bank IDs. The caller supplies a shared inclusive
d-spacing range; the composer regenerates one common reflection topology and
retains ordered provenance for every bank. This avoids silently unioning
different bank-visible topologies or choosing a facility-specific range. Each
provenance record includes both the source bank number and the exact public
`bank_id`; input validation requires those IDs to align in bank order.

`StructuralTofRefinementOptions` mirrors only the dense native solver controls.
The result returns the updated phase and banks, immutable calculated/profile/
background and reflection-intensity arrays, the stable physical parameter set,
accepted history, bounded termination reason, evaluation count, and an opaque
exact-resume checkpoint. `StructuralTofCancellation` reuses the thread-safe TOF
cancellation token. End-to-end Python tests recover two distinct bank scales
and Zero terms, resume bitwise-identical accepted state after cancellation, and
recover a shared cubic lattice parameter. The structural real-data/oracle
acceptance described below completes the final Unit 38d gate.

Symmetry-constrained coordinates use the same local-tangent convention as the
general structural solver. A special-position `q` value describes the current
accepted step, not an absolute fractional coordinate: the accepted physical
structure retains the displacement while the next solver/checkpoint parameter
state resets that local coordinate to zero. Structural TOF trials apply
`q_after - q_before` exactly once. A mirror-site regression checks accepted
motion, final checkpoint validation, and bitwise-identical continuation.

`StructuralTofMultiBankProjectState` is the Python-free application-host
boundary for retaining these analyses. It requires unique analysis IDs and
disjoint bank ownership, resolves every bank ID to an exact project TOF
pattern/instrument record, requires each member histogram to reference exactly
the one shared built-in phase, and revalidates an optional solver checkpoint.
Native project format 5 persists that facade through
`save_structural_tof_multibank_project` and
`load_structural_tof_multibank_project`. The wire contract references the
project-owned observations, initial instruments, and phase definition rather
than duplicating them. It retains explicit bank geometry/correction, stable
site IDs, every refinement selection and bound, numerical controls, and the
complete last-accepted checkpoint. Load reconstructs typed domain state and
requires exact checkpoint/request validation. Formats 1--4 remain readable
with no structural TOF analyses.

## TOF roadmap completion audit

The completed scope is the facility-neutral back-to-back-exponential/TCH TOF
model and the explicit adapters/corrections listed here. Completion does not
assert compatibility with an unimplemented beamline-specific profile or
reduction convention.

| Roadmap requirement | Authoritative implementation | Review evidence |
| --- | --- | --- |
| Atomic multi-bank fixed-cell refinement | `tof_multibank` workflow with bank-local observations, profiles, backgrounds, and intensities | Distinct-bank recovery, aggregate metrics, atomic cancellation, and exact resume in `crates/phasesmith-workflows/tests/tof_multibank.rs` |
| Shared-cell analytical motion | Reciprocal-metric `dd/dp` composed with each bank's analytical TOF calibration/profile rows | Crystal-system finite differences and shared-cell recovery in `tof_multibank_lattice.rs`; moving-support derivative tests in `phasesmith-core` and the NumPy reference |
| Bank-local analytical instrument motion | Stable 15-row instrument order, selected bounded columns, rank/correlation diagnostics | Recovery, finite-difference kernel coverage, rank-deficiency, and restart tests in `tof_multibank_instrument.rs` |
| Joint lattice plus instrument refinement | One combined weighted system and atomic bounded/backtracked acceptance | Cross-family recovery, diagnostics, and exact continuation in `tof_multibank_geometry.rs` |
| Application facades and durability | Python `tof_multibank`/`tof_structural` facades and native project formats 3--5 | Python recovery/resume tests and deterministic format-3/4/5 round trips in `crates/phasesmith-persistence/tests/tof_project.rs` |
| Real-data and isolated-oracle coverage | POWGEN LaB6 plus LANL nickel bank 2--4 native/public workflows | Checksum-pinned validation reports and pinned public-scripting-only GSAS-II comparisons in `tests/test_real_data_integration.py` and `tests/test_real_data_oracle.py` |
| Structural TOF Rietveld | Fused structural TOF values/dense/JVP/VJP products, shared structural solver, explicit incident/Lorentz/background contracts | Structural finite differences/adjoint tests, checkpoint/special-position tests, LANL cell/Uiso oracle, and POWGEN LaB6 structural acceptance |

Unsupported profile functions, calibration syntaxes, and named absorption,
extinction, or texture models remain explicit errors. Adding one is a new
equation-, derivative-, provenance-, benchmark-, and oracle-backed increment;
it is not an unfinished part of Units 26 or 28--38.

## Unit 38 delivery sequence

The structural extension is split into reviewable numerical increments:

1. Add the explicit TOF neutron Lorentz correction with an independent NumPy
   equation, analytical reciprocal-metric derivative, invalid-angle tests, and
   centered finite differences.
2. **Complete:** add typed TOF bank geometry and parse the independently
   documented scattering angle from bounded legacy instrument input.
3. **Complete:** add a Rust structural-TOF calculation primitive that evaluates
   values and structural/profile derivatives through the same finite-support
   accumulation. Dense, JVP, VJP, independent-reference, and realistic
   multi-reflection benchmark gates are present.
4. **Complete:** compose the primitive into a guarded
   multi-bank structural objective and solver with shared structure/cell and
   bank-local scale, background, instrument, geometry, masks, and uncertainties.
5. **Complete:** expose the same native contract through Python and project
   persistence, then validate a checksum-pinned real structural dataset against
   an isolated pinned oracle. Python refinement, format-5 persistence, native
   LANL nickel acceptance, public incident-spectrum exposure, and the
   pinned-GSAS-II structural comparison all pass.

The isolated worker uses only GSAS-II's public scripting API and exports plain
JSON/NPZ records. It creates temporary checksum-derived one-bank RAW files
before import because the pinned scripting loader otherwise selects the first
dataset on repeated reads of this legacy multi-bank file. GSAS-II refines 44
live variables: shared cell and Ni Uiso, three local scale and Zero terms, and
36 local background coefficients. It reaches joint Rwp 0.03367362, minimum
bank correlation 0.99730908, a=3.52368699 A, and Uiso=0.00401813 A^2 on 13,290
selected centers. PhaseSmith's 13,293-center result differs by 0.00004414 A and
0.00005306 A^2 in cell and Uiso. Both profiles pass independent quality gates;
Rwp equality is not asserted across their different optimizer and background
contracts.

No stage may claim “all TOF beamlines” merely because another profile function
fits. Acceptance requires a declared reduction/correction convention and bank
metadata for every dataset.

## ORNL/POWGEN acceptance boundary

For an ORNL/POWGEN structural request to be scientifically complete, it must
contain or reference:

- the measured TOF grid, mask, uncertainty, and background convention;
- the bank's calibrated profile and `tof(d)` coefficients;
- the bank scattering angle;
- a crystallographic phase and neutron scattering identities;
- whether vanadium normalization removed detector efficiency and the incident
  wavelength distribution;
- the selected Lorentz convention;
- any absorption, extinction, texture, or other sample corrections; and
- provenance and checksums for the data, calibration, structure, and reduction.

The checksum-pinned POWGEN bank now exercises this contract as an explicit
one-bank structural acceptance case. Its header identifies the sample and
vanadium runs, bin-width scaling, and proton-charge normalization; the
calibration has no type-4 incident-spectrum record, so the request declares
the observations already normalized rather than applying a second spectrum
division. Bank 2 supplies `two_theta = 90 deg`, and the request explicitly
selects the one-dimensional TOF-neutron Lorentz factor. A staged
scale/background/geometry, boron-coordinate, and isotropic-displacement solve
reaches Rwp 0.14415481, background-subtracted profile correlation 0.97638031,
`a = 4.15792396 A`, and `x(B) = 0.19954072`. The lattice and boron coordinate
agree with the published POWGEN SRM-660b refinement within 0.000424 A and
0.000060, respectively. This complements the stricter LANL multi-bank
structural oracle; it does not claim that every POWGEN reduction or sample can
inherit this specific normalization and correction declaration.
For application requests, the reduced GSA header is also the reduction record;
its digest therefore anchors both the observations and the embedded reduction
declarations. A separate reduction log or recipe can instead be supplied and
hashed independently.
The public file composer is exercised directly on these pinned POWGEN files
with an inline, citation-annotated LaB6 initializer. It returns 6,824 samples,
330 generated reflection families, the 90-degree geometry, exact registered
pattern/calibration digests, a deterministic structure digest, and all explicit
physics declarations. The native structural acceptance above then gates the
full staged refinement.
The same public path is also gated on checksum-pinned LANL nickel bank 2, a
different facility/profile/input convention. An explicit 1101.6--8189.6 us
interval selects 4,431 samples before type-4 incident normalization, imports
the 88.05-degree geometry, and generates 100 families whose centers lie inside
that interval. The native LANL structural acceptance uses a deliberately wider
186-family shared 0.2--3.0 A d-space topology, so reflection-count equality is
not asserted across those two declared boundary conventions. Combining public
bank 2--4 requests with that same d-spacing declaration reproduces the
186-family shared topology and retains three ordered provenance records.
Estimating Smooth Brückner independently on each normalized bank, passing it as
`fixed_background_domain="normalized"`, and refining the three public bank
scales converges in four accepted steps to 0.04029, 0.04012, and 0.04481. The
three bank Rwp values are 0.1163--0.1243. This is an application-facade smoke
gate; the native structural acceptance remains the stricter lattice/Uiso/Zero
fit and pinned-oracle comparison.

## References

- P. Jacobs et al., “A Rietveld refinement method for angular- and
  wavelength-dispersive neutron time-of-flight powder diffraction data,”
  *Journal of Applied Crystallography* **48** (2015), 1627–1636,
  <https://doi.org/10.1107/S1600576715016520>.
- J. Liu et al., “Lorentz factor for time-of-flight neutron Bragg and total
  scattering,” *Journal of Applied Crystallography* **56** (2023),
  <https://doi.org/10.1107/S1600576723002127>.
- A. C. Larson and R. B. Von Dreele, *General Structure Analysis System
  (GSAS)*, Los Alamos National Laboratory Report LAUR 86-748, 2004, pp.
  127--129 and 222--223,
  <https://subversion.xray.aps.anl.gov/EXPGUI/gsas/all/GSAS%20Manual.pdf>.
- A. Huq et al., “POWGEN: rebuild of a third-generation powder diffractometer
  at the Spallation Neutron Source,” *Journal of Applied Crystallography* **52**
  (2019), 1189–1201, <https://doi.org/10.1107/S160057671900833X>.
