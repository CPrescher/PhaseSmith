# Real-data validation plan

## Purpose

Real patterns test boundaries that synthetic and scalar profile cases do not:
text-format rounding, background preprocessing, reflection coverage, realistic
overlap, refinement stability, phase-scale interpretation, and complete script
ergonomics. External data remains opt-in and checksum-pinned. No validation
workflow imports GSAS-II or makes it a PhaseSmith runtime dependency.

## Implemented checkpoint

The implemented checkpoint provides Rust-owned layers with Python adapters:

1. `phasesmith-io` reads columns, GSAS FXYE, and GSAS STD into validated owned
   records; Python reborrows those arrays as NumPy.
2. `phasesmith-validation` owns dataset provenance, byte sizes, SHA-256 hashes,
   stable reports, and all native real-data runners.
3. `phasesmith-workflows::quantitative_phase_analysis` implements the published
   Hill--Howard `S Z M V` conversion for compatible phase scales.
4. `phasesmith-validation` provides a standalone JSON CLI. Public Python calls
   delegate ordinary runs through PyO3 while independent reference/callback
   paths remain available.

The official APS 11-BM sucrose tutorial is the currently supported complete
case: monochromatic FXYE input, a fixed Smooth Bruckner baseline plus one
refinable constant Chebyshev residual, P 21 reflection generation,
non-negative Le Bail extraction, and analytical U/V/W/X/Y refinement. The
oracle driver passes the same fixed baseline as plain GSAS-II input and
overrides the legacy instrument import with the same symmetric starting state.
The gate checks profile improvement,
`Rwp <= 0.22`, observed/calculated correlation of at least 0.97, and finite
non-negative integrated intensities. The threshold is a PhaseSmith regression
gate for the present model, not an equivalence tolerance against another
program.

The IUCr QARR 1g case now runs the real 5–150 degree pattern, Cu K-alpha
doublet, three CIF structures, native structural values/JVP/VJP products, and
Hill--Howard QPA. No monochromatic approximation is substituted for the
doublet.

The expanded external matrix adds four distinct forms of evidence rather than
treating every public file as the same kind of golden result:

1. NIST SRM 660c checks the official 20-specimen archive, certified lattice
   interval, and released reference profiles as an external-oracle integrity
   gate. A matched PhaseSmith profile comparison awaits equivalent source and
   optics models.
2. The ANSTO Echidna LaB6 pattern passes a native constant-wavelength neutron
   Le Bail smoke gate using deposited counting uncertainties.
3. QARR 1h is a deliberately unchanged-setup holdout. Its QPA gate passes, but
   its profile gates fail, showing that the accepted 1g setup is not yet a
   transferable three-phase model.
4. The official POWGEN tutorial data and bank calibration are checksum-pinned,
   parsed into a typed microsecond record, and checked against the production
   TOF position kernel and its analytical calibration derivative. Full TOF
   structural refinement remains explicitly blocked at the application
   workflow boundary.

Each manifest declares whether it is an acceptance, oracle-integrity, holdout,
or capability case and records its reviewed expected status. Schema-2 suite
records include pinned file identities, explicit criteria, and one scientific
SHA-256 fingerprint per case; host timing is excluded from that fingerprint.
Expected holdout failure and capability blocking therefore remain visible
without making the reviewed suite itself fail.

## Pinned external-oracle matrix

The controlled `gsasii-oracle` runner now executes the exact pinned GSAS-II
revision for every applicable real-data workflow:

- sucrose and Echidna use temporary GSAS-II Le Bail projects and gate selected
  samples, reflection families, weighted Rwp, and profile correlation;
- QARR 1g and the unchanged-setup 1h holdout gate phase fractions, both Rwp
  conventions, correlation, and reflection/sample counts;
- PbSO4 retains its paired X-ray/neutron joint-workflow gates;
- POWGEN gates the legacy bank translation, selected TOF profiles, and a
  same-reflection, same-intensity reconstruction of the peak-only pattern.

The workers import no PhaseSmith module and return plain finite JSON to the
comparison drivers. Normal CI does not install GSAS-II. NIST SRM 660c remains
certified-data-primary: its published values and released fits are the oracle,
not a new GSAS-II fit.

The pinned live run confirms QARR 1g parity. QARR 1h remains a reviewed oracle
failure: its phase-fraction delta passes (maximum 0.00979), while Poisson Rwp,
unit-weight Rwp, and correlation deltas (0.09434, 0.11009, and 0.01297) exceed
their gates. The controlled test requires this explicit `failed` status rather
than skipping the holdout or silently accepting the discrepancy.

## Independent laboratory follow-up cases

Two non-GSAS-II deposits now extend the laboratory X-ray matrix without
changing the production profile model.

The University of Bath zeolite-L archive is a conversion-fidelity capability
failure. It contains raw Rigaku SmartLab scans, legacy GSAS projects, released
profile curves and final multi-block CIFs. The converter validates the packed
ASC axes and counts, exports only the primary phase, repairs the GSAS2CIF
`O-`/`O-2` truncation visibly, and preserves the released background exactly.
The raw headers contradict the deposited README: K-LTL declares a Ge(220)x2
monochromator and K-alpha1, while Li-LTL and Cs-LTL declare a 2.5-degree Soller
slit and K-alpha. The header and archived experiment records take precedence.

Using the same primary publication CIF, released observed curve, exact fixed
background and archived effective profile initializer gives:

| sample | released legacy GSAS Rwp | PhaseSmith Rwp | pinned GSAS-II Rwp |
| --- | ---: | ---: | ---: |
| K-LTL | 7.791% | 21.972% | 21.303% |
| Li-LTL | 5.201% | 23.559% | 19.493% |
| Cs-LTL | 3.492% | 15.062% | 12.080% |

GSAS-II reproduces the deposited background with zero pointwise difference but
does not reproduce the legacy curves from the publication CIF either. The
missing information is therefore recorded as legacy-project conversion
semantics, not attributed to a missing PhaseSmith fundamental-parameters term.

The XRED anatase/rutile pattern is a phase-identification capability case, not
a certified QPA case. XRED states that its background was removed, but provides
neither instrument metadata nor a certified composition. The benchmark
therefore discloses an assumed single Cu K-alpha1 model and treats fitted phase
fractions only as cross-program diagnostics. Its COD anatase CIF uses the
nonstandard `I 41/a m d S` symbol; PhaseSmith reads the explicit operations,
while the converter emits the equivalent origin-choice-1 declaration required
by GSAS-II.

PhaseSmith reaches Poisson Rwp 22.256%, unit Rwp 14.740%, and correlation
0.98749. Pinned GSAS-II's staged run reaches 44.801%, 32.354%, and 0.95097.
Despite those optimizer minima, the inferred anatase fractions agree closely:
86.049% versus 85.864%, a 0.185 percentage-point difference. This supports the
shared structure-factor, phase-scale and Hill--Howard paths while leaving the
unknown instrument and optimizer recipe outside strict parity acceptance.

Reviewed machine-readable results are stored in
`validation/results/2026-08-11-bath-ltl-gsasii.json` and
`validation/results/2026-08-11-xred-tio2-gsasii.json`.

The in-house internal-standard capability case uses the IUCr dicesium hydrogen
citrate supplementary CIF (`wm5358sup1.cif`). It is registered by exact size
and SHA-256 and converted to one 2,820-point observed/legacy-calculated/fixed-
background table plus three phase CIFs. The converter independently reproduces
the deposited legacy-GSAS Rwp of 6.226% and retains its 60.03/27.00/13.02 wt%
phase fractions. With the archived instrument zero fixed, fixed-cell Si alone
first calibrates specimen displacement from its isolated (220), (311), and
(400) windows, and that value is frozen for the multiphase fit. Under the
disclosed common isotropic model, PhaseSmith gives 10.180% Rwp and 14.593 wt%
Si; pinned GSAS-II gives 8.894% and 14.621 wt% Si. The reviewed result is
`validation/results/2026-08-11-campaign-iucr-si-standard.json`.

The broader 2026-08-11 pinned-oracle audit, including fresh Rowles, QARR,
NIST, sucrose, Echidna, PbSO4, POWGEN, Bath, XRED, and mixed-Si records, is tracked in
`docs/gsasii-benchmark-campaign.md` and summarized in
`docs/gsasii-benchmark-report.md`.

## Implemented fixed-spectrum structural checkpoint

The fixed-spectrum prerequisite now provides:

1. Typed fixed `ComponentRadiation` with positive component intensities and a
   reference-wavelength profile instrument.
2. Component-specific Bragg positions, Lorentz--polarization values, source
   weights, and optional sample physics. Structure factors retain their
   wavelength-independent scattering-vector convention.
3. One fused native structural value/JVP/VJP batch per component. Python loops
   only over the small spectrum and never over reflections or samples.
4. Fixed-cell CIF reflection generation using the exact union visible in any
   component while keeping every shared family physical for every wavelength.
5. Fixed-spectrum Rietveld scale/site/profile/background/sample derivatives,
   component-indexed reports, and format-6 persistence with formats 1--5
   migration.
6. Explicit rejection of component-wavelength and guarded lattice refinement
   until their shared topology derivative contract is implemented.

The QARR prerequisite review additionally identified and implemented two typed
structural-intensity inputs: caller-supplied fixed X-ray `f' + i f''` offsets
and polarized Bragg--Brentano LP with the instrument polarization mapped
directly to `P`. Both execute in the native value/JVP/VJP path. Persistence
format 7 stores these models and loads formats 1--6.

## Accepted QARR checkpoint

The deterministic pure-Rust three-stage workflow first refines phase scales plus
shared U/V/W/zero, then adds isotropic displacement, size, microstrain, and
preferred orientation, and finally polishes the three linear phase scales with
nonlinear parameters fixed. The fixed-anisotropic reviewed result is:

- Al2O3 30.460%, ZnO 34.113%, CaF2 35.427%;
- maximum absolute weighed-fraction error 1.004 percentage points (limit 2);
- Poisson-weighted Rwp 0.19828 (limit 0.20);
- unit-weight Rwp 0.13179 (limit 0.15);
- background-subtracted profile correlation 0.99062 (limit 0.98).

The two residual gates are intentionally separate: assigning
`sigma=sqrt(max(counts, 1))` changes the weighting and must not be compared to a
unit-weight prototype number. Current approximations are recorded in the
machine-readable report: Al2O3 CIF tensors are evaluated directly and fixed,
sites lacking CIF displacement values start at `Uiso=0.005 Å²` and refine,
Cu K-alpha1 fixed dispersion is reused for K-alpha2, SH/L=0.002 uses the
documented equal-height FCJ mapping, and absorption is not yet active. The
final scale-only covariance is propagated analytically through the
Hill--Howard normalization and reported as a phase-fraction uncertainty. The
reviewed release run ends safely under explicit stage budgets. One test
repeats the full native workflow exactly and another compares its scientific
measurements with the independent Python implementation.

PbSO4 now covers both probes in Rust: the monochromatic neutron runner and an
exact fixed-doublet X-ray runner retaining all 383 reflection families. The
joint example combines this spectrum with the neutron histogram, and the
analytical mixed-radiation objective has dedicated Rust coverage.

## Remaining QARR sequence

1. After fixed-component real-data acceptance, implement refinable wavelength
   ratios and a multi-wavelength lattice guard. Move the component-level loop
   across the PyO3 boundary only if benchmark evidence justifies the ABI.

## Remaining TOF sequence

The single-bank fixed-instrument TOF Le Bail sequence is complete. The pinned POWGEN LaB6
case now covers the typed TOF record, reflection positions, fused accumulation,
all analytical instrument derivatives, normalization, finite differences, and
nonnegative intensity extraction. Its optional 16-term Chebyshev residual is
added above the fixed Smooth Bruckner baseline and updated analytically in every
cycle. A separate live GSAS-II comparison checks
real-bank parameter translation and reconstructs the oracle's peak-only pattern
from the same extracted intensities; it also gates the two Chebyshev-enabled
native workflows' Rwp and correlation deltas. Native format 3 separately
round-trips explicit microsecond histograms and resumable TOF Le Bail state. A
reviewed atomic multi-bank model now shares exact phase/reflection topology
while preserving bank-local observations, instruments, backgrounds, scales,
and intensities. Its analytical shared-cell extension has synthetic recovery,
finite-difference, and exact-continuation gates. The next real-data increment
must identify a multi-bank dataset with citable cell truth and plain-array
oracle output before adding a golden result. Selected bank-local instrument
motion follows; structural TOF Rietveld refinement remains future scope and is
not implied by this acceptance.

## Review gates

Every new real-data case must document the source URL, immutable revision where
available, hashes, redistribution decision, radiation, geometry, coordinate
units, included range, preprocessing, refined parameters, stopping reason, and
acceptance criteria. A failed or unsupported capability is reported as such;
golden results are never silently regenerated or relaxed.
