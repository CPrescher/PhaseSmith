# MgSiO3 reduction diagnosis and raw-frame recovery

Two interacting validity errors explain the fresh v4 failure. Correcting the
example's export policy, without changing integrated amplitudes or uncertainties,
recovers the complete structure in a new original-frame run. This is exposed
MgSiO3 development evidence with supplied cell, symmetry and element types.

## Comparison and controls

The original HKL reflection block is separated from its appended SHELX model
instructions before any solve. A new PhaseSmith control on those 5,389 source
observations recovers Mg16 Si16 O48 at R1=0.0481402, with maximum/RMS coordinate
errors 0.0120033/0.0068192 A. No atomic counts or coordinates enter that search.

Comparison uses Pbca/Friedel equivalent families and inverse-sigma-squared means.
One global rescale accounts for arbitrary intensity units in diagnostics only;
source intensities never replace measured counts. On the identical 1,253 common
families, correlation improves from 0.77503 to 0.99866. For their 693 strong source
families it improves from 0.75677 to 0.99856. The strong-source condition I>10sigma
is a diagnostic population definition, not an export significance cut.

## Error 1: signal-dependent warning censoring

The native profile MISMATCH bit is set only for I>5sigma and reduced chi-square>4.
A slightly imperfect empirical template can therefore flag a bright, reliable
amplitude while an empty/background aperture passes. The example's unconditional
`flags == 0` export discards 711 otherwise usable measurements, carrying 71.27%
of the positive integrated signal in otherwise usable rows. Several entire
strong families disappear; others retain only their shadowed, near-zero copies.

Those rows have median profile correlation 0.9073 and median reduced chi-square
7.621. These are genuine residual warnings, not flags to erase. Version 5 retains
MISMATCH as an explicit warning while every other flag still excludes. Original
values, uncertainties, flags and selected-row membership remain separately saved.
No native chi-square threshold or profile estimator changed.

## Error 2: unmasked DAC-occulted crossings

The raw HDF5 mask handled reserved detector values, not moving DAC shadows.
Prediction on the detector footprint was mistaken for actual illumination.
For 1,645 otherwise usable strong-equivalent measurements, 133 have less than
one tenth of their family median; every one has exit angle >35.86 degrees in the
sample frame. None of those nearly blank copies lies inside 35 degrees. Their
other equivalents retain normal intensities. The sharp attenuation boundary and
source notes about seat/body frames identify a missing aperture-validity model.

The diagnostic uses a conservative 35-degree half-opening, with the DAC axis
along the incident beam at declared omega zero. Incident and outgoing unit rays
are rotated by R(axis,-omega) into the sample frame. Both absolute ray/axis dot
products must be >=cos(35 degrees). Boundaries are closed within eight float64
epsilons in cosine. This is geometric exclusion, not fitting a scale factor to
recover blocked counts. It does not model finite-rocking partiality or absorption.
The axis/opening remain dataset-specific, exposed-data assumptions rather than
independent DAC metrology. In particular, some bright measurements beyond the
conservative boundary are also excluded; low intensity alone never defines a mask.

## Causal checks

All variants retain original v4 amplitudes and sigmas. Search controls, seed,
trials, cycles and resource policy stay fixed; references are read after results
are saved. The 35-degree cone was declared before these model searches.

| Export policy | Observations | PhaseSmith result | SHELXT result |
| --- | ---: | --- | --- |
| Original flags==0, no DAC mask | 5,708 | No candidate | No candidate within 60 s |
| Retain MISMATCH, no DAC mask | 6,419 | No candidate | No candidate within 60 s |
| DAC mask, original flags==0 | 3,625 | No candidate | Not run |
| DAC mask and retain MISMATCH | 4,310 | Correct; R1=0.0413175 | Correct; reported R1=0.038 |

A separate source-HKL membership control also solves using fresh amplitudes
(PhaseSmith R1=0.0423940). It is explicitly an attribution control: vendor row
membership is forbidden as a substitute for visibility in the complete workflow.
The geometric variant and v5 worker use no vendor membership or intensities.

## Fresh complete confirmation

Frozen protocol `validation/enstatite-raw-workflow-v5.json`, implementation
`98662b0`, ran again from all 142 selected original HDF5 frames in a fresh external
directory. It retained the 278-teacher bank, identical positional calibration,
and all 7,277 reduced target rows. Every original amplitude, sigma, statistical
sigma, profile diagnostic, parameter, aperture value and flag is bitwise identical
to v4. Only explicit search eligibility changes.

The geometric mask excludes 2,404 rows; other quality exclusions remove 563 more.
The resulting 4,310 observations include 233 negative intensities and 685 rows
retaining their MISMATCH warning. PhaseSmith independently infers Mg16 Si16 O48,
reaches R1=0.0413175, and passes post-search labelled-coordinate recovery at
maximum/RMS errors 0.0078273/0.0050056 A. The full worker takes 100.42 seconds;
search takes 0.608 seconds. These are run records, not a speed comparison.
The selected observations match the geometric diagnostic bitwise.

Small records are `benchmarks/enstatite-reduction-diagnosis-macos.json` and
`benchmarks/enstatite-raw-workflow-macos.json`; the preserved negative result is
`benchmarks/enstatite-raw-workflow-v4-macos.json`. Local paths and SHA-256 hashes
locate the comparison scripts, PNG/PDF plots, observations and candidate CIFs.
All large products and original files remain outside Git and source files are
unchanged. Native equations, default builds and frozen holdout results are unchanged.
There is no standalone structural refinement or universal-recovery claim.

The [shared-mask proposal](single-crystal-mask-proposal.md) describes replacing
example-only shadow handling with reusable per-frame validity and Dioptas controls.
