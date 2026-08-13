# PhaseSmith versus GSAS-II: real-data benchmark report

**Campaign date:** 2026-08-11
**GSAS-II oracle revision:**
`c0bc79b259cdf0065480b5fbd57674ddf12c4a23`

## Executive conclusion

PhaseSmith has reached practical GSAS-II parity for the common profile,
structure-factor, QPA, and refinement models exercised by the qualified
benchmarks. Every fresh strict or matched-workflow comparison passed: Rowles
1a/1e, QARR 1g/1h, NIST SRM 660c at the normal SH/L value, APS sucrose,
PbSO4 X-ray/neutron, and POWGEN. The strongest laboratory-X-ray cases differ
from GSAS-II by only 0.069–0.370 Rwp percentage points, and their phase
fractions agree within 0.415 percentage points.

The poor absolute fits are not one recurring missing PhaseSmith equation.
They separate into three causes:

1. incomplete or contradictory deposited project metadata (Bath, XRED,
   Echidna, opXRD);
2. deliberately unmatched or specialized physics (large-FCJ NIST stress and
   the released full-FPA NIST curve);
3. historical conversion semantics that current GSAS-II cannot reproduce
   either (Bath).

The evidence therefore does **not** justify adding TOPAS-style fundamental
parameters, Soller, or spectral-band models now. The highest-value next work is
better project/data conversion and a unified benchmark runner. The newly
onboarded IUCr dicesium-citrate example supplies the previously missing
in-house silicon-standard test. With Si made authoritative for specimen
displacement, PhaseSmith recovers 14.593 wt% Si, pinned GSAS-II 14.621 wt%, and
the deposited legacy refinement reports 13.02 wt%.

## Method

Each result was classified before comparison. Strict parity requires a matched
common model; workflow comparisons disclose differing optimizers or staged
recipes; capability cases diagnose real data without claiming strict parity;
and kernel/translation cases compare prepared numerical quantities directly.
GSAS-II ran only in its isolated pinned interpreter and returned plain records.
PhaseSmith never imports GSAS-II or consumes its internal objects.

Fresh campaign runs use one deterministic release repetition. This is enough
for scientific comparison; host timing medians remain covered by the separate
performance harness. All numerical gates were declared in the comparison
drivers before these runs.

## Results

Rwp values are Poisson-weighted percentages. A “qualified pass” means that a
predeclared wider workflow tolerance passed; it is not strict model parity.

| Dataset | PhaseSmith Rwp | GSAS-II Rwp | Outcome | Main diagnosis |
| --- | ---: | ---: | --- | --- |
| Rowles 1a | 8.782 | 9.085 | pass | Matched common workflow; maximum phase-fraction delta 0.134 percentage points. |
| Rowles 1e | 8.264 | 8.195 | pass | Matched common workflow; maximum phase-fraction delta 0.240 percentage points. |
| QARR 1g | 18.226 | 18.395 | pass | All residual, correlation, and fraction gates pass. |
| QARR 1h | 18.358 | 18.637 | pass | Matched holdout also passes; maximum fraction delta 0.415 percentage points. |
| NIST 660c, SH/L=0.002 | 20.495 | 20.865 | pass | Normal common-subset FCJ parity. |
| NIST 660c, SH/L=0.02 | 18.912 | 17.026 | expected failure | Deliberate large-asymmetry stress exposes FCJ implementation differences. |
| APS sucrose | 14.184 | 14.483 | pass | Matched high-resolution Le Bail workflow. |
| Echidna LaB6 | 40.858 | 59.410 | qualified pass | Deposited wavelength is approximate; not strict parity. |
| PbSO4 X-ray | 10.346 | 10.573 | pass | Independent PhaseSmith versus joint GSAS-II refinement still agrees closely. |
| PbSO4 neutron | 4.217 | 4.535 | pass | Correlation and refined-cell gates also pass. |
| POWGEN LaB6 | 26.192 | 26.297 | pass | All direct translation, profile, reconstruction, and workflow gates pass. |
| Bath K-LTL | 21.972 | 21.303 | capability failure | Released legacy GSAS curve is 7.791%; neither current program reconstructs it. |
| Bath Li-LTL | 23.559 | 19.493 | capability failure | Released legacy GSAS curve is 5.201%; missing project semantics dominate. |
| Bath Cs-LTL | 15.062 | 12.080 | capability failure | Released legacy GSAS curve is 3.492%; same conversion limitation. |
| XRED TiO2 | 22.256 | 44.801 | diagnostic only | Unknown instrument/composition invalidate residual parity; phase fractions agree. |
| IUCr citrate + NIST Si 640b | 10.180 | 8.894 | qualified capability pass | Deposited legacy GSAS is 6.226%; displacement-anchored Si is 14.593/14.621/13.02 wt% in PhaseSmith/GSAS-II/legacy. |
| IUCr sodium citrate + NIST Si 640b | 18.183 | 18.945 | qualified holdout pass | All seven common-subset gates pass; Si is 22.173/21.651/18.74 wt% in PhaseSmith/GSAS-II/deposited GSAS. |

### IUCr dicesium citrate with NIST silicon internal standard

The checksum-pinned IUCr supplementary CIF is unusually complete: it contains
3,217 raw laboratory counts from a Bruker D2 Phaser, both Cu Kα wavelengths,
all three phase structures, the NIST SRM 640b identity, the deposited
background and calculated curve, and the legacy-GSAS phase fractions. The
converter reconstructs the exact 2,820-point refinement window and independently
recomputes the archived 6.226% Rwp and 4.966% Rp before either current program
runs. The ambiguous silicon Hall declaration is visibly normalized to IT 227.

Both current workflows fix the deposited background and share the doublet,
U/V/W/X/Y initializer, FCJ SH/L, structures, and per-phase isotropic size and
microstrain. Before the multiphase fit, the archived instrument zero of
-0.0448° is fixed and fixed-cell Si alone calibrates specimen displacement in
the isolated (220), (311), and (400) windows; (111) is excluded because it
overlaps the dominant citrate peak. The calibrated displacement is -0.156 mm
in PhaseSmith and -0.235 mm in GSAS-II and is frozen thereafter. Holding zero
fixed avoids the strong zero/displacement correlation exposed by an initial
two-parameter calibration trial.

PhaseSmith reaches 10.180% Rwp and fractions 60.352% dicesium citrate, 25.055%
cesium citrate, and 14.593% Si. Pinned GSAS-II reaches 8.894% and 59.463%,
25.916%, and 14.621%, respectively. Their maximum mutual fraction difference
is 0.889 percentage points, while their Si fractions differ by only 0.028
percentage points. The independently fitted displacement values differ by
0.078 mm, showing that profile conventions still affect the inferred specimen
height even when their quantitative Si result agrees closely.

The earlier global-zero workflow reached apparently better agreement with the
deposited Si fraction, but citrate peaks were allowed to steer the positional
calibration. Making Si authoritative exposes the actual common-model gap: both
programs now overestimate the deposited Si fraction and remain above the
legacy 6.226% curve. The remaining discrepancy is consistent with omitted
phase-specific legacy profiles and Stephens anisotropic broadening.

**Issue:** standard anchoring is mandatory for this workflow. After anchoring,
profile-model compression and specimen-phase physics dominate; no new
fundamental-parameters or spectral-band term is isolated yet.

### Rowles laboratory QPA

Both mixtures pass the matched common-subset contract. For 1a the
PhaseSmith/GSAS-II fractions are Al2O3 1.533/1.399%, ZnO 3.303/3.304%, and
CaF2 95.164/95.297%. For 1e they are 57.278/57.038%, 14.013/14.078%, and
28.709/28.884%. This validates the fixed Cu Kα doublet, structure factors,
Hill–Howard conversion, profile terms, and staged optimizer at the intended
GSAS-II parity level.

The separate geometry-compression diagnostic makes the feature decision
clearer. GSAS-II's empirical Rowles profile gives 9.085% and 8.195% Rwp, while
the profile compressed from the deposited geometry gives 13.928% and 11.710%.
Adding more specialized Rowles optics would currently move away from the best
GSAS-II result, not toward it.

**Issue:** none for GSAS-II parity. The remaining gap to the deposited TOPAS
result belongs to specialized instrument/source modeling explicitly outside
this campaign.

### IUCr QARR 1g and 1h

Both fresh matched comparisons pass all four gates. The Rwp deltas are 0.169
and 0.279 percentage points; profile-correlation deltas are 0.00061 and
0.00056. The largest phase-fraction deltas are 0.355 and 0.415 percentage
points.

An older 1h failure used a different native parameterization. It must not be
compared with this deliberately matched trace-mean-isotropic contract. The
fresh result demonstrates that the held-out pattern transfers when the common
model is aligned.

**Issue:** small optimizer/background parameterization differences only; no
new physical term is indicated.

### NIST SRM 660c laboratory X-ray

At SH/L=0.002, all five strict common-subset gates pass. PhaseSmith and GSAS-II
have correlations 0.95959 and 0.95982 and differ by 0.370 Rwp percentage
points. At the deliberately large SH/L=0.02, Rwp, unit-Rwp, and correlation
gates fail exactly as predeclared. This isolates the difference between
PhaseSmith's published continuous equal-height FCJ mapping and GSAS-II's
discretized implementation.

Both common-subset refinements remain far above the released NIST full-FPA
curve at 6.055% Rwp. That does not invalidate GSAS-II parity: the released fit
contains richer source and instrument physics not present in either matched
workflow.

**Issue:** large-asymmetry behavior is a bounded, known parity gap. It should
only be prioritized if future ordinary laboratory datasets repeatedly require
SH/L values near 0.02.

### APS sucrose

The 23,003-bin, 811-reflection matched Le Bail workflow passes with a 0.299
percentage-point Rwp delta and a 0.01739 correlation delta. Both programs use
the same fixed Smooth Bruckner array, constant Chebyshev residual, wavelength,
and profile start. Pinned GSAS-II reports its internal SH/L floor of 0.0005
when the requested value is zero; PhaseSmith's zero is exactly symmetric.

**Issue:** no material parity gap. This is a valuable high-resolution control,
but it does not add in-house geometry coverage.

### Echidna LaB6

This neutron case passes only its intentionally wide workflow tolerances.
PhaseSmith has the lower Rwp, while GSAS-II has the higher profile correlation.
The source's wavelength is explicitly approximate, so the residual difference
cannot identify which program has the more accurate physical model.

**Issue:** calibration metadata, not actionable profile physics.

### PbSO4 X-ray and neutron

Both probes pass despite the disclosed workflow difference: PhaseSmith fits
the histograms independently, while GSAS-II jointly refines a shared structure.
X-ray correlations are 0.99555/0.99554 and neutron correlations are
0.99664/0.99669. The maximum relative refined-cell difference is 0.000424.

**Issue:** no current parity issue. Joint multi-histogram parameter sharing is
a product/workflow enhancement, not a missing peak-profile equation.

### POWGEN LaB6

This is the campaign's cleanest direct numerical translation test. Peak
positions agree exactly; alpha and beta errors are at floating-point roundoff;
the variance error is `1.82e-12 us^2`; selected profiles differ by at most
0.131%; and the same-extracted-intensity reconstruction correlates at
0.999994. All nine gates pass.

**Issue:** none in the supported TOF profile and bank-translation scope. This
POWGEN case alone does not define a structural observation/correction contract;
the later LANL nickel multi-bank gate now validates the separate structural TOF
workflow against the same pinned GSAS-II revision.

### Bath zeolite L

The K, Li, and Cs results are stable fresh reproductions. GSAS-II matches the
deposited background point-for-point but still cannot reconstruct the released
legacy GSAS curves from the publication CIF and archived initializer. The
original files also contain optics descriptions that conflict with the
deposited README.

**Issue:** legacy project conversion fidelity and missing semantics. Adding
new profile physics to PhaseSmith would be unjustified until the historical
model can be reconstructed in another current program.

### XRED TiO2

The dataset supplies a background-subtracted pattern and COD structures but no
instrument metadata or certified composition. PhaseSmith and GSAS-II reach
very different residual minima, yet their anatase fractions are 86.049% and
85.864%, a 0.185 percentage-point difference. GSAS-II also requires an explicit
origin-choice translation for the deposited anatase CIF.

**Issue:** missing metadata and optimizer recipe. The close phase fractions
support the structure-factor and QPA paths; the residuals cannot be used as a
strict parity metric.

## Candidate-source audit

NIST describes SRM 1979 measurements on both APS 11-BM and a laboratory
diffractometer with a Johansson monochromator and position-sensitive detector.
However, the linked public supplemental archive contains nine 11-BM CIF
patterns. The material additionally needs anisotropic crystallite-size,
stacking-fault, and band-pass modeling. It was therefore not converted into a
laboratory GSAS-II parity case. See the [NIST publication and supplemental-data
description](https://pmc.ncbi.nlm.nih.gov/articles/PMC11239193/).

The [opXRD deposit](https://zenodo.org/records/14279434) contains 92,552
patterns in two archives totaling 988 MB, but only 2,179 have even partial
structural information. Its authors explicitly state that further metadata
annotation is needed. It is useful for machine-learning robustness, not as a
bulk Rietveld oracle. The already benchmarked XRED TiO2 subset remains the
tractable labeled case.

The [APS 11-BM standards page](https://wiki-ext.aps.anl.gov/ug11bm/index.php/Standards_Data)
provides raw patterns, CIFs, and GSAS-II instrument files for several standards,
including pure Si 640c/640d. These are good future synchrotron controls but do
not satisfy the request for another in-house pattern or a specimen containing
Si as an internal standard.

The initial search missed the multi-block supplementary CIF for the IUCr
dicesium hydrogen citrate study. It does combine raw laboratory counts, all
phase structures, the Si 640b fraction, instrument metadata, and a deposited
legacy-GSAS result, and is now onboarded. The related sodium dihydrogen citrate
polymorph-II deposit is also checksum-registered, converted, and compared with
the exact pinned GSAS-II revision. Trial refinements correctly rejected two
non-identifiable models: free W/X/Y made the oracle's standalone Lorentzian
width nonphysical, while fixed-profile isotropic width refinement drove its Si
microstrain negative. The accepted preferred-orientation common subset fixes
those terms and refines only scales, a constant residual, and March--Dollase
(001). PhaseSmith/GSAS-II return 18.183%/18.945% Rwp, 0.95589/0.94997
correlation, 22.173%/21.651% Si, and March ratios 0.63851/0.63237. All seven
cross-implementation gates pass. The deposited 8.433% and 18.74 wt% Si remain
the richer-model reference, not a parity target, because generalized spherical
harmonics, Stephens anisotropy, and Suortti roughness are omitted.

The anhydrous tripotassium-citrate deposit is now also complete. Its 1.44 wt%
Si content is too weak to transfer the sodium displacement-calibration
contract: PhaseSmith and pinned GSAS-II infer -0.02492 and +0.00180 mm from the
same 128 selected Si-window points; the PhaseSmith millimetre value is
conditional on the disclosed 141.5 mm radius assumption because the source
does not deposit a radius. That calibration is therefore diagnostic
and is not applied. With zero displacement and the same fixed dominant-phase
profile, the implementations give 23.965%/23.933% Rwp, 0.61846/0.62079
correlation, and 4.450%/4.343% Si. The close common-subset agreement is a
qualified parity pass, while the large gap to deposited 4.853% Rwp and 1.44 wt%
Si is the expected phase-specific-model and weak-anchor failure. The
trirubidium citrate anhydrous/monohydrate pair was therefore source-audited
before assuming that the Si-anchor contract transfers.

The anhydrous trirubidium case is now source-audited and corrected. Legacy GSAS
profile-function-4 `LX=3.634` is a Lorentzian size term and maps to the current
`X/cos(theta)` axis, not `Y*tan(theta)`. Its `shft=-8.7503` centidegree
coefficient maps analytically, at the deposited 141.5 mm radius, to
-0.1080505 mm in the current PhaseSmith/GSAS-II Bragg--Brentano displacement
parameter. The legacy manual's physical sample-shift variable has the opposite
sign (+0.1080505 mm). With those source terms fixed, PhaseSmith and pinned
GSAS-II reach 9.579%/9.735% Rwp,
0.94732/0.94248 correlation, and 2.575%/2.548% Si; every common-model parity
gate passes. The 2.15 wt% Si-only windows still infer -0.07519/-0.11031 mm, so
their 0.03512 mm disagreement remains a warning against replacing the directly
translated source value with a weak-window calibration. The monohydrate remains
lower priority because its 1.30 wt% Si, absent radius, phase-specific profiles,
Stephens anisotropy, and Suortti roughness add no cleaner discriminator.

The deposited and Poisson-weighted legacy-curve residuals were also recomputed
separately to exclude weighting vocabulary as an explanation for these gaps.
They are effectively identical: potassium is 4.852875% deposited versus
4.852883% Poisson, and anhydrous rubidium is 2.458338% versus 2.458326%.
Therefore the residual discrepancy cannot be attributed to the Rwp weight
convention. For rubidium, the largest diagnosed limitation was instead a
profile-function translation error plus omitted source displacement.

### Isotropic size/strain ablation and Stephens decision

A controlled phase-local ablation now tests whether PhaseSmith's existing
isotropic coherent-domain size, Gaussian RMS microstrain, and Lorentzian
microstrain terms can explain the potassium and anhydrous-rubidium gaps. The
instrument, wavelength doublet, FCJ geometry, zero/displacement, cells,
structures, and all silicon width terms remain fixed. Each candidate starts
from the exact two-scale/one-constant-background solution, is run from three
separated initial values, and finishes with the sample term, both scales, and
background in one weighted Jacobian. Acceptance requires full rank, no
zero-width boundary, repeatable Rwp within 0.005 percentage points, and less
than 0.98 absolute sample-versus-linear column correlation.

For potassium, finite size is the only candidate that passes every gate. It refines to
90.1877 nm, lowers Poisson Rwp from 23.9648% to 20.7374%, raises profile
correlation from 0.61846 to 0.77766, and changes Si from 4.4503% to 2.0456%.
After correcting rubidium `LX` and `shft`, its base Rwp is already 9.5794%.
Size, Gaussian strain, Lorentzian strain, and both size/strain combinations can
produce best-run values near 8.06--8.18%, but none reaches the same basin from
three starts. The earlier accepted 56.23 nm rubidium size solution is therefore
invalidated: it was compensating for an omitted position/profile translation,
not demonstrating transferable sample physics.

This still falsifies the stronger hypothesis that current isotropic width
models reproduce either deposited target, but it also demonstrates that model
selection performed before correct source translation can assign a false
physical meaning to a compensating width parameter.

GSAS-II does implement this model. In the pinned revision
`c0bc79b259cdf0065480b5fbd57674ddf12c4a23`, the phase `Mustrain` selector
offers isotropic, uniaxial, and `generalized`; the latter is explicitly the
P. W. Stephens model and uses Laue-class-dependent fourth-order polynomials in
`h`, `k`, and `l`. This is distinct from GSAS-II `HStrain`, which modifies the
lattice metric and therefore peak positions. PhaseSmith's production version
must be independently derived from Stephens, J. Appl. Cryst. 32 (1999)
281–289, DOI `10.1107/S0021889898006001`, with value and analytical derivatives
in the same pass; GSAS-II remains only the pinned black-box oracle.

The reviewed record is
`validation/results/2026-08-13-citrate-isotropic-broadening-ablation.json`.
The intentionally slow external campaign can be repeated with
`oracle/scripts/benchmark_citrate_isotropic_broadening.py`; it is not part of
the ordinary test suite.

The orthorhombic Stephens convention is now separately pinned. A Pnma oracle
fixture at exact GSAS-II revision `c0bc79b259cdf0065480b5fbd57674ddf12c4a23`
contains 395 public reflection rows and three normalized profile probes. The
independently derived adapter maps GSAS-II's generalized stored coefficients
to physical ångström⁻⁴ variance coefficients using `1e-12/(8 ln 2)` for pure
fourth powers and `3e-12/(8 ln 2)` for its factor-three mixed basis. PhaseSmith
matches every oracle Gaussian variance to `2.3e-15` centidegree² absolute and
every Lorentzian FWHM to `3.4e-16` centidegree absolute; selected complete TCH
profiles differ by at most `2.40e-6` of peak height. This validates the
convention and profile composition, but is deliberately separate from whether
Stephens parameters are identifiable or sufficient on the citrate patterns.
The fixture is `oracle/fixtures/stephens_orthorhombic_v1` and its explicit
generator is `oracle/scripts/generate_stephens_orthorhombic.py`.

The corresponding real-data sufficiency test is also complete. It fixes the
instrument, wavelength doublet, FCJ geometry, zero/displacement, cells,
structures, silicon widths, six deposited Stephens coefficient ratios, and
deposited Gaussian/Lorentzian mixing fraction. With only the source Stephens
width active, potassium changes from 23.9648% to 23.9250% Poisson Rwp. On the
corrected rubidium contract it changes 9.5794% to 9.5711%. Those changes are
far too small to explain the deposited 4.8529% and 2.4583% curves.

Allowing one common multiplier of the deposited Stephens shape reaches only
23.5954% for potassium and 9.4158% for rubidium while requiring 94.4 and
382.1 times the deposited amplitudes; the potassium result also misses the
three-start repeatability gate. On corrected rubidium, size plus fixed Stephens
also fails repeatability; this invalidates the former 56.26 nm conclusion.
Joint size/amplitude refinement is not repeatable for either pattern. Thus the
implementation and GSAS-II convention are verified, but Stephens broadening is
not the missing mechanism that closes these citrate targets under the fixed
nuisance model. The reviewed record is
`validation/results/2026-08-13-citrate-stephens-ablation.json`; rerun it with
`oracle/scripts/benchmark_citrate_stephens.py`.

### Rubidium legacy-profile residual forensics

A 16-case full factorial pinned-GSAS-II ablation separates legacy `LX`, `shft`,
`trns`, and deposited Stephens terms while refining only two phase scales and
one residual-background constant. Translating `shft` to -108.0505 micrometres
dominates the recoverable weighted-SSE improvement (87.3% Shapley share).
Correcting `LX` contributes 18.0%; Stephens contributes 0.07%. The deposited
transparency translation has a negative 5.36% contribution and worsens every
matched fixed-structure branch in which it is enabled.

This sign is not an adapter guess. The
[legacy GSAS technical manual](https://subversion.xray.aps.anl.gov/EXPGUI/gsas/all/GSAS%20Manual.pdf)
defines the profile argument as
`delta_T' = (T - T_phase) + shft*cos(theta) + trns*sin(2theta)`;
the peak-center motion is therefore the negative of the stored coefficient.
It also defines `mu_eff = -9000/(pi*R*trns)`, so the deposited positive
`trns=1.30` has a formally negative effective absorption and is an empirical
signed position correction rather than a physical transparency measurement.
A sign/scale sensitivity check confirms the distinction: reversing the
deposited sign lowers Rwp to 9.3426%, but contradicts the source coefficient;
the deposited sign raises it to 10.2240%. The worker consequently preserves
the numerical source convention and does not promote transparency into the
PhaseSmith production model.

The best fixed subset is corrected `LX` plus source shift plus Stephens at
9.7261% Rwp and 0.94259 correlation. It closes 73.19% of the weighted-SSE gap
between the former 18.338% common model and the 2.4583% deposited curve, but
does not reconstruct the latter. Of its remaining weighted SSE, 65.8% lies in
17--30 degrees; the residual correlates more strongly with the profile's width
mode (0.467) than its position mode (-0.291). This points to low-angle
profile/intensity/conversion behavior, not another global position correction.
The checked record is
`validation/results/2026-08-13-citrate-rubidium-residual-forensics.json`,
generated by `oracle/scripts/benchmark_citrate_residual_forensics.py`.

### Rubidium low-angle component audit

The follow-up audit restricts refinement weight to the 643 samples from
17.004916 degrees through 30 degrees while retaining the complete pattern in
the GSAS-II project. This avoids changing the observable fixed-background
indexing. Each controlled GSAS-II case re-refines only the two phase scales and
a constant residual background unless the position probe explicitly adds the
specimen shift. The background-shape checks are independent weighted Legendre
projections on the audited interval. Unconstrained profile fits are rejected
because they drive `W`, `X`, and `SH/L` negative; the reviewed result instead
uses fixed non-negative scans.

The source-translated profile gives 9.8275% local Poisson Rwp after low-angle
scale/background adjustment, compared with 2.7107% for the deposited curve.
No isolated component closes half of that weighted-SSE gap: specimen shift
closes 45.21%, the axial-profile scan 36.29%, `W` 27.04%, peak-group amplitude
projection 22.65%, residual-background shape 19.07%, and `X` 15.66%. The best
fixed values are `W=20`, `X=5`, and the pinned GSAS-II observable `SH/L=0.002`
lower boundary. The latter is not evidence for zero physical axial divergence:
the pinned evaluator applies `max(SH/L, 0.002)`, while the deposited legacy
model records separate `S/L=H/L=0.0097` and modern GSAS-II compresses their sum
into one field.

Combining the three best profile values with a rank-four, condition-2.54
Legendre residual-background projection reaches 4.9772% Rwp and closes 80.47%
of the gap, but remains well above the deposited 2.7107%. Deposited/current
peak-group comparisons have a median area ratio of 1.8006, RMS-width ratio of
1.6200, absolute centroid shift of 0.02466 degrees, and absolute skewness
difference of 0.3906. The evidence is
therefore coupled across reflection intensity/scale, background, position,
symmetric width, and axial-profile compression. It does not justify another
production broadening term. The next diagnostic is a source-native comparison
of reflection intensities and legacy profile-function-4 axial shapes. The
checked record is
`validation/results/2026-08-13-citrate-rubidium-low-angle-forensics.json`,
generated by `oracle/scripts/benchmark_citrate_low_angle_forensics.py`.

### Rubidium source-reflection fidelity

The source pdCIF provides 1,197 reflection rows rather than only a fitted
profile. The converter now preserves their Miller indices, phase and wavelength
IDs, measured and calculated F-squared, calculated phase, d-spacing, and I100
as a neutral CSV. The two wavelength rows have identical calculated F-squared,
leaving 600 unique reflections: 591 rubidium-citrate and nine silicon. Twenty
rubidium-citrate reflections lie in the audited 17.004916–30 degree interval.

Using the source-deposited Cromer–Mann coefficients with the converted
structures reproduces those 20 low-angle F-squared values with correlation
0.99999994, 0.0397% median absolute relative error, 0.0289%
source-intensity-weighted L1 error, and 0.0301% source-normalized RMS error.
Converted d-spacings differ by no more than 1.54e-5 angstrom, consistent with
the source's five-decimal deposit. This independently clears the converted
coordinates, symmetry, isotropic displacement values, and source scattering
contract as the dominant low-angle failure.

PhaseSmith's production Waasmaier–Kirfel table gives 0.406% median and 0.301%
source-weighted L1 error over the same reflections. Its 6.53% p95 relative error
is concentrated in weak cancellation-sensitive reflections; the normalized RMS
error remains 0.296%. The production scattering choice is therefore measurable
but much too small to explain the seven-percentage-point low-angle Rwp gap. The
remaining source-native checkpoint is the legacy profile-function-4 axial
shape. The checked record is
`validation/results/2026-08-13-citrate-rubidium-source-reflection-fidelity.json`,
generated by `benchmarks/audit_citrate_source_reflections.py`.

## Recommended priorities

1. **Unify campaign execution — complete.**
   `validation/benchmark-campaign.json` and
   `tools/run_benchmark_campaign.py` now run the available comparison drivers,
   validate dataset hashes and the pinned revision, enforce case-level result
   assertions, and write a non-destructive aggregate report without silently
   regenerating reviewed goldens.
2. **Improve conversion diagnostics.** Report missing phases, contradictory
   optics, unsupported legacy records, origin choices, and assumed radiation
   models before refinement. Bath and XRED show that this will improve user
   outcomes more than another profile term.
3. **Expand the citrate/Si holdout series.** Sodium passes its restricted
   preferred-orientation common subset; potassium passes implementation parity
   but demonstrates that a 1.44 wt% Si displacement anchor is not transferable.
   Anhydrous rubidium independently confirms that conclusion at 2.15 wt% with a
   source-deposited radius. Keep the monohydrate as a lower-priority model-rich
   holdout rather than treating its 1.30 wt% Si as an accepted position anchor.
4. **Keep large-FCJ parity as a monitored holdout.** Do not replace the current
   published PhaseSmith mapping merely to mimic one pinned GSAS-II
   discretization. Reconsider only if multiple normal-use datasets fail for the
   same reason.
5. **Defer TOPAS FPA, Soller, and spectral-band expansion.** Current data do not
   show that these terms improve the targeted GSAS-II-parity workflows.

## Reproducibility artifacts

The detailed working ledger is `docs/gsasii-benchmark-campaign.md`. Fresh JSON
records are:

- `validation/results/2026-08-11-campaign-rowles-gsasii.json`
- `validation/results/2026-08-11-campaign-qarr-1g.json`
- `validation/results/2026-08-11-campaign-qarr-1h.json`
- `validation/results/2026-08-11-campaign-nist-srm660c.json`
- `validation/results/2026-08-11-campaign-sucrose.json`
- `validation/results/2026-08-13-campaign-iucr-sodium-citrate-si.json`
- `validation/results/2026-08-13-campaign-iucr-tripotassium-citrate-si.json`
- `validation/results/2026-08-13-campaign-iucr-trirubidium-citrate-si.json`
- `validation/results/2026-08-13-citrate-rubidium-residual-forensics.json`
- `validation/results/2026-08-13-citrate-rubidium-low-angle-forensics.json`
- `validation/results/2026-08-13-citrate-rubidium-source-reflection-fidelity.json`
- `validation/results/2026-08-11-campaign-echidna.json`
- `validation/results/2026-08-11-campaign-pbso4.json`
- `validation/results/2026-08-11-campaign-powgen.json`
- `validation/results/2026-08-11-campaign-bath-ltl.json`
- `validation/results/2026-08-11-campaign-xred-tio2.json`
- `validation/results/2026-08-11-campaign-iucr-si-standard.json`

The Rowles FPA diagnostic remains in
`validation/results/2026-08-11-rowles-gsasii-fpa.json`.
