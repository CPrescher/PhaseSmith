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

**Issue:** none in the supported TOF profile and bank-translation scope.
Structural TOF Rietveld refinement remains separate future functionality.

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
legacy-GSAS result, and is now onboarded. Related public citrate/Si deposits
are now explicitly queued: sodium dihydrogen citrate polymorph II, anhydrous
tripotassium citrate, and the trirubidium citrate anhydrous/monohydrate pair.
They will be onboarded individually because preferred orientation, hydration,
and phase-specific profile records differ even though the pdCIF layout is
similar.

## Recommended priorities

1. **Unify campaign execution.** Add one manifest-driven command that runs the
   available comparison drivers, validates the pinned revision, and writes a
   summary without regenerating reviewed goldens silently.
2. **Improve conversion diagnostics.** Report missing phases, contradictory
   optics, unsupported legacy records, origin choices, and assumed radiation
   models before refinement. Bath and XRED show that this will improve user
   outcomes more than another profile term.
3. **Expand the citrate/Si holdout series.** Reuse the narrow reviewed adapter
   only where the related IUCr deposits expose the same complete counts,
   structures, instrument, standard fraction, and deposited-fit contract.
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
- `validation/results/2026-08-11-campaign-echidna.json`
- `validation/results/2026-08-11-campaign-pbso4.json`
- `validation/results/2026-08-11-campaign-powgen.json`
- `validation/results/2026-08-11-campaign-bath-ltl.json`
- `validation/results/2026-08-11-campaign-xred-tio2.json`
- `validation/results/2026-08-11-campaign-iucr-si-standard.json`

The Rowles FPA diagnostic remains in
`validation/results/2026-08-11-rowles-gsasii-fpa.json`.
