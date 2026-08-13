# PhaseSmith–GSAS-II real-data benchmark campaign

This is the working ledger for the real-data comparison campaign requested on
2026-08-11. It is updated after each audited example. The final interpretation
belongs in `gsasii-benchmark-report.md`; this file records what was actually
run, what remains unavailable, and why.

The reviewed cases are now also machine-readable in
`validation/benchmark-campaign.json`. `tools/run_benchmark_campaign.py` verifies
the exact GSAS-II revision and registered dataset checksums, runs any selected
comparison drivers, checks their declared result contracts, and writes one
non-destructive aggregate JSON report. This orchestration changes no numerical
case and never regenerates the reviewed result files implicitly.

## Comparison contract

All GSAS-II results must come from revision
`c0bc79b259cdf0065480b5fbd57674ddf12c4a23` in its isolated interpreter.
GSAS-II remains an optional black-box oracle and is never a PhaseSmith runtime
dependency. A benchmark is classified before its residual is interpreted:

- **parity** — both programs receive a deliberately matched model and recipe;
- **workflow** — both programs solve the same data with disclosed native
  workflows, so only case-local outcome tolerances are meaningful;
- **capability** — incomplete metadata or a deposited reference prevents a
  strict parity claim, but the example still diagnoses a real use case;
- **kernel/translation** — prepared parameters, profiles, or intensities are
  compared directly, independently of optimizer behavior;
- **blocked** — the required raw pattern, structure, metadata, licensing, or
  oracle translation is not available.

`Rwp` values below are Poisson-weighted and are percentages unless explicitly
labelled otherwise. “Pass” means the predeclared case-local comparison passed;
it does not mean that the physical model is complete.

## Progress ledger

| Example | Probe / source | Contract | PhaseSmith / GSAS-II | Status | Current issue or conclusion | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Rowles 1a | laboratory Cu Kα X-ray, TOPAS deposit | matched workflow | 8.782 / 9.085 | fresh pass | Fractions differ by at most 0.134 percentage points; specialized TOPAS optics are not needed for GSAS-II-level parity. | `validation/results/2026-08-11-campaign-rowles-gsasii.json` |
| Rowles 1e | laboratory Cu Kα X-ray, TOPAS deposit | matched workflow | 8.264 / 8.195 | fresh pass | Fractions differ by at most 0.240 percentage points. | `validation/results/2026-08-11-campaign-rowles-gsasii.json` |
| Rowles FPA diagnostic | laboratory Cu Kα X-ray, deposited geometry | capability / profile compression | — / 13.928 (1a), 11.710 (1e) | audited diagnostic | The GSAS-II-compressed physical profile worsens both real fits relative to its empirical profile; do not add specialized Rowles FPA terms now. | `validation/results/2026-08-11-rowles-gsasii-fpa.json` |
| IUCr QARR 1g | laboratory Cu Kα X-ray | parity | 18.226 / 18.395 | fresh pass | Largest phase-fraction delta is 0.355 percentage points; all matched-workflow gates pass. | `validation/results/2026-08-11-campaign-qarr-1g.json` |
| IUCr QARR 1h | laboratory Cu Kα X-ray | parity / holdout | 18.358 / 18.637 | fresh pass | Largest phase-fraction delta is 0.415 percentage points; all matched-workflow gates pass. The older failure belongs to a different, unmatched recipe. | `validation/results/2026-08-11-campaign-qarr-1h.json` |
| NIST SRM 660c 100a, matched SH/L | laboratory Cu Kα X-ray | parity | 20.495 / 20.865 | fresh pass | All five gates pass; profile correlation differs by 0.00023. Both are still far above the released NIST FPA curve's 6.055% Rwp because this is only the empirical common subset. | `validation/results/2026-08-11-campaign-nist-srm660c.json` |
| NIST SRM 660c 100a, large SH/L | laboratory Cu Kα X-ray | stress | 18.912 / 17.026 | expected failure | Rwp, unit-Rwp, and correlation gates fail at SH/L=0.02, isolating the known difference between the published continuous PhaseSmith mapping and GSAS-II's discretized FCJ implementation. | `validation/results/2026-08-11-campaign-nist-srm660c.json` |
| Bath K-LTL | Rigaku SmartLab laboratory X-ray | capability / conversion | 21.972 / 21.303; released GSAS 7.791 | fresh failure | Neither program reconstructs the released legacy curve from the publication CIF; missing legacy project semantics dominate. | `validation/results/2026-08-11-campaign-bath-ltl.json` |
| Bath Li-LTL | Rigaku SmartLab laboratory X-ray | capability / conversion | 23.559 / 19.493; released GSAS 5.201 | fresh failure | Same conversion-fidelity limitation; the raw header also contradicts the deposited README about optics. | `validation/results/2026-08-11-campaign-bath-ltl.json` |
| Bath Cs-LTL | Rigaku SmartLab laboratory X-ray | capability / conversion | 15.062 / 12.080; released GSAS 3.492 | fresh failure | Same conversion-fidelity limitation. | `validation/results/2026-08-11-campaign-bath-ltl.json` |
| XRED TiO2 | laboratory X-ray | capability | 22.256 / 44.801 | fresh diagnostic | Unknown instrument and composition prevent residual parity, but anatase fractions agree within 0.185 percentage points after explicit origin-choice translation. | `validation/results/2026-08-11-campaign-xred-tio2.json` |
| APS sucrose | 11-BM synchrotron X-ray | matched workflow | 14.184 / 14.483 | fresh pass | Both programs share the fixed background and profile start; the remaining 0.299 percentage-point Rwp delta passes. GSAS-II's internal SH/L floor is disclosed. | `validation/results/2026-08-11-campaign-sucrose.json` |
| ANSTO Echidna LaB6 | constant-wavelength neutron | workflow | 40.858 / 59.410 | fresh qualified pass | Correlation is 0.90621 / 0.94608. Wide case-local tolerances are retained because the deposited wavelength is explicitly approximate; this is not strict model parity. | `validation/results/2026-08-11-campaign-echidna.json` |
| GSAS-II PbSO4 X-ray | laboratory-style Cu Kα tutorial pattern | workflow | 10.346 / 10.573 | fresh pass | Rwp delta is 0.227 percentage points and correlation delta is about 0.00001, despite PhaseSmith's independent fit versus GSAS-II's joint two-probe refinement. | `validation/results/2026-08-11-campaign-pbso4.json` |
| GSAS-II PbSO4 neutron | constant-wavelength neutron | workflow | 4.217 / 4.535 | fresh pass | Rwp delta is 0.318 percentage points; correlations differ by about 0.00005 and the maximum relative cell delta is 0.000424. | `validation/results/2026-08-11-campaign-pbso4.json` |
| POWGEN LaB6 | TOF neutron | kernel/translation and workflow | 26.192 / 26.297 | fresh pass | All nine gates pass. Positions are exact, parameter errors are near roundoff, selected-profile error is 0.131%, and the same-intensity reconstructed pattern correlates at 0.999994. | `validation/results/2026-08-11-campaign-powgen.json` |
| NIST SRM 1979 | laboratory and 11-BM line-profile standard | source-audited, not onboarded | — | blocked for requested lab comparison | NIST documents the laboratory experiment, but the linked public archive contains nine APS 11-BM CIF patterns. The decisive lab patterns are not exposed there, and the material also requires stacking-fault/anisotropic-size and band-pass modeling. | [NIST article and supplement](https://pmc.ncbi.nlm.nih.gov/articles/PMC11239193/) |
| opXRD examples | mixed public powder-XRD collection | source-audited, not onboarded | — | blocked for parity | The 988 MB deposit is primarily a machine-learning collection: only 2,179 of 92,552 patterns have even partial structural labels, and its authors say further metadata annotation is needed. XRED is retained as the tractable labeled subset already benchmarked. | [opXRD record](https://zenodo.org/records/14279434), [paper](https://doi.org/10.1002/aidi.202500044) |
| APS standards collection | synchrotron standards | source-audited control pool | — | deferred | Raw patterns, CIFs, and GSAS-II instrument files exist for LaB6, Si, Al2O3, ZnO, Cr2O3, CeO2, and NAC, but these are 11-BM synchrotron controls rather than new in-house coverage. Sucrose and NIST 660c already exercise that control class. | [11-BM standards](https://wiki-ext.aps.anl.gov/ug11bm/index.php/Standards_Data) |
| IUCr dicesium hydrogen citrate + Si 640b | Bruker D2 Phaser laboratory Cu Kα X-ray | standard-anchored capability / mixed-standard QPA | 10.180 / 8.894; deposited GSAS 6.226 | fresh qualified pass | The archived instrument zero is fixed, and fixed-cell Si alone calibrates specimen displacement to -0.156/-0.235 mm in PhaseSmith/GSAS-II before it is frozen. The deposited 13.02 wt% Si then refines to 14.593/14.621%; the two current programs differ by only 0.028 percentage points for Si. | `validation/results/2026-08-11-campaign-iucr-si-standard.json` |
| IUCr sodium dihydrogen citrate polymorph + Si 640b | Bruker D2 Phaser laboratory Cu Kα X-ray | preferred-orientation common-subset holdout | 18.183 / 18.945; deposited GSAS 8.433 | fresh qualified pass | The checksum-pinned `hb7585sup1.cif` converts 4,452 refined points and both phases. Fixed-cell Si calibrates displacement to -0.140/-0.201 mm in PhaseSmith/GSAS-II (conditional on the disclosed 141.5 mm PhaseSmith radius assumption). With U/V/W/X/Y, SH/L, and size/strain fixed, the March--Dollase (001) ratios are 0.63851/0.63237 and Si is 22.173/21.651 wt% versus 18.74 wt% deposited. All seven cross-implementation gates pass; generalized spherical harmonics, Stephens width, and Suortti roughness remain explicitly unmatched. | `validation/results/2026-08-13-campaign-iucr-sodium-citrate-si.json` |
| IUCr anhydrous tripotassium citrate + Si internal standard | Bruker D2 Phaser laboratory Cu Kα X-ray | fixed-geometry common subset / Si-anchor transferability | 23.965 / 23.933; deposited GSAS 4.853 | qualified parity / expected transfer failure | All four full-pattern parity gates pass: Rwp delta 0.032 percentage points, correlation delta 0.00233, and Si-fraction delta 0.107 percentage points. Both restricted fits overestimate Si at 4.450/4.343 wt% versus 1.44 wt% deposited. The weak 128-point Si anchor gives similar calibration Rwp but incompatible displacement, -0.02492/+0.00180 mm, conditional on the disclosed 141.5 mm PhaseSmith radius assumption; that diagnostic result is not applied. | `validation/results/2026-08-13-campaign-iucr-tripotassium-citrate-si.json` |
| IUCr anhydrous trirubidium citrate + Si 640b | Bruker D2 Phaser laboratory Cu Kα X-ray | fixed-geometry common subset / Si-anchor transferability | 18.265 / 18.338; deposited GSAS 2.458 | qualified parity / expected transfer failure | All four full-pattern parity gates pass: Rwp delta 0.074 percentage points, correlation delta 0.00396, and Si-fraction delta 0.255 percentage points. Restricted Si is 3.182/2.926 wt% versus 2.15 wt% deposited. Even with the source-deposited 141.5 mm radius, the 128-point Si fits infer incompatible displacement, -0.07469/-0.10874 mm; that diagnostic result is not applied. | `validation/results/2026-08-13-campaign-iucr-trirubidium-citrate-si.json` |
| Citrate isotropic-width ablation | same potassium/rubidium patterns | phase-local model sufficiency | K: 23.965 -> 20.737; Rb: 18.265 -> 14.823 | finite-size pass / isotropic insufficiency | With instrument, geometry, structure, and silicon width fixed, three-start finite-size fits are full rank and reproducible at 90.19 nm (K) and 56.23 nm (Rb). Gaussian/Lorentzian alternatives do not pass the fixed-budget repeatability gate; size+Lorentzian drives strain to zero. The retained isotropic gains remain far from the deposited 4.853%/2.458% curves. | `validation/results/2026-08-13-citrate-isotropic-broadening-ablation.json` |
| Citrate orthorhombic-Stephens ablation | same potassium/rubidium patterns | phase-local anisotropic-width sufficiency | fixed source K: 23.925; Rb: 18.259; with size K: 20.737; Rb: 14.823 | convention pass / physical insufficiency | The six deposited coefficient ratios and mixing fractions use the separately oracle-verified GSAS-II translation. Their fixed widths barely improve the common model, and adding them to size reproduces the size-only solutions. A Stephens-only common multiplier needs 94x/382x the deposited amplitudes for only small gains; potassium also fails the three-start repeatability gate. Joint size/amplitude refinement is not repeatable. | `validation/results/2026-08-13-citrate-stephens-ablation.json` |
| IUCr trirubidium citrate monohydrate + Si 640b | Bruker D2 Phaser laboratory Cu Kα X-ray | source-qualified holdout | —; deposited GSAS 1.932 | registered / deferred behind anhydrous case | The checksum-pinned `hb7648sup1.cif` contains 6,185 counts and a contiguous 5,986-point deposited range, but its 1.30 wt% Si, absent goniometer radius, strongly phase-specific profiles, Stephens anisotropy, and fixed Suortti roughness repeat the potassium weak-anchor/model limitations. | [IUCr article](https://journals.iucr.org/e/issues/2017/02/00/hb7648/) |

## Issue taxonomy

Each completed row will receive one primary diagnosis:

1. **model parity** — a PhaseSmith equation or parameter convention differs;
2. **optimizer/recipe** — the same broad model reaches a different minimum or
   refines different parameter families;
3. **conversion fidelity** — the deposited project cannot be reconstructed
   from the released pattern/CIF/metadata;
4. **missing metadata** — wavelength, optics, composition, uncertainty, or
   specimen geometry is unknown;
5. **reference mismatch** — GSAS-II is not reproducing the claimed historical
   result either;
6. **out of current scope** — useful control data, but not laboratory X-ray
   Rietveld evidence for the present decision.

## Run log

| Date | Example | Action | Result |
| --- | --- | --- | --- |
| 2026-08-11 | Rowles 1a/1e | Audited checked-in pinned-GSAS-II workflow result. | Both matched comparisons pass. |
| 2026-08-11 | Rowles FPA | Audited checked-in compression diagnostic. | Physical-profile compression worsens both real patterns. |
| 2026-08-11 | Bath K/Li/Cs | Audited checked-in comparison against PhaseSmith, pinned GSAS-II, and released legacy GSAS curves. | Conversion-fidelity capability failure, common to both current programs. |
| 2026-08-11 | XRED TiO2 | Audited checked-in comparison. | QPA fractions agree; residual parity is invalid without instrument metadata. |
| 2026-08-11 | IUCr QARR 1g | Fresh one-repetition release comparison with the pinned oracle. | Passed all four gates; Rwp delta 0.169 percentage points. |
| 2026-08-11 | IUCr QARR 1h | Fresh one-repetition release comparison with the pinned oracle. | Passed all four gates; Rwp delta 0.279 percentage points. |
| 2026-08-11 | NIST SRM 660c 100a | Fresh one-repetition matched and large-FCJ release comparison. | Matched case passes; the predeclared large-FCJ holdout fails exactly its three expected profile gates. |
| 2026-08-11 | APS sucrose | Fresh 20-cycle matched Le Bail comparison. | Passed; Rwp delta 0.299 percentage points and correlation delta 0.01739. |
| 2026-08-11 | ANSTO Echidna LaB6 | Fresh 20-cycle native Le Bail comparison. | Qualified pass under the existing approximate-wavelength workflow tolerances. |
| 2026-08-11 | PbSO4 X-ray/neutron | Fresh one-repetition release comparison of both PhaseSmith fits with the pinned GSAS-II joint refinement. | Both probe-specific residual/correlation gates and the neutron-cell gate pass. |
| 2026-08-11 | POWGEN LaB6 | Fresh pinned-oracle parameter, profile, reconstructed-pattern, and native-workflow comparison. | All nine gates pass; native Rwp delta is 0.105 percentage points. |
| 2026-08-11 | Bath K/Li/Cs | Fresh rerun from the checksum-verified acquired archives. | Reproduced all prior residuals exactly. |
| 2026-08-11 | XRED TiO2 | Fresh rerun from the checksum-verified acquired files. | Reproduced the residuals and 0.185 percentage-point anatase-fraction delta. |
| 2026-08-11 | Rowles 1a/1e | Fresh rerun from the registered TOPAS deposit. | Reproduced both passing matched-workflow comparisons. |
| 2026-08-11 | NIST 1979, opXRD, APS standards | Audited authoritative data sources and metadata suitability. | These sources remain blocked or deferred for the recorded reasons. |
| 2026-08-11 | IUCr dicesium hydrogen citrate + NIST Si 640b | Registered the checksum-pinned supplementary CIF, converted its pattern and three phase blocks, and ran PhaseSmith, pinned GSAS-II, and the deposited legacy-GSAS curve. | Displacement-anchored result: PhaseSmith 10.180%, GSAS-II 8.894%, legacy 6.226% Rwp; Si 14.593/14.621/13.02 wt%. Si(111) is excluded because it overlaps the dominant citrate peak. |
| 2026-08-13 | IUCr sodium dihydrogen citrate + NIST Si 640b | Registered and converted the official supplementary CIF, reviewed the unequal-axial to equal-height translation, rejected non-identifiable width refinements, and ran the exact-revision common subset. | All seven preferred-orientation parity gates pass: Rwp delta 0.762 percentage points, correlation delta 0.00592, Si-fraction delta 0.522 percentage points, and March-ratio delta 0.00614. |
| 2026-08-13 | IUCr anhydrous tripotassium citrate + Si internal standard | Registered and converted the official supplementary pdCIF, then ran the fixed-geometry PhaseSmith and exact-revision GSAS-II workflows plus independent Si-window diagnostics. | PhaseSmith/GSAS-II reach 23.965/23.933% Rwp and pass all common-subset parity gates. The 0.02672 mm displacement disagreement triggers the predeclared weak-anchor failure; neither diagnostic displacement is applied, and the deposited 4.853% richer-model curve remains a reference rather than this comparison's target. |
| 2026-08-13 | IUCr anhydrous trirubidium citrate + NIST Si 640b | Registered and converted the official pdCIF with its exact spillover mask and deposited radius, then ran the fixed common profile in PhaseSmith and exact-revision GSAS-II. | PhaseSmith/GSAS-II reach 18.265/18.338% Rwp and pass all common-subset parity gates. The 0.03405 mm Si-window displacement disagreement confirms the weak-anchor failure without a radius assumption; neither displacement is applied. |
| 2026-08-13 | Potassium/rubidium citrate isotropic-width ablation | Held all non-width nonlinear terms fixed, fitted each dominant phase with size, Gaussian strain, Lorentzian strain, and size/strain combinations from three separated starts, then formed joint rank/correlation diagnostics with scales and one residual-background constant. | Finite size is the only repeatable identifiable isotropic improvement: 90.19 nm and 20.737% Rwp for potassium; 56.23 nm and 14.823% for rubidium. This is useful missing physics but does not approach either deposited curve, justifying a separately reviewed Stephens implementation. |
| 2026-08-13 | Potassium/rubidium citrate Stephens ablation | Kept every nuisance term fixed, preserved the six deposited orthorhombic coefficient ratios and mixing fraction, and tested fixed source widths, one common Stephens amplitude, finite size plus fixed source widths, and the joint size/amplitude model from three starts. | Fixed source Stephens changes Rwp by only 0.040/0.006 percentage points. With size it returns 90.57/56.26 nm and 20.737/14.823% Rwp, indistinguishable in scientific effect from size alone. Free amplitudes are 94x/382x the source values and joint size/amplitude fits are not repeatable, so Stephens does not explain the deposited targets. |
