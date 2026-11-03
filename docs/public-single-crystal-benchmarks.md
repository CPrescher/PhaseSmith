# Public single crystal benchmarks for solid materials

Availability checked **27 October 2026**. Scope: measured **X-ray diffraction rotation images** from
non-biological solids, with an independently documented reduction or structure.
Electron-diffraction datasets are outside the current scope.
No biological, amino-acid, protein, simulated-pattern, or structure-only examples
are included. This is a research shortlist and proposed protocol; no complete
dataset has yet been reduced with PhaseSmith in this investigation.

**Start with thiourea at 130 K**, a functional molecular solid with a complete
Bruker/SAINT/SADABS/SHELX archive. Use **tungsten hexacarbonyl** as the next X-ray
transfer experiment, conditional on obtaining or generating the missing reference
integration products. FeO is the most direct high-pressure mineral example but
needs provenance and geometry repair.

## Ranked shortlist

Ranks reflect readiness for the current integration question, not scientific value.
MB/GB below use decimal units and refer to compressed downloads.

| Rank | Material and measurement | Download | Verified reference and main limitation |
| --- | --- | --- | --- |
| 1 | Thiourea, CH₄N₂S, 130 K; laboratory X-ray rotation | 318.38 MB; 607 acquisition SFRM frames in six runs | SAINT 8.38A observation files, SADABS output, final CIF and SHELX files actually present. Best first case; corrected reference intensities are not raw ADU sums. |
| 2 | W(CO)₆; Bruker Ag X-ray rotation, 150 K | 326.64 MB; 586 main acquisition frames, plus setup/auxiliary images | RODIN reference structure CCDC 2360289, published APEX4 workflow. ZIP has images only: no archived integration table. |
| 3 | Fe₀.₉₄O under pressure; synchrotron X-ray rotation | Four ZIPs, 2.542 GB total; smallest p39_s1 is 304.11 MB with 76 ESPERANTO images | Published CrysAlisPro/SHELX structures and downloadable CIFs; inspected image archive has conflicting calibration records and no HKL observation table. Deferred intensity benchmark. |

All three Zenodo records expose anonymous downloads and declare **CC BY 4.0** in
their API metadata. RODIN additionally asks users not to republish or redeposit
its already-known structures as new determinations; retain that notice and the
original attribution. Access to a source program is separate from access to its
data: SAINT/APEX, CrysAlisPro and SHELX have their own distribution/use conditions.
No vendor software or data is introduced into PhaseSmith's runtime.

### 1. Thiourea at 130 K

Sources: [Zenodo record and README](https://zenodo.org/records/18910433),
[direct ZIP](https://zenodo.org/records/18910433/files/2_T130K_18MPC001.zip),
[methods paper, section 4.2](https://doi.org/10.1039/D6TA00623J).
Structure: **CCDC 2039156**, with the final CIF also inside the ZIP.

- **Acquisition:** Mo Kα, λ = 0.71073 Å, Bruker D8 VENTURE. The paper, README
  and archive CIF identify PHOTON III C14, whereas the actual image header says
  `CMOS-PHOTONII` and `Booster[A110877] with KappaStage`. Preserve this discrepancy;
  use the verified file encoding and measured response metadata rather than
  silently assigning a detector model.
- **Images:** Bruker FORMAT 100 `.sfrm`; all six representative run-start images
  decode with FabIO as 1024 × 768, signed int32 arrays. Acquisition runs contain
  **100, 91, 77, 158, 90 and 91 frames**, totaling **607**. The ZIP has 650 SFRM
  members altogether: the remainder include auxiliary, background and reconstructed
  precession images. Do not glob every SFRM into one rotation series.
- **Rotation/exposure:** run-start headers and the SAINT log agree on 1° and
  4 s per frame. Run 4 scans phi (`AXIS=3`); the others scan omega (`AXIS=2`).
  Goniometer settings vary between runs. Headers include ANGLES, ENDING, START,
  CENTER, DETPAR, DISTANC and CCDPARM. There is no verified ready-to-use PONI.
- **Geometry/response:** `work/2_T130K_18MPC001_0m.p4p` records cell, orientation,
  beam centre and refined detector corrections. The SAINT `01._ls` log records
  read noise 2.89 electrons, 36.60 electrons/ADU, 259.92 electrons/X-ray, base
  offset 64, full scale 163809, and the face-to-imaging-plane offset. These are
  archived calibration values, not an independent new gain measurement.
- **Reference products:** SAINT **8.38A**, SADABS **2016/2**, SHELXT **2018/2** and
  SHELXL **2018/3**. The inspected `0m.raw` contains unmerged observations with
  intensities/errors and detector/frame-related columns; `_ls` contains processing
  settings; `18MPC001T130K.hkl` contains **9,047 nonzero-HKL records**, including
  **544 negative intensities**, with repeated HKLs and σ(I). It is an unmerged,
  corrected SHELX input, not a unique merged list or a raw-count table. Its lack
  of explicit frame IDs prevents reliable one-to-one observation matching by
  HKL alone. The RAW format must be mapped using a documented schema first.
- **Structure:** `work/archive_18MPC001T130K.cif` gives **P 21 m a**,
  a = 7.4962(6), b = 8.5211(7), c = 5.4695(4) Å, all angles 90°;
  1,134 refinement reflections, R1(observed) = 0.0332 and wR2(all) = 0.0731.
  Use this complete archive CIF; the similarly named working CIF contains
  missing metadata. The reported reference population differs from the raw HKL
  record count because of reduction, filtering and merging.
- **Count comparability:** the SAINT log explicitly multiplies output intensities
  by **15** to normalize to 1 min/degree. LP, absorption, detector and profile
  corrections also require an audit. Dividing by 15 alone does not recover raw
  detected counts. A fresh DIALS reduction on identical decoded pixels can supply
  a detector-space comparator; archived SAINT/SADABS products supply an independent
  corrected-intensity and structural check.

### 2. Tungsten hexacarbonyl from RODIN

Sources: [Bruker record](https://zenodo.org/records/11965992),
[direct ZIP](https://zenodo.org/records/11965992/files/CCDC_2360289.zip),
[RODIN methods paper](https://doi.org/10.1021/acs.jchemed.4c00797).
Structure: [CCDC 2360289 / KOVSOD09](https://doi.org/10.5517/ccdc.csd.cc2k72c8).

- **Acquisition:** non-biological metal carbonyl solid, W(CO)₆, at 150 K;
  Bruker D8 Venture, PHOTON II, Ag Kα. Landing-page wavelength is 0.56086 Å;
  inspected headers give 0.560870 Å. Preserve the respective provenance.
- **Images:** 618 SFRM members and one crystal-image VZS. The main series are
  `ag_pgw240059_01` (**180** frames, phi, **1°/1 s**) and
  `ag_pgw240059_02` (**406** frames, omega, **0.5°/10 s**). There are also
  two 12-frame matrix runs, auxiliary files and strategy masks. Main acquisition
  frames decode as 1024 × 768 int32 arrays; binning and overflow tables must be
  preserved. A matrix-run image has values reaching the header full-scale range,
  useful for testing saturation flagging rather than treating it as a valid sum.
- **Calibration:** headers contain centre, two distance fields, detector corrections,
  exposure, gain/response and axis information. CELL/MATRIX are placeholders in
  inspected images, so they are not a frozen indexed solution. Pixel import works;
  calibrated indexing and complete Dioptas processing remain untested.
- **Reference:** the paper describes APEX4 reduction, numerical absorption
  correction based on crystal faces, XT/XL solution/refinement through Olex2.
  The record gives acquisition software BIS 6.2.6, not a pinned integration
  version. **The ZIP contains no HKL, RAW, P4P or CIF.** The CCDC structure link
  is present, but automated CIF retrieval did not succeed during this check.
  Pnma is documented for the KOVSOD family; the exact deposited cell and final
  numerical statistics still need to be fetched. Do not substitute a cell from
  another wavelength/crystal as this dataset's archived result.
- **Comparison level:** fresh integration needed for a detector-count comparator.
  A deposited final structure is structural validation, not an archived
  per-observation integration reference. This is therefore the second acquisition
  experiment, not a replacement for thiourea's more complete control.
- **Transfer opportunity:** heavy-atom scattering, absorption, saturation and the
  tenfold exposure difference challenge the frozen extraction without changing
  the SFRM reader. Other measurements are independently available on
  [Rigaku Mo, 26.70 MB](https://zenodo.org/records/11926636),
  [Rigaku Cu short exposure, 69.75 MB](https://zenodo.org/records/11925164),
  [Rigaku Cu long exposure, 102.35 MB](https://zenodo.org/records/11925895),
  [STOE, 225.44 MB](https://zenodo.org/records/12568724), and
  [Diamond, 1.316/0.836 GB](https://zenodo.org/records/11949828).
  Rigaku uses proprietary RODHYPIX and STOE XI: their metadata were read, but
  their pixel import was not verified. These are different acquisitions, not
  identical photon observations that should share absolute counts.

### 3. High pressure FeO

Sources: [Zenodo](https://zenodo.org/records/12594216),
[smallest image ZIP](https://zenodo.org/records/12594216/files/p39_s1.zip),
[logbook](https://zenodo.org/records/12594216/files/logbook.xlsx),
[Li et al., Communications Physics 2024](https://doi.org/10.1038/s42005-024-01797-1).

- **Downloads:** p01_s1 914.32 MB, p11_s1 892.75 MB, p38_s1 430.65 MB,
  p39_s1 304.11 MB, plus a 17.7 kB logbook. Only p39_s1 was inspected internally:
  **76 ESPERANTO frames**, `.par` calibration files and CrysAlis logs. No HKL,
  RAW observation table or CIF is in that ZIP's inventory. Other ZIP contents
  and image numbers remain unverified.
- **Pixels/scan:** first ESPERANTO frame decodes as 2048 × 2048 int32,
  **−3049…605769**, 1 s, omega −38° to −37°, 0.2 mm pixels. These are detector-image
  exports retaining signed values, not verified untouched native detector files.
  The header overflow threshold is 64000 despite larger stored values, so that
  field cannot simply be used as the physical saturation threshold.
- **Conflicting geometry:** image header has λ = 0.29520 Å, distance 430 mm,
  centre (1028,976); the later `cracker.par` has λ = 0.29660 Å, distance
  391.38476 mm, centre approximately (964.1944,1077.3000). Another `.par` contains
  a 1024-pixel/60 mm placeholder geometry. The paper describes both MAR555/ESRF
  and PerkinElmer XRD1621/DESY experiments, at 0.4100 and 0.2902 Å respectively.
  The 2048-pixel/0.2 mm archive is consistent with a flat-panel configuration,
  but this does not prove the detector assignment or resolve the wavelength.
- **Reference:** archive calibration logs identify CrysAlis versions 1.171.37.35
  and 40.57a; final published CIFs name **1.171.43.67a**. Do not treat these as
  one version-pinned reduction. The paper supplies
  [Supplementary Data 2, CIF](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42005-024-01797-1/MediaObjects/42005_2024_1797_MOESM5_ESM.cif)
  and [Supplementary Data 3, CIF](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42005-024-01797-1/MediaObjects/42005_2024_1797_MOESM6_ESM.cif),
  both retrieved successfully. They describe P21/m and C2/m structures at
  λ = 0.2966 Å, not a per-frame detector-count reduction. For example the C2/m
  CIF gives a = 4.8359(16), b = 2.7233(11), c = 2.9920(19) Å,
  β = 124.06(7)°. Its exact mapping to the archived p39 scan was not established.
- **Readiness:** the logbook contains p01/p11 conditions and reduction statistics,
  but no p38/p39 labels were found in the inspected worksheets. A filename p39
  must not be interpreted as 39 GPa. Resolve scan-to-pressure/CIF mapping,
  calibration and native-image provenance before adopting this as a quantitative
  intensity benchmark. It remains an attractive later DAC transfer case.

## What the comparisons can establish

| Evidence | Permitted conclusion | Not established |
| --- | --- | --- |
| Stored image array | Measured detector response, after the instrument's own documented processing | Expected signal in each reflection or independent photon truth |
| Same-pixel, uncorrected external integration with σ and support records | Agreement/disagreement between extraction algorithms | Which algorithm is physically correct on real data |
| SAINT/SADABS, XDS_ASCII, scaled DIALS or merged HKL | Agreement after an explicitly matched correction/scaling pipeline | Direct equivalence to finite-support detector ADU sums |
| Deposited CIF and successful blind solution/refinement | Structural usefulness and consistency with a known solid | Unique absolute intensities or absence of extraction bias |

The independent known-count/null fixtures already used in PhaseSmith remain the
way to establish estimator bias and uncertainty coverage when the generating
signal is known. Real data supply different evidence. Do not train or tune shape,
support, background or uncertainty models against comparator intensities or
known atomic coordinates.

## Proposed first acquisition and validation protocol

### 1. Freeze acquisition and reference artifacts

Download **only** `2_T130K_18MPC001.zip` initially. Verify the published MD5
`8fd65514a2bcd5b7332b401795c350a4`, calculate SHA256, retain the archive unchanged
and produce a member manifest. Select the six acquisition patterns
`mo_18MPC001_rampT160K_0[1-6]_####.sfrm`; require the six counts above.
Reject auxiliary `_xa`, work/background snapshots and reconstructed precession
images from the acquisition list. A small consecutive wedge is sufficient for
an import/geometry pilot; use all six complete sweeps for the eventual structural
test. Preserve RAW, P4P, logs, HKL, INS/RES and the archive CIF separately.

Freeze the current PhaseSmith/Dioptas implementation identifiers, local diff
hashes, dependencies and all settings. Freeze external software builds and
commands, including a fresh isolated DIALS integration; do not call the vendor
reference a DIALS-independent count reference when its count convention is
unrecovered. No benchmark integration or software installation is part of this
research task.

### 2. Establish common pixels and geometry before intensities

Compare decoded shapes, exact array hashes and pixel values between readers,
including underflow/overflow reconstruction. Dioptas's existing `FabioLoader`
returns `data[::-1]`: define that vertical flip **once**, and transform beam
centre, panel axes, masks and centroids with it. Record whether coordinates refer
to pixel centres or corners, x/y versus row/column, zero versus one-based image
numbers, degree versus radian angles, laboratory handedness, rotation signs,
fixed kappa/chi/phi settings and the selected scan axis.

Treat each sweep's detector pose and goniometer configuration explicitly. Resolve
the two Bruker DISTANC fields and faceplate offset against P4P/SAINT calibration;
do not interpret the first number as the final scattering-plane distance.
Validate metadata calibration first, then refine geometry using **positions
only** on a declared training subset and report held-out positional residuals.
Check the archived response/gain, offset, saturation and firmware correction
conventions; do not assume photon-counting Poisson statistics or subtract a
pedestal twice. No new powder calibrant/PONI acquisition is claimed available.

Current [dxtbx Bruker reader](https://github.com/cctbx/dxtbx/blob/main/src/dxtbx/format/FormatBrukerModern.py)
recognises CMOS-PHOTONI* and FORMAT 100, but contains detector assumptions and a
fixed scan-exposure value. Pin and test the actual installed reader: nominal
format support is insufficient evidence of correct geometry or exposure.
Pixel decoding was verified here; end-to-end DIALS import and calibrated
PhaseSmith/Dioptas processing were not.

### 3. Freeze indexing and separate training from evaluation

Discover/refine cell and orientation from spot positions, determine and record
the change of basis to the archived P21ma setting, and freeze HKLs, geometry,
per-sweep orientation and predicted observation identities. A comparator geometry
may be shared for a specifically labelled extraction-only experiment. Also retain
a separate native-indexing end-to-end result; do not confuse the two questions.

Before looking at reference intensities, partition reflection families and
reserved rotation blocks deterministically. Strong image observations can teach
empirical profiles; target families and their symmetry equivalents must not teach
their own profile. Supply nearby independent teachers throughout the scan, with
guard bands around held-out observations, and keep a separate extrapolation
stress test. Freeze teacher selection, support and model mixtures before fitting
signed target amplitudes. Keep the CIF's atomic coordinates sealed until
integration outputs and validation settings have been frozen.

### 4. Compare individual observations at matching correction levels

Retain both native signed summation and frozen empirical-profile results, with
conditional σ, background covariance, support coverage, masks and per-frame
records. Match external observations by sweep, HKL and rocking position (plus
panel/centroid where available), never HKL alone. Keep unmatched and rejected
observations in the accounting.

For a detector-space DIALS comparator, inspect the pinned `.refl` value/variance,
LP, detector-QE, partiality and export conventions and verify the correction
chain with at least one raw shoebox recomputation. A file named `integrated` is
not proof that its intensities are uncorrected. Prefer extracting an explicitly
uncorrected external estimate; if conversion is necessary, apply only known
invertible factors and propagate σ consistently. If the chain cannot be
recovered, mark raw-count comparison unavailable and compare only at the
corrected level. Audit SAINT's RAW schema and time normalization independently.

Report signed bias, median/p90 absolute relative differences for a predeclared
strong, fully recorded, isolated population, correlation, slope and residuals
versus scan angle, resolution, detector position, background and strength.
Retain the existing Fe7C3 diagnostic gate as a **software-agreement** check:
at least 30 pairs, |median bias| ≤10%, p90 absolute difference ≤30%, r ≥0.95,
with exclusions and denominator convention declared before evaluation.
Evaluate each sweep separately and the pooled population. Any fitted diagnostic
scale must be learned on a separate calibration subset, then frozen and reported;
it does not turn normalized intensities into an absolute count comparison.

### 5. Keep difficult observations visible

- **Weak/negative:** retain signed estimates without I/σ selection or clipping.
  Use absolute/σ-normalized residuals, negative fractions and intensity-bin plots;
  relative differences with near-zero denominators are not valid diagnostics.
  A combined-error residual is descriptive, not an independent standard-normal
  pull, because estimators share the same noisy images. Use known-count trials
  for absolute uncertainty coverage.
- **Partial/boundary:** report observed exposure intervals, coverage and truncated
  support. Compare matched recorded counts separately from any full-reflection
  extrapolation. Do not silently renormalize finite profiles or extend data across
  missing exposures. Fix the endpoint convention and test image/scan boundaries.
- **Saturated/overflow:** distinguish encoding overflow, reconstructed large
  values, rescan substitutions and physical detector saturation. Exclude invalid
  measurements from the main gate but report their frequency and flag agreement.
- **Overlapping/masked:** declare neighbor supports and bad-pixel regions using
  frozen geometry. Report isolated observations and overlap/edge strata separately;
  an unimplemented deconvolution must not be counted as successful extraction.

### 6. Validate structural usefulness independently

Apply a documented, equivalent correction/scaling/merging protocol to each
integration result, with common populations and frozen rejection rules where
possible. Retain intermediate unmerged intensities and σ, multiplicity,
completeness, equivalent-reflection consistency and all rejection counts. Positive
amplitudes derived for phasing must not replace the retained signed intensity
records.

Run blind SHELXT solution with **composition**, cell and defensible symmetry
only; no archived atom coordinates or structure-derived intensities enter
integration. Then refine the obtained structure with the same SHELXL version,
resolution, weighting policy, H-atom treatment and restraint policy as the
reference where recoverable. Report solution success, symmetry/basis mapping,
coordinates/bond geometry, displacement parameters, R1/wR2/GoF and difference
density. Treat the archived R1=0.0332 as a reference result, not a guarantee
that every reprocessing choice reproduces that number. Different populations
and corrections must be disclosed. A low R value alone does not validate
detector-count bias or gain.

### 7. Test transfer without retuning

Freeze the thiourea extraction policy before moving to W(CO)₆. Import only its
two main series; retain the 1°/1 s and 0.5°/10 s exposure conventions and their
different finite-support/partiality behavior. Obtain the CCDC CIF and create a
version-pinned reference reduction, preserving the fact that it is a new
reprocessing rather than an archived author integration. Test heavy-atom
absorption, dynamic range and exposure dependence with the same reporting.
Do not transfer absolute intensity scales between distinct crystals/wavelengths.

For an explicitly mineral target, FeO is the later X-ray extension after its
calibration/scan identity is resolved. The present shortlist and transfer protocol
are restricted to X-ray diffraction.

## Verification artifacts and remaining work

[Machine-readable availability evidence](research/public-single-crystal-benchmarks-20261027/availability.json)
records repository byte sizes/MD5s, access/license fields, inspected image-member
SHA256s, decoded-array hashes, selected headers and identities of sampled text
products. ZIP inventories and small source files were read with bounded HTTP
ranges; **no complete multi-gigabyte archive was downloaded**. The archive MD5s
are repository declarations, not local whole-archive checksums. Temporary HTTP
429 responses occurred; successful later bounded reads establish accessibility,
not unrestricted future download throughput.

The existing Dioptas `FabioLoader.get_image()` was exercised directly on one
sample from every shortlisted material: arrays exactly match the decoded source
with one vertical flip. This checks pixel loading, not geometry import or the GUI
workflow. No numerical source was changed, so numerical test suites were not run.

Working evidence and acquisition probes remain locally under
`runs/public-benchmark-research-20261027/` (ignored). The requested Obsidian
PhaseSmith directory did not exist on this machine; the research note creates it
at the explicitly requested path.

The active checkout is `/Users/clemens/Programming/PhaseSmith`, confirmed from
its current brief and Fe7C3 work. `/Users/clemens/Documents/PhaseSmith` is a
different, older checkout whose Git HEAD could not be read. Existing edits in
both checkouts were preserved. This task adds documentation/evidence only;
no numerical implementation, benchmark result, commit, push or release is implied.
