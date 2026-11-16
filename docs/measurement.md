# Reflection measurement for refined lattices

**Native Rust API:** `phasesmith_single_crystal::measurement`, behind the
`grains` feature. No Python binding yet. This is the step after
[lattice symmetry and constrained cells](lattice-symmetry.md): given refined
lattices, it predicts every reflection over the rotation scan and measures its
intensity on the frames, by summation and optionally by profile fitting. It is
the entry point a GUI such as Dioptas calls. It needs no cell prior, no space
group and no recipe settings; the native workflow (`workflows::single_crystal`)
keeps its own recipe but now shares the prediction, ray geometry and
Lorentz–polarisation code with this module (see *Shared code* below).

Inputs:

- frames, through the `FrameReader` trait (any `Fn(usize) -> Result<Frame>`
  that is `Sync`): counts as `f64` and a per-frame mask (`true` excludes a
  pixel);
- the detector `spots::reciprocal::Geometry` (one laboratory frame, metres,
  wavelength in Å);
- the `Scan`: frame `i` is exposed from `start + i·step` to that plus
  `exposure` degrees; `excluded` frames are never read;
- one or more `Lattice`s: a direct basis as rows in Å (or the reciprocal basis
  in 1/Å without 2π) and a `lattice_symmetry::Bravais` type. The basis must be
  the conventional cell of that type, so that centring absences and holohedry
  operations apply to its indices.

Several lattices (grains) share one geometry and are measured in one pass over
the frames.

```rust
use phasesmith_single_crystal::lattice_symmetry::{Bravais, LatticeSystem};
use phasesmith_single_crystal::measurement::{
    self as ms, Lattice, MeasureOptions, MergingOptions, NoMonitor, Observation,
    PredictOptions, ProfileOptions, Scan,
};

let scan = Scan::new(-35.0, 0.5, 0.5, 142)?.with_excluded(vec![])?;
let lattices = [Lattice::new(direct_basis, Bravais::OrthorhombicP)?];
let d_min = ms::detector_resolution(&geometry, shape);

// Data-derived rocking width and box size (one summation pass).
let shape_estimate = ms::estimate_shape(
    &read_frame, &geometry, &scan, shape, &lattices, d_min,
    &MeasureOptions::default(), &NoMonitor,
)?;

let reflections = ms::predict(
    &lattices, &geometry, &scan, shape,
    &PredictOptions::new(d_min, shape_estimate.rocking_width),
    None, // or Some(&|frame, x, y| visible) to hide reflections under masks
)?;
let mut options = MeasureOptions::default();
options.box_size = shape_estimate.box_size.unwrap_or(9);
options.profile = Some(ProfileOptions::default());
let measured = ms::measure(
    &read_frame, &geometry, &scan, shape, &reflections, &options,
    &|_stage, done, total| { report(done, total); !cancel_requested() },
)?;

let observations: Vec<Observation> = measured
    .reflections
    .iter()
    .filter(|r| r.reflection.lattice == 0 && r.usable())
    .map(|r| {
        let i = r.corrected_summation();
        Observation::new(r.reflection.hkl, i.value, i.sigma, r.reflection.d)
    })
    .collect();
let mut merging = MergingOptions::default();
merging.d_min = Some(d_min);
merging.lattice = Some(lattices[0]);
let stats = ms::merging_statistics(
    &observations, &LatticeSystem::Orthorhombic.holohedry(), &merging,
)?;
```

All option and result structs are `#[non_exhaustive]`: build options from
`Default` (or `PredictOptions::new`) and assign fields.

## Prediction

`crossings(lattice, geometry, shape, d_min, hint)` enumerates every nonzero
`hkl` that the lattice's centring allows (`Centring::allows`) with
`|h·B| ≤ 1/d_min`, predicts both Ewald crossings with
`spots::prediction::predict` (ω unwrapped to the turn nearest `hint`) and keeps
those on the detector (`−0.5 ≤ x < columns − 0.5`, same for y).

`predict(lattices, geometry, scan, shape, options, visible)` keeps crossings
inside the scan's rotation range (one entry per turn for scans longer than
360°) and attaches:

- `zeta`, the rocking factor `ζ = (s × s₀)·axis / |s × s₀|` at the predicted
  pixel;
- the rocking window: all frames whose exposure overlaps
  `ω ± rocking_width / (2 max(|ζ|, axial_zeta/2))`, the half width capped at
  `max(max_window_frames · step, 5 · rocking_width)`;
- the centre frame: the exposure holding ω, or the nearest exposure when ω
  falls between exposures (`exposure < step`), always inside the window;
- flags known before measuring: `AXIAL` (`|ζ| < axial_zeta`, default 0.1),
  `PARTIAL` (window cut by the scan ends or an excluded frame), `HIDDEN`
  (centre frame excluded, or `visible(frame, x, y)` false).

Defaults: `axial_zeta` 0.1, `max_window_frames` 20, at most one million
reflections. `detector_resolution` gives the d of the farthest detector corner.

## Summation

`measure` reads each frame that some window covers once, on
`MeasureOptions::workers` threads: 0 (default) runs in the caller's current
rayon pool (rayon's global pool outside one), so a host's worker pool bounds the
threads and the frames held; 1–64 builds a private pool for the call. On a frame, every reflection whose window covers it gets a square peak box
of `box_size` pixels (odd, default 9) centred on the rounded predicted pixel
and clipped to the detector. `spots::local_background::extract` then fits, per
box, a background plane to a ring `margin` pixels wide (default 5) outside a
guard of `gap` pixels (default 2), leaving out the guards of *all* boxes of the
frame (every lattice) and masked and saturated pixels, with iterative clipping
at `clip` robust standard deviations (default 4) and at least `minimum` ring
pixels (default 20). The background-subtracted box sum and its
counting-statistics variance (the peak pixels' variance plus the propagated
plane covariance) are added over the frames with
`spots::integration::accumulate`. Nothing is cut on I/σ: weak and negative
intensities stay.

Pixel variance (`Noise::variance`):

- `PixelVariance::CountsOrBackground` (default): `max(counts, 0)/gain +
  read_noise²` per pixel, which is unbiased for Poisson counts; a box sum's
  variance is at least that of its fitted background and at least one count²,
  so σ stays positive on empty boxes.
- `PixelVariance::FlooredAtOne`: every pixel at least one count², the rule of
  `local_background::extract` and the native workflow. It overestimates σ on
  near-empty photon-counting pixels.

The background fit itself (clipping, residual RMS, `MODEL`) uses the floored
pixel variance in both cases. Pull SD of (I − I_true)/σ for weak reflections
(30–230 counts, 9 px box) with Poisson backgrounds, averaged over four noise
seeds of about 90 reflections each (spread about ±0.1):

| background (counts/pixel) | `FlooredAtOne` | `CountsOrBackground` |
|---|---|---|
| 0.02 | 0.46 | 1.01 |
| 0.2 | 0.60 | 0.98 |
| 2 | 0.94 | 0.96 |
| 20 | 1.05 | 1.05 |

Exclusions are those of the scan given to `measure`: a reflection whose window
holds a frame that scan excludes is `PARTIAL`, and one whose centre frame it
excludes is `HIDDEN`, whatever scan `predict` saw.

Flags (`FLAG_NAMES`; zero means usable):

| flag | bit | meaning |
|---|---|---|
| `EDGE` | 1 | peak box truncated by the detector edge |
| `MASKED` | 2 | masked pixel in a peak box on a measured frame |
| `SATURATED` | 4 | pixel at or above `Noise::saturation` in a peak box |
| `OVERLAP` | 8 | box shares pixels with another reflection's box on a common frame (same or other lattice); not deblended |
| `BACKGROUND` | 16 | too few ring pixels on some frame; intensity NaN |
| `AXIAL` | 128 | close to the rotation axis |
| `PARTIAL` | 256 | window cut by the scan ends or a frame excluded in the predicting or measuring scan |
| `HIDDEN` | 512 | centre frame excluded (predicting or measuring scan), centre pixel masked on the centre frame, or hidden by `visible` |
| `NO_DATA` | 1024 | no frame of the window was measured |
| `BOUNDARY` | 2048 | significant signal in the outermost pixels of a box |
| `MODEL` | 4096 | the background plane fits its ring poorly |

Each `MeasuredReflection` carries the prediction, the box, the raw summation
`Intensity` (value, σ), mean background per pixel and frame, pixel counts,
frames measured, the observed centroid (x, y, ω of the positive excess), the
per-frame sums (`partials`), the flags, the Lorentz–polarisation
`Corrections`, the incident and diffracted beam directions in the crystal
frame (the laboratory frame at ω = 0, where the lattice basis is given;
`(diffracted − incident)/λ = h·B`), and the profile fit when requested. That is
everything an unmerged reflection-file writer (for example SHELX HKLF 4 with
direction cosines) needs; file formats stay in adapters.

Memory: frames are held only while they are processed, at most `workers` at a
time. Per frame, the background step keeps only box-local arrays: a bucket
index of the guarded boxes replaces the two full-image `u32` coverage arrays
the extraction used to allocate on every call (`extract` keeps its signature
and its results bit for bit). Profile fitting keeps the shoebox pixels of all
(reflection, frame) pairs, 25 bytes per pixel, checked against
`MeasureOptions::max_samples` (default twenty million) before any frame is
read. These pixels are kept for every reflection until the end of the pass,
because learning needs all teachers before any fit: memory grows with the
scan (about `25 · box_size² · (window + 2 · margin_frames)` bytes per
reflection; 9 px boxes and four frames make 8 kB, so 50 000 reflections take
about 400 MB). Summation alone keeps nothing per pixel. Budgets: one million
reflections, and `MeasureOptions::max_pairs`
(default one million, at most twenty million) reflection–frame pairs; the
error names the option.

Progress and cancellation: the `Monitor` (any `Fn(Stage, usize, usize) -> bool
+ Sync`) is called after each frame and around the profile stages; returning
`false` stops the measurement with a cancellation error. A frame-read error
stops it with that error. Errors stay `spots::SpotError`; `error_kind(&e)`
tells a cancellation (`ErrorKind::Cancelled`, message `"measurement
cancelled"`) and a reader failure (`ErrorKind::FrameRead`, message
`"frame read failed at frame N: "` followed by the reader's own text) from
everything else. A host that needs its reader's typed error can keep it in its
reader closure.

`usable()` requires no summation flag, a measured intensity and a finite
Lorentz–polarisation correction (`corrections` is `None` where it is
singular).

## Profile fitting

With `MeasureOptions::profile = Some(ProfileOptions)`, the same pass also keeps
the shoebox pixels of `margin_frames` (default 1) extra frames on each side of
every window, so the profiles see their tails. Summation is unaffected: its
boxes, overlaps and backgrounds use the rocking windows only, and its results
are identical with or without profile fitting.

The profile-fit input (`spots::profiles::Observations`) is built from these
shoeboxes for every reflection that is not `AXIAL`: centre = predicted
x, y, ω; local coordinates = the two angular directions perpendicular to the
diffracted ray in degrees (`RayGeometry::planar`) and ω·ζ in degrees, so one
profile serves the whole detector and every ζ. The default grid has
`grid_shape` = 31 × 31 × 23 voxels spanning the peak box (median angular pixel
size) and the median extended window times |ζ|; `ProfileOptions::grid` sets it
explicitly.

Teachers: unflagged reflections with the highest summed I/σ, at most
`max_teachers` (default 300); `spots::profiles::learn` keeps those over at
least three frames with summed I/σ above `teacher_snr` (default 20) whose first
and last frames hold below 5 % of the total and whose shoeboxes carry no flag
on any frame, the extra profile frames included (so a neighbour overlapping only
there excludes the teacher). One global profile and one per
region (detector halves × scan halves, `regions` = 2 × 2 × 2) are learned; a
region with fewer than `minimum_teachers` (8) uses the global profile. No
space group or family partition is involved. `spots::profiles::fit` then fits
each reflection's amplitude by generalized least squares with the background
plane covariance and an expected-count noise model, returning
`ProfileIntensity` (I, σ, flags, profile mass in the shoebox, reduced χ²,
correlation, profile used). Profile flags: `profiles::PARTIAL` (mass below
`minimum_mass`, default 0.9), `FAILED`, `NO_PROFILE`, `MISMATCH` (I > 5σ with
χ² > 4), and overlap, edge, partial, background, masked or saturated where
they touch the profile's footprint; `PROFILE_FLAG_NAMES` names them.
`profile_usable()` requires no profile flag, no `AXIAL`, `HIDDEN` or
`NO_DATA`, and a finite correction.

If learning or fitting fails (for example too few teachers), `measure` still
succeeds with the summation and reports `ProfileStatus::Failed(reason)`.

The geometry cache (`geometry_cache_bytes`, default 256 MiB) decides whether
learning runs on several threads: learning falls back to one thread when its
cache does not fit (`ProfileModel::learning` reports what was used).

## Corrections

`lorentz_polarization(xy, geometry, polarization)` returns the rotation Lorentz
factor `L = 1/|(s × s₀)·axis| = 1/(sin 2θ |ζ|)`, the polarisation factor
`P = f (1 − (s·e)²) + (1 − f)(1 − (s·e⊥)²)` (`e` the electric vector,
`e⊥ = s₀ × e`, `f` the fraction) and `factor = 1/(L P)`, which multiplies a
measured intensity and its σ. `Polarization::default()` is 0.98 along
laboratory y; `Polarization::unpolarized()` is 0.5. `ray_geometry` returns the
ray, the local angular pixel Jacobian and ζ at a pixel.

## Data-derived rocking width and spot size

Recommended path: measure with profile fitting (its `margin_frames` read the
frames around each window) and call `estimate_shape_from_partials` on that
measurement; re-measure only when the suggestion differs from the settings
used. On the enstatite scan this gave σ 0.0235° and spot σ 0.90 px from the
measuring pass itself, against 0.0248° and 0.88 px from `estimate_shape`'s
extra pass, with the same suggested width (0.5°) and box (9 px). A rocking
curve wider than window plus margins is truncated and underestimated; use
`estimate_shape` (one extra summation pass with a wide window) for a first
estimate or when in doubt. `ShapeEstimate::at_step_floor` says that `6σ` is
below one step, so the suggested width is the step itself.

- `estimate_shape(frames, …)` measures the lattices' reflections by summation
  with a generous window (eight steps or 2°, whichever is wider) and calls
  `estimate_shape_from_partials`.
- `estimate_shape_from_partials(measurement, scan)` takes unflagged
  reflections with summed I/σ ≥ 10, `|ζ| ≥ 0.2` and at least two measured
  frames (at most the 2000 strongest). Each frame's share of the sum is
  compared with a Gaussian rocking curve of σ/|ζ| integrated over the frame's
  exposure, centred within half a step of the predicted ω (fitted per
  reflection); one σ minimises the total robust χ² on a logarithmic grid from
  0.01 to 10 steps, refined by a parabola. Widths below the step are resolved
  through reflections that straddle frame boundaries. The suggested
  `rocking_width` is `max(6σ, step)`. The spot σ is the median second-moment
  width of the strongest reflections (I/σ ≥ 30); the suggested box holds ±4σ.
- `estimate_rocking_from_tracks(tracks, geometry, scan)` uses linked spot
  tracks (`[x, y, ω, weight]` per detection): per track the weighted ω spread
  less `exposure²/12`, times |ζ|; median over tracks. Tracks on one frame count
  as zero width, so with sub-frame widths it floors at the step
  (`at_step_floor`).

## Holohedry operations and merging statistics

`LatticeSystem::holohedry()` (and `Bravais::holohedry()`) returns the integer
rotations of the lattice point group in the conventional setting of
`lattice_symmetry::search` — `-1`, `2/m` (unique b), `mmm`, `4/mmm`, `-3m`
(hexagonal axes, obverse; twofold axes along a, b, a + b), `6/mmm`, `m-3m` —
inversion included, as many as `Bravais::holohedry_order()`. They act on
Miller indices as row vectors, `h' = h W`; `transform_index` applies one and
`equivalent_representative` picks the largest member of an orbit (Friedel
mates share it).

`merge(observations, operations)` merges equivalent observations (unweighted
mean, σ = √(Σσ²)/n). `merging_statistics(observations, operations, options)`
leaves out observations with d below `MergingOptions::d_min` (counted in
`beyond_d_min`) and reports, overall, in `shells` resolution shells of equal
reciprocal volume (default 4, from the largest observed d, or with a lattice the
largest possible d, to `d_min`) and in bins of the mean I/σ of each unique reflection (default
edges 3, 10, 30): observations, unique reflections, redundancy,
`R_int = Σ|I − ⟨I⟩| / Σ I` and `R_meas` (with √(n/(n−1))) over unique
reflections observed at least twice, mean I/σ, and, when
`MergingOptions::lattice` is given, completeness against the centring-allowed
reflections to `d_min`. Systematic absences are not removed: no space group
is assumed.

## Shared code with the native workflow

The native workflow's results are unchanged (tests in
`workflows::single_crystal::measurement` keep the former implementations and
require identical bits):

- reflection enumeration and Ewald prediction: the recipe's `predictions` now
  calls `measurement::crossings` (primitive centring, `d_min` 0.5 Å) and adds
  its own rotation range and frame assignment;
- `profile_geometry` is `measurement::ray_geometry`;
- `physical_factors` is `measurement::lorentz_polarization` with
  `Polarization::new(0.98, [0, 1, 0])`;
- both use `profiles::extract_frame`, `local_background` (now box-local),
  `integration::accumulate`, `profiles::learn_with_execution` and
  `profiles::fit_with_execution`.

Not shared: the recipe's frame driver (`FrameSource` with its validated
`frame_input::Plan`), its fixed 15 × 15 boxes and `0.9/|ζ| + 0.5` windows,
its space-group absences and competitor crossings, its family-partition
teacher rule and fixed grid, its profile-shape refinement
(`profile_refinement::refine`) and empirical scaling. Folding these into the
public options would change the recipe's results.

## Limits

- Summation does not deblend overlaps; they are flagged.
- Profile fitting uses a fixed learned profile per region (no per-reflection
  shape refinement); strong reflections whose shape departs from it are
  flagged `MISMATCH`, and summation is the better value for them. On the
  enstatite scan, profile intensities where profile-usable and summation
  otherwise gave the lowest R_int.
- No scaling, absorption or decay correction; R_int is unscaled.
- Completeness counts all centring-allowed unique reflections, including
  space-group absences and regions a cell body blocks.
- Frames with dead time (exposure < step) are not flagged partial when ω falls
  between exposures.

## Enstatite check

On the MgSiO3 orthoenstatite DAC scan (`eiger_0000.h5`, frames 2–143, 0.5°
steps; geometry, orientation and shadow masks from the native workflow's own
stages through indexing; cell 8.8101 × 5.1824 × 18.2288 Å, oP; d ≥ 0.45 Å;
no diamond-reflection mask):

| | summation | profile | profile, else summation |
|---|---|---|---|
| usable | 5027 | 4161 | 5058 |
| R_int (mmm, unscaled) | 12.1 % | 12.8 % | 9.9 % |
| R_int shells (low → high) | 10.4 / 20.3 / 27.1 / 36.8 % | 11.0 / 14.6 / 18.4 / 22.3 % | 9.4 / 12.1 / 15.5 / 17.6 % |
| Pbca absences, mean I/σ (n > 3σ) | 0.18–1.18 (8 of 739) | −0.37–0.43 (5 of 738) | −0.37–0.54 (5 of 741) |
| Pearson with CrysAlis `dom1_v9.hkl` (merged in mmm) | 0.999 (one outlier removed: 0.44 with it) | 0.987 | 0.999 |

`estimate_shape` gave a rocking σ of 0.025° (most spots lie within one 0.5°
frame, so the suggested width is the step) and a spot σ of 0.87 px (box 9).
The summation pass takes about 2 s; with profile fitting about 5 s (8 threads).
