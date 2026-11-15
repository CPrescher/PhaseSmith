//! Structure-independent single-crystal detector and profile measurements.
//!
//! Native and deterministic; detector kernels are single-threaded. The opt-in
//! FFT grain search uses a bounded private Rayon pool and `RustFFT`. No file
//! formats, GUI, interpreter, or global thread pool.
//! See `docs/experimental-spots.md` for equations and uncertainty limitations.

// Pixel counts/coordinates are validated below 2^32, exactly representable in f64.
#![allow(clippy::cast_precision_loss)]
// Short indices and sums follow the documented numerical equations.
#![allow(clippy::many_single_char_names)]

use std::fmt::{Display, Formatter};

pub mod box_optimizer;
pub mod empirical_fit;
pub mod fft;
mod fourier_refinement;
pub mod frame_input;
pub mod frame_validity;
mod geometry_linear;
pub mod geometry_refinement;
pub mod integration;
pub mod intensity_model;
pub mod local_background;
pub mod omega_background;
pub mod prediction;
mod profile_line_search;
pub mod profile_refinement;
pub mod profiles;
mod profiles_affine;
mod profiles_cache;
mod profiles_fit;
pub mod profiles_variation;
pub mod reciprocal;
pub mod reduction;
pub mod rotation;
pub mod scan_scaling;

/// Truncated aperture at a detector boundary.
pub const EDGE: u32 = 1;
/// Excluded pixels in a signal aperture.
pub const MASKED: u32 = 2;
/// Signal aperture contains a saturated pixel.
pub const SATURATED: u32 = 4;
/// Signal apertures overlap; intensities are not deblended.
pub const OVERLAP: u32 = 8;
/// Too few independent background pixels; intensity and uncertainty are NaN.
pub const BACKGROUND: u32 = 16;
/// Detected component exceeds the declared size bound.
pub const LARGE: u32 = 32;
/// No positive background-subtracted weight for a centroid.
pub const NO_CENTROID: u32 = 64;

/// Invalid input, unsupported range or explicit safety budget exceeded.
#[derive(Debug, Clone)]
pub struct SpotError(pub String);
impl Display for SpotError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for SpotError {}
fn invalid(message: &str) -> SpotError {
    SpotError(message.to_owned())
}

/// Half-open integer pixel bounds, indexed as image[row=y, column=x].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bounds {
    /// Inclusive left column.
    pub x0: usize,
    /// Inclusive bottom row in array coordinates.
    pub y0: usize,
    /// Exclusive right column.
    pub x1: usize,
    /// Exclusive top row in array coordinates.
    pub y1: usize,
}
impl Bounds {
    /// Padded by `radius` on every side, clipped to a `width` × `height`
    /// detector: the aperture [`detect`] measures for a component.
    #[must_use]
    pub fn expanded(self, radius: usize, width: usize, height: usize) -> Self {
        Self {
            x0: self.x0.saturating_sub(radius),
            y0: self.y0.saturating_sub(radius),
            x1: self.x1.saturating_add(radius).min(width),
            y1: self.y1.saturating_add(radius).min(height),
        }
    }
}

/// Borrowed row-major detector observations with explicit noise assumptions.
pub struct Image<'a> {
    /// Flattened values in detector count units.
    pub values: &'a [f64],
    /// Rows.
    pub height: usize,
    /// Columns.
    pub width: usize,
    /// True excludes a pixel. Masked nonfinite data are permitted.
    pub mask: &'a [bool],
    /// Optional supplied independent per-pixel variance, in squared count units.
    pub variance: Option<&'a [f64]>,
    /// Photons/electrons per count for the default Poisson approximation.
    pub gain: f64,
    /// Additive read-noise standard deviation in count units.
    pub read_noise: f64,
    /// Optional detector saturation level, inclusive.
    pub saturation: Option<f64>,
}
impl Image<'_> {
    /// Validate all lengths and unmasked numerical inputs before processing.
    ///
    /// # Errors
    /// Returns an error for invalid dimensions, noise assumptions or nonfinite unmasked data.
    pub fn validate(&self) -> Result<(), SpotError> {
        let n = self
            .height
            .checked_mul(self.width)
            .ok_or_else(|| invalid("image too large"))?;
        if n == 0 || n > u32::MAX as usize || self.values.len() != n || self.mask.len() != n {
            return Err(invalid(
                "image must be nonempty with consistent shape and <= 2^32-1 pixels",
            ));
        }
        if !self.gain.is_finite()
            || self.gain <= 0.0
            || !self.read_noise.is_finite()
            || self.read_noise < 0.0
            || !self.read_noise.powi(2).is_finite()
            || self.saturation.is_some_and(|v| !v.is_finite())
        {
            return Err(invalid("invalid gain, read noise or saturation"));
        }
        if self.variance.is_some_and(|v| v.len() != n) {
            return Err(invalid("variance shape differs from image"));
        }
        for i in 0..n {
            if !self.mask[i]
                && (!self.values[i].is_finite()
                    || self
                        .variance
                        .is_some_and(|v| !v[i].is_finite() || v[i] <= 0.0)
                    || !self.pixel_variance(i).is_finite())
            {
                return Err(invalid(
                    "unmasked image must be finite; variance must be finite and positive",
                ));
            }
        }
        Ok(())
    }
    fn pixel_variance(&self, i: usize) -> f64 {
        self.variance.map_or_else(
            || (self.values[i].max(0.0) / self.gain + self.read_noise.powi(2)).max(1.0),
            |v| v[i],
        )
    }
    fn saturated(&self, i: usize) -> bool {
        self.saturation.is_some_and(|s| self.values[i] >= s)
    }
}

/// Fixed detector support and detection budgets. Radii are in pixels.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Signal significance above an annular local mean.
    pub threshold_sigma: f64,
    /// Additional absolute count excess required, inclusive rejection at equality.
    pub threshold_absolute: f64,
    /// Annular mean excludes this central square.
    pub inner_radius: usize,
    /// Outer square radius for local thresholding.
    pub outer_radius: usize,
    /// Minimum 8-connected component size.
    pub min_pixels: usize,
    /// Larger components are returned flagged rather than promoted as spots.
    pub max_pixels: usize,
    /// Component bounding box padding for intensity extraction.
    pub padding: usize,
    /// Background ring width outside every signal box.
    pub background_margin: usize,
    /// Minimum usable background samples for detection and extraction.
    pub min_background: usize,
    /// Fail explicitly above this number of retained components.
    pub max_spots: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            threshold_sigma: 5.0,
            threshold_absolute: 0.0,
            inner_radius: 2,
            outer_radius: 8,
            min_pixels: 3,
            max_pixels: 4096,
            padding: 2,
            background_margin: 4,
            min_background: 12,
            max_spots: 10000,
        }
    }
}
impl Options {
    /// Reject invalid and unbounded support settings.
    ///
    /// # Errors
    /// Returns an error if supports, thresholds or budgets are outside their declared ranges.
    pub fn validate(self) -> Result<(), SpotError> {
        if !self.threshold_sigma.is_finite()
            || self.threshold_sigma <= 0.0
            || !self.threshold_absolute.is_finite()
            || self.threshold_absolute < 0.0
            || self.inner_radius >= self.outer_radius
            || self.outer_radius > 128
            || self.padding > 128
            || self.background_margin == 0
            || self.background_margin > 128
            || self.min_pixels == 0
            || self.max_pixels < self.min_pixels
            || self.min_background == 0
            || self.max_spots == 0
            || self.max_spots > 1_000_000
            || self.min_background > u32::MAX as usize
        {
            return Err(invalid(
                "invalid spot support, thresholds or component budgets",
            ));
        }
        Ok(())
    }
}

/// One finite-aperture measurement. Values preserve signed intensities.
#[derive(Clone, Debug)]
pub struct Spot {
    /// Measurement box, including padding.
    pub bounds: Bounds,
    /// x, y, I, sigma(I), background, sigma(background), maximum count,
    /// positive-weight RMS width x/y, valid signal pixels, background pixels.
    pub values: [f64; 11],
    /// Bitwise combination of documented quality flags.
    pub flags: u32,
}
/// All retained candidates in deterministic row-major discovery order.
#[derive(Debug)]
pub struct Detection {
    /// Measured spots, including flagged records.
    pub spots: Vec<Spot>,
    /// Connected components below the declared minimum size.
    pub rejected_small: usize,
}

struct Integral {
    width: usize,
    sum: Vec<f64>,
    variance: Vec<f64>,
    count: Vec<u32>,
}

// Interior columns have fixed window widths. Slice each corner stream once per
// row, so iteration proves its bounds without checking 24 unrelated indices at
// every pixel. Corner subtraction order matches the clipped scalar path.
fn integral_corners<T>(
    values: &[T],
    rows: (usize, usize),
    columns: (usize, usize),
    radius: usize,
) -> impl Iterator<Item = [&T; 4]> {
    let (start, end) = columns;
    let left = (start - radius, end - radius);
    let right = (start + radius + 1, end + radius + 1);
    values[rows.0 + left.0..rows.0 + left.1]
        .iter()
        .zip(&values[rows.0 + right.0..rows.0 + right.1])
        .zip(&values[rows.1 + left.0..rows.1 + left.1])
        .zip(&values[rows.1 + right.0..rows.1 + right.1])
        .map(|(((a, c), d), e)| [a, c, d, e])
}

fn classify_row(
    statistics: (&[i64], &[f64], &[f64]),
    values: &[f64],
    mask: &[bool],
    strong: &mut [bool],
    noise: impl Iterator<Item = f64>,
    options: Options,
) {
    // Independent lanes permit vectorization without reassociating arithmetic.
    // Evaluate all predicates, then gate ineligible/masked lanes explicitly;
    // intermediate NaNs on those lanes cannot become strong pixels.
    let minimum = i64::try_from(options.min_background).unwrap_or(i64::MAX);
    for ((((((&n, &s), &v), &value), &masked), dst), pixel_variance) in statistics
        .0
        .iter()
        .zip(statistics.1)
        .zip(statistics.2)
        .zip(values)
        .zip(mask)
        .zip(strong)
        .zip(noise)
    {
        let nf = n as f64;
        let excess = value - s / nf;
        let threshold = options.threshold_sigma * (pixel_variance + v.max(0.0) / (nf * nf)).sqrt();
        // Integer masks preserve eager predicate evaluation for independent lanes.
        *dst = (u8::from(!masked)
            & u8::from(n >= minimum)
            & u8::from(excess > options.threshold_absolute)
            & u8::from(excess > threshold))
            != 0;
    }
}

// Compensated accumulation avoids a frame-size-dependent centroid drift for
// broad connected features. Intensity sums remain in their original order.
#[derive(Default)]
struct MomentSum {
    value: f64,
    correction: f64,
}
impl MomentSum {
    fn add(&mut self, value: f64) {
        let adjusted = value - self.correction;
        let next = self.value + adjusted;
        self.correction = (next - self.value) - adjusted;
        self.value = next;
    }
}
impl Integral {
    fn new(image: &Image<'_>) -> Self {
        let n = (image.width + 1) * (image.height + 1);
        Self {
            width: image.width + 1,
            sum: vec![0.0; n],
            variance: vec![0.0; n],
            count: vec![0; n],
        }
    }
    /// Prefix sums of the included pixels. Returns whether they are exact:
    /// every included value and variance an integer, each absolute total
    /// below 2^53. Then every prefix entry and every window difference is the
    /// exact integer, whatever the summation order.
    #[allow(clippy::float_cmp)] // Exact integer test.
    fn rebuild(&mut self, image: &Image<'_>, exclude: &[bool]) -> Result<bool, SpotError> {
        const EXACT: f64 = 9_007_199_254_740_992.0; // 2^53
        let (mut integers, mut total_value, mut total_variance) = (true, 0.0_f64, 0.0_f64);
        // Each interior entry is overwritten; the zero first row/column stay zero.
        for y in 0..image.height {
            let (mut s, mut v, mut c) = (0.0, 0.0, 0);
            for x in 0..image.width {
                let i = y * image.width + x;
                if !image.mask[i] && !exclude[i] && !image.saturated(i) {
                    let (value, variance) = (image.values[i], image.pixel_variance(i));
                    s += value;
                    v += variance;
                    c += 1;
                    integers &= value.trunc() == value && variance.trunc() == variance;
                    total_value += value.abs();
                    total_variance += variance;
                }
                let j = (y + 1) * self.width + x + 1;
                self.sum[j] = self.sum[j - self.width] + s;
                self.variance[j] = self.variance[j - self.width] + v;
                self.count[j] = self.count[j - self.width] + c;
                if !self.sum[j].is_finite() || !self.variance[j].is_finite() {
                    return Err(invalid("numerical overflow in background sums"));
                }
            }
        }
        Ok(integers && total_value < EXACT && total_variance < EXACT)
    }
    /// Pass two from exact pass-one sums. Excluding a pass-one strong pixel
    /// changes the annulus sums only of the pixels whose annulus holds it,
    /// that is, the pixels in its own annulus (square windows: Chebyshev
    /// distance above the inner and at most the outer radius). Elsewhere the
    /// exact sums, and so the classification, are those of pass one. Each
    /// strong pixel scatters its contribution to its annulus; the annulus
    /// sums less those are exactly the sums a rebuild gives. Bit-identical to
    /// `rebuild(strong)` then `classify` whenever `rebuild` reported exact.
    fn reclassify_near(&self, image: &Image<'_>, options: Options, strong: &mut [bool]) {
        let (w, h) = (image.width, image.height);
        let (ro, ri) = (options.outer_radius, options.inner_radius);
        // Removed (count, value, variance) per affected pixel, in first-touch order.
        let mut slot = vec![u32::MAX; w * h];
        let mut affected: Vec<(usize, i64, f64, f64)> = Vec::new();
        for (p, _) in strong.iter().enumerate().filter(|(_, s)| **s) {
            if image.mask[p] || image.saturated(p) {
                continue;
            }
            let (value, variance) = (image.values[p], image.pixel_variance(p));
            let (px, py) = (p % w, p / w);
            for y in py.saturating_sub(ro)..=(py + ro).min(h - 1) {
                let near_row = y.abs_diff(py) <= ri;
                for x in px.saturating_sub(ro)..=(px + ro).min(w - 1) {
                    if near_row && x.abs_diff(px) <= ri {
                        continue; // Inner window: excluded from both sums.
                    }
                    let i = y * w + x;
                    if slot[i] == u32::MAX {
                        slot[i] = u32::try_from(affected.len()).unwrap_or(u32::MAX);
                        affected.push((i, 0, 0.0, 0.0));
                    }
                    let entry = &mut affected[slot[i] as usize];
                    entry.1 += 1;
                    entry.2 += value;
                    entry.3 += variance;
                }
            }
        }
        let minimum = i64::try_from(options.min_background).unwrap_or(i64::MAX);
        for (i, removed, removed_sum, removed_variance) in affected {
            if image.mask[i] {
                continue;
            }
            let (x, y) = (i % w, i / w);
            let corners = |r: usize| {
                let (x0, x1) = (x.saturating_sub(r), (x + 1).saturating_add(r).min(w));
                let (y0, y1) = (y.saturating_sub(r), (y + 1).saturating_add(r).min(h));
                [
                    y0 * self.width + x0,
                    y0 * self.width + x1,
                    y1 * self.width + x0,
                    y1 * self.width + x1,
                ]
            };
            let (outer, inner) = (corners(ro), corners(ri));
            let count = |[a, c, d, e]: [usize; 4]| {
                i64::from(self.count[e]) - i64::from(self.count[c]) - i64::from(self.count[d])
                    + i64::from(self.count[a])
            };
            let n = count(outer) - count(inner) - removed;
            if n < minimum {
                strong[i] = false;
                continue;
            }
            let window = |values: &[f64], [a, c, d, e]: [usize; 4]| {
                values[e] - values[c] - values[d] + values[a]
            };
            let s = (window(&self.sum, outer) - window(&self.sum, inner)) - removed_sum;
            let v =
                (window(&self.variance, outer) - window(&self.variance, inner)) - removed_variance;
            let nf = n as f64;
            let excess = image.values[i] - s / nf;
            strong[i] = excess > options.threshold_absolute
                && excess
                    > options.threshold_sigma
                        * (image.pixel_variance(i) + v.max(0.0) / (nf * nf)).sqrt();
        }
    }
    fn classify_clipped(
        &self,
        image: &Image<'_>,
        options: Options,
        i: usize,
        [outer, inner]: [[usize; 4]; 2],
    ) -> bool {
        if image.mask[i] {
            return false;
        }
        let count = |[a, c, d, e]: [usize; 4]| {
            i64::from(self.count[e]) - i64::from(self.count[c]) - i64::from(self.count[d])
                + i64::from(self.count[a])
        };
        let n = count(outer) - count(inner);
        if n < i64::try_from(options.min_background).unwrap_or(i64::MAX) {
            return false;
        }
        let sum = |values: &[f64], [a, c, d, e]: [usize; 4]| {
            values[e] - values[c] - values[d] + values[a]
        };
        let nf = n as f64;
        let excess = image.values[i] - (sum(&self.sum, outer) - sum(&self.sum, inner)) / nf;
        if excess > options.threshold_absolute {
            let v = sum(&self.variance, outer) - sum(&self.variance, inner);
            excess
                > options.threshold_sigma
                    * (image.pixel_variance(i) + v.max(0.0) / (nf * nf)).sqrt()
        } else {
            false
        }
    }

    // Column/row supports depend only on image shape and radii. Hoist clipping
    // and row-stride multiplication out of the per-pixel threshold arithmetic.
    fn classify(&self, image: &Image<'_>, options: Options, strong: &mut [bool]) {
        let interior = if image.width > 2 * options.outer_radius {
            (options.outer_radius, image.width - options.outer_radius)
        } else {
            (0, 0)
        };
        let columns: Vec<_> = (0..interior.0)
            .chain(interior.1..image.width)
            .map(|x| {
                (
                    x,
                    [options.outer_radius, options.inner_radius].map(|r| {
                        (
                            x.saturating_sub(r),
                            (x + 1).saturating_add(r).min(image.width),
                        )
                    }),
                )
            })
            .collect();
        let length = interior.1 - interior.0;
        let mut count_row = vec![0_i64; length];
        let mut sum_row = vec![0.0; length];
        let mut variance_row = vec![0.0; length];
        for y in 0..image.height {
            let rows = [options.outer_radius, options.inner_radius].map(|r| {
                (
                    y.saturating_sub(r) * self.width,
                    (y + 1).saturating_add(r).min(image.height) * self.width,
                )
            });
            for &(x, col) in &columns {
                let i = y * image.width + x;
                let indices = std::array::from_fn::<_, 2, _>(|k| {
                    [
                        rows[k].0 + col[k].0,
                        rows[k].0 + col[k].1,
                        rows[k].1 + col[k].0,
                        rows[k].1 + col[k].1,
                    ]
                });
                strong[i] = self.classify_clipped(image, options, i, indices);
            }
            if interior.0 == interior.1 {
                continue;
            }
            let sums = integral_corners(&self.sum, rows[0], interior, options.outer_radius).zip(
                integral_corners(&self.sum, rows[1], interior, options.inner_radius),
            );
            let variances =
                integral_corners(&self.variance, rows[0], interior, options.outer_radius).zip(
                    integral_corners(&self.variance, rows[1], interior, options.inner_radius),
                );
            let counts =
                integral_corners(&self.count, rows[0], interior, options.outer_radius).zip(
                    integral_corners(&self.count, rows[1], interior, options.inner_radius),
                );
            let first = y * image.width + interior.0;
            let last = y * image.width + interior.1;
            for ((outer, inner), dst) in counts.zip(&mut count_row) {
                let count = |[a, c, d, e]: [&u32; 4]| {
                    i64::from(*e) - i64::from(*c) - i64::from(*d) + i64::from(*a)
                };
                *dst = count(outer) - count(inner);
            }
            let sum = |[a, c, d, e]: [&f64; 4]| *e - *c - *d + *a;
            for ((outer, inner), dst) in sums.zip(&mut sum_row) {
                *dst = sum(outer) - sum(inner);
            }
            for ((outer, inner), dst) in variances.zip(&mut variance_row) {
                *dst = sum(outer) - sum(inner);
            }
            let statistics = (&count_row[..], &sum_row[..], &variance_row[..]);
            let values = &image.values[first..last];
            let mask = &image.mask[first..last];
            let dst = &mut strong[first..last];
            match image.variance {
                Some(variance) => classify_row(
                    statistics,
                    values,
                    mask,
                    dst,
                    variance[first..last].iter().copied(),
                    options,
                ),
                None => classify_row(
                    statistics,
                    values,
                    mask,
                    dst,
                    values.iter().map(|value| {
                        (value.max(0.0) / image.gain + image.read_noise.powi(2)).max(1.0)
                    }),
                    options,
                ),
            }
        }
    }
}

/// Find 8-connected spots using two annular-mean threshold passes, then sum.
///
/// Pass two excludes pass-one strong pixels from the background only. This is
/// not iterative model fitting, indexing, a calibrated false-discovery test or
/// a complete-reflection/partiality correction.
///
/// # Errors
/// Returns an error for invalid image/options or exceeding the component budget.
pub fn detect(image: &Image<'_>, options: Options) -> Result<Detection, SpotError> {
    image.validate()?;
    options.validate()?;
    let strong = strong_pixels(image, options, true)?;
    measure_components(image, &components(image, strong), options)
}

/// One 8-connected component of strong pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Component {
    /// Tight bounds of its pixels, without padding.
    pub bounds: Bounds,
    /// Strong pixels in it.
    pub pixels: usize,
}

/// Both threshold passes and every 8-connected component, in [`detect`]'s
/// row-major discovery order, before the `min_pixels`, `max_pixels` and
/// `max_spots` rules and extraction. With [`measure_components`], a host can
/// change `min_pixels` (or padding, background margin, `max_pixels`) without
/// detecting again.
///
/// # Errors
/// Returns an error for invalid image/options or numerical overflow.
pub fn detect_components(image: &Image<'_>, options: Options) -> Result<Vec<Component>, SpotError> {
    image.validate()?;
    options.validate()?;
    let strong = strong_pixels(image, options, true)?;
    Ok(components(image, strong))
}

/// The rest of [`detect`] on components from [`detect_components`]: drop
/// those below `min_pixels` (counted as `rejected_small`), pad the rest,
/// extract them with every kept box excluded from every background, and flag
/// those above `max_pixels`. `detect` is exactly `detect_components` then
/// this, so the result equals a fresh `detect` with these options whenever
/// the threshold and support options match those of the detection.
///
/// # Errors
/// Returns an error for invalid image/options, or above `max_spots` kept
/// components.
pub fn measure_components(
    image: &Image<'_>,
    components: &[Component],
    options: Options,
) -> Result<Detection, SpotError> {
    image.validate()?;
    options.validate()?;
    let (mut boxes, mut flags, mut rejected_small) = (Vec::new(), Vec::new(), 0);
    for c in components {
        let b = c.bounds;
        if c.pixels == 0
            || b.x0 >= b.x1
            || b.y0 >= b.y1
            || b.x1 > image.width
            || b.y1 > image.height
        {
            return Err(invalid("invalid component"));
        }
        if c.pixels < options.min_pixels {
            rejected_small += 1;
            continue;
        }
        if boxes.len() >= options.max_spots {
            return Err(invalid(
                "max_spots exceeded; no candidates silently truncated",
            ));
        }
        boxes.push(b.expanded(options.padding, image.width, image.height));
        flags.push(if c.pixels > options.max_pixels {
            LARGE
        } else {
            0
        });
    }
    let mut spots = extract_validated(image, &boxes, options)?;
    for (spot, flag) in spots.iter_mut().zip(flags) {
        spot.flags |= flag;
    }
    Ok(Detection {
        spots,
        rejected_small,
    })
}

/// [`detect`] with pass two always rebuilt over the whole frame, as before
/// the local pass two. The same result; for verification and timing.
///
/// # Errors
/// As [`detect`].
pub fn detect_two_pass(image: &Image<'_>, options: Options) -> Result<Detection, SpotError> {
    image.validate()?;
    options.validate()?;
    let strong = strong_pixels(image, options, false)?;
    measure_components(image, &components(image, strong), options)
}

/// Both threshold passes. Pass two is local (`Integral::reclassify_near`)
/// when `local` and pass one's sums are exact; otherwise a full rebuild.
fn strong_pixels(image: &Image<'_>, options: Options, local: bool) -> Result<Vec<bool>, SpotError> {
    let mut strong = vec![false; image.values.len()];
    let mut integral = Integral::new(image);
    let exact = integral.rebuild(image, &strong)?;
    integral.classify(image, options, &mut strong);
    if local && exact {
        integral.reclassify_near(image, options, &mut strong);
    } else {
        integral.rebuild(image, &strong)?;
        integral.classify(image, options, &mut strong);
    }
    Ok(strong)
}

/// 8-connected components of the strong pixels, in row-major discovery order.
fn components(image: &Image<'_>, mut strong: Vec<bool>) -> Vec<Component> {
    let (mut found, mut queue) = (Vec::new(), Vec::new());
    for i in 0..strong.len() {
        if !strong[i] {
            continue;
        }
        strong[i] = false;
        queue.clear();
        queue.push(i);
        let (x, y) = (i % image.width, i / image.width);
        let mut bounds = Bounds {
            x0: x,
            y0: y,
            x1: x + 1,
            y1: y + 1,
        };
        let mut head = 0;
        while head < queue.len() {
            let p = queue[head];
            head += 1;
            let (x, y) = (p % image.width, p / image.width);
            bounds.x0 = bounds.x0.min(x);
            bounds.y0 = bounds.y0.min(y);
            bounds.x1 = bounds.x1.max(x + 1);
            bounds.y1 = bounds.y1.max(y + 1);
            for yy in y.saturating_sub(1)..=(y + 1).min(image.height - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(image.width - 1) {
                    let j = yy * image.width + xx;
                    if strong[j] {
                        strong[j] = false;
                        queue.push(j);
                    }
                }
            }
        }
        found.push(Component {
            bounds,
            pixels: queue.len(),
        });
    }
    found
}

/// Sum caller-supplied boxes, including weak/negative observations without
/// running a positive-signal detection step. Boxes must be nonempty and in bounds.
///
/// # Errors
/// Returns an error for invalid image/options, bounds or exceeding the box budget.
pub fn extract(
    image: &Image<'_>,
    boxes: &[Bounds],
    options: Options,
) -> Result<Vec<Spot>, SpotError> {
    image.validate()?;
    options.validate()?;
    extract_validated(image, boxes, options)
}
#[allow(clippy::too_many_lines)] // One ordered aperture pass; arithmetic is audited against NumPy.
fn extract_validated(
    image: &Image<'_>,
    boxes: &[Bounds],
    options: Options,
) -> Result<Vec<Spot>, SpotError> {
    if boxes.len() > options.max_spots {
        return Err(invalid("max_spots exceeded"));
    }
    let mut coverage = vec![0_u32; image.values.len()];
    for b in boxes {
        if b.x0 >= b.x1 || b.y0 >= b.y1 || b.x1 > image.width || b.y1 > image.height {
            return Err(invalid(
                "boxes must be nonempty half-open bounds inside the image",
            ));
        }
        for y in b.y0..b.y1 {
            for x in b.x0..b.x1 {
                coverage[y * image.width + x] += 1;
            }
        }
    }
    let mut result = Vec::with_capacity(boxes.len());
    for &b in boxes {
        let mut flags = if b.x0 == 0 || b.y0 == 0 || b.x1 == image.width || b.y1 == image.height {
            EDGE
        } else {
            0
        };
        let outside = b.expanded(options.background_margin, image.width, image.height);
        let (mut bg_sum, mut bg_variance, mut nb) = (0.0, 0.0, 0_usize);
        for y in outside.y0..outside.y1 {
            for x in outside.x0..outside.x1 {
                let i = y * image.width + x;
                if coverage[i] == 0 && !image.mask[i] && !image.saturated(i) {
                    bg_sum += image.values[i];
                    bg_variance += image.pixel_variance(i);
                    nb += 1;
                }
            }
        }
        let (bg, bg_var) = if nb >= options.min_background {
            (bg_sum / nb as f64, bg_variance / (nb as f64).powi(2))
        } else {
            flags |= BACKGROUND;
            (f64::NAN, f64::NAN)
        };
        let (mut sum, mut variance, mut ns, mut maximum) = (0.0, 0.0, 0_usize, f64::NEG_INFINITY);
        let (mut weight, mut sx, mut sy) = (
            MomentSum::default(),
            MomentSum::default(),
            MomentSum::default(),
        );
        for y in b.y0..b.y1 {
            for x in b.x0..b.x1 {
                let i = y * image.width + x;
                if coverage[i] > 1 {
                    flags |= OVERLAP;
                }
                if image.mask[i] {
                    flags |= MASKED;
                    continue;
                }
                if image.saturated(i) {
                    flags |= SATURATED;
                }
                let d = image.values[i];
                sum += d;
                variance += image.pixel_variance(i);
                ns += 1;
                maximum = maximum.max(d);
                let w = if bg.is_finite() {
                    (d - bg).max(0.0)
                } else {
                    0.0
                };
                weight.add(w);
                sx.add(w * x as f64);
                sy.add(w * y as f64);
            }
        }
        let weight = weight.value;
        let (cx, cy) = if weight > 0.0 {
            (sx.value / weight, sy.value / weight)
        } else {
            flags |= NO_CENTROID;
            (
                (b.x0 + b.x1 - 1) as f64 / 2.0,
                (b.y0 + b.y1 - 1) as f64 / 2.0,
            )
        };
        let (mut vx, mut vy) = (MomentSum::default(), MomentSum::default());
        if weight > 0.0 {
            for y in b.y0..b.y1 {
                for x in b.x0..b.x1 {
                    let i = y * image.width + x;
                    if !image.mask[i] {
                        let w = (image.values[i] - bg).max(0.0);
                        vx.add(w * (x as f64 - cx).powi(2));
                        vy.add(w * (y as f64 - cy).powi(2));
                    }
                }
            }
        }
        let intensity = if ns > 0 {
            sum - ns as f64 * bg
        } else {
            f64::NAN
        };
        let sigma = if ns > 0 {
            (variance + (ns as f64).powi(2) * bg_var).sqrt()
        } else {
            f64::NAN
        };
        if !sum.is_finite()
            || !variance.is_finite()
            || !weight.is_finite()
            || (ns > 0 && bg.is_finite() && (!intensity.is_finite() || !sigma.is_finite()))
        {
            return Err(invalid("numerical overflow in aperture sums"));
        }
        result.push(Spot {
            bounds: b,
            flags,
            values: [
                cx,
                cy,
                intensity,
                sigma,
                bg,
                bg_var.sqrt(),
                if ns > 0 { maximum } else { f64::NAN },
                if weight > 0.0 {
                    (vx.value / weight).sqrt()
                } else {
                    f64::NAN
                },
                if weight > 0.0 {
                    (vy.value / weight).sqrt()
                } else {
                    f64::NAN
                },
                ns as f64,
                nb as f64,
            ],
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_sum_and_background_variance() {
        let mut values = vec![10.0; 25 * 25];
        values[12 * 25 + 12] = 7.0;
        let mask = vec![false; values.len()];
        let variance = vec![4.0; values.len()];
        let image = Image {
            values: &values,
            height: 25,
            width: 25,
            mask: &mask,
            variance: Some(&variance),
            gain: 1.0,
            read_noise: 0.0,
            saturation: None,
        };
        let b = Bounds {
            x0: 12,
            y0: 12,
            x1: 13,
            y1: 13,
        };
        let s = extract(&image, &[b], Options::default()).unwrap();
        assert!((s[0].values[2] + 3.0).abs() < 1e-14);
        let expected = (4.0_f64 + 4.0 / 80.0).sqrt();
        assert!((s[0].values[3] - expected).abs() < 1e-14);
    }
    /// Deterministic integer frames: Poisson-like background, spots of
    /// several sizes and brightnesses, masked and saturated pixels.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss
    )] // Small synthetic coordinates.
    fn frame(seed: u64, width: usize, height: usize, spots: usize) -> (Vec<f64>, Vec<bool>) {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let mut values: Vec<f64> = (0..width * height)
            .map(|_| (3.0 + 4.0 * next() + 3.0 * next()).floor())
            .collect();
        for _ in 0..spots {
            let (cx, cy) = (next() * width as f64, next() * height as f64);
            let (peak, sigma) = (20.0 + 3000.0 * next() * next(), 0.6 + 2.0 * next());
            let r = (4.0 * sigma).ceil() as isize;
            for dy in -r..=r {
                for dx in -r..=r {
                    let (x, y) = (cx as isize + dx, cy as isize + dy);
                    if x < 0 || y < 0 || x >= width as isize || y >= height as isize {
                        continue;
                    }
                    let d2 = (dx * dx + dy * dy) as f64;
                    values[y as usize * width + x as usize] +=
                        (peak * (-d2 / (2.0 * sigma * sigma)).exp()).floor();
                }
            }
        }
        let mask = (0..width * height).map(|_| next() < 0.03).collect();
        (values, mask)
    }
    fn same(a: &Detection, b: &Detection) {
        assert_eq!(a.rejected_small, b.rejected_small);
        assert_eq!(a.spots.len(), b.spots.len());
        for (x, y) in a.spots.iter().zip(&b.spots) {
            assert_eq!(x.bounds, y.bounds);
            assert_eq!(x.flags, y.flags);
            assert_eq!(x.values.map(f64::to_bits), y.values.map(f64::to_bits));
        }
    }
    #[test]
    fn local_second_pass_is_bit_identical_to_the_full_rebuild() {
        let mut checked_local = 0;
        for (seed, (width, height, spots)) in [
            (32, 32, 2),
            (200, 150, 40),
            (17, 300, 12),
            (300, 9, 6),
            (129, 130, 60),
            (64, 64, 0),
            (1, 1, 0),
        ]
        .into_iter()
        .enumerate()
        {
            let (values, mask) = frame(seed as u64 + 1, width, height, spots);
            let supplied: Vec<f64> = values.iter().map(|v| v.max(1.0)).collect();
            let fractional: Vec<f64> = values.iter().map(|v| v.max(1.0) + 0.25).collect();
            for (variance, gain, read_noise, saturation) in [
                (None, 1.0, 0.0, None),
                (None, 1.0, 0.0, Some(800.0)),
                (Some(&supplied[..]), 1.0, 0.0, None),
                // Not exact: pass two is the full rebuild.
                (Some(&fractional[..]), 1.0, 0.0, None),
                (None, 1.7, 1.3, None),
            ] {
                for options in [
                    Options::default(),
                    Options {
                        inner_radius: 0,
                        outer_radius: 1,
                        min_background: 3,
                        ..Options::default()
                    },
                    Options {
                        inner_radius: 3,
                        outer_radius: 70,
                        threshold_sigma: 3.0,
                        min_pixels: 1,
                        ..Options::default()
                    },
                    Options {
                        threshold_absolute: 15.0,
                        ..Options::default()
                    },
                ] {
                    let image = Image {
                        values: &values,
                        height,
                        width,
                        mask: &mask,
                        variance,
                        gain,
                        read_noise,
                        saturation,
                    };
                    let local = strong_pixels(&image, options, true).unwrap();
                    let full = strong_pixels(&image, options, false).unwrap();
                    assert_eq!(local, full, "{width}x{height} {options:?} {variance:?}");
                    match (detect(&image, options), detect_two_pass(&image, options)) {
                        (Ok(a), Ok(b)) => same(&a, &b),
                        (Err(a), Err(b)) => assert_eq!(a.0, b.0),
                        (a, b) => panic!("{a:?} {b:?}"),
                    }
                    let mut probe = Integral::new(&image);
                    if probe.rebuild(&image, &vec![false; values.len()]).unwrap()
                        && local.iter().any(|&s| s)
                    {
                        checked_local += 1;
                    }
                }
            }
        }
        // The exact path ran with strong pixels present in many cases.
        assert!(checked_local > 40, "{checked_local}");
    }
    #[test]
    fn remeasured_components_equal_a_fresh_detection_at_each_minimum() {
        let (values, mask) = frame(5, 240, 200, 50);
        let image = Image {
            values: &values,
            height: 200,
            width: 240,
            mask: &mask,
            variance: None,
            gain: 1.0,
            read_noise: 0.0,
            saturation: Some(2500.0),
        };
        let base = Options {
            min_pixels: 1,
            max_pixels: 40,
            ..Options::default()
        };
        let found = detect_components(&image, base).unwrap();
        assert!(found.iter().any(|c| c.pixels == 1) && found.iter().any(|c| c.pixels > 40));
        for min_pixels in [1, 2, 3, 5, 9, 25] {
            let options = Options { min_pixels, ..base };
            let remeasured = measure_components(&image, &found, options).unwrap();
            same(&remeasured, &detect(&image, options).unwrap());
            // The same by hand: filter, pad, public `extract`, LARGE flag.
            let kept: Vec<_> = found.iter().filter(|c| c.pixels >= min_pixels).collect();
            let boxes: Vec<_> = kept
                .iter()
                .map(|c| c.bounds.expanded(options.padding, 240, 200))
                .collect();
            let mut spots = extract(&image, &boxes, options).unwrap();
            for (spot, c) in spots.iter_mut().zip(&kept) {
                if c.pixels > options.max_pixels {
                    spot.flags |= LARGE;
                }
            }
            same(
                &Detection {
                    spots,
                    rejected_small: found.len() - kept.len(),
                },
                &remeasured,
            );
        }
        // The budget still applies to the kept components.
        let tight = Options {
            min_pixels: 1,
            max_spots: 3,
            ..base
        };
        assert!(measure_components(&image, &found, tight).is_err());
        assert!(detect(&image, tight).is_err());
    }
    #[test]
    fn exactness_needs_integers_and_bounded_totals() {
        let values = vec![3.0, 4.0, 5.0, 6.0];
        let mask = vec![false; 4];
        let image = |values: &[f64], gain, read_noise| Image {
            values: &values.to_vec().leak()[..],
            height: 2,
            width: 2,
            mask: &mask.clone().leak()[..],
            variance: None,
            gain,
            read_noise,
            saturation: None,
        };
        let exact = |image: &Image<'_>| Integral::new(image).rebuild(image, &[false; 4]).unwrap();
        assert!(exact(&image(&values, 1.0, 0.0)));
        assert!(!exact(&image(&[3.5, 4.0, 5.0, 6.0], 1.0, 0.0)));
        assert!(!exact(&image(&values, 2.0, 0.0)), "variance v/2");
        assert!(!exact(&image(&values, 1.0, 0.5)), "variance + 0.25");
        assert!(
            !exact(&image(&[4.0e15, 4.0e15, 4.0e15, 4.0e15], 1.0, 0.0)),
            "beyond 2^53"
        );
    }
    #[test]
    fn detects_unknown_spot_and_preserves_mask_flag() {
        let mut values = vec![5.0; 32 * 32];
        for y in 12..15 {
            for x in 13..16 {
                values[y * 32 + x] += 100.0;
            }
        }
        let mut mask = vec![false; values.len()];
        mask[11 * 32 + 12] = true;
        let image = Image {
            values: &values,
            height: 32,
            width: 32,
            mask: &mask,
            variance: None,
            gain: 1.0,
            read_noise: 0.0,
            saturation: None,
        };
        let result = detect(&image, Options::default()).unwrap();
        assert_eq!(result.spots.len(), 1);
        assert!((result.spots[0].values[2] - 900.0).abs() < 1e-12);
        assert_eq!(result.spots[0].flags, MASKED);
    }
}
