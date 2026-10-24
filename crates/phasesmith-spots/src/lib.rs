//! Experimental, structure-independent detector spot measurements.
//!
//! Native, deterministic and single-threaded. No file formats, GUI, crystal
//! model, interpreter, global thread pool or external numerical dependencies.
//! See `docs/experimental-spots.md` for equations and uncertainty limitations.

// Pixel counts/coordinates are validated below 2^32, exactly representable in f64.
#![allow(clippy::cast_precision_loss)]
// Short indices and sums follow the documented numerical equations.
#![allow(clippy::many_single_char_names)]

use std::fmt::{Display, Formatter};

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
    fn expanded(self, radius: usize, width: usize, height: usize) -> Self {
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
    fn rebuild(&mut self, image: &Image<'_>, exclude: &[bool]) -> Result<(), SpotError> {
        // Each interior entry is overwritten; the zero first row/column stay zero.
        for y in 0..image.height {
            let (mut s, mut v, mut c) = (0.0, 0.0, 0);
            for x in 0..image.width {
                let i = y * image.width + x;
                if !image.mask[i] && !exclude[i] && !image.saturated(i) {
                    s += image.values[i];
                    v += image.pixel_variance(i);
                    c += 1;
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
        Ok(())
    }
    fn rectangle(&self, b: Bounds) -> (f64, f64, i64) {
        let (a, c, d, e) = (
            b.y0 * self.width + b.x0,
            b.y0 * self.width + b.x1,
            b.y1 * self.width + b.x0,
            b.y1 * self.width + b.x1,
        );
        (
            self.sum[e] - self.sum[c] - self.sum[d] + self.sum[a],
            self.variance[e] - self.variance[c] - self.variance[d] + self.variance[a],
            i64::from(self.count[e]) - i64::from(self.count[c]) - i64::from(self.count[d])
                + i64::from(self.count[a]),
        )
    }
    fn strong(&self, image: &Image<'_>, options: Options, x: usize, y: usize) -> bool {
        let i = y * image.width + x;
        if image.mask[i] {
            return false;
        }
        let center = Bounds {
            x0: x,
            y0: y,
            x1: x + 1,
            y1: y + 1,
        };
        let (s, v, n) =
            self.rectangle(center.expanded(options.outer_radius, image.width, image.height));
        let (si, vi, ni) =
            self.rectangle(center.expanded(options.inner_radius, image.width, image.height));
        let n = n - ni;
        if n < i64::try_from(options.min_background).unwrap_or(i64::MAX) {
            return false;
        }
        let nf = n as f64;
        let excess = image.values[i] - (s - si) / nf;
        excess > options.threshold_absolute
            && excess
                > options.threshold_sigma
                    * (image.pixel_variance(i) + (v - vi).max(0.0) / (nf * nf)).sqrt()
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
    let mut strong = vec![false; image.values.len()];
    let mut integral = Integral::new(image);
    integral.rebuild(image, &strong)?;
    for y in 0..image.height {
        for x in 0..image.width {
            strong[y * image.width + x] = integral.strong(image, options, x, y);
        }
    }
    integral.rebuild(image, &strong)?;
    for y in 0..image.height {
        for x in 0..image.width {
            strong[y * image.width + x] = integral.strong(image, options, x, y);
        }
    }
    drop(integral);
    let (mut boxes, mut flags, mut queue, mut rejected_small) =
        (Vec::new(), Vec::new(), Vec::new(), 0);
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
        if queue.len() < options.min_pixels {
            rejected_small += 1;
            continue;
        }
        if boxes.len() >= options.max_spots {
            return Err(invalid(
                "max_spots exceeded; no candidates silently truncated",
            ));
        }
        boxes.push(bounds.expanded(options.padding, image.width, image.height));
        flags.push(if queue.len() > options.max_pixels {
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
