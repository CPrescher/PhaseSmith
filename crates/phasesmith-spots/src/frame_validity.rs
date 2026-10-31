//! Original-count shadow estimation and full-exposure conical DAC visibility.
//!
//! No interpreter, image reader, `SciPy`, GUI or global worker pool is involved.
//! Illumination-ratio behavior is compared with the MIT Dioptas estimator;
//! Gaussian smoothing, connectivity and finite-support geometry are explicit.

// All dimensions are validated below 1e8; float32 is the declared filter field.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use crate::reciprocal::Geometry;
use std::collections::VecDeque;
use std::f64::consts::TAU;
use std::fmt::{Display, Formatter};

/// Maximum retained coarse cells across a scan.
pub const MAX_CELLS: usize = 50_000_000;

/// Invalid numerical input or insufficient original-frame information.
#[derive(Clone, Debug)]
pub enum Error {
    /// Invalid shape, parameter, count or memory budget.
    Invalid(&'static str),
    /// The illumination estimator cannot establish a mask.
    Unavailable(&'static str),
}
impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(s) | Self::Unavailable(s) => f.write_str(s),
        }
    }
}
impl std::error::Error for Error {}

/// Detector dimensions and whole-block exclusion convention.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    /// Detector rows.
    pub height: usize,
    /// Detector columns.
    pub width: usize,
    /// Pixels along each coarse block side, including partial edge blocks.
    pub block: usize,
}
impl Shape {
    /// Validate dimensions and return coarse rows/columns.
    /// # Errors
    /// Rejects empty, oversized or overflowing dimensions.
    pub fn coarse(self) -> Result<[usize; 2], Error> {
        if self.height == 0
            || self.width == 0
            || self.height > 100_000
            || self.width > 100_000
            || self
                .height
                .checked_mul(self.width)
                .is_none_or(|n| n > 100_000_000)
            || !(1..=1024).contains(&self.block)
        {
            return Err(Error::Invalid("invalid detector shape or block size"));
        }
        Ok([
            self.height.div_ceil(self.block),
            self.width.div_ceil(self.block),
        ])
    }
}

/// Owned detector-block mask. True means unavailable, never zero intensity.
#[derive(Clone, Debug)]
pub struct Mask {
    /// Detector shape and exclusion block size.
    pub shape: Shape,
    /// Row-major coarse boolean exclusions.
    pub excluded: Vec<bool>,
}
impl Mask {
    /// Expand exactly one full detector mask.
    /// # Errors
    /// Rejects inconsistent shape/array lengths.
    pub fn expand(&self) -> Result<Vec<bool>, Error> {
        let [h, w] = self.shape.coarse()?;
        if self.excluded.len() != h * w {
            return Err(Error::Invalid("invalid coarse mask length"));
        }
        Ok((0..self.shape.height * self.shape.width)
            .map(|i| {
                self.excluded[(i / self.shape.width / self.shape.block) * w
                    + (i % self.shape.width / self.shape.block)]
            })
            .collect())
    }
}

/// A coarse original-frame snapshot and whether any input pixel was available.
pub struct CoarseFrame {
    /// Row-major block medians; an entirely unavailable block has value zero.
    pub values: Vec<f64>,
    /// False only when all source pixels were explicitly excluded/nonfinite.
    pub available: bool,
}

#[allow(clippy::manual_midpoint)] // Counts <=1e30 cannot overflow; preserve reference rounding.
fn median(values: &mut [f64]) -> f64 {
    let m = values.len() / 2;
    let even = values.len() % 2 == 0;
    let (lower, middle, _) = values.select_nth_unstable_by(m, f64::total_cmp);
    if even {
        let low = lower
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .expect("nonempty lower half");
        (low + *middle) / 2.
    } else {
        *middle
    }
}

/// Compute medians directly from original count arrays without changing pixels.
/// # Errors
/// Rejects malformed input or available counts outside `[0,1e30]`. Nonfinite or
/// explicitly masked samples do not contribute. Real zeros remain available.
pub fn coarsen(image: &[f64], excluded: &[bool], shape: Shape) -> Result<CoarseFrame, Error> {
    let [h, w] = shape.coarse()?;
    if image.len() != shape.height * shape.width || excluded.len() != image.len() {
        return Err(Error::Invalid("original frame/mask shape mismatch"));
    }
    let mut values = vec![0.; h * w];
    let mut scratch = Vec::with_capacity(shape.block * shape.block);
    let mut available = false;
    for y in 0..h {
        for x in 0..w {
            scratch.clear();
            for row in y * shape.block..((y + 1) * shape.block).min(shape.height) {
                for col in x * shape.block..((x + 1) * shape.block).min(shape.width) {
                    let i = row * shape.width + col;
                    if !excluded[i] && image[i].is_finite() {
                        if !(0.0..=1e30).contains(&image[i]) {
                            return Err(Error::Invalid(
                                "original nonnegative counts must lie in [0,1e30]",
                            ));
                        }
                        scratch.push(image[i]);
                    }
                }
            }
            if !scratch.is_empty() {
                available = true;
                values[y * w + x] = median(&mut scratch);
            }
        }
    }
    Ok(CoarseFrame { values, available })
}

/// Parameters of the original-image illumination-ratio estimator.
#[derive(Clone, Copy, Debug)]
pub struct SeatOptions {
    /// Inclusive illuminated reference start; None uses round-to-even 37%.
    pub reference_first: Option<usize>,
    /// Inclusive illuminated reference end; None uses round-to-even 59%.
    pub reference_last: Option<usize>,
    /// Strict normalized attenuation threshold, between zero and one.
    pub threshold: f64,
    /// Manhattan dilation guard in detector pixels, rounded up to whole blocks.
    pub guard_pixels: usize,
}
impl Default for SeatOptions {
    fn default() -> Self {
        Self {
            reference_first: None,
            reference_last: None,
            threshold: 0.85,
            guard_pixels: 32,
        }
    }
}

/// Complete original-frame shadow estimate, independent of any frontend.
pub struct SeatEstimate {
    /// Frame-major, then row-major coarse exclusions.
    pub excluded: Vec<bool>,
    /// Detection-only flux normalizations; not corrections to measured counts.
    pub detection_scales: Vec<f64>,
    /// Actual illuminated reference frame indices.
    pub reference_indices: Vec<usize>,
    /// Original inputs already known unavailable through their pixel masks.
    pub known_unavailable_frames: Vec<usize>,
}

fn reflect(i: isize, length: usize) -> usize {
    let n = isize::try_from(length).expect("bounded detector dimension");
    let j = i.rem_euclid(2 * n);
    usize::try_from(if j >= n { 2 * n - 1 - j } else { j }).expect("nonnegative reflected index")
}

fn smooth(input: &[f64], frames: usize, h: usize, w: usize) -> Vec<f32> {
    let n = h * w;
    let raw: Vec<_> = input.iter().map(|&v| v as f32).collect();
    let mut filtered = vec![0_f32; raw.len()];
    for f in 0..frames {
        for y in 0..h {
            for x in 0..w {
                let mut stencil = [0_f32; 9];
                for dy in -1_isize..=1 {
                    for dx in -1_isize..=1 {
                        let j =
                            usize::try_from((dy + 1) * 3 + dx + 1).expect("nonnegative stencil");
                        stencil[j] = raw
                            [f * n + reflect(y as isize + dy, h) * w + reflect(x as isize + dx, w)];
                    }
                }
                stencil.sort_unstable_by(f32::total_cmp);
                filtered[f * n + y * w + x] = stencil[4];
            }
        }
    }
    let mut weights: [f64; 9] = std::array::from_fn(|i| (-0.5 * (i as f64 - 4.).powi(2)).exp());
    let total: f64 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= total;
    }
    let mut out = vec![0_f32; raw.len()];
    for vertical in [true, false] {
        for f in 0..frames {
            for y in 0..h {
                for x in 0..w {
                    let mut sum = 0.;
                    for (j, &weight) in weights.iter().enumerate() {
                        let d = j as isize - 4;
                        let yy = if vertical {
                            reflect(y as isize + d, h)
                        } else {
                            y
                        };
                        let xx = if vertical {
                            x
                        } else {
                            reflect(x as isize + d, w)
                        };
                        sum += f64::from(filtered[f * n + yy * w + xx]) * weight;
                    }
                    out[f * n + y * w + x] = sum as f32;
                }
            }
        }
        std::mem::swap(&mut filtered, &mut out);
    }
    filtered
}

fn neighbors(i: usize, h: usize, w: usize) -> impl Iterator<Item = usize> {
    let mut v = [None; 4];
    if i >= w {
        v[0] = Some(i - w);
    }
    if i + w < h * w {
        v[1] = Some(i + w);
    }
    if i % w != 0 {
        v[2] = Some(i - 1);
    }
    if i % w + 1 < w {
        v[3] = Some(i + 1);
    }
    v.into_iter().flatten()
}

fn dilate(mask: &[bool], h: usize, w: usize, radius: usize) -> Vec<bool> {
    if radius == 0 {
        return mask.to_vec();
    }
    let mut distance = vec![usize::MAX; mask.len()];
    let mut queue = VecDeque::new();
    for (i, &v) in mask.iter().enumerate() {
        if v {
            distance[i] = 0;
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        if distance[i] >= radius {
            continue;
        }
        for j in neighbors(i, h, w) {
            if distance[j] == usize::MAX {
                distance[j] = distance[i] + 1;
                queue.push_back(j);
            }
        }
    }
    distance.iter().map(|&d| d <= radius).collect()
}

fn close_and_fill(mask: &[bool], h: usize, w: usize) -> Vec<bool> {
    let expanded = dilate(mask, h, w, 1);
    let mut closed = vec![false; mask.len()];
    for i in 0..mask.len() {
        let inner = i >= w && i + w < h * w && i % w != 0 && i % w + 1 < w;
        closed[i] = inner && expanded[i] && neighbors(i, h, w).all(|j| expanded[j]);
    }
    let mut exterior = vec![false; mask.len()];
    let mut queue = VecDeque::new();
    for i in 0..mask.len() {
        if (i < w || i + w >= h * w || i % w == 0 || i % w + 1 == w) && !closed[i] {
            exterior[i] = true;
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        for j in neighbors(i, h, w) {
            if !closed[j] && !exterior[j] {
                exterior[j] = true;
                queue.push_back(j);
            }
        }
    }
    exterior.iter().map(|v| !v).collect()
}

fn components(mask: &[bool], shape: Shape, h: usize, w: usize) -> Vec<bool> {
    let mut seen = vec![false; mask.len()];
    let mut keep = vec![false; mask.len()];
    let mut queue = VecDeque::new();
    for seed in 0..mask.len() {
        if !mask[seed] || seen[seed] {
            continue;
        }
        let mut cells = Vec::new();
        let mut side = false;
        seen[seed] = true;
        queue.push_back(seed);
        while let Some(i) = queue.pop_front() {
            cells.push(i);
            let x = ((i % w) * shape.block) as f64;
            side |= x < shape.width as f64 * 350. / 2048. || x > shape.width as f64 * 1700. / 2048.;
            for j in neighbors(i, h, w) {
                if mask[j] && !seen[j] {
                    seen[j] = true;
                    queue.push_back(j);
                }
            }
        }
        if side && cells.len() >= 20 {
            for i in cells {
                keep[i] = true;
            }
        }
    }
    keep
}

fn quantile(values: &mut [f64], q: f64) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    let index = q * (values.len() - 1) as f64;
    let low = index.floor() as usize;
    let high = index.ceil() as usize;
    let t = index - low as f64;
    if t < 0.5 {
        values[low] + (values[high] - values[low]) * t
    } else {
        values[high] - (values[high] - values[low]) * (1. - t)
    }
}

/// Estimate moving seat shadows from a complete frame-major block-median array.
/// All numerical operations, including filters, percentiles and morphology,
/// execute in Rust. Uniform positive dimming is not a geometric exclusion.
/// # Errors
/// Rejects invalid input, unavailable illuminated references or an unclassified
/// wholly dark exposure. No all-valid fallback is returned.
pub fn estimate(
    coarse: &[f64],
    available: &[bool],
    shape: Shape,
    options: SeatOptions,
) -> Result<SeatEstimate, Error> {
    let [h, w] = shape.coarse()?;
    let frames = available.len();
    if frames < 5 {
        return Err(Error::Unavailable(
            "seat-shadow estimation needs at least five frames",
        ));
    }
    if frames > 1_000_000
        || frames.checked_mul(h * w).is_none_or(|n| n > MAX_CELLS)
        || coarse.len() != frames * h * w
        || coarse
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1e30).contains(v))
        || !options.threshold.is_finite()
        || !(0.0..1.0).contains(&options.threshold)
        || options.threshold == 0.
        || options.guard_pixels > 4096
    {
        return Err(Error::Invalid(
            "invalid coarse frames, threshold, guard or cell budget",
        ));
    }
    let first = options
        .reference_first
        .unwrap_or_else(|| (0.37 * (frames - 1) as f64).round_ties_even() as usize);
    let last = options
        .reference_last
        .unwrap_or_else(|| (0.59 * (frames - 1) as f64).round_ties_even() as usize);
    if first >= last || last >= frames {
        return Err(Error::Invalid("invalid inclusive reference range"));
    }
    let count = 5.min(last - first + 1);
    let reference_indices: Vec<_> = (0..count)
        .map(|i| first + i * (last - first) / (count - 1))
        .collect();
    let filtered = smooth(coarse, frames, h, w);
    let cells = h * w;
    let mut reference = vec![0_f32; cells];
    let mut scratch = Vec::with_capacity(count);
    for i in 0..cells {
        scratch.clear();
        scratch.extend(
            reference_indices
                .iter()
                .map(|&f| f64::from(filtered[f * cells + i])),
        );
        reference[i] = median(&mut scratch) as f32;
    }
    if !reference.iter().any(|&v| v > 40.) {
        return Err(Error::Unavailable(
            "illuminated reference background must exceed 40 counts",
        ));
    }
    let mut excluded = Vec::with_capacity(coarse.len());
    let mut detection_scales = Vec::with_capacity(frames);
    let mut known_unavailable_frames = Vec::new();
    let mut ratios = vec![0_f32; cells];
    let mut samples = Vec::with_capacity(cells);
    for f in 0..frames {
        samples.clear();
        for i in 0..cells {
            ratios[i] = filtered[f * cells + i] / reference[i].max(1.);
            if reference[i] > 40. {
                samples.push(f64::from(ratios[i]));
            }
        }
        let scale = quantile(&mut samples, 0.9);
        if scale <= 0. && available[f] {
            return Err(Error::Unavailable(
                "wholly dark unmasked frames have unknown visibility; supply reviewed masks or geometry",
            ));
        }
        detection_scales.push(scale);
        if !available[f] {
            known_unavailable_frames.push(f);
            excluded.extend(vec![true; cells]);
            continue;
        }
        let candidate: Vec<_> = (0..cells)
            .map(|i| {
                ratios[i] = (f64::from(ratios[i]) / scale.max(1e-6)) as f32;
                ratios[i] < options.threshold as f32 && reference[i] > 25.
            })
            .collect();
        let closed = close_and_fill(&candidate, h, w);
        let keep = components(&closed, shape, h, w);
        excluded.extend(dilate(
            &keep,
            h,
            w,
            options.guard_pixels.div_ceil(shape.block),
        ));
    }
    Ok(SeatEstimate {
        excluded,
        detection_scales,
        reference_indices,
        known_unavailable_frames,
    })
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn unit(v: [f64; 3]) -> Result<[f64; 3], Error> {
    let scale = v.into_iter().map(f64::abs).fold(0., f64::max);
    if !v.into_iter().all(f64::is_finite) || scale == 0. {
        return Err(Error::Invalid("axis/ray must be a finite nonzero triple"));
    }
    let scaled = v.map(|x| x / scale);
    let norm = dot(scaled, scaled).sqrt();
    Ok(scaled.map(|x| x / norm))
}

/// Exact closed-interval extrema of `ray dot R(axis,omega)normal`.
/// Axis and normal are unit vectors; angular bounds are radians, low<=high.
#[must_use]
pub fn dot_range(ray: [f64; 3], axis: [f64; 3], normal: [f64; 3], low: f64, high: f64) -> [f64; 2] {
    let parallel = axis.map(|v| v * dot(axis, normal));
    let c = dot(ray, parallel);
    let a = dot(ray, std::array::from_fn(|j| normal[j] - parallel[j]));
    let b = dot(ray, cross(axis, normal));
    let value = |x: f64| c + a * x.cos() + b * x.sin();
    let mut range = [value(low).min(value(high)), value(low).max(value(high))];
    let phase = b.atan2(a);
    let radius = a.hypot(b);
    for (point, extremum, upper) in [
        (phase, c + radius, true),
        (phase + std::f64::consts::PI, c - radius, false),
    ] {
        if ((low - point) / TAU).ceil() <= ((high - point) / TAU).floor() {
            if upper {
                range[1] = range[1].max(extremum);
            } else {
                range[0] = range[0].min(extremum);
            }
        }
    }
    range
}

/// Explicit conical aperture parameters; there is no universal opening default.
#[derive(Clone, Copy, Debug)]
pub struct DacOptions {
    /// Sample-fixed DAC axis in lab coordinates at omega zero.
    pub axis_at_zero: [f64; 3],
    /// Strictly positive half-opening below 90 degrees.
    pub half_opening_degrees: f64,
}

/// Mask every pixel/block whose full footprint or exposure leaves the DAC cone.
/// Four affine footprint corners must remain in the same cone over the closed
/// exposure. The cosine boundary admits eight float64 epsilons.
/// # Errors
/// Rejects invalid geometry, exposure, axis, opening or dimensions.
#[allow(clippy::float_cmp)] // Exact zero-width exposures are invalid; no fuzzy angular cutoff.
pub fn dac_mask(
    geometry: Geometry,
    bounds: [f64; 2],
    shape: Shape,
    options: DacOptions,
) -> Result<Mask, Error> {
    let [h, w] = shape.coarse()?;
    geometry
        .convert(&[])
        .map_err(|_| Error::Invalid("invalid affine detector geometry"))?;
    if bounds.iter().any(|v| !v.is_finite() || v.abs() > 1e8)
        || bounds[0] == bounds[1]
        || !options.half_opening_degrees.is_finite()
        || !(0.0..90.0).contains(&options.half_opening_degrees)
        || options.half_opening_degrees == 0.
    {
        return Err(Error::Invalid("invalid aperture opening or exposure"));
    }
    let axis = unit(geometry.axis)?;
    let normal = unit(options.axis_at_zero)?;
    let beam = unit(geometry.beam)?;
    let low = bounds[0].min(bounds[1]).to_radians();
    let high = bounds[0].max(bounds[1]).to_radians();
    let limit = options.half_opening_degrees.to_radians().cos() - 8. * f64::EPSILON;
    let incoming = dot_range(beam, axis, normal, low, high);
    let mut excluded = vec![true; h * w];
    if incoming[0] < limit && incoming[1] > -limit {
        return Ok(Mask { shape, excluded });
    }
    for y in 0..h {
        for x in 0..w {
            let mut minimum = f64::INFINITY;
            let mut maximum = f64::NEG_INFINITY;
            for (right, bottom) in [(false, false), (false, true), (true, false), (true, true)] {
                let xx = ((x + usize::from(right)) * shape.block).min(shape.width) as f64 - 0.5;
                let yy = ((y + usize::from(bottom)) * shape.block).min(shape.height) as f64 - 0.5;
                let ray = unit(std::array::from_fn(|j| {
                    geometry.origin[j] + xx * geometry.fast[j] + yy * geometry.slow[j]
                }))?;
                let range = dot_range(ray, axis, normal, low, high);
                minimum = minimum.min(range[0]);
                maximum = maximum.max(range[1]);
            }
            excluded[y * w + x] = minimum < limit && maximum > -limit;
        }
    }
    Ok(Mask { shape, excluded })
}
