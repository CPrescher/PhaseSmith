//! Guarded rectangular summation with a clipped, unweighted local background plane.
//! Derived from linear least squares and independent-pixel covariance propagation.
use crate::spots::{
    BACKGROUND, Bounds, EDGE, Image, MASKED, OVERLAP, SATURATED, Spot, SpotError, invalid,
};

/// Significant signed excess remains in the outermost signal-pixel strip.
pub const BOUNDARY: u32 = 2048;
/// Extensive background rejection or a poor planar residual model.
pub const MODEL: u32 = 4096;

type Matrix = [[f64; 3]; 3];
#[derive(Clone, Copy)]
struct Sample {
    x: [f64; 3],
    value: f64,
    variance: f64,
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn mv(a: Matrix, b: [f64; 3]) -> [f64; 3] {
    a.map(|r| dot(r, b))
}
fn inverse(a: Matrix) -> Option<Matrix> {
    let mut rows = [[0.0; 6]; 3];
    for i in 0..3 {
        rows[i][..3].copy_from_slice(&a[i]);
        rows[i][i + 3] = 1.0;
    }
    for c in 0..3 {
        let pivot = (c..3).max_by(|&i, &j| rows[i][c].abs().total_cmp(&rows[j][c].abs()))?;
        rows.swap(c, pivot);
        let d = rows[c][c];
        if !d.is_finite() || d.abs() < 1e-10 {
            return None;
        }
        for v in &mut rows[c] {
            *v /= d;
        }
        for r in 0..3 {
            if r != c {
                let f = rows[r][c];
                let pivot_row = rows[c];
                for (value, pivot) in rows[r].iter_mut().zip(pivot_row) {
                    *value -= f * pivot;
                }
            }
        }
    }
    Some(std::array::from_fn(|i| {
        std::array::from_fn(|j| rows[i][j + 3])
    }))
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n.is_multiple_of(2) {
        v[n / 2 - 1].midpoint(v[n / 2])
    } else {
        v[n / 2]
    }
}
fn fit(samples: &[Sample], keep: &[bool]) -> Option<([f64; 3], Matrix)> {
    let mut a = [[0.0; 3]; 3];
    let mut rhs = [0.0; 3];
    for (s, &yes) in samples.iter().zip(keep) {
        if yes {
            for i in 0..3 {
                rhs[i] += s.x[i] * s.value;
                for (value, &coordinate) in a[i].iter_mut().zip(&s.x) {
                    *value += s.x[i] * coordinate;
                }
            }
        }
    }
    let inv = inverse(a)?;
    Some((mv(inv, rhs), inv))
}

/// Diagnostics: plane coefficients (normalised x/y), dI/dcoeff (three), candidate
/// and rejected background counts, residual RMS, boundary I/sigma/fraction.
/// The covariance is conditional on fixed clipping membership and support.
///
/// # Errors
/// Invalid arrays, supports, numerical overflow or unsupported work budgets.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn extract(
    image: &Image<'_>,
    boxes: &[Bounds],
    gap: usize,
    margin: usize,
    minimum: usize,
    clip: f64,
) -> Result<(Vec<Spot>, Vec<[f64; 12]>), SpotError> {
    let (spots, diagnostics, _) = extract_full(image, boxes, gap, margin, minimum, clip)?;
    Ok((spots, diagnostics))
}

/// Pixel variance entering the σ of sums (peak, plane covariance, border).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SumVariance {
    /// `max(max(counts, 0)/gain + read², 1)`: the former rule, kept for
    /// [`extract`] and the native workflow.
    Floored,
    /// `max(counts, 0)/gain + read²` per pixel (unbiased for Poisson counts);
    /// a peak or border sum's variance is at least that of its fitted
    /// background and at least one count², so σ stays positive.
    CountsOrBackground,
}

/// Internal profile-collection variant retaining the full fitted-plane covariance.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity
)]
pub(crate) fn extract_full(
    image: &Image<'_>,
    boxes: &[Bounds],
    gap: usize,
    margin: usize,
    minimum: usize,
    clip: f64,
) -> Result<(Vec<Spot>, Vec<[f64; 12]>, Vec<[f64; 9]>), SpotError> {
    extract_full_with(
        image,
        boxes,
        gap,
        margin,
        minimum,
        clip,
        SumVariance::Floored,
    )
}

/// [`extract_full`] with a choice of the variance of sums. Clipping, the
/// residual RMS and all flags except [`BOUNDARY`] use the floored pixel
/// variance in either case.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity
)]
pub(crate) fn extract_full_with(
    image: &Image<'_>,
    boxes: &[Bounds],
    gap: usize,
    margin: usize,
    minimum: usize,
    clip: f64,
    sum: SumVariance,
) -> Result<(Vec<Spot>, Vec<[f64; 12]>, Vec<[f64; 9]>), SpotError> {
    image.validate()?;
    if boxes.len() > 1_000_000
        || gap > 128
        || !(1..=128).contains(&margin)
        || !(3..=1_000_000).contains(&minimum)
        || !clip.is_finite()
        || !(2.0..=10.0).contains(&clip)
    {
        return Err(invalid("invalid guarded plane background options"));
    }
    for &b in boxes {
        if b.x0 >= b.x1 || b.y0 >= b.y1 || b.x1 > image.width || b.y1 > image.height {
            return Err(invalid("invalid signal box"));
        }
    }
    let coverage = SparseCoverage::new(boxes, gap, image.width, image.height);
    extract_boxes(image, boxes, gap, margin, minimum, clip, &coverage, sum)
}

/// How many signal boxes and guarded boxes cover each pixel near one box.
pub(crate) trait Coverage {
    /// Signal-box and guard counts of every pixel of `region`, row-major.
    fn counts(&self, region: Bounds) -> (Vec<u32>, Vec<u32>);
}

/// Box-local coverage from a bucket index of the guarded boxes; no
/// full-image array. Counts equal those of a dense per-pixel accumulation.
pub(crate) struct SparseCoverage<'a> {
    boxes: &'a [Bounds],
    guards: Vec<Bounds>,
    cells: std::collections::HashMap<usize, Vec<u32>>,
    columns: usize,
}
/// Bucket side, pixels.
const CELL: usize = 32;
impl<'a> SparseCoverage<'a> {
    /// Heap bytes of the index (guards and bucket lists), excluding the
    /// hash table's control bytes.
    #[cfg(test)]
    pub(crate) fn heap_bytes(&self) -> usize {
        self.guards.capacity() * std::mem::size_of::<Bounds>()
            + self
                .cells
                .values()
                .map(|v| v.capacity() * 4 + 32)
                .sum::<usize>()
    }
    pub(crate) fn new(boxes: &'a [Bounds], gap: usize, width: usize, height: usize) -> Self {
        let columns = width.div_ceil(CELL);
        let guards: Vec<Bounds> = boxes
            .iter()
            .map(|b| b.expanded(gap, width, height))
            .collect();
        let mut cells: std::collections::HashMap<usize, Vec<u32>> =
            std::collections::HashMap::new();
        for (i, g) in guards.iter().enumerate() {
            for cy in g.y0 / CELL..=(g.y1 - 1) / CELL {
                for cx in g.x0 / CELL..=(g.x1 - 1) / CELL {
                    #[allow(clippy::cast_possible_truncation)] // At most 1e6 boxes.
                    cells.entry(cy * columns + cx).or_default().push(i as u32);
                }
            }
        }
        Self {
            boxes,
            guards,
            cells,
            columns,
        }
    }
}
fn add(counts: &mut [u32], region: Bounds, rect: Bounds) {
    let (x0, x1) = (rect.x0.max(region.x0), rect.x1.min(region.x1));
    let (y0, y1) = (rect.y0.max(region.y0), rect.y1.min(region.y1));
    let w = region.x1 - region.x0;
    for y in y0..y1 {
        for x in x0..x1 {
            counts[(y - region.y0) * w + x - region.x0] += 1;
        }
    }
}
impl Coverage for SparseCoverage<'_> {
    fn counts(&self, region: Bounds) -> (Vec<u32>, Vec<u32>) {
        let mut candidates = Vec::new();
        for cy in region.y0 / CELL..=(region.y1 - 1) / CELL {
            for cx in region.x0 / CELL..=(region.x1 - 1) / CELL {
                if let Some(list) = self.cells.get(&(cy * self.columns + cx)) {
                    candidates.extend_from_slice(list);
                }
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        let n = (region.x1 - region.x0) * (region.y1 - region.y0);
        let (mut signal, mut guard) = (vec![0; n], vec![0; n]);
        for &i in &candidates {
            add(&mut signal, region, self.boxes[i as usize]);
            add(&mut guard, region, self.guards[i as usize]);
        }
        (signal, guard)
    }
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity
)]
fn extract_boxes(
    image: &Image<'_>,
    boxes: &[Bounds],
    gap: usize,
    margin: usize,
    minimum: usize,
    clip: f64,
    coverage: &impl Coverage,
    sum: SumVariance,
) -> Result<(Vec<Spot>, Vec<[f64; 12]>, Vec<[f64; 9]>), SpotError> {
    // Variance of a pixel in a sum, given its floored variance and fitted
    // background. A supplied variance image is always used as given.
    let counts = sum == SumVariance::CountsOrBackground && image.variance.is_none();
    let read = image.read_noise.powi(2);
    let sum_variance = |value: f64, floored: f64| {
        if counts {
            value.max(0.0) / image.gain + read
        } else {
            floored
        }
    };
    let mut out = Vec::with_capacity(boxes.len());
    let mut diagnostics = Vec::with_capacity(boxes.len());
    let mut covariances = Vec::with_capacity(boxes.len());
    for &b in boxes {
        let cx = (b.x0 + b.x1 - 1) as f64 / 2.0;
        let cy = (b.y0 + b.y1 - 1) as f64 / 2.0;
        let sx = ((b.x1 - b.x0) as f64 / 2.0).max(1.0);
        let sy = ((b.y1 - b.y0) as f64 / 2.0).max(1.0);
        let basis = |x: usize, y: usize| [1.0, (x as f64 - cx) / sx, (y as f64 - cy) / sy];
        let outer = b.expanded(gap + margin, image.width, image.height);
        let (signal, guard) = coverage.counts(outer);
        let local = |x: usize, y: usize| (y - outer.y0) * (outer.x1 - outer.x0) + x - outer.x0;
        let mut samples = Vec::new();
        for y in outer.y0..outer.y1 {
            for x in outer.x0..outer.x1 {
                let i = y * image.width + x;
                if guard[local(x, y)] == 0 && !image.mask[i] && !image.saturated(i) {
                    samples.push(Sample {
                        x: basis(x, y),
                        value: image.values[i],
                        variance: image.pixel_variance(i),
                    });
                }
            }
        }
        let mut keep = vec![true; samples.len()];
        let mut model = None;
        if samples.len() >= minimum {
            for iteration in 0..9 {
                if keep.iter().filter(|&&v| v).count() < minimum {
                    model = None;
                    break;
                }
                model = fit(&samples, &keep);
                let Some((coeff, _)) = model else {
                    break;
                };
                if iteration == 8 {
                    break;
                }
                let residuals: Vec<f64> = samples
                    .iter()
                    .zip(&keep)
                    .filter(|(_, k)| **k)
                    .map(|(s, _)| (s.value - dot(s.x, coeff)) / s.variance.sqrt())
                    .collect();
                let centre = median(residuals.clone());
                let scale = (1.4826
                    * median(residuals.iter().map(|r| (r - centre).abs()).collect()))
                .max(1.0);
                let mut changed = false;
                for (s, k) in samples.iter().zip(&mut keep) {
                    if *k
                        && ((s.value - dot(s.x, coeff)) / s.variance.sqrt() - centre).abs()
                            > clip * scale
                    {
                        *k = false;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
        }
        let mut flags = if b.x0 == 0 || b.y0 == 0 || b.x1 == image.width || b.y1 == image.height {
            EDGE
        } else {
            0
        };
        let n = keep.iter().filter(|&&v| v).count();
        let mut diag = [f64::NAN; 12];
        diag[6] = samples.len() as f64;
        diag[7] = (samples.len() - n) as f64;
        let (coeff, inv) = model.unwrap_or_else(|| {
            flags |= BACKGROUND;
            ([f64::NAN; 3], [[f64::NAN; 3]; 3])
        });
        let mut cov = [[0.0; 3]; 3];
        let mut residual = 0.0;
        for (s, &yes) in samples.iter().zip(&keep) {
            if yes {
                let var = sum_variance(s.value, s.variance);
                let v = mv(inv, s.x);
                for i in 0..3 {
                    for j in 0..3 {
                        cov[i][j] += var * v[i] * v[j];
                    }
                }
                residual += ((s.value - dot(s.x, coeff)) / s.variance.sqrt()).powi(2);
            }
        }
        let rms = (residual / (n.saturating_sub(3).max(1)) as f64).sqrt();
        if n * 4 < samples.len() * 3 || rms > 3.0 {
            flags |= MODEL;
        }
        let (mut count, mut sum, mut variance, mut maximum) =
            (0_usize, 0.0, 0.0, f64::NEG_INFINITY);
        let (mut q, mut eq) = ([0.0; 3], [0.0; 3]);
        let (mut edge, mut edge_var) = (0.0, 0.0);
        let (mut background_variance, mut edge_background) = (0.0, 0.0);
        let (mut weight, mut wx, mut wy, mut wxx, mut wyy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for y in b.y0..b.y1 {
            for x in b.x0..b.x1 {
                let i = y * image.width + x;
                if signal[local(x, y)] > 1 {
                    flags |= OVERLAP;
                }
                if image.mask[i] {
                    flags |= MASKED;
                    continue;
                }
                if image.saturated(i) {
                    flags |= SATURATED;
                }
                let p = basis(x, y);
                let d = image.values[i];
                let v = sum_variance(d, image.pixel_variance(i));
                let expected = dot(p, coeff).max(0.0) / image.gain + read;
                count += 1;
                sum += d;
                variance += v;
                background_variance += expected;
                maximum = maximum.max(d);
                for j in 0..3 {
                    q[j] += p[j];
                }
                let w = (d - dot(p, coeff)).max(0.0);
                weight += w;
                wx += w * x as f64;
                wy += w * y as f64;
                wxx += w * (x as f64).powi(2);
                wyy += w * (y as f64).powi(2);
                if x == b.x0 || x + 1 == b.x1 || y == b.y0 || y + 1 == b.y1 {
                    edge += d;
                    edge_var += v;
                    edge_background += expected;
                    for j in 0..3 {
                        eq[j] += p[j];
                    }
                }
            }
        }
        if counts && count > 0 {
            // NaN backgrounds (no plane) leave the counts' variance.
            variance = variance.max(background_variance).max(1.0);
            edge_var = edge_var.max(edge_background);
        }
        let intensity = sum - dot(q, coeff);
        let bg_var = if model.is_some() {
            dot(q, mv(cov, q)).max(0.0)
        } else {
            f64::NAN
        };
        let sigma = (variance + bg_var).sqrt();
        let edge_intensity = edge - dot(eq, coeff);
        let edge_sigma = (edge_var + dot(eq, mv(cov, eq)).max(0.0)).sqrt();
        if intensity > 3.0 * sigma
            && edge_intensity > 3.0 * edge_sigma
            && edge_intensity > 0.02 * intensity
        {
            flags |= BOUNDARY;
        }
        diag[..3].copy_from_slice(&coeff);
        diag[3..6].copy_from_slice(&q.map(|v| -v));
        diag[8] = rms;
        diag[9] = edge_intensity;
        diag[10] = edge_sigma;
        diag[11] = if intensity > 0.0 {
            edge_intensity / intensity
        } else {
            f64::NAN
        };
        let mx = if weight > 0.0 { wx / weight } else { cx };
        let my = if weight > 0.0 { wy / weight } else { cy };
        let values = [
            mx,
            my,
            if count > 0 { intensity } else { f64::NAN },
            if count > 0 { sigma } else { f64::NAN },
            if count > 0 {
                dot(q, coeff) / count as f64
            } else {
                f64::NAN
            },
            if count > 0 {
                bg_var.sqrt() / count as f64
            } else {
                f64::NAN
            },
            if count > 0 { maximum } else { f64::NAN },
            if weight > 0.0 {
                (wxx / weight - mx * mx).max(0.0).sqrt()
            } else {
                f64::NAN
            },
            if weight > 0.0 {
                (wyy / weight - my * my).max(0.0).sqrt()
            } else {
                f64::NAN
            },
            count as f64,
            n as f64,
        ];
        if !sum.is_finite()
            || !variance.is_finite()
            || (model.is_some() && count > 0 && (!intensity.is_finite() || !sigma.is_finite()))
        {
            return Err(invalid("overflow in guarded plane extraction"));
        }
        out.push(Spot {
            bounds: b,
            values,
            flags,
        });
        diagnostics.push(diag);
        covariances.push(std::array::from_fn(|i| cov[i / 3][i % 3]));
    }
    Ok((out, diagnostics, covariances))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The former full-image accumulation, kept as the reference.
    struct DenseCoverage {
        signal: Vec<u32>,
        guard: Vec<u32>,
        width: usize,
    }
    impl DenseCoverage {
        fn new(boxes: &[Bounds], gap: usize, width: usize, height: usize) -> Self {
            let mut signal = vec![0_u32; width * height];
            let mut guard = vec![0_u32; width * height];
            for &b in boxes {
                for y in b.y0..b.y1 {
                    for x in b.x0..b.x1 {
                        signal[y * width + x] += 1;
                    }
                }
                let g = b.expanded(gap, width, height);
                for y in g.y0..g.y1 {
                    for x in g.x0..g.x1 {
                        guard[y * width + x] += 1;
                    }
                }
            }
            Self {
                signal,
                guard,
                width,
            }
        }
    }
    impl Coverage for DenseCoverage {
        fn counts(&self, r: Bounds) -> (Vec<u32>, Vec<u32>) {
            let mut s = Vec::new();
            let mut g = Vec::new();
            for y in r.y0..r.y1 {
                for x in r.x0..r.x1 {
                    s.push(self.signal[y * self.width + x]);
                    g.push(self.guard[y * self.width + x]);
                }
            }
            (s, g)
        }
    }

    #[test]
    fn box_local_coverage_reproduces_the_full_image_accumulation_bit_for_bit() {
        let (width, height) = (157, 101);
        let mut state = 12_345_u64;
        let mut next = |n: usize| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(clippy::cast_possible_truncation)]
            let v = (state >> 33) as usize % n;
            v
        };
        let mut values = vec![0.; width * height];
        let mut mask = vec![false; width * height];
        for i in 0..values.len() {
            #[allow(clippy::cast_precision_loss)]
            let v = 20. + (next(1000) as f64) * 0.01 + (i % width) as f64 * 0.05;
            values[i] = v;
            mask[i] = next(97) == 0;
        }
        for _ in 0..40 {
            let (x, y) = (next(width), next(height));
            for dy in 0..3 {
                for dx in 0..3 {
                    if x + dx < width && y + dy < height {
                        values[(y + dy) * width + x + dx] += 400.;
                    }
                }
            }
        }
        let image = Image {
            values: &values,
            height,
            width,
            mask: &mask,
            variance: None,
            gain: 1.,
            read_noise: 0.,
            saturation: Some(1e9),
        };
        for gap in [0, 2, 3] {
            // Clustered, overlapping and edge boxes of several sizes.
            let boxes: Vec<Bounds> = (0..120)
                .map(|i| {
                    let s = 3 + next(9);
                    let (x, y) = if i % 3 == 0 {
                        (60 + next(20), 40 + next(15))
                    } else {
                        (next(width - 1), next(height - 1))
                    };
                    Bounds {
                        x0: x,
                        y0: y,
                        x1: (x + s).min(width),
                        y1: (y + s).min(height),
                    }
                })
                .collect();
            let dense = DenseCoverage::new(&boxes, gap, width, height);
            let sparse = SparseCoverage::new(&boxes, gap, width, height);
            let a = extract_boxes(&image, &boxes, gap, 5, 20, 4., &dense, SumVariance::Floored)
                .unwrap();
            let b = extract_boxes(
                &image,
                &boxes,
                gap,
                5,
                20,
                4.,
                &sparse,
                SumVariance::Floored,
            )
            .unwrap();
            let bits = |r: &(Vec<Spot>, Vec<[f64; 12]>, Vec<[f64; 9]>)| {
                r.0.iter()
                    .flat_map(|s| {
                        s.values
                            .iter()
                            .map(|v| v.to_bits())
                            .chain([u64::from(s.flags)])
                    })
                    .chain(r.1.iter().flatten().map(|v| v.to_bits()))
                    .chain(r.2.iter().flatten().map(|v| v.to_bits()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(bits(&a), bits(&b));
            assert!(a.0.iter().any(|s| s.flags & OVERLAP != 0));
            let full = extract_full(&image, &boxes, gap, 5, 20, 4.).unwrap();
            assert_eq!(bits(&a), bits(&full));
        }
    }
}
