//! Independent finite-voxel 3D profile learning and signed GLS intensity fitting.
use crate::{Bounds, Image, Spot, SpotError, invalid, local_background};
use std::collections::HashMap;

/// Less than the required finite learned-profile mass was observed.
pub const PARTIAL: u32 = 8192;
/// A conditional covariance solve failed.
pub const FAILED: u32 = 16384;
/// No identifiable profile amplitude from the available observations.
pub const NO_PROFILE: u32 = 32768;
/// Strong fitted signal has an excessive conditional residual chi-square.
pub const MISMATCH: u32 = 131_072;
type Matrix = [[f64; 3]; 3];

#[derive(Clone, Copy)]
/// Odd finite voxel dimensions in x, y, omega order.
pub struct Grid {
    /// Dimensions; storage index is (omega * ny + y) * nx + x.
    pub shape: [usize; 3],
    /// Voxel widths in detector pixels, detector pixels and degrees.
    pub step: [f64; 3],
}
impl Grid {
    /// Number of finite voxels, saturating before validation.
    #[must_use]
    pub fn size(self) -> usize {
        self.shape.into_iter().fold(1_usize, usize::saturating_mul)
    }
    /// Validate bounded, positive finite grid dimensions and widths.
    /// # Errors
    /// Invalid dimensions, widths or the voxel budget is exceeded.
    pub fn validate(self) -> Result<(), SpotError> {
        if self
            .shape
            .iter()
            .any(|&v| !(3..=65).contains(&v) || v % 2 == 0)
            || self.step.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || self.size() > 300_000
        {
            return Err(invalid("invalid finite profile grid"));
        }
        Ok(())
    }
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn mv(a: Matrix, b: [f64; 3]) -> [f64; 3] {
    a.map(|r| dot(r, b))
}
pub(super) fn inverse(a: Matrix) -> Option<Matrix> {
    let mut r = [[0.; 6]; 3];
    for i in 0..3 {
        r[i][..3].copy_from_slice(&a[i]);
        r[i][i + 3] = 1.;
    }
    for c in 0..3 {
        let p = (c..3).max_by(|&i, &j| r[i][c].abs().total_cmp(&r[j][c].abs()))?;
        r.swap(c, p);
        let d = r[c][c];
        if !d.is_finite() || d.abs() < 1e-20 {
            return None;
        }
        for v in &mut r[c] {
            *v /= d;
        }
        let pivot = r[c];
        for (i, row) in r.iter_mut().enumerate() {
            if i != c {
                let f = row[c];
                for (v, p) in row.iter_mut().zip(pivot) {
                    *v -= f * p;
                }
            }
        }
    }
    Some(std::array::from_fn(|i| {
        std::array::from_fn(|j| r[i][j + 3])
    }))
}

/// Visit exact box/voxel overlap fractions and analytical centre derivatives.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Grid ranges are clipped before conversion.
pub(super) fn weights(
    grid: Grid,
    b: [f64; 6],
    centre: [f64; 3],
    mut visit: impl FnMut(usize, f64, [f64; 3]),
) {
    let origin =
        std::array::from_fn::<_, 3, _>(|i| centre[i] - grid.shape[i] as f64 * grid.step[i] / 2.);
    let mut ranges = [(0, 0); 3];
    for i in 0..3 {
        if b[2 * i + 1] <= origin[i] || b[2 * i] >= origin[i] + grid.shape[i] as f64 * grid.step[i]
        {
            return;
        }
        let lo = ((b[2 * i] - origin[i]) / grid.step[i]).floor().max(0.) as usize;
        let hi = (((b[2 * i + 1] - origin[i]) / grid.step[i]).floor().max(0.) as usize + 1)
            .min(grid.shape[i]);
        ranges[i] = (lo.min(grid.shape[i]), hi);
    }
    let axis = |i: usize, j: usize| {
        let lo = origin[i] + j as f64 * grid.step[i];
        let hi = lo + grid.step[i];
        let length = b[2 * i + 1].min(hi) - b[2 * i].max(lo);
        if length <= 0. {
            (0., 0.)
        } else {
            (
                length / grid.step[i],
                (f64::from(hi < b[2 * i + 1]) - f64::from(lo > b[2 * i])) / grid.step[i],
            )
        }
    };
    for z in ranges[2].0..ranges[2].1 {
        let (wz, dz) = axis(2, z);
        for y in ranges[1].0..ranges[1].1 {
            let (wy, dy) = axis(1, y);
            for x in ranges[0].0..ranges[0].1 {
                let (wx, dx) = axis(0, x);
                let w = wx * wy * wz;
                if w > 0. {
                    visit(
                        (z * grid.shape[1] + y) * grid.shape[0] + x,
                        w,
                        [dx * wy * wz, wx * dy * wz, wx * wy * dz],
                    );
                }
            }
        }
    }
}

fn transformed_weights(
    grid: Grid,
    b: [f64; 6],
    centre: [f64; 3],
    scale: [f64; 3],
    mut visit: impl FnMut(usize, f64, [f64; 3]),
) {
    let mut bounds = [0.; 6];
    for j in 0..3 {
        let a = (b[2 * j] - centre[j]) * scale[j];
        let c = (b[2 * j + 1] - centre[j]) * scale[j];
        bounds[2 * j] = a.min(c);
        bounds[2 * j + 1] = a.max(c);
    }
    weights(grid, bounds, [0.; 3], |i, w, d| {
        visit(i, w, std::array::from_fn(|j| d[j] * scale[j]));
    });
}

/// Finite profile mass and three centre partials, computed in one voxel pass.
/// # Errors
/// Invalid mass grid, sample intervals, centre or work budget.
pub fn evaluate(
    profile: &[f64],
    grid: Grid,
    bounds: &[[f64; 6]],
    centre: [f64; 3],
) -> Result<Vec<[f64; 4]>, SpotError> {
    validate_evaluation(profile, grid, bounds, centre)?;
    Ok(bounds
        .iter()
        .map(|&b| {
            let mut v = [0.; 4];
            weights(grid, b, centre, |i, w, d| {
                v[0] += profile[i] * w;
                for j in 0..3 {
                    v[j + 1] += profile[i] * d[j];
                }
            });
            v
        })
        .collect())
}

/// Raw sample rows are signed signal, independent variance and fitted background.
pub struct Observations<'a> {
    /// Signed signal, independent pixel variance and fitted background.
    pub samples: &'a [[f64; 3]],
    /// Samples actually available for a fit; no significance threshold.
    pub valid: &'a [bool],
    /// Row-major sample offsets for each reflection/exposure group plus sentinel.
    pub offsets: &'a [usize],
    /// Half-open detector-pixel signal rectangles, one per group.
    pub boxes: &'a [Bounds],
    /// Reflection IDs of each group.
    pub ids: &'a [usize],
    /// Increasing full exposure bounds, in degrees.
    pub angles: &'a [[f64; 2]],
    /// Row-major 3-by-3 background-plane covariance for each group.
    pub covariance: &'a [[f64; 9]],
    /// Frozen predicted or independently measured x/y/omega per reflection.
    pub centres: &'a [[f64; 3]],
    /// Signed laboratory-to-local coordinate scales per reflection (e.g. zeta).
    pub scales: &'a [[f64; 3]],
    /// Row-major local-plane Jacobians per reflection, applied before scales.
    pub planar: &'a [[f64; 4]],
    /// Explicit measurement-quality bits per group.
    pub flags: &'a [u32],
}
impl Observations<'_> {
    /// Check dimensions, intervals, valid samples and bounded work.
    /// # Errors
    /// Inconsistent arrays, nonfinite valid data or work-budget violations.
    pub fn validate(&self) -> Result<(), SpotError> {
        let n = self.ids.len();
        if self.centres.len() > 1_000_000
            || self.planar.len() != self.centres.len()
            || self
                .planar
                .iter()
                .any(|&a| !crate::profiles_affine::valid(a))
            || self.scales.len() != self.centres.len()
            || self
                .scales
                .iter()
                .flatten()
                .any(|v| !v.is_finite() || v.abs() < 1e-8)
            || self.samples.len() > 20_000_000
            || self.valid.len() != self.samples.len()
            || self.offsets.len() != n + 1
            || self.boxes.len() != n
            || self.angles.len() != n
            || self.covariance.len() != n
            || self.flags.len() != n
            || self.offsets.first() != Some(&0)
            || self.offsets.last() != Some(&self.samples.len())
            || self.offsets.windows(2).any(|w| w[0] > w[1])
            || self.ids.iter().any(|&i| i >= self.centres.len())
            || self.centres.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(invalid("invalid profile observation shapes or budget"));
        }
        for g in 0..n {
            let b = self.boxes[g];
            let a = self.angles[g];
            if b.x0 >= b.x1
                || b.y0 >= b.y1
                || b.x1 > u32::MAX as usize
                || b.y1 > u32::MAX as usize
                || self.offsets[g + 1] - self.offsets[g] != (b.x1 - b.x0) * (b.y1 - b.y0)
                || !a[0].is_finite()
                || !a[1].is_finite()
                || a[1] <= a[0]
            {
                return Err(invalid("invalid profile shoebox"));
            }
            if self.valid[self.offsets[g]..self.offsets[g + 1]]
                .iter()
                .any(|&v| v)
                && self.covariance[g].iter().any(|v| !v.is_finite())
            {
                return Err(invalid("nonfinite valid background covariance"));
            }
        }
        for (s, &ok) in self.samples.iter().zip(self.valid) {
            if ok && (!s.iter().all(|v| v.is_finite()) || s[1] <= 0.) {
                return Err(invalid("invalid profile samples"));
            }
        }
        Ok(())
    }
    fn matrix(&self, id: usize) -> [f64; 4] {
        let a = self.planar[id];
        [
            a[0] * self.scales[id][0],
            a[1] * self.scales[id][0],
            a[2] * self.scales[id][1],
            a[3] * self.scales[id][1],
        ]
    }
    pub(super) fn shifted(&self, id: usize, shift: [f64; 3]) -> [f64; 3] {
        let a = self.matrix(id);
        let det = a[0] * a[3] - a[1] * a[2];
        [
            self.centres[id][0] + (a[3] * shift[0] - a[1] * shift[1]) / det,
            self.centres[id][1] + (-a[2] * shift[0] + a[0] * shift[1]) / det,
            self.centres[id][2] + shift[2] / self.scales[id][2],
        ]
    }
    fn local_shift(&self, id: usize, delta: [f64; 3]) -> [f64; 3] {
        let a = self.matrix(id);
        [
            a[0] * delta[0] + a[1] * delta[1],
            a[2] * delta[0] + a[3] * delta[1],
            delta[2] * self.scales[id][2],
        ]
    }
    #[allow(clippy::float_cmp)] // Exact identity selects the separable operator, without changing near-identity geometry.
    pub(super) fn weights(
        &self,
        grid: Grid,
        b: [f64; 6],
        centre: [f64; 3],
        id: usize,
        visit: impl FnMut(usize, f64, [f64; 3]),
    ) {
        if self.planar[id] == [1., 0., 0., 1.] {
            transformed_weights(grid, b, centre, self.scales[id], visit);
        } else {
            crate::profiles_affine::weights(
                grid,
                b,
                centre,
                self.matrix(id),
                self.scales[id][2],
                visit,
            );
        }
    }
    pub(super) fn bounds(&self, g: usize, i: usize) -> [f64; 6] {
        let b = self.boxes[g];
        let j = i - self.offsets[g];
        let x = (b.x0 + j % (b.x1 - b.x0)) as f64;
        let y = (b.y0 + j / (b.x1 - b.x0)) as f64;
        [
            x - 0.5,
            x + 0.5,
            y - 0.5,
            y + 0.5,
            self.angles[g][0],
            self.angles[g][1],
        ]
    }
    pub(super) fn basis(&self, g: usize, i: usize) -> [f64; 3] {
        let b = self.boxes[g];
        let v = self.bounds(g, i);
        [
            1.,
            (v[0].midpoint(v[1]) - (b.x0 + b.x1 - 1) as f64 / 2.)
                / ((b.x1 - b.x0) as f64 / 2.).max(1.),
            (v[2].midpoint(v[3]) - (b.y0 + b.y1 - 1) as f64 / 2.)
                / ((b.y1 - b.y0) as f64 / 2.).max(1.),
        ]
    }
}

/// Streaming image-to-shoebox extraction; covariance and masks remain explicit.
/// # Errors
/// Invalid image/background inputs or the sample budget is exceeded.
#[allow(clippy::type_complexity)]
pub fn extract_frame(
    image: &Image<'_>,
    boxes: &[Bounds],
    gap: usize,
    margin: usize,
    minimum: usize,
    clip: f64,
) -> Result<
    (
        Vec<Spot>,
        Vec<[f64; 12]>,
        Vec<[f64; 9]>,
        Vec<[f64; 3]>,
        Vec<bool>,
    ),
    SpotError,
> {
    let (spots, diagnostics, covariance) =
        local_background::extract_full(image, boxes, gap, margin, minimum, clip)?;
    let size: usize = boxes.iter().map(|b| (b.x1 - b.x0) * (b.y1 - b.y0)).sum();
    if size > 20_000_000 {
        return Err(invalid("profile frame sample budget exceeded"));
    }
    let mut samples = Vec::with_capacity(size);
    let mut valid = Vec::with_capacity(size);
    for (b, d) in boxes.iter().zip(&diagnostics) {
        for y in b.y0..b.y1 {
            for x in b.x0..b.x1 {
                let i = y * image.width + x;
                let basis = [
                    1.,
                    (x as f64 - (b.x0 + b.x1 - 1) as f64 / 2.)
                        / ((b.x1 - b.x0) as f64 / 2.).max(1.),
                    (y as f64 - (b.y0 + b.y1 - 1) as f64 / 2.)
                        / ((b.y1 - b.y0) as f64 / 2.).max(1.),
                ];
                let bg = dot(basis, [d[0], d[1], d[2]]);
                let ok = !image.mask[i] && !image.saturated(i) && bg.is_finite();
                samples.push(if ok {
                    [image.values[i] - bg, image.pixel_variance(i), bg]
                } else {
                    [0., 1., 0.]
                });
                valid.push(ok);
            }
        }
    }
    Ok((spots, diagnostics, covariance, samples, valid))
}

/// Performance controls only; scientific equations and traversal order are unchanged.
#[derive(Clone, Copy)]
pub struct Execution {
    /// Requested native worker count, in 1..4.
    pub workers: usize,
    /// Additional geometry-cache byte cap, 0 disables caching, at most 1 GiB.
    pub geometry_cache_bytes: usize,
}
impl Default for Execution {
    fn default() -> Self {
        Self {
            workers: 1,
            geometry_cache_bytes: 64 * 1024 * 1024,
        }
    }
}
impl Execution {
    /// Validate explicit resource controls.
    /// # Errors
    /// Invalid worker count or geometry-cache cap.
    pub fn validate(self) -> Result<(), SpotError> {
        if !(1..=4).contains(&self.workers) || self.geometry_cache_bytes > 1024 * 1024 * 1024 {
            return Err(invalid("invalid profile execution workers/cache budget"));
        }
        Ok(())
    }
}
/// Actual admitted learning cache; independent of the learned numerical arrays.
#[derive(Clone, Copy, Default)]
pub struct LearningExecution {
    /// Actual native workers (serial cache implementation: one).
    pub workers: usize,
    /// Caller-declared additional-cache cap.
    pub cache_budget_bytes: usize,
    /// Actual admitted vector-capacity bytes, zero on streaming fallback.
    pub cache_bytes: usize,
    /// Cached sample rows.
    pub cache_rows: usize,
    /// Cached nonzero voxel intersections.
    pub cache_entries: usize,
}

/// Finite local/global profile bank and auditable reference membership.
pub struct Learned {
    /// Contiguous model-major finite voxel masses, each nonempty model sums to one.
    pub profiles: Vec<f64>,
    /// Reference counts, including cells too small to learn a local profile.
    pub counts: Vec<usize>,
    /// Median reference-centroid offset for each model, in local grid units.
    pub shifts: Vec<[f64; 3]>,
    /// Reflections actually used for profile training.
    pub used: Vec<bool>,
    /// Independently measured spatial reference centres, angular predictions retained.
    pub reference_centres: Vec<[f64; 3]>,
    /// Actual numerical execution resources, not part of the profile model.
    pub execution: LearningExecution,
}
fn median(mut a: Vec<f64>) -> f64 {
    a.sort_by(f64::total_cmp);
    if a.len() % 2 == 0 {
        a[a.len() / 2 - 1].midpoint(a[a.len() / 2])
    } else {
        a[a.len() / 2]
    }
}

/// Train finite local voxel masses using a nonnegative diagonally majorized solve.
/// # Errors
/// Invalid observations, regions, parameters or a missing usable global profile.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn learn(
    o: &Observations<'_>,
    regions: &[usize],
    references: &[bool],
    grid: Grid,
    models: usize,
    minimum: usize,
    snr: f64,
    cut: f64,
    iterations: usize,
    variance_weighted: bool,
) -> Result<Learned, SpotError> {
    learn_with_execution(
        o,
        regions,
        references,
        grid,
        models,
        minimum,
        snr,
        cut,
        iterations,
        variance_weighted,
        Execution::default(),
    )
}

/// Same finite-profile equations with bounded, immutable geometry reuse.
/// # Errors
/// Rejects the same learning inputs and invalid execution controls.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn learn_with_execution(
    o: &Observations<'_>,
    regions: &[usize],
    references: &[bool],
    grid: Grid,
    models: usize,
    minimum: usize,
    snr: f64,
    cut: f64,
    iterations: usize,
    variance_weighted: bool,
    execution: Execution,
) -> Result<Learned, SpotError> {
    execution.validate()?;
    o.validate()?;
    grid.validate()?;
    let n = o.centres.len();
    let k = grid.size();
    if regions.len() != n
        || references.len() != n
        || !(2..=65).contains(&models)
        || regions.iter().any(|&v| v >= models - 1)
        || minimum < 2
        || !snr.is_finite()
        || snr < 3.
        || !cut.is_finite()
        || !(0.0..0.2).contains(&cut)
        || !(1..=100).contains(&iterations)
    {
        return Err(invalid("invalid profile training parameters"));
    }
    let mut total = vec![0.; n];
    let mut variance = vec![0.; n];
    let mut moment = vec![[0.; 3]; n];
    let mut bad = vec![false; n];
    let mut ends = vec![[0.; 2]; n];
    let mut groups = vec![0; n];
    for g in 0..o.ids.len() {
        let id = o.ids[g];
        bad[id] |= o.flags[g] != 0;
        let mut q = [0.; 3];
        let mut sum = 0.;
        for i in o.offsets[g]..o.offsets[g + 1] {
            if o.valid[i] {
                let d = o.samples[i][0];
                let b = o.bounds(g, i);
                sum += d;
                total[id] += d;
                variance[id] += o.samples[i][1];
                for j in 0..3 {
                    moment[id][j] += d * (b[2 * j] + b[2 * j + 1]) / 2.;
                    q[j] += o.basis(g, i)[j];
                }
            } else {
                bad[id] = true;
            }
        }
        let cov = std::array::from_fn(|i| std::array::from_fn(|j| o.covariance[g][i * 3 + j]));
        variance[id] += dot(q, mv(cov, q));
        if groups[id] == 0 {
            ends[id][0] = sum;
        }
        ends[id][1] = sum;
        groups[id] += 1;
    }
    let mut used = vec![false; n];
    let mut centres = o.centres.to_vec();
    let mut counts = vec![0; models];
    let mut offsets = vec![Vec::<[f64; 3]>::new(); models];
    for id in 0..n {
        if references[id]
            && !bad[id]
            && groups[id] >= 3
            && total[id] > snr * variance[id].sqrt()
            && ends[id].iter().all(|v| *v < 0.05 * total[id])
        {
            centres[id] = moment[id].map(|v| v / total[id]);
            let measured_shift =
                std::array::from_fn::<_, 3, _>(|j| centres[id][j] - o.centres[id][j]);
            // A coarse-exposure angular centroid is not an estimate of the
            // Bragg maximum. Preserve the geometric angle during training.
            centres[id][2] = o.centres[id][2];
            let shift = o.local_shift(
                id,
                std::array::from_fn(|j| centres[id][j] - o.centres[id][j]),
            );
            if measured_shift[0].abs() <= 3.
                && measured_shift[1].abs() <= 3.
                && measured_shift[2].abs() <= 1.5
            {
                used[id] = true;
                for m in [0, regions[id] + 1] {
                    counts[m] += 1;
                    offsets[m].push(shift);
                }
            }
        }
    }
    if counts[0] < minimum {
        return Err(invalid("too few isolated strong training profiles"));
    }
    let mut reference_weights = vec![1.; n];
    if variance_weighted {
        for id in 0..n {
            reference_weights[id] = if used[id] && variance[id] > 0. {
                total[id] / variance[id].sqrt()
            } else {
                0.
            };
        }
        let max = reference_weights.iter().copied().fold(0., f64::max);
        for w in &mut reference_weights {
            *w = (*w / max).powi(2);
        }
    }
    let shifts = offsets
        .iter()
        .map(|v| {
            if v.is_empty() {
                [0.; 3]
            } else {
                std::array::from_fn(|j| median(v.iter().map(|x| x[j]).collect()))
            }
        })
        .collect();
    let mut profile = vec![0.; models * k];
    let mut denominator = vec![0.; models * k];
    let cache = crate::profiles_cache::LearningGeometry::new(
        o,
        grid,
        &centres,
        &used,
        execution.geometry_cache_bytes,
    );
    let execution_stats = LearningExecution {
        workers: 1,
        cache_budget_bytes: execution.geometry_cache_bytes,
        cache_bytes: cache
            .as_ref()
            .map_or(0, crate::profiles_cache::LearningGeometry::bytes),
        cache_rows: cache
            .as_ref()
            .map_or(0, crate::profiles_cache::LearningGeometry::rows),
        cache_entries: cache
            .as_ref()
            .map_or(0, crate::profiles_cache::LearningGeometry::entries),
    };
    if let Some(cache) = &cache {
        cache.initialize(
            o,
            regions,
            &total,
            &reference_weights,
            k,
            &mut profile,
            &mut denominator,
        );
    } else {
        for g in 0..o.ids.len() {
            let id = o.ids[g];
            if !used[id] {
                continue;
            }
            for i in o.offsets[g]..o.offsets[g + 1] {
                let value = o.samples[i][0] / total[id];
                let mut row = 0.;
                o.weights(grid, o.bounds(g, i), centres[id], id, |_, w, _| row += w);
                o.weights(grid, o.bounds(g, i), centres[id], id, |j, w, _| {
                    for m in [0, regions[id] + 1] {
                        profile[m * k + j] += reference_weights[id] * w * value;
                        denominator[m * k + j] += reference_weights[id] * w * row;
                    }
                });
            }
        }
    }
    for (p, &d) in profile.iter_mut().zip(&denominator) {
        *p = if d > 0. { (*p / d).max(0.) } else { 0. };
    }
    for _ in 0..iterations {
        let mut gradient = vec![0.; models * k];
        if let Some(cache) = &cache {
            cache.gradient(
                o,
                regions,
                &total,
                &reference_weights,
                k,
                &profile,
                &mut gradient,
            );
        } else {
            for g in 0..o.ids.len() {
                let id = o.ids[g];
                if !used[id] {
                    continue;
                }
                for i in o.offsets[g]..o.offsets[g + 1] {
                    let b = o.bounds(g, i);
                    let value = o.samples[i][0] / total[id];
                    let mut expected = [0.; 2];
                    o.weights(grid, b, centres[id], id, |j, w, _| {
                        expected[0] += w * profile[j];
                        expected[1] += w * profile[(regions[id] + 1) * k + j];
                    });
                    o.weights(grid, b, centres[id], id, |j, w, _| {
                        for (slot, m) in [0, regions[id] + 1].into_iter().enumerate() {
                            gradient[m * k + j] +=
                                reference_weights[id] * w * (value - expected[slot]);
                        }
                    });
                }
            }
        }
        for ((p, &d), &g) in profile.iter_mut().zip(&denominator).zip(&gradient) {
            if d > 0. {
                *p = (*p + 0.9 * g / d).max(0.);
            }
        }
    }
    // Keep the central connected learned component, explicitly discarding noisy islands.
    for (m, p) in profile.chunks_mut(k).enumerate() {
        if counts[m] < minimum {
            p.fill(0.);
            continue;
        }
        let peak = p.iter().copied().fold(0., f64::max);
        let threshold = cut * peak;
        let seed = p
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                let x = i % grid.shape[0];
                let y = (i / grid.shape[0]) % grid.shape[1];
                let z = i / (grid.shape[0] * grid.shape[1]);
                x.abs_diff(grid.shape[0] / 2) <= 2
                    && y.abs_diff(grid.shape[1] / 2) <= 2
                    && z.abs_diff(grid.shape[2] / 2) <= 2
            })
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(k / 2, |(i, _)| i);
        let mut keep = vec![false; k];
        let mut stack = vec![seed];
        while let Some(i) = stack.pop() {
            if keep[i] || p[i] < threshold || p[i] <= 0. {
                continue;
            }
            keep[i] = true;
            let x = i % grid.shape[0];
            let y = (i / grid.shape[0]) % grid.shape[1];
            let z = i / (grid.shape[0] * grid.shape[1]);
            for (position, step, limit) in [
                (x, 1, grid.shape[0]),
                (y, grid.shape[0], grid.shape[1]),
                (z, grid.shape[0] * grid.shape[1], grid.shape[2]),
            ] {
                if position > 0 {
                    stack.push(i - step);
                }
                if position + 1 < limit {
                    stack.push(i + step);
                }
            }
        }
        for (v, &yes) in p.iter_mut().zip(&keep) {
            if !yes {
                *v = 0.;
            }
        }
        let mass: f64 = p.iter().sum();
        if mass > 0. {
            for v in p {
                *v /= mass;
            }
        }
    }
    if profile[..k].iter().sum::<f64>() < 0.99 {
        return Err(invalid("empty learned global profile"));
    }
    Ok(Learned {
        profiles: profile,
        counts,
        shifts,
        used,
        reference_centres: centres,
        execution: execution_stats,
    })
}

/// Fit values: I, sigma, observed mass, chi-square/dof, valid samples, correlation,
/// and four conditional objective derivatives (I, centre x/y/omega).
/// # Errors
/// Invalid arrays, grid, assignments, covariance or noise parameters.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn fit(
    o: &Observations<'_>,
    profiles: &[f64],
    grid: Grid,
    assignments: &[usize],
    shifts: &[[f64; 3]],
    minimum_mass: f64,
    expected_noise: bool,
    gain: f64,
    read_noise: f64,
) -> Result<(Vec<[f64; 10]>, Vec<u32>), SpotError> {
    validate_fit(
        o,
        profiles,
        grid,
        assignments,
        shifts,
        minimum_mass,
        gain,
        read_noise,
    )?;
    let n = o.centres.len();
    let k = grid.size();
    let mut probability = vec![0.; o.samples.len()];
    let mut mass = vec![0.; n];
    let mut flags = vec![0; n];
    let mut coverage = HashMap::<(u64, u64, usize, usize), u32>::new();
    for g in 0..o.ids.len() {
        let id = o.ids[g];
        let model = assignments[id];
        let centre = o.shifted(id, shifts[model]);
        for (i, p) in probability
            .iter_mut()
            .enumerate()
            .take(o.offsets[g + 1])
            .skip(o.offsets[g])
        {
            o.weights(grid, o.bounds(g, i), centre, id, |j, w, _| {
                *p += w * profiles[model * k + j];
            });
            if *p > 0. {
                let b = o.boxes[g];
                let local = i - o.offsets[g];
                let xy = (b.x0 + local % (b.x1 - b.x0), b.y0 + local / (b.x1 - b.x0));
                *coverage
                    .entry((
                        o.angles[g][0].to_bits(),
                        o.angles[g][1].to_bits(),
                        xy.0,
                        xy.1,
                    ))
                    .or_insert(0) += 1;
                flags[id] |= o.flags[g] & (crate::BACKGROUND | local_background::MODEL);
                if !o.valid[i] {
                    flags[id] |= o.flags[g] & (crate::MASKED | crate::SATURATED);
                }
            }
            if o.valid[i] {
                mass[id] += *p;
            }
        }
    }
    // The generous collection rectangles are not the learned foreground.
    // Integer occupancy counts affect only flags, never floating-point order.
    for g in 0..o.ids.len() {
        for (i, &p) in probability
            .iter()
            .enumerate()
            .take(o.offsets[g + 1])
            .skip(o.offsets[g])
        {
            if p <= 0. {
                continue;
            }
            let b = o.boxes[g];
            let local = i - o.offsets[g];
            let xy = (b.x0 + local % (b.x1 - b.x0), b.y0 + local / (b.x1 - b.x0));
            if coverage[&(
                o.angles[g][0].to_bits(),
                o.angles[g][1].to_bits(),
                xy.0,
                xy.1,
            )] > 1
            {
                flags[o.ids[g]] |= crate::OVERLAP;
            }
        }
        if mass[o.ids[g]] < 0.999 {
            flags[o.ids[g]] |= o.flags[g] & (crate::EDGE | 256);
        }
    }
    let mut values = vec![[f64::NAN; 10]; n];
    let mut intensity = vec![0.0_f64; n];
    for iteration in 0..if expected_noise { 8 } else { 1 } {
        let mut num = vec![0.; n];
        let mut den = vec![0.; n];
        let mut chi = vec![0.; n];
        let mut counts = vec![0_usize; n];
        let mut moments = vec![[0.; 5]; n];
        let mut derivatives = vec![[0.; 4]; n];
        let mut shape_raw = vec![[0.; 3]; n];
        let mut shape_normal = vec![[0.; 3]; n];
        for g in 0..o.ids.len() {
            let id = o.ids[g];
            if !(o.offsets[g]..o.offsets[g + 1]).any(|i| o.valid[i] && probability[i] > 0.) {
                continue;
            }
            let mut b_inverse =
                std::array::from_fn(|i| std::array::from_fn(|j| o.covariance[g][i * 3 + j]));
            // A positive plane covariance is guaranteed by the collector; zero is a valid no-background fixture.
            let zero = b_inverse.iter().flatten().all(|v| v.abs() < 1e-25);
            if !zero {
                let Some(inv) = inverse(b_inverse) else {
                    flags[id] |= FAILED;
                    continue;
                };
                b_inverse = inv;
            }
            let (mut pu, mut du) = ([0.; 3], [0.; 3]);
            let (mut pp, mut pd, mut dd) = (0., 0., 0.);
            let mut shape_u = [[0.; 3]; 3];
            let mut shape_score = [0.; 3];
            let mut shape_p = [0.; 3];
            let model = assignments[id];
            let centre = o.shifted(id, shifts[model]);
            for (i, &p) in probability
                .iter()
                .enumerate()
                .take(o.offsets[g + 1])
                .skip(o.offsets[g])
            {
                if !o.valid[i] || p <= 0. {
                    continue;
                }
                let s = o.samples[i];
                let x = o.basis(g, i);
                let variance = if expected_noise && iteration > 0 {
                    ((s[2] + intensity[id].max(0.) * p).max(0.) / gain + read_noise.powi(2)).max(1.)
                } else {
                    s[1]
                };
                let w = 1. / variance;
                pp += w * p * p;
                pd += w * p * s[0];
                dd += w * s[0] * s[0];
                counts[id] += 1;
                for a in 0..3 {
                    pu[a] += w * p * x[a];
                    du[a] += w * s[0] * x[a];
                    for b in 0..3 {
                        if !zero {
                            b_inverse[a][b] += w * x[a] * x[b];
                        }
                    }
                }
                let residual = s[0] - intensity[id] * p;
                o.weights(grid, o.bounds(g, i), centre, id, |j, _, dp| {
                    for a in 0..3 {
                        let v = profiles[model * k + j] * dp[a];
                        shape_score[a] += w * v * residual;
                        shape_p[a] += w * v * p;
                        for b in 0..3 {
                            shape_u[a][b] += w * v * x[b];
                        }
                    }
                });
                let mm = &mut moments[id];
                mm[0] += p;
                mm[1] += s[0];
                mm[2] += p * p;
                mm[3] += s[0] * s[0];
                mm[4] += p * s[0];
            }
            let inv = if zero {
                [[0.; 3]; 3]
            } else {
                let Some(inv) = inverse(b_inverse) else {
                    flags[id] |= FAILED;
                    continue;
                };
                inv
            };
            let gp = mv(inv, pu);
            let gd = mv(inv, du);
            num[id] += pd - dot(pu, gd);
            den[id] += pp - dot(pu, gp);
            chi[id] += dd - 2. * intensity[id] * pd + intensity[id].powi(2) * pp
                - dot(
                    std::array::from_fn(|j| du[j] - intensity[id] * pu[j]),
                    mv(inv, std::array::from_fn(|j| du[j] - intensity[id] * pu[j])),
                );
            for a in 0..3 {
                shape_raw[id][a] +=
                    shape_score[a] + intensity[id] * shape_p[a] - dot(shape_u[a], gd);
                shape_normal[id][a] += shape_p[a] - dot(shape_u[a], gp);
            }
        }
        for id in 0..n {
            if den[id] <= 1e-30 || !den[id].is_finite() {
                flags[id] |= NO_PROFILE;
                continue;
            }
            let old = intensity[id];
            intensity[id] = num[id] / den[id];
            let sigma = (1. / den[id]).sqrt();
            let dof = counts[id].saturating_sub(1).max(1) as f64;
            let cost = (chi[id]
                + 2. * (old - intensity[id]) * (num[id] - old * den[id])
                + (intensity[id] - old).powi(2) * den[id])
                .max(0.);
            let m = moments[id];
            let count = counts[id] as f64;
            let correlation = (m[4] - m[0] * m[1] / count)
                / ((m[2] - m[0] * m[0] / count).max(0.) * (m[3] - m[1] * m[1] / count).max(0.))
                    .sqrt();
            derivatives[id][0] = 2. * (intensity[id] * den[id] - num[id]);
            for a in 0..3 {
                derivatives[id][a + 1] =
                    -2. * intensity[id] * (shape_raw[id][a] - intensity[id] * shape_normal[id][a]);
            }
            values[id] = [
                intensity[id],
                sigma,
                mass[id],
                cost / dof,
                count,
                correlation,
                derivatives[id][0],
                derivatives[id][1],
                derivatives[id][2],
                derivatives[id][3],
            ];
        }
    }
    for id in 0..n {
        if mass[id] < minimum_mass {
            flags[id] |= PARTIAL;
        }
        if values[id][0] > 5. * values[id][1] && values[id][3] > 4. {
            flags[id] |= MISMATCH;
        }
    }
    Ok((values, flags))
}

/// Exact affine-pixel profile mass and laboratory-centre derivatives.
/// # Errors
/// Invalid array shapes, singular transform or invalid finite grid.
pub fn evaluate_affine(
    profile: &[f64],
    grid: Grid,
    bounds: &[[f64; 6]],
    centre: [f64; 3],
    planar: [f64; 4],
    scales: [f64; 3],
) -> Result<Vec<[f64; 4]>, SpotError> {
    validate_evaluation(profile, grid, bounds, centre)?;
    if !crate::profiles_affine::valid(planar)
        || scales.iter().any(|v| !v.is_finite() || v.abs() < 1e-8)
    {
        return Err(invalid("invalid profile affine coordinates"));
    }
    let a = [
        planar[0] * scales[0],
        planar[1] * scales[0],
        planar[2] * scales[1],
        planar[3] * scales[1],
    ];
    Ok(bounds
        .iter()
        .map(|&b| {
            let mut out = [0.; 4];
            crate::profiles_affine::weights(grid, b, centre, a, scales[2], |i, w, d| {
                out[0] += profile[i] * w;
                for j in 0..3 {
                    out[j + 1] += profile[i] * d[j];
                }
            });
            out
        })
        .collect())
}

/// Mass, three centre partials and log spatial/rocking-width partials in one pass.
/// Widths scale the finite grid and its normalized voxel densities together.
/// # Errors
/// Invalid grid, profile, bounds, coordinate transform or scales.
pub fn evaluate_shape(
    profile: &[f64],
    grid: Grid,
    bounds: &[[f64; 6]],
    centre: [f64; 3],
    planar: [f64; 4],
    scales: [f64; 3],
) -> Result<Vec<[f64; 6]>, SpotError> {
    validate_evaluation(profile, grid, bounds, centre)?;
    if !crate::profiles_affine::valid(planar)
        || scales.iter().any(|v| !v.is_finite() || v.abs() < 1e-8)
    {
        return Err(invalid("invalid profile affine coordinates"));
    }
    let a = [
        planar[0] * scales[0],
        planar[1] * scales[0],
        planar[2] * scales[1],
        planar[3] * scales[1],
    ];
    Ok(bounds
        .iter()
        .map(|&b| {
            let mut out = [0.; 6];
            crate::profiles_affine::weights_with_widths(
                grid,
                b,
                centre,
                a,
                scales[2],
                |i, w, d| {
                    out[0] += profile[i] * w;
                    for j in 0..5 {
                        out[j + 1] += profile[i] * d[j];
                    }
                },
            );
            out
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_fit(
    o: &Observations<'_>,
    profiles: &[f64],
    grid: Grid,
    assignments: &[usize],
    shifts: &[[f64; 3]],
    minimum_mass: f64,
    gain: f64,
    read_noise: f64,
) -> Result<(), SpotError> {
    o.validate()?;
    grid.validate()?;
    let n = o.centres.len();
    let k = grid.size();
    if profiles.len() != shifts.len() * k
        || assignments.len() != n
        || assignments.iter().any(|&i| i >= shifts.len())
        || profiles.iter().any(|v| !v.is_finite() || *v < 0.)
        || shifts.iter().flatten().any(|v| !v.is_finite())
        || !minimum_mass.is_finite()
        || !(0.0..=1.0).contains(&minimum_mass)
        || !gain.is_finite()
        || gain <= 0.
        || !read_noise.is_finite()
        || read_noise < 0.
    {
        return Err(invalid("invalid profile fit parameters"));
    }
    Ok(())
}

fn validate_evaluation(
    profile: &[f64],
    grid: Grid,
    bounds: &[[f64; 6]],
    centre: [f64; 3],
) -> Result<(), SpotError> {
    grid.validate()?;
    if profile.len() != grid.size()
        || profile.iter().any(|v| !v.is_finite() || *v < 0.)
        || centre.iter().any(|v| !v.is_finite())
        || bounds.len() > 20_000_000
        || bounds
            .iter()
            .any(|b| b.iter().any(|v| !v.is_finite()) || (0..3).any(|j| b[2 * j] >= b[2 * j + 1]))
    {
        return Err(invalid("invalid profile evaluation arrays"));
    }
    Ok(())
}
