//! Native strong-profile variable projection, frozen weak extraction and joint sigma.
use crate::{
    SpotError, box_optimizer as optimizer,
    geometry_linear::eigen,
    intensity_model, invalid,
    profiles::{self, Grid, Observations},
};
use rayon::prelude::*;

/// Declared independent-pixel counting model.
#[derive(Clone, Copy)]
pub struct Noise {
    /// Expected count variance (otherwise use supplied independent variance).
    pub expected: bool,
    /// Positive photons/electrons per count.
    pub gain: f64,
    /// Nonnegative read noise in count units.
    pub read: f64,
}
/// Explicit strong/frozen-weak controls, with unchanged finite profile support.
#[derive(Clone, Copy)]
pub struct Options {
    /// Private native worker count, 1..4.
    pub workers: usize,
    /// Fixed count/noise outer iteration count, 1..10.
    pub iterations: usize,
    /// Maximum accepted inner iterations, 1..200.
    pub maximum_iterations: usize,
    /// Closed x/y displacement bound in pixels.
    pub centre_pixels: f64,
    /// Closed omega displacement bound in degrees.
    pub centre_degrees: f64,
    /// Spatial multiplicative width interval, bracketing one in [.25,3].
    pub spatial_range: [f64; 2],
    /// Rocking multiplicative width interval, bracketing one in [.25,3].
    pub rocking_range: [f64; 2],
    /// Nonnegative neutral penalties for x/y/omega/log widths.
    pub regularizer: [f64; 5],
    /// Frozen expected intensity used to design a weak matched filter.
    pub weak_design_counts: f64,
    /// Frozen weak spatial width multiplier.
    pub weak_spatial_width: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            workers: 1,
            iterations: 3,
            maximum_iterations: 80,
            centre_pixels: 1.,
            centre_degrees: 0.4,
            spatial_range: [0.65, 1.5],
            rocking_range: [0.5, 1.5],
            regularizer: [0.01, 0.01, 0.16, 0.04, 0.04],
            weak_design_counts: 1000.,
            weak_spatial_width: 0.95,
        }
    }
}
impl Options {
    fn validate(self) -> Result<(), SpotError> {
        if !(1..=4).contains(&self.workers)
            || !(1..=10).contains(&self.iterations)
            || !(1..=200).contains(&self.maximum_iterations)
            || [
                self.centre_pixels,
                self.centre_degrees,
                self.weak_design_counts,
                self.weak_spatial_width,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.)
            || self.regularizer.iter().any(|v| !v.is_finite() || *v < 0.)
            || [self.spatial_range, self.rocking_range].iter().any(|p| {
                p.iter().any(|v| !v.is_finite())
                    || p[0] < 0.25
                    || p[0] >= 1.
                    || p[1] <= 1.
                    || p[1] > 3.
            })
        {
            return Err(invalid("invalid native profile-refinement options"));
        }
        Ok(())
    }
}
/// An actual conditional per-reflection refinement result.
pub struct Row {
    /// The ten existing intensity/quality diagnostic columns.
    pub values: [f64; 10],
    /// Existing source, support and conditional fitting quality flags.
    pub flags: u32,
    /// x/y/omega shifts and log spatial/rocking multipliers.
    pub parameters: [f64; 5],
    /// All requested inner solves reached a convergence criterion.
    pub converged: bool,
    /// Actual inner objective evaluations across noise iterations.
    pub evaluations: usize,
}
/// Owned single-reflection working data. Construction copies only its selected groups.
pub struct PreparedSpot {
    samples: Vec<[f64; 3]>,
    valid: Vec<bool>,
    offsets: Vec<usize>,
    boxes: Vec<crate::Bounds>,
    angles: Vec<[f64; 2]>,
    covariance: Vec<[f64; 9]>,
    flags: Vec<u32>,
    ids: Vec<usize>,
    centre: [f64; 3],
    actual: [f64; 3],
    scales: [f64; 3],
    planar: [f64; 4],
    profile: Vec<f64>,
    grid: Grid,
    shift: [f64; 3],
    minimum_mass: f64,
    noise: Noise,
    signal: Vec<f64>,
    basis: Vec<[f64; 3]>,
    bounds: Vec<[f64; 6]>,
    weights: Vec<f64>,
}
/// Read-only working arrays for independent diagnostics and numerical tests.
pub struct Snapshot<'a> {
    /// Signed available signal, zero at excluded pixels.
    pub signal: &'a [f64],
    /// Current frozen inverse independent variance (empty before `set_noise`).
    pub weights: &'a [f64],
    /// Detector-background plane basis for each collected sample.
    pub basis: &'a [[f64; 3]],
    /// Sample offsets for the selected reflection's exposure groups.
    pub offsets: &'a [usize],
    /// Finite background-plane covariance, one per group.
    pub covariance: &'a [[f64; 9]],
    /// Original sample footprints, including full exposure bounds.
    pub bounds: &'a [[f64; 6]],
}
impl PreparedSpot {
    /// Prepare one selected reflection without changing original arrays.
    /// # Errors
    /// Invalid observations, bank/grid/noise parameters or an out-of-range ID.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        o: &Observations<'_>,
        id: usize,
        profiles: &[f64],
        grid: Grid,
        assignment: usize,
        shifts: &[[f64; 3]],
        minimum_mass: f64,
        noise: Noise,
    ) -> Result<Self, SpotError> {
        let assignments = vec![assignment; o.centres.len()];
        profiles::validate_fit(
            o,
            profiles,
            grid,
            &assignments,
            shifts,
            minimum_mass,
            noise.gain,
            noise.read,
        )?;
        if id >= o.centres.len() {
            return Err(invalid("profile ID outside observations"));
        }
        let groups: Vec<_> = (0..o.ids.len()).filter(|&g| o.ids[g] == id).collect();
        Ok(Self::selected(
            o,
            id,
            &groups,
            &profiles[assignment * grid.size()..(assignment + 1) * grid.size()],
            grid,
            shifts[assignment],
            minimum_mass,
            noise,
        ))
    }
    #[allow(clippy::too_many_arguments)]
    fn selected(
        o: &Observations<'_>,
        id: usize,
        groups: &[usize],
        profile: &[f64],
        grid: Grid,
        shift: [f64; 3],
        minimum_mass: f64,
        noise: Noise,
    ) -> Self {
        let mut s = Self {
            samples: Vec::new(),
            valid: Vec::new(),
            offsets: vec![0],
            boxes: Vec::new(),
            angles: Vec::new(),
            covariance: Vec::new(),
            flags: Vec::new(),
            ids: vec![0; groups.len()],
            centre: o.centres[id],
            actual: o.shifted(id, shift),
            scales: o.scales[id],
            planar: o.planar[id],
            profile: profile.to_vec(),
            grid,
            shift,
            minimum_mass,
            noise,
            signal: Vec::new(),
            basis: Vec::new(),
            bounds: Vec::new(),
            weights: Vec::new(),
        };
        for &g in groups {
            s.boxes.push(o.boxes[g]);
            s.angles.push(o.angles[g]);
            s.covariance
                .push(o.covariance[g].map(|v| if v.is_finite() { v } else { 0. }));
            s.flags.push(o.flags[g]);
            for i in o.offsets[g]..o.offsets[g + 1] {
                s.samples.push(o.samples[i]);
                s.valid.push(o.valid[i]);
                s.signal.push(if o.valid[i] { o.samples[i][0] } else { 0. });
                s.basis.push(o.basis(g, i));
                s.bounds.push(o.bounds(g, i));
            }
            s.offsets.push(s.samples.len());
        }
        s
    }
    /// Read-only numerical snapshots; measured input arrays remain untouched.
    #[must_use]
    pub fn snapshot(&self) -> Snapshot<'_> {
        Snapshot {
            signal: &self.signal,
            weights: &self.weights,
            basis: &self.basis,
            offsets: &self.offsets,
            covariance: &self.covariance,
            bounds: &self.bounds,
        }
    }
    fn grid_at(&self, p: [f64; 5]) -> Grid {
        Grid {
            shape: self.grid.shape,
            step: [
                self.grid.step[0] * p[3].exp(),
                self.grid.step[1] * p[3].exp(),
                self.grid.step[2] * p[4].exp(),
            ],
        }
    }
    /// Profile mass and all five analytical partials with the current finite support.
    /// # Errors
    /// Nonfinite parameters or an invalid scaled grid.
    pub fn mass(&self, p: [f64; 5]) -> Result<Vec<[f64; 6]>, SpotError> {
        if p.iter().any(|v| !v.is_finite()) {
            return Err(invalid("nonfinite profile shape parameters"));
        }
        profiles::evaluate_shape(
            &self.profile,
            self.grid_at(p),
            &self.bounds,
            std::array::from_fn(|j| self.actual[j] + p[j]),
            self.planar,
            self.scales,
        )
    }
    /// Freeze expected/supplied independent variances for the next conditional fit.
    /// # Errors
    /// Invalid mass, variance or intensity; unavailable pixels always have zero weight.
    pub fn set_noise(
        &mut self,
        mass: &[f64],
        intensity: f64,
        foreground: bool,
    ) -> Result<(), SpotError> {
        if mass.len() != self.samples.len()
            || !intensity.is_finite()
            || mass.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("invalid profile noise inputs"));
        }
        self.weights = mass
            .iter()
            .enumerate()
            .map(|(i, &m)| {
                if !self.valid[i] || (foreground && m <= 0.) {
                    return 0.;
                }
                let variance = if self.noise.expected {
                    ((self.samples[i][2] + intensity.max(0.) * m).max(0.) / self.noise.gain
                        + self.noise.read.powi(2))
                    .max(1.)
                } else {
                    self.samples[i][1]
                };
                1. / variance
            })
            .collect();
        if self.weights.iter().any(|v| !v.is_finite() || *v < 0.) {
            return Err(invalid("invalid profile counting variance"));
        }
        Ok(())
    }
    /// Conditional GLS Gram matrix including y, with background covariance retained.
    /// # Errors
    /// Invalid column count/design, unfrozen noise or a singular update.
    pub fn normal(&self, design: &[f64], columns: usize) -> Result<Vec<f64>, SpotError> {
        intensity_model::normal(
            &self.signal,
            &self.weights,
            &self.basis,
            &self.offsets,
            &self.covariance,
            design,
            columns,
        )
    }
    /// Profiled-amplitude SSE, exact shape gradient and variable-projection GN normal.
    /// # Errors
    /// Nonidentifiable amplitude, invalid parameters/regularizer or nonfinite arithmetic.
    #[allow(clippy::manual_midpoint)] // Preserve the previous average and operation order.
    pub fn objective(
        &self,
        p: [f64; 5],
        regularizer: [f64; 5],
    ) -> Result<optimizer::Model, SpotError> {
        if regularizer.iter().any(|v| !v.is_finite() || *v < 0.) {
            return Err(invalid("invalid profile shape regularizer"));
        }
        let mass = self.mass(p)?;
        let gram = self.normal(&mass.concat(), 6)?;
        let den = gram[0];
        let num = gram[6];
        if den <= 0. {
            return Err(invalid("nonidentifiable strong profile"));
        }
        let amp = num / den;
        let cost = gram[48] - num * amp + (0..5).map(|j| regularizer[j] * p[j] * p[j]).sum::<f64>();
        let residual: [f64; 5] =
            std::array::from_fn(|j| gram[(j + 1) * 7 + 6] - amp * gram[(j + 1) * 7]);
        let gradient = (0..5)
            .map(|j| -2. * amp * residual[j] + 2. * regularizer[j] * p[j])
            .collect();
        let mut normal = vec![0.; 25];
        for j in 0..5 {
            for k in 0..=j {
                let raw = 0.5 * (gram[(j + 1) * 7 + k + 1] + gram[(k + 1) * 7 + j + 1]);
                let value = 2.
                    * (amp * amp * (raw - gram[(j + 1) * 7] * gram[(k + 1) * 7] / den)
                        + residual[j] * residual[k] / den)
                    + if j == k { 2. * regularizer[j] } else { 0. };
                normal[j * 5 + k] = value;
                normal[k * 5 + j] = value;
            }
        }
        Ok(optimizer::Model {
            cost,
            gradient,
            normal,
        })
    }
    /// Existing native fixed-shape integration at declared position/width parameters.
    /// # Errors
    /// Invalid parameters, unsupported finite grid or failed native fit.
    pub fn native_fit(&self, p: [f64; 5]) -> Result<([f64; 10], u32), SpotError> {
        let centres = [std::array::from_fn(|j| self.centre[j] + p[j])];
        let scales = [self.scales];
        let planar = [self.planar];
        let o = Observations {
            samples: &self.samples,
            valid: &self.valid,
            offsets: &self.offsets,
            boxes: &self.boxes,
            ids: &self.ids,
            angles: &self.angles,
            covariance: &self.covariance,
            centres: &centres,
            scales: &scales,
            planar: &planar,
            flags: &self.flags,
        };
        let (values, flags) = profiles::fit(
            &o,
            &self.profile,
            self.grid_at(p),
            &[0],
            &[self.shift],
            self.minimum_mass,
            self.noise.expected,
            self.noise.gain,
            self.noise.read,
        )?;
        Ok((values[0], flags[0]))
    }
    /// Apply frozen strong shape and inflate amplitude sigma by conditional joint curvature.
    /// # Errors
    /// Invalid shape, covariance or a nonpositive scaled joint normal.
    #[allow(clippy::manual_midpoint)] // Preserve the previous average and operation order.
    pub fn finish(&mut self, p: [f64; 5], choice: u8) -> Result<([f64; 10], u32), SpotError> {
        if choice != 1 && choice != 2 {
            return Err(invalid("strong profile choice must be one or two"));
        }
        let (mut values, flags) = self.native_fit(p)?;
        let mass = self.mass(p)?;
        let probability: Vec<_> = mass.iter().map(|m| m[0]).collect();
        self.set_noise(&probability, values[0], true)?;
        let n = if choice == 2 { 3 } else { 6 };
        let mut design = Vec::with_capacity(mass.len() * n);
        for m in mass {
            design.push(m[0]);
            design.extend(m[1..n].iter().map(|v| values[0] * v));
        }
        let gram = self.normal(&design, n)?;
        let diag: Vec<_> = (0..n)
            .map(|j| gram[j * (n + 1) + j].max(1e-30).sqrt())
            .collect();
        let scaled: Vec<_> = (0..n * n)
            .map(|i| {
                let a = i / n;
                let b = i % n;
                0.5 * (gram[a * (n + 1) + b] / diag[a] / diag[b]
                    + gram[b * (n + 1) + a] / diag[b] / diag[a])
            })
            .collect();
        let (eig, vectors) = eigen(&scaled, n)?;
        if eig[0] < -1e-7 {
            return Err(invalid("nonpositive profile normal matrix"));
        }
        let variance: f64 = (0..n)
            .filter(|&j| eig[j] > 1e-9)
            .map(|j| vectors[j] * vectors[j] / eig[j])
            .sum();
        values[1] = values[1].max(variance.max(0.).sqrt() / diag[0]);
        Ok((values, flags))
    }
    /// Refine an explicitly selected strong peak using fixed outer noise iterations.
    /// # Errors
    /// Invalid controls, nonidentifiable amplitudes or an invalid final joint fit.
    pub fn refine(&mut self, initial: f64, choice: u8, options: Options) -> Result<Row, SpotError> {
        options.validate()?;
        if !initial.is_finite() || (choice != 1 && choice != 2) {
            return Err(invalid("invalid strong profile input"));
        }
        let mut p = [0.; 5];
        let mut lower = [
            -options.centre_pixels,
            -options.centre_pixels,
            -options.centre_degrees,
            options.spatial_range[0].ln(),
            options.rocking_range[0].ln(),
        ];
        let mut upper = [
            options.centre_pixels,
            options.centre_pixels,
            options.centre_degrees,
            options.spatial_range[1].ln(),
            options.rocking_range[1].ln(),
        ];
        if choice == 2 {
            lower[2..].fill(0.);
            upper[2..].fill(0.);
        }
        let mut intensity = initial;
        let mut converged = true;
        let mut evaluations = 0;
        for _ in 0..options.iterations {
            let mass = self.mass(p)?;
            let probability: Vec<_> = mass.iter().map(|m| m[0]).collect();
            self.set_noise(&probability, intensity, false)?;
            let before = self.objective(p, options.regularizer)?.cost;
            let result = optimizer::quasi_newton(
                &p,
                &lower,
                &upper,
                optimizer::Options {
                    iterations: options.maximum_iterations,
                    evaluations: options.maximum_iterations * 30 + 1,
                    ftol: 1e-9,
                    gtol: 1e-4,
                    xtol: 1e-13,
                    cost_floor: 1.,
                },
                |v| self.objective(std::array::from_fn(|j| v[j]), options.regularizer),
            )?;
            converged &= result.converged;
            evaluations += result.evaluations;
            if result.model.cost <= before + 1e-5 {
                p.copy_from_slice(&result.parameters);
            }
            let mass = self.mass(p)?;
            let gram = self.normal(&mass.concat(), 6)?;
            if gram[0] <= 0. {
                return Err(invalid("nonidentifiable strong profile"));
            }
            intensity = gram[6] / gram[0];
        }
        let (values, flags) = self.finish(p, choice)?;
        Ok(Row {
            values,
            flags,
            parameters: p,
            converged,
            evaluations,
        })
    }
    /// Frozen weak matched filter; signed/null amplitudes never fit their own shape/noise.
    /// # Errors
    /// Invalid declared design noise or a nonidentifiable matched filter.
    pub fn fixed(&mut self, p: [f64; 5], design_counts: f64) -> Result<[f64; 3], SpotError> {
        let mass: Vec<_> = self.mass(p)?.iter().map(|m| m[0]).collect();
        self.set_noise(&mass, design_counts, true)?;
        let variance: Vec<_> = self
            .samples
            .iter()
            .zip(&self.valid)
            .map(|(s, &v)| if v { s[1] } else { 0. })
            .collect();
        let result = intensity_model::fixed_filter(
            &self.signal,
            &variance,
            &self.weights,
            &self.basis,
            &self.offsets,
            &self.covariance,
            &mass,
        )?;
        let observed_mass = mass
            .iter()
            .zip(&self.valid)
            .filter_map(|(m, &v)| v.then_some(m))
            .sum();
        Ok([result[0], result[1], observed_mass])
    }
}

/// Physical template centroid displacement for every assigned profile.
/// # Errors
/// Rejects inconsistent dimensions, maps, profiles or nonfinite scales/widths.
#[allow(clippy::too_many_arguments)]
pub fn centroid_offsets(
    profiles: &[f64],
    grid: Grid,
    shifts: &[[f64; 3]],
    assignments: &[usize],
    scales: &[[f64; 3]],
    planar: &[[f64; 4]],
    log_width: &[f64],
) -> Result<Vec<[f64; 2]>, SpotError> {
    grid.validate()?;
    let n = assignments.len();
    let k = grid.size();
    if profiles.len() != shifts.len() * k
        || scales.len() != n
        || planar.len() != n
        || log_width.len() != n
        || assignments.iter().any(|&i| i >= shifts.len())
        || profiles.iter().any(|v| !v.is_finite() || *v < 0.)
        || shifts.iter().flatten().any(|v| !v.is_finite())
        || scales
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || v.abs() < 1e-8)
        || log_width
            .iter()
            .any(|v| !v.is_finite() || !v.exp().is_finite())
        || planar.iter().any(|&v| !crate::profiles_affine::valid(v))
    {
        return Err(invalid("invalid profile centroid inputs"));
    }
    let mut moments = vec![[0.; 2]; shifts.len()];
    for bank in 0..shifts.len() {
        for i in 0..k {
            let x = i % grid.shape[0];
            let y = i / grid.shape[0] % grid.shape[1];
            moments[bank][0] += profiles[bank * k + i]
                * (x as f64 - (grid.shape[0] - 1) as f64 / 2.)
                * grid.step[0];
            moments[bank][1] += profiles[bank * k + i]
                * (y as f64 - (grid.shape[1] - 1) as f64 / 2.)
                * grid.step[1];
        }
    }
    Ok((0..n)
        .map(|i| {
            let bank = assignments[i];
            let u = (shifts[bank][0] + moments[bank][0] * log_width[i].exp()) / scales[i][0];
            let v = (shifts[bank][1] + moments[bank][1] * log_width[i].exp()) / scales[i][1];
            let a = planar[i];
            let det = a[0] * a[3] - a[1] * a[2];
            [(a[3] * u - a[1] * v) / det, (-a[2] * u + a[0] * v) / det]
        })
        .collect())
}

/// Optional already-computed baseline, preserving host result metadata without refitting.
pub struct Baseline<'a> {
    /// Native fixed-profile ten-column values.
    pub values: &'a [[f64; 10]],
    /// Native fixed-profile flags.
    pub flags: &'a [u32],
}
/// Frozen batch inputs; no peak/family selection is inferred by refinement.
pub struct Input<'a> {
    /// Original shoebox observations.
    pub observations: Observations<'a>,
    /// Flat bank profiles in grid order.
    pub profiles: &'a [f64],
    /// Original finite profile grid.
    pub grid: Grid,
    /// Profile-bank assignment for every reflection.
    pub assignments: &'a [usize],
    /// Fixed bank shifts in local profile coordinates.
    pub shifts: &'a [[f64; 3]],
    /// Minimum supported profile mass for existing quality flags.
    pub minimum_mass: f64,
    /// Counting model.
    pub noise: Noise,
    /// 0=fixed, 1=strong five parameters, 2=strong x/y only.
    pub choices: &'a [u8],
    /// Explicit weak fixed-filter update mask, disjoint from strong choices.
    pub weak: &'a [bool],
    /// Declared physical sensor centroid offsets, one per reflection.
    pub sensor_offsets: &'a [[f64; 2]],
    /// Optional precomputed strong parameters, one five-vector per reflection.
    pub frozen: Option<&'a [[f64; 5]]>,
    /// Optional precomputed native baseline; otherwise calculated once here.
    pub baseline: Option<Baseline<'a>>,
}
/// Native batch values, retained baselines, parameters and actual convergence status.
pub struct Fit {
    /// Original fixed-profile values.
    pub baseline_values: Vec<[f64; 10]>,
    /// Original fixed-profile flags.
    pub baseline_flags: Vec<u32>,
    /// Final values; every source row remains present.
    pub values: Vec<[f64; 10]>,
    /// Final flags, including FAILED on conditional-fit errors.
    pub flags: Vec<u32>,
    /// Actual fitted/frozen parameters; zero for untouched rows.
    pub parameters: Vec<[f64; 5]>,
    /// Actual optimizer success, including row failures.
    pub converged: Vec<bool>,
    /// Actual objective evaluations for each source row.
    pub evaluations: Vec<usize>,
}
/// Refine explicitly selected profiles using a bounded private native worker pool.
/// # Errors
/// Invalid arrays/controls or a cancelled progress callback. Individual conditional
/// fit failures retain their baseline row and add FAILED; no observation is deleted.
#[allow(clippy::too_many_lines)]
pub fn refine(
    input: &Input<'_>,
    options: Options,
    mut progress: impl FnMut(usize, usize) -> Result<(), SpotError>,
) -> Result<Fit, SpotError> {
    options.validate()?;
    let o = &input.observations;
    let n = o.centres.len();
    profiles::validate_fit(
        o,
        input.profiles,
        input.grid,
        input.assignments,
        input.shifts,
        input.minimum_mass,
        input.noise.gain,
        input.noise.read,
    )?;
    if input.choices.len() != n
        || input.weak.len() != n
        || input.sensor_offsets.len() != n
        || input.choices.iter().any(|v| *v > 2)
        || (0..n).any(|i| input.weak[i] && input.choices[i] != 0)
        || input
            .sensor_offsets
            .iter()
            .flatten()
            .any(|v| !v.is_finite())
        || input
            .frozen
            .is_some_and(|p| p.len() != n || p.iter().flatten().any(|v| !v.is_finite()))
    {
        return Err(invalid("invalid frozen profile choices/offsets/parameters"));
    }
    let (baseline_values, baseline_flags) = if let Some(b) = &input.baseline {
        if b.values.len() != n || b.flags.len() != n {
            return Err(invalid("invalid baseline profile dimensions"));
        }
        (b.values.to_vec(), b.flags.to_vec())
    } else {
        profiles::fit(
            o,
            input.profiles,
            input.grid,
            input.assignments,
            input.shifts,
            input.minimum_mass,
            input.noise.expected,
            input.noise.gain,
            input.noise.read,
        )?
    };
    let mut values = baseline_values.clone();
    let mut flags = baseline_flags.clone();
    let mut parameters = vec![[0.; 5]; n];
    let mut converged = vec![true; n];
    let mut evaluations = vec![0; n];
    let selected: Vec<_> = (0..n)
        .filter(|&i| input.choices[i] != 0 || input.weak[i])
        .collect();
    let centroids = if input.weak.iter().any(|v| *v) {
        centroid_offsets(
            input.profiles,
            input.grid,
            input.shifts,
            input.assignments,
            o.scales,
            o.planar,
            &vec![options.weak_spatial_width.ln(); n],
        )?
    } else {
        vec![[0.; 2]; n]
    };
    let mut offsets = vec![0_usize; n + 1];
    for &id in o.ids {
        offsets[id + 1] += 1;
    }
    for i in 0..n {
        offsets[i + 1] += offsets[i];
    }
    let mut cursor = offsets[..n].to_vec();
    let mut groups = vec![0; o.ids.len()];
    for (g, &id) in o.ids.iter().enumerate() {
        groups[cursor[id]] = g;
        cursor[id] += 1;
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.workers)
        .build()
        .map_err(|_| invalid("could not build profile worker pool"))?;
    let mut done = 0;
    for chunk in selected.chunks(options.workers) {
        let rows: Vec<_> = pool.install(|| {
            chunk
                .par_iter()
                .map(|&i| {
                    let bank = input.assignments[i];
                    let k = input.grid.size();
                    let mut spot = PreparedSpot::selected(
                        o,
                        i,
                        &groups[offsets[i]..offsets[i + 1]],
                        &input.profiles[bank * k..(bank + 1) * k],
                        input.grid,
                        input.shifts[bank],
                        input.minimum_mass,
                        input.noise,
                    );
                    let mut p = [0.; 5];
                    let result = if input.choices[i] != 0 {
                        if let Some(frozen) = input.frozen {
                            p = frozen[i];
                            spot.finish(p, input.choices[i]).map(|(values, flags)| Row {
                                values,
                                flags,
                                parameters: p,
                                converged: true,
                                evaluations: 0,
                            })
                        } else {
                            spot.refine(baseline_values[i][0], input.choices[i], options)
                        }
                    } else {
                        p[3] = options.weak_spatial_width.ln();
                        p[0] = input.sensor_offsets[i][0] - centroids[i][0];
                        p[1] = input.sensor_offsets[i][1] - centroids[i][1];
                        spot.fixed(p, options.weak_design_counts).map(|fixed| {
                            let mut values = baseline_values[i];
                            values[..3].copy_from_slice(&fixed);
                            values[3..].fill(f64::NAN);
                            Row {
                                values,
                                flags: baseline_flags[i],
                                parameters: p,
                                converged: true,
                                evaluations: 0,
                            }
                        })
                    };
                    (i, result, p)
                })
                .collect()
        });
        for (i, result, p) in rows {
            if let Ok(row) = result {
                values[i] = row.values;
                flags[i] = row.flags;
                parameters[i] = row.parameters;
                converged[i] = row.converged;
                evaluations[i] = row.evaluations;
            } else {
                flags[i] |= profiles::FAILED;
                parameters[i] = p;
                converged[i] = false;
            }
            done += 1;
            progress(done, selected.len())?;
        }
    }
    Ok(Fit {
        baseline_values,
        baseline_flags,
        values,
        flags,
        parameters,
        converged,
        evaluations,
    })
}
