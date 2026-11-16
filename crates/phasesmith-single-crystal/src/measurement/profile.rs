//! Profile fitting on the shoeboxes of [`super::measure`]: a bank of 3D
//! profiles learned from strong reflections, then a signed least-squares
//! amplitude fit of every reflection.
use super::{
    AXIAL, Result, Scan,
    geometry::ray_geometry,
    integrate::{Groups, Intensity, MeasureOptions, MeasuredReflection, Monitor, Stage},
    invalid, median,
};
use crate::spots::{Bounds, profiles, reciprocal::Geometry};

/// Controls of profile fitting. Non-exhaustive: start from `Default` and
/// assign fields.
///
/// Local coordinates: the two angular directions perpendicular to the
/// diffracted ray (degrees, from [`super::RayGeometry::planar`]) and the
/// rotation times `ζ` (degrees), so one profile serves the whole detector and
/// every `ζ`.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ProfileOptions {
    /// Frames read on each side of the rocking window, so that profiles see
    /// the tails and teachers can be checked for empty ends. Default 1.
    pub margin_frames: usize,
    /// Voxel grid; `None` (default) derives it: `grid_shape` voxels spanning
    /// the peak box (median angular pixel size) and the median extended
    /// window times `|ζ|`.
    pub grid: Option<profiles::Grid>,
    /// Voxel counts x, y, ω of the derived grid (odd, 3–65). Default
    /// `[31, 31, 23]`.
    pub grid_shape: [usize; 3],
    /// Local profiles: detector split into `regions[0] × regions[1]` and the
    /// scan into `regions[2]` parts, plus one global profile (at most 64
    /// regions). Default `[2, 2, 2]`.
    pub regions: [usize; 3],
    /// Teachers needed for a profile (global or local, ≥ 2); a region with
    /// fewer uses the global profile. Default 8.
    pub minimum_teachers: usize,
    /// Teachers needed in all. Default 20.
    pub minimum_total_teachers: usize,
    /// Candidate teachers offered to learning: the unflagged reflections
    /// with the highest summed I/σ, at most this many (≥ 2). Fewer teachers
    /// learn faster and need a smaller geometry cache. Default 300.
    pub max_teachers: usize,
    /// Teacher threshold in summed I/σ (≥ 3). Teachers are unflagged
    /// reflections over at least three frames whose first and last frame
    /// hold below 5 % of the total. Default 20.
    pub teacher_snr: f64,
    /// Profile mass inside the measured shoebox below which a fit is
    /// [`crate::spots::profiles::PARTIAL`]. Default 0.9.
    pub minimum_mass: f64,
    /// Learning iterations (1–100). Default 100.
    pub iterations: usize,
    /// Geometry cache for learning and fitting, bytes (≤ 1 GiB); learning
    /// runs on several threads only when its cache fits. Default 256 MiB.
    pub geometry_cache_bytes: usize,
}
impl Default for ProfileOptions {
    fn default() -> Self {
        Self {
            margin_frames: 1,
            grid: None,
            grid_shape: [31, 31, 23],
            regions: [2, 2, 2],
            minimum_teachers: 8,
            minimum_total_teachers: 20,
            max_teachers: 300,
            teacher_snr: 20.,
            minimum_mass: 0.9,
            iterations: 100,
            geometry_cache_bytes: 256 * 1024 * 1024,
        }
    }
}
impl ProfileOptions {
    pub(super) fn check(&self) -> Result<()> {
        let models: usize = self.regions.iter().product();
        if self.margin_frames > 20
            || self.grid.is_some_and(|g| g.validate().is_err())
            || (profiles::Grid {
                shape: self.grid_shape,
                step: [1.; 3],
            })
            .validate()
            .is_err()
            || self.regions.contains(&0)
            || models > 64
            || self.minimum_teachers < 2
            || self.max_teachers < 2
            || !self.teacher_snr.is_finite()
            || self.teacher_snr < 3.
            || !(0.0..=1.0).contains(&self.minimum_mass)
            || !(1..=100).contains(&self.iterations)
            || self.geometry_cache_bytes > 1024 * 1024 * 1024
        {
            return Err(invalid("invalid profile options"));
        }
        Ok(())
    }
}

/// A profile-fitted intensity.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct ProfileIntensity {
    /// Fitted amplitude (total counts under the profile) and its σ, without
    /// corrections.
    pub intensity: Intensity,
    /// Fit flags: [`crate::spots::profiles::PARTIAL`] (profile mass below
    /// the minimum), [`crate::spots::profiles::FAILED`],
    /// [`crate::spots::profiles::NO_PROFILE`],
    /// [`crate::spots::profiles::MISMATCH`], and overlap, edge, partial,
    /// background, masked or saturated where they touch the profile's
    /// footprint. Zero means usable.
    pub flags: u32,
    /// Profile mass inside the measured shoebox pixels.
    pub mass: f64,
    /// Reduced χ² of the fit.
    pub chi2: f64,
    /// Correlation of data and profile.
    pub correlation: f64,
    /// Profile used: 0 global, `1 + region` local.
    pub model: usize,
}

/// The learned profile bank.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ProfileModel {
    /// Voxel grid (x, y in degrees perpendicular to the ray, ω times ζ in
    /// degrees).
    pub grid: profiles::Grid,
    /// Model-major voxel masses; each nonempty profile sums to one.
    pub profiles: Vec<f64>,
    /// Teachers per profile (global first).
    pub counts: Vec<usize>,
    /// Teachers used in all.
    pub teachers: usize,
    /// Reflections fitted.
    pub fitted: usize,
    /// Resources the learning actually used (workers, geometry cache).
    pub learning: profiles::LearningExecution,
    /// Resources the fit actually used.
    pub fitting: profiles::FitExecution,
}

/// Outcome of profile fitting.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ProfileStatus {
    /// Profile fitting was not requested.
    NotRequested,
    /// Profiles were learned and fitted.
    Fitted(ProfileModel),
    /// Learning or fitting failed (for example too few teachers); the
    /// summation results stand.
    Failed(String),
}

/// Learn and fit; fills [`MeasuredReflection::profile`].
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn fit(
    groups: &Groups,
    out: &mut [MeasuredReflection],
    geometry: &Geometry,
    scan: &Scan,
    shape: [usize; 2],
    options: &MeasureOptions,
    p: &ProfileOptions,
    monitor: &impl Monitor,
) -> Result<ProfileStatus> {
    // Reflections a profile can describe: off the rotation axis, finite
    // local geometry and at least one shoebox.
    let mut has_group = vec![false; out.len()];
    for &id in &groups.ids {
        has_group[id] = true;
    }
    let mut index = vec![usize::MAX; out.len()];
    let mut selected = Vec::new();
    let mut planar = Vec::new();
    let mut scales = Vec::new();
    let mut centres = Vec::new();
    for (i, m) in out.iter().enumerate() {
        let r = &m.reflection;
        let ray = ray_geometry([r.xyz[0], r.xyz[1]], geometry);
        let a = ray.planar;
        let det = a[0] * a[3] - a[1] * a[2];
        if has_group[i]
            && r.flags & AXIAL == 0
            && ray.zeta.is_finite()
            && ray.zeta.abs() >= 1e-6
            && a.iter().all(|v| v.is_finite())
            && det.abs() > 1e-8 * a[0].hypot(a[1]) * a[2].hypot(a[3])
        {
            index[i] = selected.len();
            selected.push(i);
            planar.push(a);
            scales.push([1., 1., ray.zeta]);
            centres.push(r.xyz);
        }
    }
    if selected.is_empty() {
        return Ok(ProfileStatus::Failed(
            "no reflection suitable for profile fitting".into(),
        ));
    }
    let mut ids = Vec::new();
    let mut boxes: Vec<Bounds> = Vec::new();
    let mut angles = Vec::new();
    let mut covariance = Vec::new();
    let mut flags = Vec::new();
    let mut offsets = vec![0];
    let mut samples = Vec::new();
    let mut valid = Vec::new();
    for g in 0..groups.ids.len() {
        let id = index[groups.ids[g]];
        if id == usize::MAX {
            continue;
        }
        let range = groups.offsets[g]..groups.offsets[g + 1];
        ids.push(id);
        boxes.push(groups.boxes[g]);
        angles.push(scan.bounds(groups.frames[g]));
        covariance.push(groups.covariance[g]);
        flags.push(groups.flags[g]);
        samples.extend_from_slice(&groups.samples[range.clone()]);
        valid.extend_from_slice(&groups.valid[range]);
        offsets.push(samples.len());
    }
    let observations = profiles::Observations {
        samples: &samples,
        valid: &valid,
        offsets: &offsets,
        boxes: &boxes,
        ids: &ids,
        angles: &angles,
        covariance: &covariance,
        centres: &centres,
        scales: &scales,
        planar: &planar,
        flags: &flags,
    };
    let grid = if let Some(g) = p.grid {
        g
    } else {
        let pixel = median(
            planar
                .iter()
                .map(|a| (a[0] * a[3] - a[1] * a[2]).abs().sqrt())
                .collect(),
        )
        .unwrap_or(1.);
        let rocking = median(
            selected
                .iter()
                .zip(&scales)
                .map(|(&i, s)| {
                    let f = out[i].reflection.frames;
                    let span = (f[1] - f[0] + 1 + 2 * p.margin_frames) as f64 * scan.step;
                    span * s[2].abs() / 2.
                })
                .collect(),
        )
        .unwrap_or(scan.step);
        let half = (options.box_size as f64 / 2.) * pixel;
        profiles::Grid {
            shape: p.grid_shape,
            step: [
                2. * half / p.grid_shape[0] as f64,
                2. * half / p.grid_shape[1] as f64,
                2. * rocking / p.grid_shape[2] as f64,
            ],
        }
    };
    if grid.validate().is_err() {
        return Ok(ProfileStatus::Failed(
            "derived profile grid is invalid".into(),
        ));
    }
    let [rx, ry, rz] = p.regions;
    let [lo, hi] = scan.range();
    let regions: Vec<usize> = centres
        .iter()
        .map(|c| {
            let part = |v: f64, n: usize| ((v * n as f64).floor().max(0.) as usize).min(n - 1);
            let x = part(c[0] / shape[1] as f64, rx);
            let y = part(c[1] / shape[0] as f64, ry);
            let z = part((c[2] - lo) / (hi - lo), rz);
            (z * ry + y) * rx + x
        })
        .collect();
    let models = rx * ry * rz + 1;
    let mut candidates: Vec<(f64, usize)> = selected
        .iter()
        .enumerate()
        .filter(|&(_, &i)| out[i].flags == 0 && out[i].summation.is_measured())
        .map(|(k, &i)| (out[i].summation.value / out[i].summation.sigma, k))
        .collect();
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut references = vec![false; selected.len()];
    for &(_, k) in candidates.iter().take(p.max_teachers) {
        references[k] = true;
    }
    let execution = profiles::Execution {
        workers: options.threads().clamp(1, 32),
        geometry_cache_bytes: p.geometry_cache_bytes,
    };
    if !monitor.progress(Stage::LearningProfiles, 0, 1) {
        return Err(invalid(super::CANCELLED));
    }
    let bank = match profiles::learn_with_execution(
        &observations,
        &regions,
        &references,
        grid,
        models,
        p.minimum_teachers,
        p.teacher_snr,
        0.,
        p.iterations,
        false,
        execution,
    ) {
        Ok(b) => b,
        Err(e) => return Ok(ProfileStatus::Failed(e.0)),
    };
    let teachers = bank.used.iter().filter(|&&u| u).count();
    if teachers < p.minimum_total_teachers {
        return Ok(ProfileStatus::Failed(format!(
            "{teachers} profile teachers, fewer than {}",
            p.minimum_total_teachers
        )));
    }
    if !monitor.progress(Stage::LearningProfiles, 1, 1)
        || !monitor.progress(Stage::FittingProfiles, 0, 1)
    {
        return Err(invalid(super::CANCELLED));
    }
    let size = grid.size();
    let assignments: Vec<usize> = regions
        .iter()
        .map(|&r| {
            if bank.counts[r + 1] >= p.minimum_teachers
                && bank.profiles[(r + 1) * size..(r + 2) * size]
                    .iter()
                    .sum::<f64>()
                    > 0.
            {
                r + 1
            } else {
                0
            }
        })
        .collect();
    let fitted = match profiles::fit_with_execution(
        &observations,
        &bank.profiles,
        grid,
        &assignments,
        &bank.shifts,
        p.minimum_mass,
        true,
        options.noise.gain,
        options.noise.read_noise,
        execution,
    ) {
        Ok(f) => f,
        Err(e) => return Ok(ProfileStatus::Failed(e.0)),
    };
    for (k, &i) in selected.iter().enumerate() {
        let v = fitted.values[k];
        out[i].profile = Some(ProfileIntensity {
            intensity: Intensity {
                value: v[0],
                sigma: v[1],
            },
            flags: fitted.flags[k],
            mass: v[2],
            chi2: v[3],
            correlation: v[5],
            model: assignments[k],
        });
    }
    if !monitor.progress(Stage::FittingProfiles, 1, 1) {
        return Err(invalid(super::CANCELLED));
    }
    Ok(ProfileStatus::Fitted(ProfileModel {
        grid,
        profiles: bank.profiles,
        counts: bank.counts,
        teachers,
        fitted: selected.len(),
        learning: bank.execution,
        fitting: fitted.execution,
    }))
}
