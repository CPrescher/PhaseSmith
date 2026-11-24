//! Merging of equivalent reflections and merging statistics.
use super::{Lattice, Result, invalid, norm};
use crate::lattice_symmetry::{Rotation, equivalent_representative};
use std::collections::BTreeMap;

/// One unmerged observation.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Observation {
    /// Indices.
    pub hkl: [i64; 3],
    /// Intensity (corrected or not; one convention throughout).
    pub intensity: f64,
    /// Standard uncertainty (positive).
    pub sigma: f64,
    /// Interplanar spacing, Å.
    pub d: f64,
}
impl Observation {
    /// An observation.
    #[must_use]
    pub fn new(hkl: [i64; 3], intensity: f64, sigma: f64, d: f64) -> Self {
        Self {
            hkl,
            intensity,
            sigma,
            d,
        }
    }
}

/// Equivalent observations merged.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Merged {
    /// Representative indices
    /// ([`crate::lattice_symmetry::equivalent_representative`]).
    pub hkl: [i64; 3],
    /// Unweighted mean intensity.
    pub intensity: f64,
    /// σ of the mean, `√(Σσ²) / n`.
    pub sigma: f64,
    /// Observations merged.
    pub observations: usize,
    /// Interplanar spacing of the first observation, Å.
    pub d: f64,
    /// Mean of the observations' I/σ.
    pub mean_i_over_sigma: f64,
}

/// Merge observations equivalent under the operations (row-vector
/// convention, `h' = h W`; include the inversion to merge Friedel mates),
/// sorted by representative.
#[must_use]
pub fn merge(observations: &[Observation], operations: &[Rotation]) -> Vec<Merged> {
    let mut groups: BTreeMap<[i64; 3], Vec<&Observation>> = BTreeMap::new();
    for o in observations {
        groups
            .entry(equivalent_representative(o.hkl, operations))
            .or_default()
            .push(o);
    }
    groups
        .into_iter()
        .map(|(hkl, g)| {
            let n = g.len() as f64;
            Merged {
                hkl,
                intensity: g.iter().map(|o| o.intensity).sum::<f64>() / n,
                sigma: g.iter().map(|o| o.sigma * o.sigma).sum::<f64>().sqrt() / n,
                observations: g.len(),
                d: g[0].d,
                mean_i_over_sigma: g.iter().map(|o| o.intensity / o.sigma).sum::<f64>() / n,
            }
        })
        .collect()
}

/// Controls of [`merging_statistics`]. Non-exhaustive: start from `Default`
/// and assign fields.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct MergingOptions {
    /// Resolution shells of equal reciprocal volume. Default 4.
    pub shells: usize,
    /// Upper limits of the bins in the mean I/σ of a unique reflection; one
    /// more bin takes the rest. Default `[3, 10, 30]`.
    pub signal_bins: Vec<f64>,
    /// Resolution limit of the shells and of completeness, Å; default the
    /// smallest observed d.
    pub d_min: Option<f64>,
    /// Lattice whose centring-allowed reflections to `d_min` are the
    /// possible ones for completeness; none (default) leaves completeness
    /// out. Its basis must be the one the indices refer to.
    pub lattice: Option<Lattice>,
}
impl Default for MergingOptions {
    fn default() -> Self {
        Self {
            shells: 4,
            signal_bins: vec![3., 10., 30.],
            d_min: None,
            lattice: None,
        }
    }
}

/// Statistics of one group of unique reflections.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct MergingBin {
    /// Bin limits: d in Å (largest first) for shells and overall, mean I/σ
    /// for signal bins.
    pub limits: [f64; 2],
    /// Observations.
    pub observations: usize,
    /// Unique reflections observed.
    pub unique: usize,
    /// Unique reflections possible, when a lattice was given.
    pub possible: Option<usize>,
    /// `unique / possible`.
    pub completeness: Option<f64>,
    /// `observations / unique`.
    pub redundancy: f64,
    /// `Σ|I − ⟨I⟩| / Σ I` over unique reflections observed at least twice
    /// (unweighted means).
    pub r_int: Option<f64>,
    /// Redundancy-independent `Σ √(n/(n−1)) Σ|I − ⟨I⟩| / Σ I`.
    pub r_meas: Option<f64>,
    /// Observations in unique reflections observed at least twice.
    pub multiples: usize,
    /// Mean over unique reflections of their mean I/σ.
    pub mean_i_over_sigma: Option<f64>,
}

/// Merging statistics overall, by resolution shell and by signal.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct MergingStatistics {
    /// All observations to `d_min` (limits: largest observed or, with a
    /// lattice, possible d to `d_min`).
    pub overall: MergingBin,
    /// Resolution shells of equal reciprocal volume from the largest observed
    /// d (with a lattice: the largest possible d) to `d_min`, low resolution
    /// first; their unique and possible reflections add up to `overall`.
    pub shells: Vec<MergingBin>,
    /// Bins in mean I/σ, weakest first.
    pub signal: Vec<MergingBin>,
    /// Observations left out because their d is below `d_min`.
    pub beyond_d_min: usize,
}

fn bin(limits: [f64; 2], groups: &[(&Merged, Vec<f64>)], possible: Option<usize>) -> MergingBin {
    let observations: usize = groups.iter().map(|g| g.1.len()).sum();
    let (mut num, mut meas, mut den, mut multiples) = (0., 0., 0., 0);
    for (m, values) in groups {
        if values.len() >= 2 {
            let n = values.len() as f64;
            let deviation: f64 = values.iter().map(|v| (v - m.intensity).abs()).sum();
            num += deviation;
            meas += (n / (n - 1.)).sqrt() * deviation;
            den += values.iter().sum::<f64>();
            multiples += values.len();
        }
    }
    let unique = groups.len();
    MergingBin {
        limits,
        observations,
        unique,
        possible,
        completeness: possible.map(|p| unique as f64 / p.max(1) as f64),
        redundancy: observations as f64 / unique.max(1) as f64,
        r_int: (den > 0.).then(|| num / den),
        r_meas: (den > 0.).then(|| meas / den),
        multiples,
        mean_i_over_sigma: (unique > 0)
            .then(|| groups.iter().map(|g| g.0.mean_i_over_sigma).sum::<f64>() / unique as f64),
    }
}

/// Unique representatives of the centring-allowed reflections to `d_min`.
fn possible(
    lattice: &Lattice,
    d_min: f64,
    operations: &[Rotation],
) -> Result<Vec<([i64; 3], f64)>> {
    let q_max = 1. / d_min;
    let limits = lattice.direct().map(|a| (norm(a) * q_max).ceil() + 1.);
    let count: f64 = limits.iter().map(|n| 2. * n + 1.).product();
    if count.is_nan() || count > 32e6 {
        return Err(invalid(
            "resolution limit gives too many possible reflections",
        ));
    }
    let limits = limits.map(|n| n as i64);
    let centring = lattice.bravais().centring();
    let mut unique = BTreeMap::new();
    for h in -limits[0]..=limits[0] {
        for k in -limits[1]..=limits[1] {
            for l in -limits[2]..=limits[2] {
                let hkl = [h, k, l];
                if hkl == [0; 3] || !centring.allows(hkl) {
                    continue;
                }
                let n = norm(lattice.q(hkl));
                if n <= q_max {
                    unique
                        .entry(equivalent_representative(hkl, operations))
                        .or_insert(1. / n);
                }
            }
        }
    }
    Ok(unique.into_iter().collect())
}

/// `R_int`, `R_meas`, redundancy and (with a lattice) completeness of the
/// observations under the operations, overall, in resolution shells of equal
/// reciprocal volume and in bins of mean I/σ. Pass only the observations to
/// merge (for example the usable ones). Observations with d below
/// `MergingOptions::d_min` are left out (counted in
/// [`MergingStatistics::beyond_d_min`]). Systematic absences are not removed:
/// no space group is assumed.
/// # Errors
/// Nonfinite intensities, nonpositive σ or d, no observation (within
/// `d_min`), no shells, or an invalid resolution limit.
pub fn merging_statistics(
    observations: &[Observation],
    operations: &[Rotation],
    options: &MergingOptions,
) -> Result<MergingStatistics> {
    if observations.is_empty()
        || options.shells == 0
        || options.shells > 100
        || options.signal_bins.iter().any(|v| !v.is_finite())
        || options.signal_bins.windows(2).any(|w| w[0] >= w[1])
        || observations.iter().any(|o| {
            !o.intensity.is_finite()
                || o.sigma.is_nan()
                || o.sigma <= 0.
                || !o.sigma.is_finite()
                || !o.d.is_finite()
                || o.d <= 0.
        })
    {
        return Err(invalid("invalid observations or merging options"));
    }
    let d_min = options.d_min.unwrap_or_else(|| {
        observations
            .iter()
            .map(|o| o.d)
            .fold(f64::INFINITY, f64::min)
    });
    if !d_min.is_finite() || d_min <= 0. {
        return Err(invalid("invalid resolution limit"));
    }
    // Relative tolerance for d computed at the limit itself.
    let floor = d_min * (1. - 1e-9);
    let all = observations.len();
    let observations: Vec<Observation> = observations
        .iter()
        .copied()
        .filter(|o| o.d >= floor)
        .collect();
    if observations.is_empty() {
        return Err(invalid("no observation within the resolution limit"));
    }
    let all_possible = options
        .lattice
        .as_ref()
        .map(|l| possible(l, d_min, operations))
        .transpose()?;
    let d_max = observations
        .iter()
        .map(|o| o.d)
        .chain(all_possible.iter().flatten().map(|p| p.1))
        .fold(0., f64::max)
        .max(d_min);
    let merged = merge(&observations, operations);
    let mut members: BTreeMap<[i64; 3], Vec<f64>> = BTreeMap::new();
    for o in &observations {
        members
            .entry(equivalent_representative(o.hkl, operations))
            .or_default()
            .push(o.intensity);
    }
    let groups: Vec<(&Merged, Vec<f64>)> = merged
        .iter()
        .map(|m| (m, members.remove(&m.hkl).unwrap_or_default()))
        .collect();
    let q = |d: f64| d.powi(-3);
    let (q0, q1) = (q(d_max), q(d_min));
    let shells = options.shells;
    let limit = |i: usize| (q0 + (q1 - q0) * i as f64 / shells as f64).cbrt().recip();
    // The outer limits are open against rounding of the shell edges; every
    // kept d lies in [d_min, d_max].
    let in_shell =
        |d: f64, s: usize| (s == 0 || d <= limit(s)) && (s + 1 == shells || d > limit(s + 1));
    let mut shell_bins = Vec::with_capacity(shells);
    for s in 0..shells {
        let chosen: Vec<_> = groups
            .iter()
            .filter(|g| in_shell(g.0.d, s))
            .map(|g| (g.0, g.1.clone()))
            .collect();
        let possible = all_possible
            .as_ref()
            .map(|p| p.iter().filter(|(_, d)| in_shell(*d, s)).count());
        shell_bins.push(bin([limit(s), limit(s + 1)], &chosen, possible));
    }
    let mut edges = vec![f64::NEG_INFINITY];
    edges.extend(&options.signal_bins);
    edges.push(f64::INFINITY);
    let signal = edges
        .windows(2)
        .map(|w| {
            let chosen: Vec<_> = groups
                .iter()
                .filter(|g| g.0.mean_i_over_sigma >= w[0] && g.0.mean_i_over_sigma < w[1])
                .map(|g| (g.0, g.1.clone()))
                .collect();
            bin([w[0], w[1]], &chosen, None)
        })
        .collect();
    let overall = bin([d_max, d_min], &groups, all_possible.as_ref().map(Vec::len));
    Ok(MergingStatistics {
        overall,
        shells: shell_bins,
        signal,
        beyond_d_min: all - observations.len(),
    })
}
