//! Exact per-reflection fixed-profile GLS with bounded first-pass derivative reuse.
use crate::spots::{
    SpotError,
    profiles::{self, Execution, FitExecution, Fitted, Grid, Observations, dot, inverse, mv},
};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};

struct Derivatives {
    offsets: Vec<usize>,
    values: Vec<[f64; 3]>,
    maximum: usize,
    budget: usize,
}
impl Derivatives {
    fn new(rows: usize, budget: usize) -> Option<Self> {
        let pointer_bytes = rows.checked_add(1)?.checked_mul(size_of::<usize>())?;
        if pointer_bytes >= budget {
            return None;
        }
        let mut offsets = Vec::new();
        offsets.try_reserve_exact(rows + 1).ok()?;
        offsets.push(0);
        let pointer_bytes = offsets.capacity() * size_of::<usize>();
        if pointer_bytes >= budget {
            return None;
        }
        let maximum = (budget - pointer_bytes) / size_of::<[f64; 3]>();
        if maximum == 0 {
            return None;
        }
        Some(Self {
            offsets,
            values: Vec::new(),
            maximum,
            budget,
        })
    }
    fn bytes(&self) -> usize {
        self.offsets.capacity() * size_of::<usize>()
            + self.values.capacity() * size_of::<[f64; 3]>()
    }
    fn push(&mut self, value: [f64; 3]) -> bool {
        if self.values.len() == self.maximum {
            return false;
        }
        if self.values.len() == self.values.capacity() {
            let target = (self.values.capacity().max(512) * 2).min(self.maximum);
            if self
                .values
                .try_reserve_exact(target - self.values.len())
                .is_err()
                || self.bytes() > self.budget
            {
                return false;
            }
        }
        self.values.push(value);
        true
    }
}
struct Context<'a, 'data> {
    o: &'a Observations<'data>,
    profiles: &'a [f64],
    grid: Grid,
    assignments: &'a [usize],
    shifts: &'a [[f64; 3]],
    minimum_mass: f64,
    expected_noise: bool,
    gain: f64,
    read_noise: f64,
    probability: &'a [f64],
    mass: &'a [f64],
    flags: &'a [u32],
}
struct Row {
    values: [f64; 10],
    flags: u32,
    cache_bytes: usize,
    cached: bool,
}

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn row(c: &Context<'_, '_>, id: usize, groups: &[usize], budget: usize) -> Row {
    let o = c.o;
    let k = c.grid.size();
    let model = c.assignments[id];
    let centre = o.shifted(id, c.shifts[model]);
    let rows: usize = groups
        .iter()
        .map(|&g| o.offsets[g + 1] - o.offsets[g])
        .sum();
    let mut cache = Derivatives::new(rows, budget);
    let mut peak = cache.as_ref().map_or(0, Derivatives::bytes);
    let mut values = [f64::NAN; 10];
    let mut intensity = 0_f64;
    let mut flags = c.flags[id];
    for iteration in 0..if c.expected_noise { 8 } else { 1 } {
        let (mut num, mut den, mut chi) = (0., 0., 0.);
        let mut count = 0_usize;
        let mut moments = [0.; 5];
        let mut shape_raw = [0.; 3];
        let mut shape_normal = [0.; 3];
        let mut cache_row = 0;
        for &g in groups {
            if !(o.offsets[g]..o.offsets[g + 1]).any(|i| o.valid[i] && c.probability[i] > 0.) {
                continue;
            }
            let mut b_inverse =
                std::array::from_fn(|i| std::array::from_fn(|j| o.covariance[g][i * 3 + j]));
            let zero = b_inverse.iter().flatten().all(|v| v.abs() < 1e-25);
            if !zero {
                let Some(inv) = inverse(b_inverse) else {
                    flags |= profiles::FAILED;
                    continue;
                };
                b_inverse = inv;
            }
            let (mut pu, mut du) = ([0.; 3], [0.; 3]);
            let (mut pp, mut pd, mut dd) = (0., 0., 0.);
            let mut shape_u = [[0.; 3]; 3];
            let mut shape_score = [0.; 3];
            let mut shape_p = [0.; 3];
            for i in o.offsets[g]..o.offsets[g + 1] {
                let p = c.probability[i];
                if !o.valid[i] || p <= 0. {
                    continue;
                }
                let s = o.samples[i];
                let x = o.basis(g, i);
                let variance = if c.expected_noise && iteration > 0 {
                    ((s[2] + intensity.max(0.) * p).max(0.) / c.gain + c.read_noise.powi(2)).max(1.)
                } else {
                    s[1]
                };
                let w = 1. / variance;
                pp += w * p * p;
                pd += w * p * s[0];
                dd += w * s[0] * s[0];
                count += 1;
                for a in 0..3 {
                    pu[a] += w * p * x[a];
                    du[a] += w * s[0] * x[a];
                    for b in 0..3 {
                        if !zero {
                            b_inverse[a][b] += w * x[a] * x[b];
                        }
                    }
                }
                let residual = s[0] - intensity * p;
                let mut accumulate = |v: [f64; 3]| {
                    for a in 0..3 {
                        shape_score[a] += w * v[a] * residual;
                        shape_p[a] += w * v[a] * p;
                        for b in 0..3 {
                            shape_u[a][b] += w * v[a] * x[b];
                        }
                    }
                };
                if iteration > 0 && cache.is_some() {
                    if let Some(cache) = &cache {
                        for &v in
                            &cache.values[cache.offsets[cache_row]..cache.offsets[cache_row + 1]]
                        {
                            accumulate(v);
                        }
                    }
                } else {
                    let mut overflow = false;
                    o.weights(c.grid, o.bounds(g, i), centre, id, |j, _, dp| {
                        let v = std::array::from_fn(|a| c.profiles[model * k + j] * dp[a]);
                        accumulate(v);
                        if iteration == 0
                            && !overflow
                            && let Some(cache) = &mut cache
                        {
                            overflow = !cache.push(v);
                        }
                    });
                    if iteration == 0 {
                        if let Some(cache) = &cache {
                            peak = peak.max(cache.bytes());
                        }
                        if overflow {
                            cache = None;
                        } else if let Some(cache) = &mut cache {
                            cache.offsets.push(cache.values.len());
                        }
                    }
                }
                cache_row += 1;
                moments[0] += p;
                moments[1] += s[0];
                moments[2] += p * p;
                moments[3] += s[0] * s[0];
                moments[4] += p * s[0];
            }
            let inv = if zero {
                [[0.; 3]; 3]
            } else {
                let Some(inv) = inverse(b_inverse) else {
                    flags |= profiles::FAILED;
                    continue;
                };
                inv
            };
            let gp = mv(inv, pu);
            let gd = mv(inv, du);
            num += pd - dot(pu, gd);
            den += pp - dot(pu, gp);
            chi += dd - 2. * intensity * pd + intensity.powi(2) * pp
                - dot(
                    std::array::from_fn(|j| du[j] - intensity * pu[j]),
                    mv(inv, std::array::from_fn(|j| du[j] - intensity * pu[j])),
                );
            for a in 0..3 {
                shape_raw[a] += shape_score[a] + intensity * shape_p[a] - dot(shape_u[a], gd);
                shape_normal[a] += shape_p[a] - dot(shape_u[a], gp);
            }
        }
        if den <= 1e-30 || !den.is_finite() {
            flags |= profiles::NO_PROFILE;
            continue;
        }
        let old = intensity;
        intensity = num / den;
        let sigma = (1. / den).sqrt();
        let dof = count.saturating_sub(1).max(1) as f64;
        let cost =
            (chi + 2. * (old - intensity) * (num - old * den) + (intensity - old).powi(2) * den)
                .max(0.);
        let m = moments;
        let count = count as f64;
        let correlation = (m[4] - m[0] * m[1] / count)
            / ((m[2] - m[0] * m[0] / count).max(0.) * (m[3] - m[1] * m[1] / count).max(0.)).sqrt();
        let mut derivatives = [0.; 4];
        derivatives[0] = 2. * (intensity * den - num);
        for a in 0..3 {
            derivatives[a + 1] = -2. * intensity * (shape_raw[a] - intensity * shape_normal[a]);
        }
        values = [
            intensity,
            sigma,
            c.mass[id],
            cost / dof,
            count,
            correlation,
            derivatives[0],
            derivatives[1],
            derivatives[2],
            derivatives[3],
        ];
    }
    if c.mass[id] < c.minimum_mass {
        flags |= profiles::PARTIAL;
    }
    if values[0] > 5. * values[1] && values[3] > 4. {
        flags |= profiles::MISMATCH;
    }
    let cached = cache.as_ref().is_some_and(|v| !v.values.is_empty());
    Row {
        values,
        flags,
        cache_bytes: peak,
        cached,
    }
}

#[allow(clippy::too_many_arguments)]
fn probability(
    o: &Observations<'_>,
    profiles: &[f64],
    grid: Grid,
    assignments: &[usize],
    shifts: &[[f64; 3]],
    groups: std::ops::Range<usize>,
    base: usize,
    output: &mut [f64],
) {
    let k = grid.size();
    for g in groups {
        let id = o.ids[g];
        let model = assignments[id];
        let centre = o.shifted(id, shifts[model]);
        for i in o.offsets[g]..o.offsets[g + 1] {
            o.weights(grid, o.bounds(g, i), centre, id, |j, w, _| {
                output[i - base] += w * profiles[model * k + j];
            });
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn fit(
    o: &Observations<'_>,
    profiles: &[f64],
    grid: Grid,
    assignments: &[usize],
    shifts: &[[f64; 3]],
    minimum_mass: f64,
    expected_noise: bool,
    gain: f64,
    read_noise: f64,
    execution: Execution,
) -> Result<Fitted, SpotError> {
    execution.validate()?;
    profiles::validate_fit(
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
    let requested = execution.workers;
    let count = requested.min(n.max(1));
    let pool = if count > 1 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(count)
            .build()
            .ok()
    } else {
        None
    };
    let workers = if pool.is_some() { count } else { 1 };
    let mut probabilities = vec![0.; o.samples.len()];
    if let Some(pool) = &pool {
        let chunk = o.ids.len().div_ceil(workers).max(1);
        let mut rest = &mut probabilities[..];
        let mut tasks = Vec::new();
        for first in (0..o.ids.len()).step_by(chunk) {
            let last = (first + chunk).min(o.ids.len());
            let base = o.offsets[first];
            let length = o.offsets[last] - base;
            let (part, tail) = rest.split_at_mut(length);
            rest = tail;
            tasks.push((first..last, base, part));
        }
        pool.install(|| {
            tasks.into_par_iter().for_each(|(groups, base, part)| {
                probability(o, profiles, grid, assignments, shifts, groups, base, part);
            });
        });
    } else {
        probability(
            o,
            profiles,
            grid,
            assignments,
            shifts,
            0..o.ids.len(),
            0,
            &mut probabilities,
        );
    }
    let mut mass = vec![0.; n];
    let mut flags = vec![0_u32; n];
    let mut frames = BTreeMap::<(u64, u64), Vec<usize>>::new();
    for g in 0..o.ids.len() {
        let id = o.ids[g];
        frames
            .entry((o.angles[g][0].to_bits(), o.angles[g][1].to_bits()))
            .or_default()
            .push(g);
        for (i, &p) in probabilities
            .iter()
            .enumerate()
            .take(o.offsets[g + 1])
            .skip(o.offsets[g])
        {
            if p > 0. {
                flags[id] |=
                    o.flags[g] & (crate::spots::BACKGROUND | crate::spots::local_background::MODEL);
                if !o.valid[i] {
                    flags[id] |= o.flags[g] & (crate::spots::MASKED | crate::spots::SATURATED);
                }
            }
            if o.valid[i] {
                mass[id] += p;
            }
        }
    }
    // Only integer overlap flags couple different reflections. Process one exact
    // exposure interval at a time instead of retaining an all-scan pixel hash.
    let mut coverage = HashMap::<(usize, usize), u32>::new();
    let mut peak_coverage = 0;
    for groups in frames.values() {
        coverage.clear();
        for &g in groups {
            let b = o.boxes[g];
            for (i, &p) in probabilities
                .iter()
                .enumerate()
                .take(o.offsets[g + 1])
                .skip(o.offsets[g])
            {
                if p > 0. {
                    let local = i - o.offsets[g];
                    let xy = (b.x0 + local % (b.x1 - b.x0), b.y0 + local / (b.x1 - b.x0));
                    *coverage.entry(xy).or_insert(0) += 1;
                }
            }
        }
        peak_coverage = peak_coverage.max(coverage.len());
        for &g in groups {
            let id = o.ids[g];
            let b = o.boxes[g];
            for (i, &p) in probabilities
                .iter()
                .enumerate()
                .take(o.offsets[g + 1])
                .skip(o.offsets[g])
            {
                if p > 0. {
                    let local = i - o.offsets[g];
                    let xy = (b.x0 + local % (b.x1 - b.x0), b.y0 + local / (b.x1 - b.x0));
                    if coverage[&xy] > 1 {
                        flags[id] |= crate::spots::OVERLAP;
                    }
                }
            }
            if mass[id] < 0.999 {
                flags[id] |= o.flags[g] & (crate::spots::EDGE | 256);
            }
        }
    }
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
    let context = Context {
        o,
        profiles,
        grid,
        assignments,
        shifts,
        minimum_mass,
        expected_noise,
        gain,
        read_noise,
        probability: &probabilities,
        mass: &mass,
        flags: &flags,
    };
    let cap = execution.geometry_cache_bytes / workers;
    let rows: Vec<_> = if let Some(pool) = &pool {
        pool.install(|| {
            (0..n)
                .into_par_iter()
                .map(|id| row(&context, id, &groups[offsets[id]..offsets[id + 1]], cap))
                .collect()
        })
    } else {
        (0..n)
            .map(|id| row(&context, id, &groups[offsets[id]..offsets[id + 1]], cap))
            .collect()
    };
    let maximum = rows.iter().map(|r| r.cache_bytes).max().unwrap_or(0);
    let cached = rows.iter().filter(|r| r.cached).count();
    let stats = FitExecution {
        workers,
        requested_workers: requested,
        cache_budget_bytes: execution.geometry_cache_bytes,
        maximum_reflection_cache_bytes: maximum,
        cache_peak_bound_bytes: maximum * workers,
        cached_reflections: cached,
        uncached_reflections: n - cached,
        maximum_exposure_overlap_pixels: peak_coverage,
    };
    let mut values = Vec::with_capacity(n);
    let mut flags = Vec::with_capacity(n);
    for row in rows {
        values.push(row.values);
        flags.push(row.flags);
    }
    Ok(Fitted {
        values,
        flags,
        execution: stats,
    })
}
