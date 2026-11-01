use super::measurement::profile_geometry;
use super::{
    Error, Measurements, Options, Result, bounds, grains,
    math::{V3, cross, dot, median, unit},
};
use phasesmith_spots::{
    empirical_fit, intensity_model, profile_refinement as refinement, profiles,
};
use std::collections::BTreeMap;
/// Signed corrected intensities and complete source quality/conditional-fit records.
pub struct Reduced {
    /// Integer Miller indices in the selected supplied setting.
    pub hkl: Vec<[i32; 3]>,
    /// Source-ordered numerical measurement rows; units follow the containing record.
    pub values: Vec<[f64; 2]>,
    /// Conditional counting/background/fitted-shape sigma before equivalent-error inflation.
    pub statistical_sigma: Vec<f64>,
    /// Explicit Lorentz/polarization correction for every retained row.
    pub physical: Vec<f64>,
    /// Fitted within-family multiplicative intensity correction.
    pub empirical: Vec<f64>,
    /// Original quality bits; source rows are retained even when excluded.
    pub flags: Vec<u32>,
    /// Final search selection; finite signed values are retained without an SNR cut.
    pub use_mask: Vec<bool>,
    /// Centre eligibility from the same shared original-image mask.
    pub visible: Vec<bool>,
    /// Ten-column profile amplitude and quality diagnostics.
    pub profiles: Vec<[f64; 10]>,
    /// Fitted x/y/omega shifts and log spatial/rocking width multipliers.
    pub parameters: Vec<[f64; 5]>,
    /// Frozen policy: zero fixed shape, one strong full shape, two strong x/y.
    pub choices: Vec<u8>,
    /// Actual robust within-family fit, including references and convergence history.
    pub scaling: empirical_fit::Fit,
    /// Nonnegative family-balanced equivalent-pair relative observation-error floor.
    pub error_floor: f64,
}
fn harmonics(v: V3) -> [f64; 8] {
    let [x, y, z] = v;
    [
        x,
        y,
        z,
        x * x - y * y,
        2. * z * z - x * x - y * y,
        2. * x * y,
        2. * x * z,
        2. * y * z,
    ]
}
fn rotate(v: V3, axis: V3, angle: f64) -> V3 {
    let c = angle.cos();
    let s = angle.sin();
    let cross = cross(axis, v);
    let d = dot(v, axis);
    std::array::from_fn(|j| c * v[j] + s * cross[j] + (1. - c) * d * axis[j])
}
struct Design {
    frames: Vec<f64>,
    detector: Vec<f64>,
    harmonics: Vec<f64>,
    penalty: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    groups: Vec<usize>,
    training: Vec<bool>,
    variance: Vec<f64>,
    widths: [usize; 3],
}
fn design(
    m: &Measurements,
    values: &[[f64; 2]],
    flags: &[u32],
    g: phasesmith_spots::reciprocal::Geometry,
    o: &Options,
) -> Result<Design> {
    let n = m.hkl.len();
    let nf = bounds(o).len();
    let width = nf - 1;
    let p = width + 24 + 8;
    if p > 256 {
        return Err(Error("empirical frame parameter budget exceeded".into()));
    }
    let mut frames = vec![0.; n * width];
    let mut detector = vec![0.; n * 24];
    let mut rays = vec![0.; n * 8];
    let angles = bounds(o);
    let axis = unit(g.axis);
    let beam = unit(g.beam);
    let [height, image_width] = m.detector_shape;
    for (i, &c) in m.data.centres.iter().enumerate() {
        let frame = angles
            .iter()
            .enumerate()
            .min_by(|a, b| {
                (c[2] - f64::midpoint(a.1[0], a.1[1]))
                    .abs()
                    .total_cmp(&(c[2] - f64::midpoint(b.1[0], b.1[1])).abs())
            })
            .unwrap()
            .0;
        let gauge = nf / 2;
        if frame != gauge {
            frames[i * width + frame - usize::from(frame > gauge)] = 1.;
        }
        let u = [
            c[0] / (image_width - 1) as f64 * 4.,
            c[1] / (height - 1) as f64 * 4.,
        ];
        let low = u.map(|x| (x.floor() as usize).min(3));
        let frac = [u[0] - low[0] as f64, u[1] - low[1] as f64];
        for dy in 0..2 {
            for dx in 0..2 {
                let node = (low[1] + dy) * 5 + low[0] + dx;
                if node != 12 {
                    let value = (if dx == 1 { frac[0] } else { 1. - frac[0] })
                        * (if dy == 1 { frac[1] } else { 1. - frac[1] });
                    detector[i * 24 + node - usize::from(node > 12)] = value;
                }
            }
        }
        let ray = profile_geometry([c[0], c[1]], g).0;
        let angle = -c[2].to_radians();
        let a = harmonics(rotate(ray, axis, angle));
        let b = harmonics(rotate(beam.map(|x| -x), axis, angle));
        for j in 0..8 {
            rays[i * 8 + j] = a[j] + b[j];
        }
    }
    let penalty = penalty_matrix(width, nf, &rays, n)?;
    let FamilyDesign {
        groups,
        mut training,
        variance,
    } = family_design(
        &m.hkl,
        values,
        flags,
        o.priors
            .group
            .as_ref()
            .ok_or_else(|| Error("scaling needs an assessed Laue hypothesis".into()))?,
    );
    if o.discovery.is_some() {
        let partition = o.partition_group();
        for (keep, &h) in training.iter_mut().zip(&m.hkl) {
            *keep &= grains::fold(grains::family(h.map(i64::from), &partition)) != 0;
        }
    }
    let mut lower = vec![-2f64.ln(); width];
    lower.extend(vec![-0.5; 24]);
    lower.extend(vec![-1.; 8]);
    let mut upper = vec![2f64.ln(); width];
    upper.extend(vec![1.; 24]);
    upper.extend(vec![1.; 8]);
    Ok(Design {
        frames,
        detector,
        harmonics: rays,
        penalty,
        lower,
        upper,
        groups,
        training,
        variance,
        widths: [width, 24, 8],
    })
}
fn error_floor(
    values: &mut [[f64; 2]],
    groups: &[usize],
    reference: &[bool],
    training: &[bool],
) -> Result<f64> {
    let mut families: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, v) in values.iter().enumerate() {
        if v.iter().all(|x| x.is_finite()) && v[1] > 0. {
            families.entry(groups[i]).or_default().push(i);
        }
    }
    let mut pairs = Vec::new();
    let mut family_mu = vec![0.; values.len()];
    for members in families.values() {
        let mu = median(&members.iter().map(|&i| values[i][0]).collect::<Vec<_>>()).max(0.);
        for &i in members {
            family_mu[i] = mu;
        }
        let ids: Vec<_> = members
            .iter()
            .copied()
            .filter(|&i| reference[i] && training[i])
            .collect();
        let size = ids.len() * (ids.len().saturating_sub(1)) / 2;
        if pairs.len() + size > 1_000_000 {
            return Err(Error("equivalent pair budget exceeded".into()));
        }
        if size > 0 {
            for a in 0..ids.len() {
                for b in a + 1..ids.len() {
                    let i = ids[a];
                    let j = ids[b];
                    pairs.push((
                        (values[i][0] - values[j][0]).abs(),
                        values[i][1].powi(2) + values[j][1].powi(2),
                        mu,
                        1. / size as f64,
                    ));
                }
            }
        }
    }
    if pairs.is_empty() {
        return Err(Error("no paired equivalent training references".into()));
    }
    let total = pairs.iter().map(|x| x.3).sum::<f64>();
    let width = |beta: f64| {
        let mut residual: Vec<_> = pairs
            .iter()
            .enumerate()
            .map(|(i, p)| (p.0 / (p.1 + 2. * (beta * p.2).powi(2)).sqrt(), i))
            .collect();
        residual.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut sum = 0.;
        for (r, i) in residual {
            sum += pairs[i].3;
            if sum >= 0.5 * total {
                return r / 0.674_489_750_196_081_7;
            }
        }
        f64::INFINITY
    };
    if width(0.5) > 1. {
        return Err(Error(
            "equivalent error exceeds declared relative bound".into(),
        ));
    }
    let (mut lo, mut hi) = (0., 0.5);
    if width(0.) <= 1. {
        hi = 0.;
    } else {
        for _ in 0..60 {
            let mid = f64::midpoint(lo, hi);
            if width(mid) > 1. {
                lo = mid;
            } else {
                hi = mid;
            }
        }
    }
    let beta = f64::midpoint(lo, hi);
    for (i, v) in values.iter_mut().enumerate() {
        if v.iter().all(|x| x.is_finite()) && v[1] > 0. {
            v[1] = v[1].hypot(beta * family_mu[i]);
        }
    }
    Ok(beta)
}
pub(super) fn reduce(
    m: &Measurements,
    o: &Options,
    g: &phasesmith_spots::reciprocal::Geometry,
) -> Result<Reduced> {
    let n = m.hkl.len();
    let (fitted, choices) = fit_profiles_native(m, o)?;
    let physical = physical_factors(m, g)?;
    let mut values: Vec<_> = fitted
        .values
        .iter()
        .zip(&physical)
        .map(|(v, &f)| [v[0] * f, v[1] * f])
        .collect();
    let d = design(m, &values, &fitted.flags, *g, o)?;
    let y: Vec<_> = values
        .iter()
        .map(|v| if v[0].is_finite() { v[0] } else { 0. })
        .collect();
    let fit = empirical_fit::fit(
        &empirical_fit::Input {
            design: intensity_model::Design {
                rows: n,
                widths: d.widths,
                matrices: [&d.frames, &d.detector, &d.harmonics],
            },
            penalty: &d.penalty,
            intensity: &y,
            variance: &d.variance,
            groups: &d.groups,
            reference: &d.training,
            lower: &d.lower,
            upper: &d.upper,
            start: &vec![0.; d.lower.len()],
        },
        empirical_options_v1(),
    )?;
    if !fit.converged {
        return Err(Error(
            "empirical scaling did not converge; retain unscaled observations".into(),
        ));
    }
    for (v, &f) in values.iter_mut().zip(&fit.correction) {
        v[0] *= f;
        v[1] *= f;
    }
    let statistical_sigma = values.iter().map(|v| v[1]).collect();
    let mut trained = vec![false; n];
    for i in 0..n {
        if d.training[i] {
            trained[d.groups[i]] = true;
        }
    }
    let train: Vec<_> = d.groups.iter().map(|&g| trained[g]).collect();
    let beta = error_floor(&mut values, &d.groups, &fit.references, &train)?;
    let use_mask: Vec<_> = values
        .iter()
        .zip(&fitted.flags)
        .zip(&m.visible)
        .map(|((v, &f), &visible)| {
            (f & !131_072) == 0 && visible && v.iter().all(|x| x.is_finite()) && v[1] > 0.
        })
        .collect();
    if use_mask.iter().filter(|&&v| v).count() < 1000 {
        return Err(Error(
            "insufficient quality-controlled signed observations".into(),
        ));
    }
    Ok(Reduced {
        hkl: m.hkl.clone(),
        values,
        statistical_sigma,
        physical,
        empirical: fit.correction.clone(),
        flags: fitted.flags,
        use_mask,
        visible: m.visible.clone(),
        profiles: fitted.values,
        parameters: fitted.parameters,
        choices,
        scaling: fit,
        error_floor: beta,
    })
}

pub(super) fn physical_factors(
    m: &Measurements,
    g: &phasesmith_spots::reciprocal::Geometry,
) -> Result<Vec<f64>> {
    let beam = unit(g.beam);
    let axis = unit(g.axis);
    let e = [0., 1., 0.];
    if dot(e, beam).abs() > 1e-8 {
        return Err(Error(
            "polarization vector must be perpendicular to beam".into(),
        ));
    }
    let orth = cross(beam, e);
    let physical: Vec<_> = m
        .data
        .centres
        .iter()
        .map(|c| {
            let ray = profile_geometry([c[0], c[1]], *g).0;
            let p = 0.98 * (1. - dot(ray, e).powi(2)) + (1. - 0.98) * (1. - dot(ray, orth).powi(2));
            dot(cross(ray, beam), axis).abs() / p
        })
        .collect();
    if physical.iter().any(|v| !v.is_finite() || *v <= 0.) {
        return Err(Error("singular physical intensity correction".into()));
    }
    Ok(physical)
}

fn penalty_matrix(width: usize, nf: usize, rays: &[f64], n: usize) -> Result<Vec<f64>> {
    let p = width + 24 + 8;
    let mut penalty = vec![];
    for i in 0..width {
        let mut r = vec![0.; p];
        r[i] = 1. / 0.25;
        penalty.extend(r);
    }
    let mut full = vec![vec![0.; p]; nf];
    for (i, row) in full.iter_mut().enumerate() {
        if i != nf / 2 {
            row[i - usize::from(i > nf / 2)] = 1.;
        }
    }
    for i in 0..nf - 2 {
        penalty.extend((0..p).map(|j| (full[i + 2][j] - 2. * full[i + 1][j] + full[i][j]) / 0.10));
    }
    for i in 0..24 {
        let mut r = vec![0.; p];
        r[width + i] = 1. / 0.15;
        penalty.extend(r);
    }
    let svd = nalgebra::DMatrix::from_fn(n, 8, |i, j| rays[i * 8 + j] / (n as f64).sqrt())
        .svd(false, true);
    let vt = svd
        .v_t
        .ok_or_else(|| Error("ray-design SVD failed".into()))?;
    for i in 0..8 {
        let mut r = vec![0.; p];
        for j in 0..8 {
            r[width + 24 + j] = svd.singular_values[i] * vt[(i, j)] / 0.25;
        }
        penalty.extend(r);
    }
    Ok(penalty)
}

fn fit_profiles_native(m: &Measurements, o: &Options) -> Result<(refinement::Fit, Vec<u8>)> {
    let n = m.hkl.len();
    let borrowed = m.data.borrowed();
    let mut base = profiles::fit_with_execution(
        &borrowed,
        &m.bank.profiles,
        m.grid,
        &m.assignments,
        &m.bank.shifts,
        0.9,
        true,
        1.,
        0.,
        profiles::Execution {
            workers: o.profile_workers,
            geometry_cache_bytes: o.geometry_cache_bytes,
        },
    )?;
    for (flag, &source) in base.flags.iter_mut().zip(&m.flags) {
        *flag |= source & 1024;
    }
    let choices: Vec<u8> = m
        .aperture
        .iter()
        .zip(&m.flags)
        .map(|(a, &f)| u8::from(f == 0 && a[1] > 0. && a[0] > 50. * a[1]))
        .collect();
    let weak = vec![false; n];
    let offsets = vec![[0.; 2]; n];
    let input = refinement::Input {
        observations: borrowed,
        profiles: &m.bank.profiles,
        grid: m.grid,
        assignments: &m.assignments,
        shifts: &m.bank.shifts,
        minimum_mass: 0.9,
        noise: refinement::Noise {
            expected: true,
            gain: 1.,
            read: 0.,
        },
        choices: &choices,
        weak: &weak,
        sensor_offsets: &offsets,
        frozen: None,
        baseline: Some(refinement::Baseline {
            values: &base.values,
            flags: &base.flags,
        }),
    };
    let fitted = refinement::refine(
        &input,
        refinement::Options {
            workers: o.profile_workers,
            iterations: 3,
            maximum_iterations: 80,
            centre_pixels: 1.,
            centre_degrees: 0.4,
            spatial_range: [0.65, 1.5],
            rocking_range: [0.5, 1.5],
            regularizer: [0.01, 0.01, 0.16, 0.04, 0.04],
            weak_design_counts: 1000.,
            weak_spatial_width: 0.95,
        },
        |_, _| o.checkpoint().map_err(|e| phasesmith_spots::SpotError(e.0)),
    )?;
    Ok((fitted, choices))
}

struct FamilyDesign {
    groups: Vec<usize>,
    training: Vec<bool>,
    variance: Vec<f64>,
}
fn family_design(
    hkl: &[[i32; 3]],
    values: &[[f64; 2]],
    flags: &[u32],
    group: &phasesmith_crystallography::SpaceGroup,
) -> FamilyDesign {
    let n = hkl.len();
    let canonical: Vec<_> = hkl
        .iter()
        .map(|&h| {
            let h = h.map(i64::from);
            group
                .rotations()
                .iter()
                .map(|r| {
                    std::array::from_fn::<_, 3, _>(|j| {
                        (0..3).map(|k| h[k] * i64::from(r[j][k])).sum()
                    })
                })
                .min()
                .expect("rotations")
        })
        .collect();
    let mut families: BTreeMap<_, usize> = canonical.iter().copied().map(|h| (h, 0)).collect();
    for (i, id) in families.values_mut().enumerate() {
        *id = i;
    }
    let groups: Vec<_> = canonical.iter().map(|h| families[h]).collect();
    let eligible: Vec<_> = values
        .iter()
        .zip(flags)
        .map(|(v, &f)| f == 0 && v[0].is_finite() && v[1] > 0. && v[0] > 10. * v[1])
        .collect();
    let mut training: Vec<_> = eligible
        .iter()
        .zip(&canonical)
        .map(|(&e, &h)| e && grains::fold(h) != 0)
        .collect();
    let mut count = vec![0usize; n];
    for i in 0..n {
        if training[i] {
            count[groups[i]] += 1;
        }
    }
    for i in 0..n {
        training[i] &= count[groups[i]] >= 2;
    }
    let variance = values
        .iter()
        .zip(&eligible)
        .map(|(v, &e)| {
            if e {
                (v[1] / v[0]).powi(2) + 0.02f64.powi(2)
            } else {
                1.
            }
        })
        .collect();
    FamilyDesign {
        groups,
        training,
        variance,
    }
}

fn empirical_options_v1() -> empirical_fit::Options {
    empirical_fit::Options {
        strength: 1000.,
        huber: 2.5,
        iterations: 100,
        minimum_component: 0.2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[allow(clippy::float_cmp)] // Unchanged integer-valued source counts are exact.
    fn equivalent_error_recovers_analytic_paired_width_without_changing_signed_intensity() {
        let mut values = vec![[90., 1.], [110., 1.], [-5., 1.], [-5., 1.]];
        let beta = error_floor(
            &mut values,
            &[0, 0, 1, 1],
            &[true, true, false, false],
            &[true; 4],
        )
        .unwrap();
        let expected =
            ((20f64 / 0.674_489_750_196_081_7).powi(2) - 2.).sqrt() / (2f64.sqrt() * 100.);
        assert!((beta - expected).abs() < 1e-14);
        assert_eq!(
            values.iter().map(|x| x[0]).collect::<Vec<_>>(),
            vec![90., 110., -5., -5.]
        );
        assert_eq!(values[2][1], 1.);
    }
    #[test]
    fn ray_svd_penalty_preserves_the_complete_physical_harmonic_gram() {
        let n = 24;
        let rays: Vec<_> = (0..n)
            .flat_map(|i| {
                let a = i as f64 * 0.3;
                harmonics([a.cos() * 0.8, a.sin() * 0.8, 0.6])
            })
            .collect();
        let nf = 8;
        let width = nf - 1;
        let p = width + 32;
        let penalty = penalty_matrix(width, nf, &rays, n).unwrap();
        let last = &penalty[penalty.len() - 8 * p..];
        for i in 0..8 {
            for j in 0..8 {
                let got = (0..8)
                    .map(|k| last[k * p + width + 24 + i] * last[k * p + width + 24 + j])
                    .sum::<f64>();
                let expected = (0..n)
                    .map(|k| rays[k * 8 + i] * rays[k * 8 + j])
                    .sum::<f64>()
                    / (n as f64 * 0.25f64.powi(2));
                assert!((got - expected).abs() < 1e-12);
            }
        }
    }
}
