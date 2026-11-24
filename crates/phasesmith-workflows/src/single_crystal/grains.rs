use super::{
    Error, Options, Positions, Result, SpaceGroup,
    math::{M3, V3, det, dot, inv_t, least, norm, row},
    positions,
};
pub use phasesmith_single_crystal::grains::Grain;
use phasesmith_single_crystal::grains::{self as search, reduce};
use phasesmith_single_crystal::spots::{SpotError, geometry_refinement as geometry};
/// Frozen discovery/geometry products; competitors are measured crossings only.
pub struct Indexed {
    /// All position-supported lattice hypotheses, including failed validations.
    pub grains: Vec<Grain>,
    /// Index of the training-supported primitive setting matching the explicit prior.
    pub selected: usize,
    /// Selected original track indices in source order.
    pub track_ids: Vec<usize>,
    /// Integer Miller indices in the selected supplied setting.
    pub hkl: Vec<[i64; 3]>,
    /// Observed x/y/omega centroids, in pixels/degrees.
    pub observed: Vec<V3>,
    /// Frozen independent training membership or retained training count.
    pub training: Vec<bool>,
    /// Frozen positional eligibility intersected with valid final predictions.
    pub eligible: Vec<bool>,
    /// Actual native geometry fit, including history and convergence status.
    pub fit: geometry::Fit,
    /// Independently position-supported measured crossings for exclusion only.
    pub competitors: Vec<Competitor>,
}
/// A position-supported alternative used only through its measured crossings.
pub struct Competitor {
    /// Original lattice hypothesis index.
    pub candidate: usize,
    /// Fitted reciprocal row basis in inverse angstroms.
    pub basis: M3,
    /// Integer Miller indices in the selected supplied setting.
    pub hkl: Vec<[i64; 3]>,
    /// Measured x/y/omega crossings; no speculative full competing lattice.
    pub crossings: Vec<V3>,
    /// Frozen independent training membership or retained training count.
    pub training: usize,
    /// Independent withheld position count.
    pub withheld: usize,
    /// Withheld reciprocal residual RMS in inverse angstroms.
    pub rms: f64,
}
#[allow(clippy::float_cmp)] // Integer HKLs and exact zero/signed determinant sentinels.
fn setting(models: &[Grain], target: [f64; 6]) -> Result<(usize, M3)> {
    let mut choices = Vec::new();
    for (id, g) in models.iter().enumerate() {
        if g.training < 100 {
            continue;
        }
        let a = reduce(g.direct);
        let mut axes = Vec::new();
        for x in -2i32..=2 {
            for y in -2i32..=2 {
                for z in -2i32..=2 {
                    let v = [x, y, z];
                    if v == [0; 3] || v.iter().find(|&&x| x != 0).is_some_and(|&x| x < 0) {
                        continue;
                    }
                    if gcd(gcd(x, y), z) != 1 {
                        continue;
                    }
                    let v = v.map(f64::from);
                    axes.push((row(v, a), v));
                }
            }
        }
        axes.sort_by(|a, b| norm(a.0).total_cmp(&norm(b.0)));
        let tolerance = 1f64.to_radians().sin();
        for i in 0..axes.len() {
            for j in i + 1..axes.len() {
                if dot(axes[i].0, axes[j].0).abs() > tolerance * norm(axes[i].0) * norm(axes[j].0) {
                    continue;
                }
                for k in j + 1..axes.len() {
                    if dot(axes[i].0, axes[k].0).abs()
                        > tolerance * norm(axes[i].0) * norm(axes[k].0)
                        || dot(axes[j].0, axes[k].0).abs()
                            > tolerance * norm(axes[j].0) * norm(axes[k].0)
                    {
                        continue;
                    }
                    let t = [axes[i].1, axes[j].1, axes[k].1];
                    if det(t).round().abs() != 1. {
                        continue;
                    }
                    let b = [axes[i].0, axes[j].0, axes[k].0];
                    for p in [
                        [0, 1, 2],
                        [0, 2, 1],
                        [1, 0, 2],
                        [1, 2, 0],
                        [2, 0, 1],
                        [2, 1, 0],
                    ] {
                        let mut b = p.map(|i| b[i]);
                        if det(b) < 0. {
                            b[2] = b[2].map(|x| -x);
                        }
                        let r = std::array::from_fn::<_, 3, _>(|i| norm(b[i]) / target[i] - 1.);
                        if r.iter().all(|x| x.abs() <= 0.03) {
                            choices.push((dot(r, r), usize::MAX - g.training, id, b));
                        }
                    }
                }
            }
        }
    }
    choices.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    choices.first().map(|x| (x.2, x.3)).ok_or_else(|| {
        Error("no training-supported primitive setting matches the declared cell prior".into())
    })
}
pub(super) fn family(h: [i64; 3], group: &SpaceGroup) -> [i64; 3] {
    group
        .rotations()
        .iter()
        .flat_map(|r| {
            let t: [i64; 3] =
                std::array::from_fn(|j| (0..3).map(|k| h[k] * i64::from(r[k][j])).sum());
            [t, t.map(|x| -x)]
        })
        .min()
        .expect("space group rotations")
}
pub(super) fn fold(h: [i64; 3]) -> usize {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    for x in h {
        hash.update(x.to_le_bytes());
    }
    let b = hash.finalize();
    u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize % 5
}
pub(super) fn index(p: &Positions, o: &Options, shape: [usize; 2]) -> Result<Indexed> {
    let (q, rms) = positions::reciprocal(p, o.geometry)?;
    if q.len() > 100_000
        || q.iter()
            .flatten()
            .any(|x| !x.is_finite() || x.abs() > 1000.)
    {
        return Err(Error("indexing point/count bound exceeded".into()));
    }
    let first = o.acquisition.first;
    let ids: Vec<_> = (0..p.values.len())
        .filter(|&i| {
            let [a, b] = p.ranges[i];
            p.flags[i] == 0
                && p.values[i][4] <= 8.
                && rms[i] <= 0.03
                && ((first + a) / 5) % 5 == ((first + b - 1) / 5) % 5
        })
        .collect();
    let q: Vec<_> = ids.iter().map(|&i| q[i]).collect();
    let training: Vec<_> = ids
        .iter()
        .map(|&i| !((p.ranges[i][0] + first) / 5).is_multiple_of(5))
        .collect();
    let train: Vec<_> = q
        .iter()
        .zip(training.iter())
        .filter_map(|(&q, &t)| t.then_some(q))
        .collect();
    if train.len() < 20 || q.len() - train.len() < 6 {
        return Err(Error("insufficient independent indexing tracks".into()));
    }
    let grains = discover_and_validate(&q, &training, &train, o)?;
    let cell = o.priors.cell;
    let target = [
        cell.a_angstrom,
        cell.b_angstrom,
        cell.c_angstrom,
        cell.alpha_deg,
        cell.beta_deg,
        cell.gamma_deg,
    ];
    let (selected, a) = setting(&grains, target)?;
    let basis = inv_t(a)?;
    let hkl: Vec<_> = q
        .iter()
        .map(|&q| a.map(|a| dot(q, a).round_ties_even() as i64))
        .collect();
    let eligible: Vec<_> = q
        .iter()
        .enumerate()
        .map(|(i, &q)| {
            let predicted = row(hkl[i].map(|h| h as f64), basis);
            grains[selected].indexed[i]
                && norm(std::array::from_fn(|j| q[j] - predicted[j])) < 0.015
                && hkl[i] != [0; 3]
        })
        .collect();
    let partition = o.partition_group();
    let fit_training: Vec<_> = training
        .iter()
        .zip(&hkl)
        .map(|(&t, &h)| t && fold(family(h, &partition)) != 0)
        .collect();
    let observed: Vec<_> = ids
        .iter()
        .map(|&i| [p.values[i][0], p.values[i][1], p.values[i][2]])
        .collect();
    let fit = geometry::fit(
        &geometry::Input {
            observed: &observed,
            hkl: &hkl,
            training: &fit_training,
            eligible: &eligible,
            basis,
            geometry: o.geometry,
            image_shape: shape,
            reference_cell: Some(target),
            refine_lengths: false,
        },
        geometry_options_v1(),
        |_, _| {
            o.checkpoint()
                .map_err(|e| phasesmith_single_crystal::spots::SpotError(e.0))
        },
    )?;
    let valid = validate_geometry(&fit, &observed, &fit_training, &eligible)?;
    let mut result = Indexed {
        grains,
        selected,
        track_ids: ids,
        hkl,
        observed,
        training: fit_training,
        eligible: valid,
        fit,
        competitors: vec![],
    };
    result.competitors = competing_domains(p, &result, &training, &eligible)?;
    Ok(result)
}

fn gcd(a: i32, b: i32) -> i32 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

fn discover_and_validate(
    q: &[V3],
    training: &[bool],
    train: &[V3],
    o: &Options,
) -> Result<Vec<Grain>> {
    let options = search::Options::default();
    let models = search::discover(
        train,
        &options,
        &o.null_rotations,
        &search::Method::Fft,
        &mut |_, _| o.checkpoint().map_err(|e| SpotError(e.0)),
    )?;
    Ok(search::validate(
        q,
        training,
        &models,
        &o.null_rotations,
        &options,
    )?)
}

fn validate_geometry(
    fit: &geometry::Fit,
    observed: &[V3],
    training: &[bool],
    eligible: &[bool],
) -> Result<Vec<bool>> {
    let valid: Vec<_> = eligible
        .iter()
        .zip(&fit.predictions)
        .map(|(&x, p)| x && p.flags == 0)
        .collect();
    let heldout: Vec<_> = (0..observed.len())
        .filter(|&i| valid[i] && !training[i])
        .collect();
    let xy = (heldout
        .iter()
        .map(|&i| {
            let p = fit.predictions[i].values;
            (p[0] - observed[i][0]).powi(2) + (p[1] - observed[i][1]).powi(2)
        })
        .sum::<f64>()
        / heldout.len() as f64)
        .sqrt();
    let omega = (heldout
        .iter()
        .map(|&i| (fit.predictions[i].values[2] - observed[i][2]).powi(2))
        .sum::<f64>()
        / heldout.len() as f64)
        .sqrt();
    if valid
        .iter()
        .zip(training.iter())
        .filter(|(x, t)| **x && **t)
        .count()
        < 100
        || heldout.len() < 30
        || xy > 1.
        || omega > 0.3
        || !fit.converged
        || fit.at_bounds
    {
        return Err(Error(format!(
            "geometry gates failed: heldout={} xy={xy} omega={omega} converged={} bounds={}",
            heldout.len(),
            fit.converged,
            fit.at_bounds
        )));
    }
    Ok(valid)
}

fn competing_domains(
    p: &Positions,
    r: &Indexed,
    training: &[bool],
    eligible: &[bool],
) -> Result<Vec<Competitor>> {
    let fit = &r.fit;
    let ids = &r.track_ids;
    let grains = &r.grains;
    let selected = r.selected;
    let observed = &r.observed;
    let updated = positions::reciprocal(p, fit.state.geometry)?.0;
    let updated: Vec<_> = ids.iter().map(|&i| updated[i]).collect();
    let mut competitors = Vec::new();
    for (i, g) in grains.iter().enumerate() {
        if i == selected {
            continue;
        }
        let owntrain = (0..updated.len())
            .filter(|&j| g.indexed[j] && training[j])
            .count();
        let overlap = (0..updated.len())
            .filter(|&j| g.indexed[j] && eligible[j] && training[j])
            .count();
        if overlap as f64 > 0.2 * owntrain.max(1) as f64 {
            continue;
        }
        let train: Vec<_> = (0..updated.len())
            .filter(|&j| g.indexed[j] && training[j])
            .collect();
        let test: Vec<_> = (0..updated.len())
            .filter(|&j| g.indexed[j] && !training[j])
            .collect();
        if train.len() < 30 || test.len() < 10 {
            continue;
        }
        let h: Vec<_> = train.iter().map(|&j| g.hkl[j].map(|x| x as f64)).collect();
        let q: Vec<_> = train.iter().map(|&j| updated[j]).collect();
        let Ok(b) = least(&h, &q) else { continue };
        let rms = (test
            .iter()
            .map(|&j| {
                let p = row(g.hkl[j].map(|x| x as f64), b);
                dot(
                    std::array::from_fn(|k| p[k] - updated[j][k]),
                    std::array::from_fn(|k| p[k] - updated[j][k]),
                )
            })
            .sum::<f64>()
            / test.len() as f64)
            .sqrt();
        if rms <= 0.015 {
            let only: Vec<_> = (0..updated.len())
                .filter(|&j| g.indexed[j] && !eligible[j])
                .collect();
            competitors.push(Competitor {
                candidate: i,
                basis: b,
                hkl: only.iter().map(|&j| g.hkl[j]).collect(),
                crossings: only.iter().map(|&j| observed[j]).collect(),
                training: train.len(),
                withheld: test.len(),
                rms,
            });
        }
    }
    competitors.sort_by_key(|g| std::cmp::Reverse(g.training));
    competitors.truncate(2);
    Ok(competitors)
}

fn geometry_options_v1() -> geometry::Options {
    geometry::Options {
        xy_scale: 0.4,
        omega_scale: 0.2,
        huber: 2.5,
        max_evaluations: 150,
        translation_limit_mm: 2.,
        rotation_limit_deg: 2.,
        log_length_limit: 0.02,
        initial_xy_gate: 5.,
        initial_omega_gate: 1.,
    }
}

#[cfg(test)]
#[path = "grains_benchmark.rs"]
mod benchmark;
