use super::{
    Error, Options, Positions, Result, SpaceGroup,
    math::{M3, V3, condition, det, dot, inv_t, least, mul, norm, row},
    positions,
};
use phasesmith_spots::{
    fft::{Engine, SupportBits},
    geometry_refinement as geometry,
};
/// Position-supported competing lattice, with independently withheld ownership.
pub struct Grain {
    /// Right-handed direct row basis in angstroms.
    pub direct: M3,
    /// Integer Miller indices in the selected supplied setting.
    pub hkl: Vec<[i64; 3]>,
    /// Frozen lattice, geometry and withheld-data evidence.
    pub indexed: Vec<bool>,
    /// Frozen independent training membership or retained training count.
    pub training: usize,
    /// Independent withheld position count.
    pub withheld: usize,
    /// Whether every frozen training, withheld and rotated-null gate passed.
    pub accepted: bool,
    /// Retained ownership counts for the 64 predeclared null rotations.
    pub null_counts: Vec<usize>,
}
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
fn valid(a: M3) -> bool {
    let l = a.map(norm);
    a.iter().flatten().all(|x| x.is_finite())
        && l.iter().all(|&x| (1.5..=30.).contains(&x))
        && (15.0..=20000.).contains(&det(a).abs())
        && condition(a) <= 20.
}
fn reduce(mut a: M3) -> M3 {
    for _ in 0..100 {
        a.sort_by(|x, y| dot(*x, *x).total_cmp(&dot(*y, *y)));
        let mut changed = false;
        for j in 1..3 {
            for i in (0..j).rev() {
                let k = (dot(a[i], a[j]) / dot(a[i], a[i])).round_ties_even();
                let t = std::array::from_fn(|n| a[j][n] - k * a[i][n]);
                if dot(t, t) < dot(a[j], a[j]) - 1e-10 {
                    a[j] = t;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    if det(a) < 0. {
        a[2] = a[2].map(|x| -x);
    }
    a
}
fn score(e: &Engine, q: &[V3], a: M3) -> Result<Vec<(V3, f64)>> {
    Ok(e.score(q, a, inv_t(a)?))
}
#[allow(clippy::float_cmp)] // Integer HKLs and exact zero/signed determinant sentinels.
fn mask(s: &[(V3, f64)]) -> Vec<bool> {
    s.iter().map(|(h, d)| *d < 0.015 && *h != [0.; 3]).collect()
}
fn assign(e: &Engine, q: &[V3], a: &[M3]) -> Result<Vec<i32>> {
    let models = a
        .iter()
        .map(|&a| Ok((a, inv_t(a)?)))
        .collect::<Result<Vec<_>>>()?;
    Ok(e.assign(q, &models, 0.015, 0.8).0)
}
#[allow(clippy::float_cmp)] // Integer HKLs and exact zero/signed determinant sentinels.
fn integer_basis(indices: &[V3]) -> Option<M3> {
    let mut basis: [Option<[i128; 3]>; 3] = [None; 3];
    for h in indices {
        if h.iter()
            .any(|x| !x.is_finite() || x.abs() > 1_000_000. || *x != x.round_ties_even())
        {
            return None;
        }
        let mut row = h.map(|x| x as i128);
        for col in 0..3 {
            while row[col] != 0 {
                if let Some(pivot) = basis[col] {
                    let k = row[col].div_euclid(pivot[col]);
                    for j in 0..3 {
                        row[j] = row[j].checked_sub(k.checked_mul(pivot[j])?)?;
                    }
                    if row[col] != 0 {
                        if row[col] < 0 {
                            row = row.map(|x| -x);
                        }
                        basis[col] = Some(row);
                        row = pivot;
                    }
                } else {
                    basis[col] = Some(if row[col] > 0 { row } else { row.map(|x| -x) });
                    row = [0; 3];
                    break;
                }
            }
        }
    }
    if basis
        .iter()
        .flatten()
        .flatten()
        .any(|x| x.abs() > 9_007_199_254_740_992)
    {
        return None;
    }
    Some([
        basis[0]?.map(|x| x as f64),
        basis[1]?.map(|x| x as f64),
        basis[2]?.map(|x| x as f64),
    ])
}
fn refine(e: &Engine, q: &[V3], mut a: M3) -> Result<Option<M3>> {
    for cycle in 0..8 {
        let s = score(e, q, a)?;
        let mut h = Vec::new();
        let mut points = Vec::new();
        for (&q, (indices, d)) in q.iter().zip(s) {
            if d < if cycle < 2 { 0.03 } else { 0.015 } && indices != [0.; 3] {
                h.push(indices);
                points.push(q);
            }
        }
        if h.len() < 6 {
            return Ok(None);
        }
        let Ok(b) = least(&h, &points) else {
            return Ok(None);
        };
        if det(b).abs() < 1e-12 {
            return Ok(None);
        }
        let updated = inv_t(b)?;
        let delta = a
            .iter()
            .flatten()
            .zip(updated.iter().flatten())
            .map(|(x, y)| (x - y).abs())
            .fold(0., f64::max);
        a = updated;
        if cycle >= 2 && delta < 1e-8 {
            break;
        }
    }
    a = reduce(a);
    if !valid(a) {
        return Ok(None);
    }
    let s = score(e, q, a)?;
    let selected = mask(&s);
    let mut ids: Vec<_> = (0..q.len()).filter(|&i| selected[i]).collect();
    ids.sort_by(|&i, &j| s[i].1.total_cmp(&s[j].1));
    let mut best = a;
    for size in [24, 48, 96] {
        if ids.len() < size {
            continue;
        }
        let hs: Vec<_> = ids[..size].iter().map(|&i| s[i].0).collect();
        if let Some(h) = integer_basis(&hs) {
            if det(h).round().abs() <= 1. {
                continue;
            }
            let t = reduce(inv_t(mul(h, inv_t(a)?))?);
            if !valid(t) {
                continue;
            }
            let m = mask(&score(e, q, t)?);
            if (0..q.len()).filter(|&i| m[i] && selected[i]).count() as f64
                >= 0.9 * selected.iter().filter(|&&v| v).count() as f64
                && det(t).abs() < det(best).abs()
            {
                best = t;
            }
        }
    }
    Ok(Some(best))
}
#[allow(clippy::float_cmp)] // Integer HKLs and exact zero/signed determinant sentinels.
fn joint(e: &Engine, q: &[V3], mut models: Vec<M3>) -> Result<(Vec<M3>, Vec<i32>)> {
    let mut labels = assign(e, q, &models)?;
    for _ in 0..8 {
        let mut updated = Vec::new();
        let mut stable = true;
        for (i, &a) in models.iter().enumerate() {
            let block: Vec<_> = q
                .iter()
                .zip(&labels)
                .filter_map(|(&q, &l)| (l == i as i32 + 1).then_some(q))
                .collect();
            let h: Vec<_> = score(e, &block, a)?.into_iter().map(|x| x.0).collect();
            let mut b = a;
            if block.len() >= 6
                && let Ok(fit) = least(&h, &block)
                && det(fit).abs() > 1e-12
            {
                let t = inv_t(fit)?;
                if valid(t) {
                    b = t;
                }
            }
            stable &= block.iter().zip(h).all(|(&q, h)| {
                std::array::from_fn::<_, 3, _>(|j| dot(q, b[j]).round_ties_even()) == h
            });
            updated.push(b);
        }
        let next = assign(e, q, &updated)?;
        let changed = labels != next;
        models = updated;
        labels = next;
        if !changed && stable {
            break;
        }
    }
    Ok((models, labels))
}
fn vectors(e: &Engine, q: &[V3]) -> Result<Vec<V3>> {
    let max = q.iter().flatten().map(|x| x.abs()).fold(0., f64::max);
    let step = (2. * max / 252.).min(1. / (2.1 * 30.));
    let (initial, _, _) = e
        .windowed_maxima(q, 256, step, 1.5, 30., 192)
        .map_err(Error)?;
    let radius = 1.5 / (256. * step);
    let refined = e.refine_fourier(q, &initial, radius)?;
    let v: Vec<_> = refined.iter().map(|r| r.0).collect();
    let m: Vec<_> = refined.iter().map(|r| r.1).collect();
    let mut order: Vec<_> = (0..v.len()).collect();
    order.sort_by(|&i, &j| m[j][0].total_cmp(&m[i][0]));
    let mut chosen: Vec<V3> = Vec::new();
    for i in order {
        let n = norm(v[i]);
        if !(1.5..=30.).contains(&n)
            || chosen
                .iter()
                .any(|&x| dot(v[i], x).abs() / (n * norm(x)) > 0.9995)
        {
            continue;
        }
        chosen.push(v[i]);
        if chosen.len() == 64 {
            break;
        }
    }
    Ok(chosen)
}
fn discover(q: &[V3], o: &Options) -> Result<Vec<M3>> {
    let e = Engine::new(4).map_err(Error)?;
    let mut models = Vec::new();
    let mut labels = vec![0; q.len()];
    for iteration in 0..5 {
        o.checkpoint()?;
        let mut explained = vec![false; q.len()];
        for &a in &models {
            for (x, m) in explained.iter_mut().zip(mask(&score(&e, q, a)?)) {
                *x |= m;
            }
        }
        let block: Vec<_> = q
            .iter()
            .enumerate()
            .filter_map(|(i, &q)| (labels[i] == 0 && !explained[i]).then_some(q))
            .collect();
        if block.len() < 20 {
            break;
        }
        let v = vectors(&e, &block)?;
        let rows = v
            .iter()
            .map(|&a| {
                let mut bits = vec![0u64; block.len().div_ceil(64)];
                for (i, &q) in block.iter().enumerate() {
                    let p = dot(q, a);
                    if (p - p.round_ties_even()).abs() < 0.15 {
                        bits[i / 64] |= 1 << (i % 64);
                    }
                }
                bits
            })
            .collect();
        let bits = SupportBits {
            rows,
            count: block.len(),
            pool: e.pool.clone(),
        };
        let keys: Vec<_> = bits
            .candidate_keys(96)
            .into_iter()
            .filter(|k| {
                let a = k.map(|i| v[i]);
                (15.0..=20000.).contains(&det(a).abs())
            })
            .collect();
        let volumes: Vec<_> = keys.iter().map(|k| det(k.map(|i| v[i])).abs()).collect();
        let (order, _) = bits.rank_triplets(&keys, &volumes);
        let mut candidates = Vec::new();
        let mut tried = 0;
        for i in order {
            let a = keys[i].map(|i| v[i]);
            if condition(a) > 20. {
                continue;
            }
            tried += 1;
            if tried > 32 {
                break;
            }
            let Some(a) = refine(&e, &block, a)? else {
                continue;
            };
            let s = score(&e, &block, a)?;
            let selected = mask(&s);
            let count = selected.iter().filter(|&&v| v).count();
            let own: Vec<_> = block
                .iter()
                .zip(&selected)
                .filter_map(|(&q, &ok)| ok.then_some(q))
                .collect();
            let mut duplicate = own.is_empty();
            for &old in &models {
                let n = mask(&score(&e, &own, old)?).iter().filter(|&&x| x).count();
                duplicate |= n as f64 >= 0.8 * own.len() as f64;
            }
            if count < 20 || duplicate {
                continue;
            }
            let volume = det(a).abs();
            let excess = count as f64
                - block.len() as f64 * volume * 4. * std::f64::consts::PI / 3. * 0.015f64.powi(3);
            let rms = (s
                .iter()
                .zip(&selected)
                .filter_map(|(s, &ok)| ok.then_some(s.1 * s.1))
                .sum::<f64>()
                / count as f64)
                .sqrt();
            candidates.push((-excess, rms, volume, tried, a));
        }
        candidates.sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then(a.1.total_cmp(&b.1))
                .then(a.2.total_cmp(&b.2))
                .then(a.3.cmp(&b.3))
        });
        let Some(best) = candidates.first() else {
            break;
        };
        models.push(best.4);
        (models, labels) = joint(&e, q, models)?;
        eprintln!(
            "native grain discovery {}/5: {} lattices",
            iteration + 1,
            models.len()
        );
    }
    Ok(models)
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
                    fn gcd(a: i32, b: i32) -> i32 {
                        if b == 0 { a.abs() } else { gcd(b, a % b) }
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
        .zip(&training)
        .filter_map(|(&q, &t)| t.then_some(q))
        .collect();
    if train.len() < 20 || q.len() - train.len() < 6 {
        return Err(Error("insufficient independent indexing tracks".into()));
    }
    let models = discover(&train, o)?;
    let e = Engine::new(4).map_err(Error)?;
    let labels = assign(&e, &q, &models)?;
    let held: Vec<_> = q
        .iter()
        .zip(&training)
        .filter_map(|(&q, &t)| (!t).then_some(q))
        .collect();
    let mut null = vec![vec![]; models.len()];
    for &rotation in &o.null_rotations {
        let rotated: Vec<_> = held.iter().map(|&q| row(q, rotation)).collect();
        let l = assign(&e, &rotated, &models)?;
        for (i, n) in null.iter_mut().enumerate() {
            n.push(l.iter().filter(|&&x| x == i as i32 + 1).count());
        }
    }
    let mut grains = Vec::new();
    for (i, &direct) in models.iter().enumerate() {
        let s = score(&e, &q, direct)?;
        let indexed: Vec<_> = labels.iter().map(|&l| l == i as i32 + 1).collect();
        let ntrain = indexed
            .iter()
            .zip(&training)
            .filter(|(a, b)| **a && **b)
            .count();
        let ntest = indexed
            .iter()
            .zip(&training)
            .filter(|(a, b)| **a && !**b)
            .count();
        let mean = null[i].iter().sum::<usize>() as f64 / 64.;
        let var = null[i]
            .iter()
            .map(|&v| (v as f64 - mean).powi(2))
            .sum::<f64>()
            / 63.;
        let accepted = ntrain >= 20
            && ntest >= 6
            && ntest > *null[i].iter().max().unwrap()
            && ntest as f64 >= 2. * mean.max(1.)
            && ntest as f64 >= mean + 4. * var.max(mean).max(1.).sqrt()
            && ntest as f64 >= 0.5 * ntrain as f64 * held.len() as f64 / train.len() as f64;
        grains.push(Grain {
            direct,
            hkl: s.iter().map(|s| s.0.map(|x| x as i64)).collect(),
            indexed,
            training: ntrain,
            withheld: ntest,
            accepted,
            null_counts: null[i].clone(),
        });
    }
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
    let fit_training: Vec<_> = training
        .iter()
        .zip(&hkl)
        .map(|(&t, &h)| t && fold(family(h, &o.priors.group)) != 0)
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
        geometry::Options::default(),
        |_, _| o.checkpoint().map_err(|e| phasesmith_spots::SpotError(e.0)),
    )?;
    let valid: Vec<_> = eligible
        .iter()
        .zip(&fit.predictions)
        .map(|(&x, p)| x && p.flags == 0)
        .collect();
    let heldout: Vec<_> = (0..hkl.len())
        .filter(|&i| valid[i] && !fit_training[i])
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
        .zip(&fit_training)
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
    let updated = positions::reciprocal(p, fit.state.geometry)?.0;
    let updated: Vec<_> = ids.iter().map(|&i| updated[i]).collect();
    let mut competitors = Vec::new();
    for (i, g) in grains.iter().enumerate() {
        if i == selected {
            continue;
        }
        let owntrain = (0..q.len())
            .filter(|&j| g.indexed[j] && training[j])
            .count();
        let overlap = (0..q.len())
            .filter(|&j| g.indexed[j] && eligible[j] && training[j])
            .count();
        if overlap as f64 > 0.2 * owntrain.max(1) as f64 {
            continue;
        }
        let train: Vec<_> = (0..q.len())
            .filter(|&j| g.indexed[j] && training[j])
            .collect();
        let test: Vec<_> = (0..q.len())
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
    Ok(Indexed {
        grains,
        selected,
        track_ids: ids,
        hkl,
        observed,
        training: fit_training,
        eligible: valid,
        fit,
        competitors,
    })
}

fn gcd(a: i32, b: i32) -> i32 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}
