//! Data-derived rocking width and spot size.
use super::{
    Lattice, Result, Scan,
    geometry::ray_geometry,
    integrate::{FrameReader, MeasureOptions, Measurement, Monitor, measure},
    invalid, median,
    predict::{PredictOptions, predict},
};
use crate::spots::{reciprocal::Geometry, rotation};

/// Rocking width and spot size estimated from the data.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct ShapeEstimate {
    /// Standard deviation of the rocking curve for `|ζ| = 1` (the crystal's
    /// mosaic spread and beam divergence), degrees.
    pub rocking_sigma: f64,
    /// Suggested [`PredictOptions::rocking_width`]: `6 σ` (± 3 σ), at least
    /// one scan step, degrees.
    pub rocking_width: f64,
    /// Median second-moment width of strong spots on the detector, x and y,
    /// pixels (`None` when not estimated).
    pub spot_sigma: Option<[f64; 2]>,
    /// Suggested odd [`MeasureOptions::box_size`] holding ± 4 spot σ
    /// (at least 5), when the spot size was estimated.
    pub box_size: Option<usize>,
    /// Reflections or tracks the rocking estimate rests on.
    pub reflections: usize,
}

/// Fewest reflections or tracks for an estimate.
const MINIMUM: usize = 5;
/// Most reflections used for the rocking fit.
const MOST: usize = 2000;

fn suggested(sigma: f64, scan: &Scan) -> f64 {
    (6. * sigma).max(scan.step)
}

/// Standard normal cumulative distribution (erfc approximation with
/// fractional error below 1.2·10⁻⁷, Numerical Recipes `erfcc`).
pub(super) fn phi(x: f64) -> f64 {
    let z = (x / std::f64::consts::SQRT_2).abs();
    let t = 1. / (1. + 0.5 * z);
    let poly = -z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let erfc = t * poly.exp();
    if x >= 0. { 1. - erfc / 2. } else { erfc / 2. }
}

struct Curve {
    /// Exposure intervals, observed fractions and their variances.
    frames: Vec<([f64; 2], f64, f64)>,
    omega: f64,
    zeta: f64,
}

/// Rocking width from the per-frame sums of strong reflections, and spot
/// size from their second moments.
///
/// Reflections without flags, with summed I/σ ≥ 10, `|ζ| ≥ 0.2` and at
/// least two measured frames (the extra profile frames count) are used, the
/// strongest 2000 at most. Each frame's share of the summed intensity is
/// compared with a Gaussian rocking curve of standard deviation `σ / |ζ|`
/// integrated over the frame's exposure, centred within half a step of the
/// predicted ω (fitted per reflection); one `σ` minimises the total robust
/// χ² over a logarithmic grid from 0.01 to 10 steps, refined by a parabola.
/// Widths well below the step are resolved only through reflections that
/// straddle frame boundaries, which is why many reflections are pooled.
///
/// Measure with a generous window ([`estimate_shape`] does this) so that the
/// tails are inside it.
/// # Errors
/// Fewer than five usable reflections.
#[allow(clippy::too_many_lines)]
pub fn estimate_shape_from_partials(m: &Measurement, scan: &Scan) -> Result<ShapeEstimate> {
    scan.check()?;
    let mut candidates: Vec<(f64, Curve, Option<[f64; 2]>)> = Vec::new();
    for r in &m.reflections {
        let s = r.summation;
        let zeta = r.reflection.zeta;
        let measured: Vec<_> = r
            .partials
            .iter()
            .filter(|p| p.intensity.is_finite() && p.sigma.is_finite() && p.sigma > 0.)
            .collect();
        if r.flags != 0
            || !s.is_measured()
            || s.value < 10. * s.sigma
            || zeta.abs() < 0.2
            || measured.len() < 2
            || measured.iter().any(|p| p.flags != 0)
        {
            continue;
        }
        let total: f64 = measured.iter().map(|p| p.intensity).sum();
        if total <= 0. {
            continue;
        }
        let frames = measured
            .iter()
            .map(|p| {
                let f = p.intensity / total;
                let v = (p.sigma / total).powi(2) + (0.02 * f).powi(2) + 1e-6;
                (scan.bounds(p.frame), f, v)
            })
            .collect();
        let (mut w, mut sx, mut sy) = (0., 0., 0.);
        for p in &measured {
            if p.intensity > 0. && p.spread.iter().all(|v| v.is_finite()) {
                w += p.intensity;
                sx += p.intensity * p.spread[0];
                sy += p.intensity * p.spread[1];
            }
        }
        let spot = (s.value >= 30. * s.sigma && w > 0.).then(|| [sx / w, sy / w]);
        candidates.push((
            s.value / s.sigma,
            Curve {
                frames,
                omega: r.reflection.xyz[2],
                zeta,
            },
            spot,
        ));
    }
    if candidates.len() < MINIMUM {
        return Err(invalid(
            "too few strong unflagged reflections to estimate the rocking width",
        ));
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    candidates.truncate(MOST);
    let sigmas: Vec<f64> = (0..=60)
        .map(|k| scan.step * 10f64.powf(-2. + 3. * f64::from(k) / 60.))
        .collect();
    let totals: Vec<f64> = sigmas
        .iter()
        .map(|&sigma| {
            candidates
                .iter()
                .map(|(_, c, _)| {
                    let s = sigma / c.zeta.abs();
                    (0..=20)
                        .map(|j| {
                            let centre = c.omega + scan.step * (f64::from(j) / 20. - 0.5);
                            let model: Vec<f64> = c
                                .frames
                                .iter()
                                .map(|(b, _, _)| {
                                    phi((b[1] - centre) / s) - phi((b[0] - centre) / s)
                                })
                                .collect();
                            let sum: f64 = model.iter().sum();
                            if sum <= 1e-12 {
                                return 25. * c.frames.len() as f64;
                            }
                            c.frames
                                .iter()
                                .zip(&model)
                                .map(|((_, f, v), m)| ((f - m / sum).powi(2) / v).min(25.))
                                .sum::<f64>()
                        })
                        .fold(f64::INFINITY, f64::min)
                })
                .sum()
        })
        .collect();
    let best = (0..totals.len())
        .min_by(|&a, &b| totals[a].total_cmp(&totals[b]))
        .unwrap_or(0);
    let mut log_sigma = sigmas[best].ln();
    if best > 0 && best + 1 < totals.len() {
        let (a, b, c) = (totals[best - 1], totals[best], totals[best + 1]);
        let curvature = a - 2. * b + c;
        if curvature > 0. {
            let h = (sigmas[best + 1].ln() - sigmas[best - 1].ln()) / 2.;
            log_sigma += h * 0.5 * (a - c) / curvature;
        }
    }
    let sigma = log_sigma.exp();
    let spots: Vec<[f64; 2]> = candidates.iter().filter_map(|c| c.2).collect();
    let spot_sigma = (spots.len() >= MINIMUM).then(|| {
        [
            median(spots.iter().map(|s| s[0]).collect()).unwrap_or(f64::NAN),
            median(spots.iter().map(|s| s[1]).collect()).unwrap_or(f64::NAN),
        ]
    });
    let box_size = spot_sigma.map(|s| {
        let half = (4. * s[0].max(s[1])).ceil().max(2.) as usize;
        2 * half + 1
    });
    Ok(ShapeEstimate {
        rocking_sigma: sigma,
        rocking_width: suggested(sigma, scan),
        spot_sigma,
        box_size,
        reflections: candidates.len(),
    })
}

/// Rocking width from linked spot tracks: each track's samples are
/// `[x, y, ω, weight]` (ω the exposure centre of the frame the spot was
/// found on, weight its intensity). Per track, the weighted ω spread less the
/// exposure's own `exposure² / 12`, times `|ζ|`, is the rocking σ; the
/// median over tracks with `|ζ| ≥ 0.2` is reported. Tracks seen on one frame
/// only count as zero width, so when most spots fit within one frame the
/// estimate floors at the step (prefer [`estimate_shape`] then). No spot
/// size.
/// # Errors
/// Fewer than five usable tracks.
pub fn estimate_rocking_from_tracks(
    tracks: &[Vec<[f64; 4]>],
    geometry: &Geometry,
    scan: &Scan,
) -> Result<ShapeEstimate> {
    scan.check()?;
    let spreads: Vec<f64> = tracks
        .iter()
        .filter_map(|t| {
            let s = rotation::summarize(t).ok()?;
            if !s[..4].iter().all(|v| v.is_finite()) {
                return None;
            }
            let zeta = ray_geometry([s[0], s[1]], geometry).zeta;
            if !zeta.is_finite() || zeta.abs() < 0.2 {
                return None;
            }
            let own = (s[3].powi(2) - scan.exposure.powi(2) / 12.).max(0.);
            Some(own.sqrt() * zeta.abs())
        })
        .collect();
    if spreads.len() < MINIMUM {
        return Err(invalid(
            "too few usable tracks to estimate the rocking width",
        ));
    }
    let count = spreads.len();
    let sigma = median(spreads).unwrap_or(0.);
    Ok(ShapeEstimate {
        rocking_sigma: sigma,
        rocking_width: suggested(sigma, scan),
        spot_sigma: None,
        box_size: None,
        reflections: count,
    })
}

/// Rocking width and spot size from the frames: predicts the lattices'
/// reflections to `d_min` with a generous window (eight steps or 2°,
/// whichever is wider), measures them by summation with `options` (no
/// profiles) and applies [`estimate_shape_from_partials`]. Costs one pass
/// over the frames.
/// # Errors
/// As [`super::predict`], [`measure`] and [`estimate_shape_from_partials`].
#[allow(clippy::too_many_arguments)]
pub fn estimate_shape(
    frames: &impl FrameReader,
    geometry: &Geometry,
    scan: &Scan,
    shape: [usize; 2],
    lattices: &[Lattice],
    d_min: f64,
    options: &MeasureOptions,
    monitor: &impl Monitor,
) -> Result<ShapeEstimate> {
    let probe = PredictOptions::new(d_min, (8. * scan.step).max(2.));
    let reflections = predict(lattices, geometry, scan, shape, &probe, None)?;
    let mut summation = options.clone();
    summation.profile = None;
    let m = measure(
        frames,
        geometry,
        scan,
        shape,
        &reflections,
        &summation,
        monitor,
    )?;
    estimate_shape_from_partials(&m, scan)
}
