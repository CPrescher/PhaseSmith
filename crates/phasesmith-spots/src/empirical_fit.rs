//! Robust native F(frame)*D(panel)*A(ray) optimization with fixed family labels.
use crate::{SpotError, box_optimizer as optimizer, intensity_model, invalid};
/// Robust scaling controls; no source observation is deleted from output.
#[derive(Clone, Copy)]
pub struct Options {
    /// Nonnegative neutral-prior strength.
    pub strength: f64,
    /// Positive Huber threshold for standardized within-family log residuals.
    pub huber: f64,
    /// Outer IRLS iteration cap, 1..200.
    pub iterations: usize,
    /// Positive physical D/A component lower limit, below one.
    pub minimum_component: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            strength: 1000.,
            huber: 2.5,
            iterations: 100,
            minimum_component: 0.2,
        }
    }
}
/// One actual robust outer iteration.
pub struct Iteration {
    /// Euclidean coefficient change.
    pub parameter_change: f64,
    /// Maximum change in weight divided by its base weight.
    pub weight_change: f64,
    /// Native inner convergence status.
    pub inner_success: bool,
    /// Actual inner objective evaluations.
    pub evaluations: usize,
}
/// Complete signed-row scaling result.
pub struct Fit {
    /// Free coefficients in supplied sector order.
    pub coefficients: Vec<f64>,
    /// Multiplicative correction for every input row, including negative/weak rows.
    pub correction: Vec<f64>,
    /// Raw detector and absorption components, two per row.
    pub components: Vec<f64>,
    /// Declared references retained only in families having at least two members.
    pub references: Vec<bool>,
    /// Both original parameter/weight IRLS criteria reached.
    pub converged: bool,
    /// All actual robust iterations.
    pub history: Vec<Iteration>,
}
/// Fit design, signed intensities and explicit independent reference/family labels.
pub struct Input<'a> {
    /// Existing fused empirical design sectors.
    pub design: intensity_model::Design<'a>,
    /// Row-major neutral-prior matrix with total parameter width.
    pub penalty: &'a [f64],
    /// Signed source intensities; only selected teachers must be positive.
    pub intensity: &'a [f64],
    /// Positive log-intensity variances for every row.
    pub variance: &'a [f64],
    /// Nonnegative family IDs, each below the input row count.
    pub groups: &'a [usize],
    /// Caller-frozen training/reference mask.
    pub reference: &'a [bool],
    /// Closed coefficient lower bounds, strictly negative.
    pub lower: &'a [f64],
    /// Closed coefficient upper bounds, strictly positive.
    pub upper: &'a [f64],
    /// Finite starting coefficients within the declared bounds.
    pub start: &'a [f64],
}
/// Optimize paired references and apply the fitted correction to all signed rows.
/// # Errors
/// Rejects malformed/nonphysical designs, source arrays, bounds or final components.
#[allow(clippy::too_many_lines)]
pub fn fit(input: &Input<'_>, options: Options) -> Result<Fit, SpotError> {
    let d = &input.design;
    let n = d.rows;
    let p: usize = d.widths.iter().sum();
    if p == 0
        || p > 256
        || !input.penalty.len().is_multiple_of(p)
        || input.penalty.len() > 20_000_000
        || [
            input.intensity.len(),
            input.variance.len(),
            input.groups.len(),
            input.reference.len(),
        ]
        .iter()
        .any(|&k| k != n)
        || input.lower.len() != p
        || input.upper.len() != p
        || input.start.len() != p
        || input
            .intensity
            .iter()
            .chain(input.variance)
            .chain(input.penalty)
            .chain(input.lower)
            .chain(input.upper)
            .any(|v| !v.is_finite())
        || input.variance.iter().any(|v| *v <= 0.)
        || input.groups.iter().any(|g| *g >= n)
        || input.lower.iter().any(|v| *v >= 0.)
        || input.upper.iter().any(|v| *v <= 0.)
        || !options.strength.is_finite()
        || options.strength < 0.
        || !options.huber.is_finite()
        || options.huber <= 0.
        || !options.minimum_component.is_finite()
        || options.minimum_component <= 0.
        || options.minimum_component >= 1.
        || options.iterations == 0
        || options.iterations > 200
    {
        return Err(invalid("invalid native empirical scaling inputs"));
    }
    intensity_model::evaluate(d, input.start, 1e-8)?;
    let mut counts = vec![0_usize; n];
    for i in 0..n {
        if input.reference[i] {
            counts[input.groups[i]] += 1;
        }
    }
    let references: Vec<_> = (0..n)
        .map(|i| input.reference[i] && counts[input.groups[i]] >= 2)
        .collect();
    let ids: Vec<_> = (0..n).filter(|&i| references[i]).collect();
    if ids.len() < 2 || ids.iter().any(|&i| input.intensity[i] <= 0.) {
        return Err(invalid("scaling requires positive paired references"));
    }
    let mut compressed = vec![usize::MAX; n];
    let mut next = 0;
    for (i, &count) in counts.iter().enumerate() {
        if count >= 2 {
            compressed[i] = next;
            next += 1;
        }
    }
    let labels: Vec<_> = ids.iter().map(|&i| compressed[input.groups[i]]).collect();
    let target: Vec<_> = ids.iter().map(|&i| input.intensity[i].ln()).collect();
    let base: Vec<_> = ids.iter().map(|&i| 1. / input.variance[i]).collect();
    let mut weights = base.clone();
    let mut b = input.start.to_vec();
    let rootreg = options.strength.sqrt();
    let mut history = Vec::new();
    let mut converged = false;
    for _ in 0..options.iterations {
        let centered_target = intensity_model::center(&target, 1, &labels, &weights)?;
        let solve = optimizer::minimize(
            &b,
            input.lower,
            input.upper,
            optimizer::Options {
                iterations: 80,
                evaluations: 80,
                ftol: 1e-10,
                gtol: 1e-7,
                xtol: 1e-13,
                cost_floor: 0.,
            },
            |parameters| {
                let (values, components, jac) = intensity_model::evaluate(d, parameters, 1e-8)?;
                let values: Vec<_> = ids.iter().map(|&i| values[i]).collect();
                let rows: Vec<_> = ids
                    .iter()
                    .flat_map(|&i| jac[i * p..(i + 1) * p].iter().copied())
                    .collect();
                let values = intensity_model::center(&values, 1, &labels, &weights)?;
                let rows = intensity_model::center(&rows, p, &labels, &weights)?;
                let mut residual = Vec::with_capacity(ids.len() + input.penalty.len() / p + 1);
                let mut full_jac =
                    Vec::with_capacity((ids.len() + input.penalty.len() / p + 1) * p);
                for i in 0..ids.len() {
                    let root = weights[i].sqrt();
                    residual.push(root * (values[i] - centered_target[i]));
                    full_jac.extend(rows[i * p..(i + 1) * p].iter().map(|v| root * v));
                }
                for row in input.penalty.chunks_exact(p) {
                    residual.push(
                        rootreg * row.iter().zip(parameters).map(|(x, b)| x * b).sum::<f64>(),
                    );
                    full_jac.extend(row.iter().map(|v| rootreg * v));
                }
                let (index, &minimum) = components
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.total_cmp(b.1))
                    .ok_or_else(|| invalid("empty empirical design"))?;
                let mut barrier = vec![0.; p];
                if minimum < options.minimum_component {
                    let sector = index % 2 + 1;
                    let offset = if sector == 1 {
                        d.widths[0]
                    } else {
                        d.widths[0] + d.widths[1]
                    };
                    let width = d.widths[sector];
                    let row = index / 2;
                    for j in 0..width {
                        barrier[offset + j] = 1000. * d.matrices[sector][row * width + j];
                    }
                }
                residual.push(1000. * (minimum - options.minimum_component).min(0.));
                full_jac.extend(barrier);
                optimizer::least_squares(&residual, &full_jac, p)
            },
        )?;
        let value = intensity_model::evaluate(d, &solve.parameters, 1e-8)?.0;
        let difference: Vec<_> = ids
            .iter()
            .enumerate()
            .map(|(j, &i)| target[j] - value[i])
            .collect();
        let r = intensity_model::center(&difference, 1, &labels, &weights)?;
        let new_weights: Vec<_> = ids
            .iter()
            .enumerate()
            .map(|(j, &i)| {
                base[j]
                    * (options.huber / (r[j].abs() / input.variance[i].sqrt()).max(1e-30)).min(1.)
            })
            .collect();
        let db = solve
            .parameters
            .iter()
            .zip(&b)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let dw = new_weights
            .iter()
            .zip(&weights)
            .zip(&base)
            .map(|((a, b), w)| (a - b).abs() / w)
            .fold(0., f64::max);
        b = solve.parameters;
        weights = new_weights;
        history.push(Iteration {
            parameter_change: db,
            weight_change: dw,
            inner_success: solve.converged,
            evaluations: solve.evaluations,
        });
        if db < 1e-6 && dw < 1e-5 {
            converged = true;
            break;
        }
    }
    let (values, components, _) = intensity_model::evaluate(d, &b, 1e-8)?;
    let correction: Vec<_> = values.iter().map(|v| (-v).exp()).collect();
    if components
        .iter()
        .any(|v| *v < options.minimum_component - 1e-6)
        || correction.iter().any(|v| !v.is_finite())
    {
        return Err(invalid(
            "empirical correction leaves its declared physical domain",
        ));
    }
    Ok(Fit {
        coefficients: b,
        correction,
        components,
        references,
        converged,
        history,
    })
}
