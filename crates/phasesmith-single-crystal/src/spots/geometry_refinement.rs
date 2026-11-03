//! Native indexed-spot geometry calibration with fused analytical derivatives.
//!
//! The objective is f² sum(sqrt(1+(r/f)²)-1), with scaled x/y/omega residuals.
//! Detector pose, crystal Euler rotations and optional log lengths are local
//! closed-box variables. No intensity, atomic model or HKL reassignment enters.
use crate::spots::fft::{M3, V3, dot, inverse_transpose};
use crate::spots::geometry_linear::{cholesky, eigen, solve};
use crate::spots::{SpotError, invalid, prediction, reciprocal::Geometry};

/// Inverse pixel mapping and exact image/plane Jacobian at one predicted point.
#[derive(Clone, Copy)]
pub struct MappedPoint {
    /// Image x/y coordinates corresponding to a plane pixel prediction.
    pub image: [f64; 2],
    /// d(image xy)/d(plane xy), row-major.
    pub jacobian: [[f64; 2]; 2],
}
/// Caller-owned pixel coordinate mapping. Native applications implement this in
/// Rust; an optional language binding may bridge an application's custom map.
/// The geometry optimizer itself never requires an interpreter.
pub trait PixelMap {
    /// Bypass all mapping work for the established affine-panel path.
    fn is_identity(&self) -> bool {
        false
    }
    /// Map a batch of valid plane coordinates, preserving order and cardinality.
    /// # Errors
    /// Reports an invalid/nonunique inverse or unavailable map support.
    fn inverse(&self, plane: &[[f64; 2]]) -> Result<Vec<MappedPoint>, SpotError>;
}
/// Identity mapping for ordinary affine panels.
pub struct IdentityMap;
impl PixelMap for IdentityMap {
    fn is_identity(&self) -> bool {
        true
    }
    fn inverse(&self, plane: &[[f64; 2]]) -> Result<Vec<MappedPoint>, SpotError> {
        Ok(plane
            .iter()
            .map(|&image| MappedPoint {
                image,
                jacobian: [[1., 0.], [0., 1.]],
            })
            .collect())
    }
}
fn map_data<'a>(
    records: impl Iterator<Item = &'a prediction::Prediction>,
    mapper: &dyn PixelMap,
) -> Result<Vec<(usize, MappedPoint)>, SpotError> {
    if mapper.is_identity() {
        return Ok(Vec::new());
    }
    let mut ids = Vec::new();
    let mut plane = Vec::new();
    for (i, p) in records.enumerate() {
        if p.flags == 0 {
            ids.push(i);
            plane.push([p.values[0], p.values[1]]);
        }
    }
    let coordinates = mapper.inverse(&plane)?;
    if coordinates.len() != ids.len()
        || coordinates.iter().any(|m| {
            m.image
                .iter()
                .chain(m.jacobian.iter().flatten())
                .any(|v| !v.is_finite())
        })
    {
        return Err(invalid(
            "pixel inverse map/Jacobian must be finite and match the input batch",
        ));
    }
    Ok(ids.into_iter().zip(coordinates).collect())
}
fn apply_map(
    p: &mut prediction::Prediction,
    detector: Option<&mut [f64; 27]>,
    mapped: MappedPoint,
) {
    p.values[..2].copy_from_slice(&mapped.image);
    let transform = |j: &mut [f64; 27]| {
        for k in 0..9 {
            let x = j[k];
            let y = j[9 + k];
            j[k] = mapped.jacobian[0][0] * x + mapped.jacobian[0][1] * y;
            j[9 + k] = mapped.jacobian[1][0] * x + mapped.jacobian[1][1] * y;
        }
    };
    transform(&mut p.jacobian);
    if let Some(d) = detector {
        transform(d);
    }
}
fn map_predictions(
    records: &mut [prediction::Prediction],
    mapper: &dyn PixelMap,
) -> Result<(), SpotError> {
    for (i, mapped) in map_data(records.iter(), mapper)? {
        apply_map(&mut records[i], None, mapped);
    }
    Ok(())
}

/// Rz*Ry*Rx and its three exact angle partials (radians).
#[must_use]
pub fn rotation(angles: V3) -> (M3, [M3; 3]) {
    let (sx, cx) = angles[0].sin_cos();
    let (sy, cy) = angles[1].sin_cos();
    let (sz, cz) = angles[2].sin_cos();
    let rx = [[1., 0., 0.], [0., cx, -sx], [0., sx, cx]];
    let ry = [[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]];
    let rz = [[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]];
    let dx = [[0., 0., 0.], [0., -sx, -cx], [0., cx, -sx]];
    let dy = [[-sy, 0., cy], [0., 0., 0.], [-cy, 0., -sy]];
    let dz = [[-sz, -cz, 0.], [cz, -sz, 0.], [0., 0., 0.]];
    (
        multiply(multiply(rz, ry), rx),
        [
            multiply(multiply(rz, ry), dx),
            multiply(multiply(rz, dy), rx),
            multiply(multiply(dz, ry), rx),
        ],
    )
}
fn transpose(a: M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}
fn multiply(a: M3, b: M3) -> M3 {
    let b = transpose(b);
    a.map(|r| b.map(|c| dot(r, c)))
}
fn vector(a: M3, b: V3) -> V3 {
    a.map(|r| dot(r, b))
}
fn finite_basis(b: M3) -> Result<(), SpotError> {
    if b.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite reciprocal basis"));
    }
    let a = inverse_transpose(b).map_err(|e| invalid(&e))?;
    if a.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid("singular reciprocal basis"));
    }
    Ok(())
}

/// Replace the metric while preserving the canonical direct-cell orientation.
/// # Errors
/// Rejects a non-right-handed starting cell or invalid reference metric.
#[allow(clippy::needless_range_loop)] // Forward triangular solve depends on earlier rows.
pub fn reference_basis(basis: M3, cell: [f64; 6]) -> Result<M3, SpotError> {
    finite_basis(basis)?;
    if cell.iter().any(|v| !v.is_finite())
        || cell[..3].iter().any(|v| *v <= 0.)
        || cell[3..].iter().any(|v| *v <= 0. || *v >= 180.)
    {
        return Err(invalid("invalid reference cell lengths/angles"));
    }
    let direct = inverse_transpose(basis).map_err(|e| invalid(&e))?;
    let cross = [
        direct[1][1] * direct[2][2] - direct[1][2] * direct[2][1],
        direct[1][2] * direct[2][0] - direct[1][0] * direct[2][2],
        direct[1][0] * direct[2][1] - direct[1][1] * direct[2][0],
    ];
    if dot(direct[0], cross) <= 0. {
        return Err(invalid(
            "reference orientation requires a right-handed basis",
        ));
    }
    let gram = multiply(direct, transpose(direct));
    let lower = cholesky(&gram.concat(), 3)?;
    let mut canonical = direct;
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..i {
                canonical[i][j] -= lower[i * 3 + k] * canonical[k][j];
            }
            canonical[i][j] /= lower[i * 3 + i];
        }
    }
    let mut gram = [[0.; 3]; 3];
    for i in 0..3 {
        gram[i][i] = cell[i] * cell[i];
    }
    for (i, j, k) in [(0, 1, 5), (0, 2, 4), (1, 2, 3)] {
        gram[i][j] = cell[i] * cell[j] * cell[k].to_radians().cos();
        gram[j][i] = gram[i][j];
    }
    let lower = cholesky(&gram.concat(), 3)
        .map_err(|_| invalid("reference cell has nonpositive volume"))?;
    let l = std::array::from_fn(|i| std::array::from_fn(|j| lower[i * 3 + j]));
    inverse_transpose(multiply(l, canonical)).map_err(|e| invalid(&e))
}

/// Transformed state and row-major physical-vector parameter chains.
pub struct State {
    /// Reciprocal basis with row vectors (1/A).
    pub basis: M3,
    /// Rigid detector pose; beam/axis/wavelength unchanged.
    pub geometry: Geometry,
    /// Nine basis components by parameter count, row-major.
    pub basis_jacobian: Vec<f64>,
    /// Origin/fast/slow components by parameter count, row-major.
    pub detector_jacobian: Vec<f64>,
}
/// Evaluate physical values and exact Euler, translation and log-length chains.
/// # Errors
/// Rejects nonfinite parameters, invalid basis/geometry or overflow.
pub fn parameter_state(
    parameters: &[f64],
    basis: M3,
    geometry: Geometry,
    pivot: V3,
    lengths: bool,
) -> Result<State, SpotError> {
    let n = if lengths { 12 } else { 9 };
    if parameters.len() != n || parameters.iter().chain(&pivot).any(|v| !v.is_finite()) {
        return Err(invalid("invalid geometry parameter vector/pivot"));
    }
    finite_basis(basis)?;
    geometry.convert(&[])?;
    let (rc, dc) = rotation([parameters[0], parameters[1], parameters[2]]);
    let (rd, dd) = rotation([parameters[6], parameters[7], parameters[8]]);
    let mut b0 = basis;
    if lengths {
        for i in 0..3 {
            b0[i] = b0[i].map(|v| v * parameters[9 + i].exp());
        }
    }
    let b = multiply(b0, transpose(rc));
    let relative = std::array::from_fn(|j| geometry.origin[j] - pivot[j]);
    let shifted = vector(rd, relative);
    let mut g = geometry;
    g.origin = std::array::from_fn(|j| pivot[j] + 0.001 * parameters[3 + j] + shifted[j]);
    g.fast = vector(rd, geometry.fast);
    g.slow = vector(rd, geometry.slow);
    let mut jb = vec![0.; 9 * n];
    let mut jg = vec![0.; 9 * n];
    for k in 0..3 {
        let db = multiply(b0, transpose(dc[k]));
        for i in 0..9 {
            jb[i * n + k] = db[i / 3][i % 3];
        }
        jg[k * n + 3 + k] = 0.001;
        for (i, derivative) in [
            vector(dd[k], relative),
            vector(dd[k], geometry.fast),
            vector(dd[k], geometry.slow),
        ]
        .iter()
        .enumerate()
        {
            for j in 0..3 {
                jg[(i * 3 + j) * n + 6 + k] = derivative[j];
            }
        }
        if lengths {
            for j in 0..3 {
                jb[(k * 3 + j) * n + 9 + k] = b[k][j];
            }
        }
    }
    if b.iter()
        .flatten()
        .chain(&g.origin)
        .chain(&jb)
        .chain(&jg)
        .any(|v| !v.is_finite())
    {
        return Err(invalid("geometry state overflow"));
    }
    Ok(State {
        basis: b,
        geometry: g,
        basis_jacobian: jb,
        detector_jacobian: jg,
    })
}

/// Fixed loss scales, initial gates, local parameter bounds and evaluation budget.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// x/y loss scale in pixels (not a calibrated uncertainty).
    pub xy_scale: f64,
    /// Omega loss scale in degrees.
    pub omega_scale: f64,
    /// Positive soft-L1 scale in standardized residual units.
    pub huber: f64,
    /// Total model evaluation budget, 1..1000, including rejected trials.
    pub max_evaluations: usize,
    /// Translation limit in laboratory millimetres, at most 100.
    pub translation_limit_mm: f64,
    /// Crystal/detector rotation limits in degrees, at most 10.
    pub rotation_limit_deg: f64,
    /// Optional log-length limit, at most .2.
    pub log_length_limit: f64,
    /// Strict initial radial x/y gate in pixels.
    pub initial_xy_gate: f64,
    /// Strict initial omega gate in degrees.
    pub initial_omega_gate: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
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
}
impl Options {
    fn validate(self) -> Result<(), SpotError> {
        if [
            self.xy_scale,
            self.omega_scale,
            self.huber,
            self.translation_limit_mm,
            self.rotation_limit_deg,
            self.log_length_limit,
            self.initial_xy_gate,
            self.initial_omega_gate,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.)
            || self.max_evaluations == 0
            || self.max_evaluations > 1000
            || self.translation_limit_mm > 100.
            || self.rotation_limit_deg > 10.
            || self.log_length_limit > 0.2
        {
            return Err(invalid("invalid geometry fitting options"));
        }
        Ok(())
    }
}
/// Immutable indexed observations and independent training/eligibility decisions.
pub struct Input<'a> {
    /// Measured x/y pixels and unwrapped omega degrees.
    pub observed: &'a [V3],
    /// Fixed indexed reflections, never reassigned by the fitter.
    pub hkl: &'a [[i64; 3]],
    /// Only selected eligible training observations teach the fit.
    pub training: &'a [bool],
    /// Caller quality mask, intersected with initial strict crossing gates.
    pub eligible: &'a [bool],
    /// Initial reciprocal row basis.
    pub basis: M3,
    /// Initial affine detector geometry.
    pub geometry: Geometry,
    /// Detector height and width; determines the detector rotation pivot.
    pub image_shape: [usize; 2],
    /// Optional declared cell metric in A/degrees.
    pub reference_cell: Option<[f64; 6]>,
    /// Refine three row lengths while preserving initial cell angles.
    pub refine_lengths: bool,
}
/// Native optimization and complete withheld diagnostics.
pub struct Fit {
    /// Actual local parameters in the documented 9/12 order.
    pub parameters: Vec<f64>,
    /// Final physical state and parameter chains.
    pub state: State,
    /// Original predictions, before metric replacement or any fitting.
    pub initial: Vec<prediction::Prediction>,
    /// Final predictions for every source observation.
    pub predictions: Vec<prediction::Prediction>,
    /// Frozen caller mask intersected with initial strict gates.
    pub eligible: Vec<bool>,
    /// Convergence criterion reached inside evaluation budget.
    pub converged: bool,
    /// Any local parameter is within 1e-11*max(1,abs(bound)) of a closed bound.
    pub at_bounds: bool,
    /// Scaled robust-Jacobian condition, infinity if unidentified.
    pub scaled_condition: f64,
    /// Largest off-diagonal conditional parameter correlation.
    pub maximum_parameter_correlation: f64,
    /// Conditional pseudoinverse of the final robust normal matrix.
    pub covariance: Vec<f64>,
    /// Actual objective evaluations including rejected/invalid trials.
    pub evaluations: usize,
    /// Accepted monotonically decreasing soft-L1 costs, including the start.
    pub history: Vec<f64>,
    /// Training and validation; each holds before/after RMS(xy,omega,scaled).
    pub rms: [[[f64; 3]; 2]; 2],
    /// Eligible training and validation counts.
    pub counts: [usize; 2],
}

struct Model {
    cost: f64,
    gradient: Vec<f64>,
    normal: Vec<f64>,
}
/// Fused robust objective/gradient/normal evaluation for explicitly selected rows.
/// # Errors
/// Rejects any lost crossing or invalid source arrays.
pub fn objective(
    input: &Input<'_>,
    base: M3,
    pivot: V3,
    selected: &[usize],
    parameters: &[f64],
    options: Options,
) -> Result<(f64, Vec<f64>, Vec<f64>), SpotError> {
    options.validate()?;
    let count = input.hkl.len();
    if input.observed.len() != count || selected.iter().any(|&i| i >= count) {
        return Err(invalid("invalid geometry objective rows"));
    }
    let m = evaluate(
        input,
        base,
        pivot,
        selected,
        parameters,
        options,
        &IdentityMap,
    )?;
    Ok((m.cost, m.gradient, m.normal))
}
fn evaluate(
    input: &Input<'_>,
    base: M3,
    pivot: V3,
    selected: &[usize],
    parameters: &[f64],
    options: Options,
    mapper: &dyn PixelMap,
) -> Result<Model, SpotError> {
    let state = parameter_state(
        parameters,
        base,
        input.geometry,
        pivot,
        input.refine_lengths,
    )?;
    let n = parameters.len();
    let h: Vec<_> = selected.iter().map(|&i| input.hkl[i]).collect();
    let hints: Vec<_> = selected.iter().map(|&i| input.observed[i][2]).collect();
    let mut predicted =
        prediction::predict_with_detector(state.geometry, state.basis, &h, &hints, false)?;
    for (i, mapped) in map_data(predicted.iter().map(|p| &p.prediction), mapper)? {
        let record = &mut predicted[i];
        apply_map(
            &mut record.prediction,
            Some(&mut record.detector_jacobian),
            mapped,
        );
    }
    let mut model = Model {
        cost: 0.,
        gradient: vec![0.; n],
        normal: vec![0.; n * n],
    };
    for (&id, p) in selected.iter().zip(predicted) {
        if p.prediction.flags != 0 {
            return Err(invalid("trial geometry loses an indexed crossing"));
        }
        for (row, scale) in [options.xy_scale, options.xy_scale, options.omega_scale]
            .into_iter()
            .enumerate()
        {
            let r = (p.prediction.values[row] - input.observed[id][row]) / scale;
            let z = (r / options.huber).powi(2);
            let root = (1. + z).sqrt();
            // Rationalized rho avoids cancellation close to an exact solution.
            model.cost += r * r / (root + 1.);
            let weight = 1. / root;
            let curvature = weight / (1. + z);
            let mut j = [0.; 12];
            for (k, value) in j.iter_mut().enumerate().take(n) {
                for c in 0..9 {
                    *value += p.prediction.jacobian[row * 9 + c] * state.basis_jacobian[c * n + k]
                        + p.detector_jacobian[row * 9 + c] * state.detector_jacobian[c * n + k];
                }
                *value /= scale;
                model.gradient[k] += weight * r * *value;
            }
            for a in 0..n {
                for b in 0..=a {
                    model.normal[a * n + b] += curvature * j[a] * j[b];
                }
            }
        }
    }
    for a in 0..n {
        for b in 0..a {
            model.normal[b * n + a] = model.normal[a * n + b];
        }
    }
    if !model.cost.is_finite()
        || model
            .gradient
            .iter()
            .chain(&model.normal)
            .any(|v| !v.is_finite())
    {
        return Err(invalid("geometry objective overflow"));
    }
    Ok(model)
}

fn spans_three(hkl: &[[i64; 3]], selected: &[usize]) -> bool {
    let Some(&first) = selected.iter().find(|&&i| hkl[i] != [0; 3]) else {
        return false;
    };
    let a = hkl[first].map(i128::from);
    for &i in selected {
        let b = hkl[i].map(i128::from);
        let cross = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        if cross != [0; 3] {
            return selected.iter().any(|&j| {
                (0..3)
                    .map(|k| cross[k] * i128::from(hkl[j][k]))
                    .sum::<i128>()
                    != 0
            });
        }
    }
    false
}

/// Fit all geometry parameters and diagnostics entirely in Rust.
/// Progress is optional host reporting; numerical evaluations never call a host.
/// # Errors
/// Rejects invalid/degenerate inputs, insufficient independent indexed evidence,
/// reference metrics, lost retained crossings or host progress cancellation.
#[allow(clippy::too_many_lines)]
pub fn fit_with_map(
    input: &Input<'_>,
    options: Options,
    mapper: &dyn PixelMap,
    mut progress: impl FnMut(usize, usize) -> Result<(), SpotError>,
) -> Result<Fit, SpotError> {
    options.validate()?;
    let m = input.hkl.len();
    if m > 1_000_000
        || input.observed.len() != m
        || input.training.len() != m
        || input.eligible.len() != m
        || input.observed.iter().flatten().any(|v| !v.is_finite())
        || input.image_shape.contains(&0)
        || input.image_shape.iter().any(|v| *v > 1_000_000)
    {
        return Err(invalid("invalid geometry fitting arrays or detector shape"));
    }
    let hints: Vec<_> = input.observed.iter().map(|v| v[2]).collect();
    let mut initial = prediction::predict(input.geometry, input.basis, input.hkl, &hints, false)?;
    map_predictions(&mut initial, mapper)?;
    let eligible: Vec<_> = (0..m)
        .map(|i| {
            input.eligible[i]
                && initial[i].flags == 0
                && (initial[i].values[0] - input.observed[i][0])
                    .hypot(initial[i].values[1] - input.observed[i][1])
                    < options.initial_xy_gate
                && (initial[i].values[2] - input.observed[i][2]).abs() < options.initial_omega_gate
        })
        .collect();
    let selected: Vec<_> = (0..m)
        .filter(|&i| eligible[i] && input.training[i])
        .collect();
    let withheld = eligible
        .iter()
        .zip(input.training)
        .filter(|(v, t)| **v && !**t)
        .count();
    if selected.len() < 30 || withheld < 10 || !spans_three(input.hkl, &selected) {
        return Err(invalid(
            "need >=30 training and >=10 withheld indexed spots spanning 3D",
        ));
    }
    let base = if let Some(cell) = input.reference_cell {
        reference_basis(input.basis, cell)?
    } else {
        input.basis
    };
    let pivot = std::array::from_fn(|j| {
        input.geometry.origin[j]
            + (input.image_shape[1] - 1) as f64 / 2. * input.geometry.fast[j]
            + (input.image_shape[0] - 1) as f64 / 2. * input.geometry.slow[j]
    });
    let n = if input.refine_lengths { 12 } else { 9 };
    let units: Vec<_> = (0..n)
        .map(|i| if (3..6).contains(&i) { 0.1 } else { 0.001 })
        .collect();
    let limits: Vec<_> = (0..n)
        .map(|i| {
            if i >= 9 {
                options.log_length_limit
            } else if (3..6).contains(&i) {
                options.translation_limit_mm
            } else {
                options.rotation_limit_deg.to_radians()
            }
        })
        .collect();
    let mut parameters = vec![0.; n];
    let mut evaluations = 1;
    progress(0, options.max_evaluations)?;
    let mut model = evaluate(input, base, pivot, &selected, &parameters, options, mapper)?;
    let mut history = vec![model.cost];
    let mut damping = 1e-3;
    let mut converged = false;
    loop {
        let free: Vec<_> = (0..n)
            .filter(|&i| {
                !((parameters[i] <= -limits[i] && model.gradient[i] > 0.)
                    || (parameters[i] >= limits[i] && model.gradient[i] < 0.))
            })
            .collect();
        let projected = free
            .iter()
            .map(|&i| (model.gradient[i] * units[i]).abs())
            .fold(0., f64::max);
        if projected < 1e-9 {
            converged = true;
            break;
        }
        if evaluations >= options.max_evaluations {
            break;
        }
        let nf = free.len();
        let mut matrix = vec![0.; nf * nf];
        let mut rhs = vec![0.; nf];
        for (a, &i) in free.iter().enumerate() {
            rhs[a] = -model.gradient[i] * units[i];
            for (b, &j) in free.iter().enumerate() {
                matrix[a * nf + b] = model.normal[i * n + j] * units[i] * units[j];
            }
            matrix[a * nf + a] += damping * matrix[a * nf + a].abs().max(1.);
        }
        let Ok(delta) = solve(&matrix, &rhs) else {
            damping *= 10.;
            if damping > 1e24 {
                break;
            }
            continue;
        };
        let mut trial = parameters.clone();
        for (a, &i) in free.iter().enumerate() {
            trial[i] = (parameters[i] + units[i] * delta[a]).clamp(-limits[i], limits[i]);
        }
        let step: Vec<_> = trial.iter().zip(&parameters).map(|(a, b)| a - b).collect();
        let stepnorm = step
            .iter()
            .zip(&units)
            .map(|(d, s)| (d / s).powi(2))
            .sum::<f64>()
            .sqrt();
        let pnorm = parameters
            .iter()
            .zip(&units)
            .map(|(p, s)| (p / s).powi(2))
            .sum::<f64>()
            .sqrt();
        if stepnorm <= 1e-11 * (1e-11 + pnorm) {
            converged = true;
            break;
        }
        progress(evaluations, options.max_evaluations)?;
        evaluations += 1;
        let Ok(next) = evaluate(input, base, pivot, &selected, &trial, options, mapper) else {
            damping *= 10.;
            continue;
        };
        let improvement = model.cost - next.cost;
        let mut predicted = -model
            .gradient
            .iter()
            .zip(&step)
            .map(|(g, d)| g * d)
            .sum::<f64>();
        for i in 0..n {
            for j in 0..n {
                predicted -= 0.5 * step[i] * model.normal[i * n + j] * step[j];
            }
        }
        let ratio = if predicted > 0. {
            improvement / predicted
        } else {
            0.
        };
        if improvement > 0. {
            let cost_converged = improvement <= 1e-11 * model.cost && ratio > 0.25;
            parameters = trial;
            model = next;
            history.push(model.cost);
            damping = if ratio > 0.75 {
                (damping / 3.).max(1e-15)
            } else if ratio < 0.25 {
                damping * 4.
            } else {
                damping
            };
            if cost_converged {
                converged = true;
                break;
            }
        } else {
            damping *= 10.;
        }
        if damping > 1e24 {
            break;
        }
    }
    let state = parameter_state(
        &parameters,
        base,
        input.geometry,
        pivot,
        input.refine_lengths,
    )?;
    let mut predictions =
        prediction::predict(state.geometry, state.basis, input.hkl, &hints, false)?;
    map_predictions(&mut predictions, mapper)?;
    if eligible
        .iter()
        .zip(&predictions)
        .any(|(e, p)| *e && p.flags != 0)
    {
        return Err(invalid("refined geometry loses withheld indexed crossings"));
    }
    let scaled: Vec<_> = (0..n * n)
        .map(|i| model.normal[i] * units[i / n] * units[i % n])
        .collect();
    let (eigenvalues, _) = eigen(&scaled, n)?;
    let scaled_condition = if eigenvalues[0] > 0. {
        (eigenvalues[n - 1] / eigenvalues[0]).sqrt()
    } else {
        f64::INFINITY
    };
    let normal_scale = model
        .normal
        .iter()
        .map(|v| v.abs())
        .fold(0., f64::max)
        .max(1e-300);
    let scaled: Vec<_> = model.normal.iter().map(|v| v / normal_scale).collect();
    let (eigenvalues, vectors) = eigen(&scaled, n)?;
    let mut covariance = vec![0.; n * n];
    for k in 0..n {
        if eigenvalues[k] > 1e-15 * eigenvalues[n - 1] {
            for i in 0..n {
                for j in 0..n {
                    covariance[i * n + j] +=
                        vectors[i * n + k] * vectors[j * n + k] / (eigenvalues[k] * normal_scale);
                }
            }
        }
    }
    let mut maximum_parameter_correlation: f64 = 0.;
    for i in 0..n {
        for j in 0..i {
            let den = (covariance[i * n + i].max(0.) * covariance[j * n + j].max(0.)).sqrt();
            if den > 0. {
                maximum_parameter_correlation =
                    maximum_parameter_correlation.max((covariance[i * n + j] / den).abs());
            }
        }
    }
    let mut rms = [[[0.; 3]; 2]; 2];
    let mut counts = [0; 2];
    for i in 0..m {
        if eligible[i] {
            let part = usize::from(!input.training[i]);
            counts[part] += 1;
            for (when, p) in [&initial[i], &predictions[i]].into_iter().enumerate() {
                let r: V3 = std::array::from_fn(|j| p.values[j] - input.observed[i][j]);
                rms[part][when][0] += r[0] * r[0] + r[1] * r[1];
                rms[part][when][1] += r[2] * r[2];
                rms[part][when][2] += (r[0] / options.xy_scale).powi(2)
                    + (r[1] / options.xy_scale).powi(2)
                    + (r[2] / options.omega_scale).powi(2);
            }
        }
    }
    for (part, estimates) in rms.iter_mut().enumerate() {
        for estimate in estimates {
            for value in estimate {
                *value = (*value / counts[part] as f64).sqrt();
            }
        }
    }
    let at_bounds = parameters
        .iter()
        .zip(&limits)
        .any(|(p, l)| l - p.abs() <= 1e-11 * l.abs().max(1.));
    Ok(Fit {
        parameters,
        state,
        initial,
        predictions,
        eligible,
        converged,
        at_bounds,
        scaled_condition,
        maximum_parameter_correlation,
        covariance,
        evaluations,
        history,
        rms,
        counts,
    })
}

/// Fit an ordinary affine-panel geometry without an external coordinate map.
/// # Errors
/// Reports the same validation, crossing and host-cancellation errors as
/// [`fit_with_map`].
pub fn fit(
    input: &Input<'_>,
    options: Options,
    progress: impl FnMut(usize, usize) -> Result<(), SpotError>,
) -> Result<Fit, SpotError> {
    fit_with_map(input, options, &IdentityMap, progress)
}

impl Fit {
    /// Advisory identification/withheld-improvement check, not a correctness certificate.
    #[must_use]
    pub fn acceptable(&self) -> bool {
        self.converged
            && !self.at_bounds
            && self.scaled_condition < 1e6
            && self.rms[1][1][2] < self.rms[1][0][2]
    }
}
