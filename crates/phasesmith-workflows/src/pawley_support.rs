//! Local one-sided geometry at a hard CW support discontinuity.
use crate::pawley::err;
use crate::{ConstraintTransform, PawleyError, PawleyEvaluation, PawleyInput, pawley_key};
use nalgebra::{DMatrix, DVector};
use phasesmith_core::{ConstantWavelengthInstrument, CwProfileParameters};
use std::collections::BTreeMap;

struct SupportGuard {
    position: f64,
    radius: f64,
    support: f64,
    grow: bool,
    shrink: bool,
    gradient: Vec<f64>,
    gap: f64,
}
fn radius_gradient(
    input: &PawleyInput,
    transform: &ConstraintTransform,
    free: &[f64],
    position: f64,
    support: f64,
) -> Result<(f64, Vec<f64>), PawleyError> {
    let values = transform.unpack(free, false).map_err(err)?;
    let mut coefficients = [0.0; 5];
    let mut indices = [0; 5];
    for (i, name) in crate::PAWLEY_PROFILE_NAMES.iter().enumerate() {
        let key = pawley_key("profile", "instrument", name)?;
        coefficients[i] = values[&key];
        indices[i] = input
            .parameters
            .index_of(&key)
            .ok_or_else(|| err("missing profile coefficient"))?;
    }
    let instrument = ConstantWavelengthInstrument {
        wavelength_angstrom: input.instrument.wavelength_angstrom,
        u_deg2: coefficients[0],
        v_deg2: coefficients[1],
        w_deg2: coefficients[2],
        x_deg: coefficients[3],
        y_deg: coefficients[4],
    };
    let profile = CwProfileParameters::from_instrument(position, instrument).map_err(err)?;
    let chain = transform.derivative_matrix().map_err(err)?;
    let mut gradient = vec![0.0; free.len()];
    for (i, &index) in indices.iter().enumerate() {
        let derivative = support
            * (profile.tch.d_total_fwhm_d_gaussian_fwhm * profile.d_gaussian_fwhm_d_instrument[i]
                + profile.tch.d_total_fwhm_d_lorentzian_fwhm
                    * profile.d_lorentzian_fwhm_d_instrument[i]);
        for (j, v) in gradient.iter_mut().enumerate() {
            *v += derivative * chain.values[index * free.len() + j];
        }
    }
    Ok((support * profile.tch.total_fwhm, gradient))
}
/// Positive jumps exclude local crossing directions. Several radius events
/// are certified together only when none has a negative jump.
pub(crate) fn support_guard(
    input: &PawleyInput,
    transform: &ConstraintTransform,
    free: &[f64],
    evaluation: &PawleyEvaluation,
    support: f64,
    uncertainty: bool,
) -> Result<Option<SupportGuards>, PawleyError> {
    if input.axial.is_some() || input.fixed_spectrum.is_some() {
        return Ok(None);
    }
    let chain = transform.derivative_matrix().map_err(err)?;
    if input.parameters.specs().iter().enumerate().any(|(i, s)| {
        s.key().module() == "pawley_lattice"
            && chain.row(i).is_some_and(|r| r.iter().any(|v| *v != 0.0))
    }) {
        return Ok(None);
    }
    let mut groups = BTreeMap::<u64, f64>::new();
    for (&position, &area) in evaluation.positions.iter().zip(&evaluation.intensities) {
        *groups.entry(position.to_bits()).or_default() += area;
    }
    let values = transform.unpack(free, false).map_err(err)?;
    let mut instrument = input.instrument;
    instrument.u_deg2 = values[&pawley_key("profile", "instrument", "u_deg2")?];
    instrument.v_deg2 = values[&pawley_key("profile", "instrument", "v_deg2")?];
    instrument.w_deg2 = values[&pawley_key("profile", "instrument", "w_deg2")?];
    instrument.x_deg = values[&pawley_key("profile", "instrument", "x_deg")?];
    instrument.y_deg = values[&pawley_key("profile", "instrument", "y_deg")?];
    let x = &input.pattern.x_deg;
    let mut events = Vec::new();
    let mut negative_jump = false;
    let mut event_samples = std::collections::BTreeSet::new();
    for (bits, area) in groups {
        if area == 0.0 {
            continue;
        }
        let position = f64::from_bits(bits);
        let profile = CwProfileParameters::from_instrument(position, instrument).map_err(err)?;
        let radius = support * profile.tch.total_fwhm;
        let tolerance = 64.0 * f64::EPSILON * (1.0 + position.abs() + radius);
        let mut candidates = Vec::new();
        for edge in [position - radius, position + radius] {
            let next = x.partition_point(|v| *v < edge);
            for i in [next.saturating_sub(1), next] {
                if i < x.len()
                    && (x[i] - edge).abs() <= tolerance
                    && evaluation.residuals.included[i]
                    && !candidates.contains(&i)
                {
                    candidates.push(i);
                }
            }
        }
        if candidates.is_empty() {
            continue;
        }
        if candidates.iter().any(|i| !event_samples.insert(*i)) {
            // Two distinct radii changing the same sample have cross terms in
            // their combined jump; separate positive jumps do not certify it.
            return Ok(None);
        }
        let gap = candidates
            .iter()
            .map(|&i| ((x[i] - position).abs() - radius).abs())
            .fold(0.0, f64::max);
        let mut grow_jump = 0.0;
        let mut shrink_jump = 0.0;
        for i in candidates {
            let inside = x[i] >= position - radius && x[i] <= position + radius;
            let weight = if uncertainty {
                input
                    .pattern
                    .uncertainty
                    .as_ref()
                    .map_or(1.0, |s| 1.0 / s[i])
            } else {
                1.0
            };
            let change = area
                * profile.tch.evaluate(x[i] - position).value
                * weight
                * if inside { -1.0 } else { 1.0 };
            let jump = 2.0 * evaluation.residuals.weighted_residual[i] * change + change * change;
            if inside {
                shrink_jump += jump;
            } else {
                grow_jump += jump;
            }
        }
        negative_jump |= grow_jump < 0.0 || shrink_jump < 0.0;
        let (_, gradient) = radius_gradient(input, transform, free, position, support)?;
        events.push(SupportGuard {
            position,
            radius,
            support,
            grow: grow_jump > 0.0,
            shrink: shrink_jump > 0.0,
            gradient,
            gap,
        });
    }
    if events.len() > 1 && negative_jump {
        return Ok(None);
    }
    events.retain(|g| (g.grow || g.shrink) && g.gradient.iter().any(|v| *v != 0.0));
    Ok((!events.is_empty()).then_some(SupportGuards(events)))
}
pub(crate) struct SupportGuards(Vec<SupportGuard>);
impl SupportGuards {
    pub(crate) fn near(&self, norms: &[f64], tolerance: f64) -> bool {
        self.0.iter().all(|g| g.near(norms, tolerance))
    }
    pub(crate) fn constrain(
        &self,
        a: &DMatrix<f64>,
        b: &DVector<f64>,
        norms: &[f64],
    ) -> (DMatrix<f64>, DVector<f64>) {
        self.0.iter().fold((a.clone(), b.clone()), |(a, b), g| {
            g.constrain(&a, &b, norms)
        })
    }
    pub(crate) fn retract(
        &self,
        input: &PawleyInput,
        transform: &ConstraintTransform,
        free: &mut [f64],
    ) -> Result<(), PawleyError> {
        // Solve all violated radius equations together so retraction at one
        // cutoff does not undo another independent cutoff constraint.
        for _ in 0..12 {
            let mut rows = Vec::new();
            let mut rhs = Vec::new();
            for g in &self.0 {
                let (radius, gradient) =
                    radius_gradient(input, transform, free, g.position, g.support)?;
                if (!g.grow || radius <= g.radius) && (!g.shrink || radius >= g.radius) {
                    continue;
                }
                let margin = 16.0 * f64::EPSILON * (1.0 + g.position.abs() + g.radius);
                let target = if g.grow && g.shrink {
                    g.radius
                } else if g.grow {
                    g.radius - margin
                } else {
                    g.radius + margin
                };
                rows.push(DVector::from_vec(gradient));
                rhs.push(target - radius);
            }
            if rows.is_empty() {
                return Ok(());
            }
            let c = DMatrix::from_columns(&rows).transpose();
            let svd = c.svd(true, true);
            let threshold = svd.singular_values.amax() * 1e-13;
            let delta = svd.solve(&DVector::from_vec(rhs), threshold).map_err(err)?;
            for (v, d) in free.iter_mut().zip(delta.iter()) {
                *v += d;
            }
        }
        Err(err("support-boundary retraction failed"))
    }
}
impl SupportGuard {
    pub(crate) fn near(&self, norms: &[f64], step_tolerance: f64) -> bool {
        let normal = self
            .gradient
            .iter()
            .zip(norms)
            .map(|(g, n)| (g / n).powi(2))
            .sum::<f64>()
            .sqrt();
        normal.is_finite() && normal > 0.0 && self.gap <= normal * step_tolerance
    }
    pub(crate) fn constrain(
        &self,
        a: &DMatrix<f64>,
        b: &DVector<f64>,
        norms: &[f64],
    ) -> (DMatrix<f64>, DVector<f64>) {
        let row = DVector::from_iterator(
            norms.len(),
            self.gradient.iter().zip(norms).map(|(g, n)| g / n),
        );
        let norm = row.norm();
        let mut signs = Vec::new();
        if self.grow {
            signs.push(-1.0);
        }
        if self.shrink {
            signs.push(1.0);
        }
        let mut extended = DMatrix::zeros(a.nrows() + signs.len(), a.ncols());
        extended.rows_mut(0, a.nrows()).copy_from(a);
        let mut rhs = DVector::zeros(extended.nrows());
        rhs.rows_mut(0, b.len()).copy_from(b);
        for (i, sign) in signs.iter().enumerate() {
            extended
                .row_mut(a.nrows() + i)
                .copy_from(&(row.transpose() * (*sign / norm)));
        }
        (extended, rhs)
    }
}

/// Cheap exact membership signature for symmetric fixed-position profiles.
/// Used only to bracket an already observed line-search discontinuity.
pub(crate) fn support_membership(
    input: &PawleyInput,
    transform: &ConstraintTransform,
    free: &[f64],
    positions: &[f64],
    support: f64,
) -> Result<Option<Vec<(usize, usize)>>, PawleyError> {
    if input.axial.is_some() || input.fixed_spectrum.is_some() {
        return Ok(None);
    }
    let chain = transform.derivative_matrix().map_err(err)?;
    if input.parameters.specs().iter().enumerate().any(|(i, s)| {
        s.key().module() == "pawley_lattice"
            && chain.row(i).is_some_and(|r| r.iter().any(|v| *v != 0.0))
    }) {
        return Ok(None);
    }
    let values = transform.unpack(free, false).map_err(err)?;
    let mut instrument = input.instrument;
    instrument.u_deg2 = values[&pawley_key("profile", "instrument", "u_deg2")?];
    instrument.v_deg2 = values[&pawley_key("profile", "instrument", "v_deg2")?];
    instrument.w_deg2 = values[&pawley_key("profile", "instrument", "w_deg2")?];
    instrument.x_deg = values[&pawley_key("profile", "instrument", "x_deg")?];
    instrument.y_deg = values[&pawley_key("profile", "instrument", "y_deg")?];
    let x = &input.pattern.x_deg;
    positions
        .iter()
        .map(|&position| {
            let radius = support
                * CwProfileParameters::from_instrument(position, instrument)
                    .map_err(err)?
                    .tch
                    .total_fwhm;
            Ok((
                x.partition_point(|v| *v < position - radius),
                x.partition_point(|v| *v <= position + radius),
            ))
        })
        .collect::<Result<Vec<_>, PawleyError>>()
        .map(Some)
}
