//! CW Pawley least squares: independent integrated areas and analytical geometry.

use crate::pawley_operator::{PawleyColumn, PawleyJacobian};
use crate::{
    BackgroundModel, Constraint, ConstraintTransform, DifferentiableBackground,
    LatticeReflectionDomain, ParameterBounds, ParameterKey, ParameterSet, ParameterSpec,
    ResidualEvaluation, ResidualOptions, cw_lattice_geometry, evaluate_residuals,
};
use nalgebra::{DMatrix, DVector};
use phasesmith_core::{
    ConstantWavelengthInstrument, CwReflectionBatchView, FcjGeometry, GridView, SupportPolicy,
    WavelengthComponentsView, accumulate_cw_batch, accumulate_cw_fcj_batch,
    cw_components_support_samples,
};
use phasesmith_model::PatternRecord;
use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};

/// Validated-workflow error; messages include the failing scientific boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct PawleyError(pub String);
impl Display for PawleyError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for PawleyError {}
pub(crate) fn err(e: impl Display) -> PawleyError {
    PawleyError(e.to_string())
}

/// One ordered powder-family list. Areas absorb multiplicity and fixed corrections.
#[derive(Clone, Debug, PartialEq)]
pub struct PawleyPhase {
    /// Stable phase identity.
    pub id: String,
    /// Stable family identities, unique within this phase.
    pub reflection_ids: Vec<String>,
    /// Ideal fixed positions; replaced analytically when a lattice domain exists.
    pub two_theta_deg: Vec<f64>,
    /// Initial areas; signed values require an explicit signed request.
    pub intensities: Vec<f64>,
    /// Miller families for the optional lattice domain.
    pub hkl: Vec<[i32; 3]>,
    /// Conservative fixed family domain; topology never changes during fitting.
    pub lattice: Option<LatticeReflectionDomain>,
}
impl PawleyPhase {
    /// Construct a fixed family superset from a validated lattice domain.
    ///
    /// # Errors
    /// Returns an error if reflection generation fails.
    pub fn from_domain(id: String, domain: LatticeReflectionDomain) -> Result<Self, PawleyError> {
        let generated = domain
            .generate(domain.parameterization().reference_cell(), None)
            .map_err(err)?;
        Ok(Self {
            id,
            reflection_ids: generated.reflection_ids,
            two_theta_deg: generated.two_theta_deg,
            intensities: generated.integrated_intensity,
            hkl: generated.hkl,
            lattice: Some(domain),
        })
    }
}
/// Native request. No structural intensity model or independent phase scales.
#[derive(Clone, Debug, PartialEq)]
pub struct PawleyInput {
    /// Observations, explicit one-sigma weights, mask and fixed background.
    pub pattern: PatternRecord,
    /// Fixed wavelength and initial CW coefficients.
    pub instrument: ConstantWavelengthInstrument,
    /// Fixed axial geometry, or a symmetric profile.
    pub axial: Option<FcjGeometry>,
    /// Ordered phases.
    pub phases: Vec<PawleyPhase>,
    /// Optional additive coefficient-invariant background.
    pub background: Option<BackgroundModel>,
    /// Allow signed fitted areas; default callers should select false.
    pub signed_intensities: bool,
    /// Complete physical parameter table, built by `pawley_parameters`.
    pub parameters: ParameterSet,
    /// Explicit fixed, affine or linear ties.
    pub constraints: Vec<Constraint>,
}
/// Ordered CW coefficient names.
pub const PAWLEY_PROFILE_NAMES: [&str; 5] = ["u_deg2", "v_deg2", "w_deg2", "x_deg", "y_deg"];
/// Construct a durable Pawley parameter key.
///
/// # Errors
/// Rejects invalid identities.
pub fn pawley_key(family: &str, owner: &str, name: &str) -> Result<ParameterKey, PawleyError> {
    ParameterKey::new(format!("pawley_{family}"), owner, name).map_err(err)
}
/// Build the complete ordered table; callers may replace specs to select or bound variables.
/// Intensities and background are initially free; profile and lattice initially fixed.
///
/// # Errors
/// Rejects invalid initial values, identities or bounds.
pub fn pawley_parameters(
    phases: &[PawleyPhase],
    instrument: ConstantWavelengthInstrument,
    background: Option<&BackgroundModel>,
    signed: bool,
) -> Result<ParameterSet, PawleyError> {
    let mut specs = Vec::new();
    let mut add = |key, value: f64, unit, bounds, refine| -> Result<(), PawleyError> {
        specs.push(
            ParameterSpec::new(key, value, unit, bounds, value.abs().max(1e-3), refine)
                .map_err(err)?,
        );
        Ok(())
    };
    let unbounded = ParameterBounds::new(f64::NEG_INFINITY, f64::INFINITY).map_err(err)?;
    for phase in phases {
        if phase.reflection_ids.len() != phase.intensities.len() {
            return Err(err("reflection/area shape mismatch"));
        }
        for (id, &value) in phase.reflection_ids.iter().zip(&phase.intensities) {
            add(
                pawley_key("intensity", &phase.id, id)?,
                value,
                "y*degree",
                ParameterBounds::new(if signed { f64::NEG_INFINITY } else { 0.0 }, f64::INFINITY)
                    .map_err(err)?,
                true,
            )?;
        }
    }
    for (name, value) in PAWLEY_PROFILE_NAMES.iter().zip(profile_values(instrument)) {
        add(
            pawley_key("profile", "instrument", name)?,
            value,
            if name.ends_with("deg2") {
                "degree^2"
            } else {
                "degree"
            },
            unbounded,
            false,
        )?;
    }
    if let Some(bg) = background {
        for ((name, value), bounds) in bg
            .parameter_names()
            .iter()
            .zip(bg.coefficients())
            .zip(bg.parameter_bounds())
        {
            add(
                pawley_key("background", bg.background_id(), name)?,
                value,
                "y",
                bounds,
                true,
            )?;
        }
    }
    for phase in phases {
        if let Some(domain) = &phase.lattice {
            let par = domain.parameterization();
            let values = par.values_from_cell(par.reference_cell()).map_err(err)?;
            for (j, (name, value)) in par.parameter_names().iter().zip(values).enumerate() {
                add(
                    pawley_key("lattice", &phase.id, name)?,
                    value,
                    if name.ends_with("deg") {
                        "degree"
                    } else {
                        "angstrom"
                    },
                    ParameterBounds::new(domain.bounds().lower()[j], domain.bounds().upper()[j])
                        .map_err(err)?,
                    false,
                )?;
            }
        }
    }
    ParameterSet::new(specs).map_err(err)
}
pub(crate) fn profile_values(i: ConstantWavelengthInstrument) -> [f64; 5] {
    [i.u_deg2, i.v_deg2, i.w_deg2, i.x_deg, i.y_deg]
}

impl PawleyInput {
    /// Validate the complete mutable adapter record immediately before calculation.
    ///
    /// # Errors
    /// Rejects malformed records, unknown parameters, illegal bounds or unsupported backgrounds.
    pub fn validate(&self) -> Result<(), PawleyError> {
        self.pattern.validate().map_err(err)?;
        self.instrument.validate().map_err(err)?;
        if self.phases.is_empty() {
            return Err(err("Pawley requires phases"));
        }
        if self
            .background
            .as_ref()
            .is_some_and(|b| !b.basis_is_invariant())
        {
            return Err(err("Pawley requires a coefficient-invariant background"));
        }
        let mut ids = BTreeSet::new();
        for p in &self.phases {
            if p.id.trim().is_empty() || p.id.contains('/') || !ids.insert(&p.id) {
                return Err(err("invalid or duplicate phase identity"));
            }
            let n = p.intensities.len();
            if n == 0
                || p.reflection_ids.len() != n
                || p.two_theta_deg.len() != n
                || (p.lattice.is_some() && p.hkl.len() != n)
            {
                return Err(err("Pawley reflection shape mismatch"));
            }
            let mut refs = BTreeSet::new();
            for id in &p.reflection_ids {
                if id.trim().is_empty() || id.contains('/') || !refs.insert(id) {
                    return Err(err("invalid or duplicate reflection identity"));
                }
            }
            if let Some(domain) = &p.lattice {
                if domain.wavelength_angstrom().to_bits()
                    != self.instrument.wavelength_angstrom.to_bits()
                {
                    return Err(err("Pawley lattice wavelength differs from instrument"));
                }
            }
            CwReflectionBatchView::new(&p.two_theta_deg, &p.intensities).map_err(err)?;
        }
        let expected = pawley_parameters(
            &self.phases,
            self.instrument,
            self.background.as_ref(),
            self.signed_intensities,
        )?;
        if expected.specs().len() != self.parameters.specs().len() {
            return Err(err("Pawley parameter table shape mismatch"));
        }
        for (a, b) in expected.specs().iter().zip(self.parameters.specs()) {
            if a.key() != b.key()
                || a.unit() != b.unit()
                || b.bounds().lower() < a.bounds().lower()
                || b.bounds().upper() > a.bounds().upper()
            {
                return Err(err(
                    "Pawley parameter identity, unit or bound contract mismatch",
                ));
            }
        }
        let t = ConstraintTransform::new(self.parameters.clone(), self.constraints.clone())
            .map_err(err)?;
        t.unpack(&t.pack().map_err(err)?, false).map_err(err)?;
        Ok(())
    }
}
/// Exactly coincident unit profiles; individual areas are not identifiable without ties.
#[derive(Clone, Debug, PartialEq)]
pub struct PawleyCoincidentGroup {
    /// Ordered stable phase/reflection identities.
    pub members: Vec<(String, String)>,
    /// Identifiable total area of this identical-profile group, conditional on other columns.
    pub total_intensity: f64,
}
/// One complete physical calculation and sample-major analytical Jacobian.
#[derive(Clone, Debug)]
pub struct PawleyEvaluation {
    /// Total calculated signal.
    pub calculated_y: Vec<f64>,
    /// Fixed plus fitted background.
    pub background_y: Vec<f64>,
    /// Integrated areas in phase/reflection order.
    pub intensities: Vec<f64>,
    /// Current ideal positions in phase/reflection order.
    pub positions: Vec<f64>,
    /// Dense Jacobian over scaled free variables, including masked samples.
    pub jacobian: Option<DMatrix<f64>>,
    /// Support-block products from the same analytical evaluation pass.
    pub jacobian_operator: PawleyJacobian,
    /// Weighted masked residuals and agreement factors.
    pub residuals: ResidualEvaluation,
    /// Free columns without any included weighted support.
    pub inactive_columns: Vec<usize>,
    /// Reflection families with no mask-included profile support.
    pub unobserved_reflections: Vec<(String, String)>,
    /// Exact coincidences only; numerical near-dependencies are reported by rank.
    pub coincident_groups: Vec<PawleyCoincidentGroup>,
}
/// Evaluate values and all selected derivatives through the native fused profile pass.
///
/// # Errors
/// Rejects invalid state and allocations above the explicit element ceiling.
#[allow(clippy::too_many_lines)]
pub fn evaluate_pawley(
    input: &PawleyInput,
    free: &[f64],
    support: f64,
    use_uncertainty: bool,
    max_elements: usize,
) -> Result<PawleyEvaluation, PawleyError> {
    evaluate_pawley_with_storage(input, free, support, use_uncertainty, max_elements, true)
}
/// Evaluate with optional dense materialization; products always remain available.
///
/// # Errors
/// Rejects invalid models and workspaces above the explicit element ceiling.
#[allow(clippy::too_many_lines)]
pub fn evaluate_pawley_with_storage(
    input: &PawleyInput,
    free: &[f64],
    support: f64,
    use_uncertainty: bool,
    max_elements: usize,
    dense: bool,
) -> Result<PawleyEvaluation, PawleyError> {
    input.validate()?;
    let transform = ConstraintTransform::new(input.parameters.clone(), input.constraints.clone())
        .map_err(err)?;
    let values = transform.unpack(free, false).map_err(err)?;
    let n = input.pattern.sample_count();
    let p = input.parameters.specs().len();
    let k = free.len();
    let reflections: usize = input.phases.iter().map(|p| p.intensities.len()).sum();
    // Includes worst-case support storage, both dense Jacobians, derivative transform and normal workspace.
    let allocation = n
        .checked_mul(
            p.checked_add(k)
                .and_then(|v| v.checked_add(reflections.checked_mul(2)?.checked_add(10)?))
                .ok_or_else(|| err("size overflow"))?,
        )
        .and_then(|v| v.checked_add(p.checked_mul(k)?))
        .and_then(|v| v.checked_add(k.checked_mul(k)?.checked_mul(8)?))
        .ok_or_else(|| err("size overflow"))?;
    if dense && allocation > max_elements {
        return Err(err("Pawley dense memory element limit exceeded"));
    }
    let get = |family, owner: &str, name: &str| -> Result<f64, PawleyError> {
        values
            .get(&pawley_key(family, owner, name)?)
            .copied()
            .ok_or_else(|| err("missing parameter"))
    };
    let mut instrument = input.instrument;
    instrument.u_deg2 = get("profile", "instrument", "u_deg2")?;
    instrument.v_deg2 = get("profile", "instrument", "v_deg2")?;
    instrument.w_deg2 = get("profile", "instrument", "w_deg2")?;
    instrument.x_deg = get("profile", "instrument", "x_deg")?;
    instrument.y_deg = get("profile", "instrument", "y_deg")?;
    let mut positions = Vec::new();
    let mut intensities = Vec::new();
    let mut chains = Vec::new();
    for phase in &input.phases {
        let geometry = if let Some(domain) = &phase.lattice {
            let par = domain.parameterization();
            let v = par
                .parameter_names()
                .iter()
                .map(|name| get("lattice", &phase.id, name))
                .collect::<Result<Vec<_>, _>>()?;
            let cell = par.to_cell(&v).map_err(err)?;
            domain.validate_cell(cell).map_err(err)?;
            Some(
                cw_lattice_geometry(par, cell, &phase.hkl, instrument.wavelength_angstrom)
                    .map_err(err)?,
            )
        } else {
            None
        };
        positions.extend(
            geometry
                .as_ref()
                .map_or(phase.two_theta_deg.as_slice(), |g| {
                    g.two_theta_deg.as_slice()
                }),
        );
        for id in &phase.reflection_ids {
            intensities.push(get("intensity", &phase.id, id)?);
        }
        chains.push(geometry);
    }
    let grid = GridView::new(&input.pattern.x_deg).map_err(err)?;
    let batch = CwReflectionBatchView::new(&positions, &intensities).map_err(err)?;
    if !dense {
        let wavelengths = [instrument.wavelength_angstrom];
        let weights = [1.0];
        let components = WavelengthComponentsView::new(&wavelengths, &weights).map_err(err)?;
        let active = cw_components_support_samples(
            grid,
            batch,
            instrument,
            components,
            input.axial.unwrap_or(FcjGeometry {
                sample_over_radius: 0.0,
                detector_over_radius: 0.0,
            }),
            SupportPolicy::FwhmMultiple(support),
        )
        .map_err(err)?;
        // Native local pairs plus stored areas, global columns, constraint chain,
        // general coupled-face workspace and two accepted/trial evaluations.
        let allocation = active
            .checked_mul(6)
            .and_then(|v| {
                v.checked_add(n.checked_mul((p - reflections).checked_mul(2)?.checked_add(24)?)?)
            })
            .and_then(|v| v.checked_add(p.checked_mul(k)?.checked_mul(8)?))
            .and_then(|v| v.checked_add(k.checked_mul(k)?.checked_mul(16)?))
            .ok_or_else(|| err("size overflow"))?;
        if allocation > max_elements {
            return Err(err("Pawley product memory element limit exceeded"));
        }
    }
    let accumulation = match input.axial {
        Some(geometry) => accumulate_cw_fcj_batch(
            grid,
            batch,
            instrument,
            geometry,
            SupportPolicy::FwhmMultiple(support),
        )
        .map_err(err)?,
        None => accumulate_cw_batch(
            grid,
            batch,
            instrument,
            SupportPolicy::FwhmMultiple(support),
        )
        .map_err(err)?,
    };
    let mut physical = vec![PawleyColumn::empty(); p];
    let local = &accumulation.derivatives.local;
    let global = accumulation
        .derivatives
        .global
        .as_ref()
        .ok_or_else(|| err("missing profile derivatives"))?;
    let index = |family, owner: &str, name: &str| -> Result<usize, PawleyError> {
        input
            .parameters
            .index_of(&pawley_key(family, owner, name)?)
            .ok_or_else(|| err("unknown parameter"))
    };
    let mut reflection = 0;
    for (phase, geometry) in input.phases.iter().zip(&chains) {
        for (r, id) in phase.reflection_ids.iter().enumerate() {
            let col = index("intensity", &phase.id, id)?;
            physical[col] = PawleyColumn {
                start: local.starts[reflection],
                values: (local.offsets[reflection]..local.offsets[reflection + 1])
                    .map(|a| local.values[a * local.parameter_count])
                    .collect(),
            };
            for a in local.offsets[reflection]..local.offsets[reflection + 1] {
                let i = local.starts[reflection] + a - local.offsets[reflection];
                if let Some(g) = geometry {
                    for (j, name) in g.parameter_names.iter().enumerate() {
                        physical[index("lattice", &phase.id, name)?].add_global(
                            i,
                            local.values[a * local.parameter_count + 1]
                                * g.d_two_theta_d_parameters[r * g.parameter_names.len() + j],
                            n,
                        );
                    }
                }
            }
            reflection += 1;
        }
    }
    for (j, name) in PAWLEY_PROFILE_NAMES.iter().enumerate() {
        let col = index("profile", "instrument", name)?;
        physical[col] = PawleyColumn {
            start: 0,
            values: global.values[j * n..(j + 1) * n].to_vec(),
        };
    }
    let mut background_y = input.pattern.background_y.clone();
    if let Some(bg) = &input.background {
        let names = bg.parameter_names();
        let coefficients = names
            .iter()
            .map(|name| get("background", bg.background_id(), name))
            .collect::<Result<Vec<_>, _>>()?;
        let bg = bg.replace_coefficients(&coefficients).map_err(err)?;
        let basis = bg.basis(&input.pattern.x_deg).map_err(err)?;
        for (i, value) in bg
            .calculate(&input.pattern.x_deg)
            .map_err(err)?
            .iter()
            .enumerate()
        {
            background_y[i] += value;
        }
        for (j, name) in names.iter().enumerate() {
            let col = index("background", bg.background_id(), name)?;
            physical[col] = PawleyColumn {
                start: 0,
                values: (0..n).map(|i| basis.values[i * names.len() + j]).collect(),
            };
        }
    }
    let calculated_y: Vec<f64> = accumulation
        .y
        .iter()
        .zip(&background_y)
        .map(|(a, b)| a + b)
        .collect();
    let chain = transform.derivative_matrix().map_err(err)?;
    let jacobian_operator = PawleyJacobian::new(n, k, physical, &chain.values)?;
    let jacobian = dense
        .then(|| jacobian_operator.materialize(max_elements))
        .transpose()?;
    let weights: Vec<f64> = (0..n)
        .map(|i| {
            if input.pattern.mask.as_ref().is_some_and(|m| !m[i]) {
                0.0
            } else {
                1.0
            }
        })
        .collect();
    let inactive_columns = jacobian_operator
        .column_norms(&weights)?
        .iter()
        .enumerate()
        .filter_map(|(i, v)| (*v == 0.0).then_some(i))
        .collect::<Vec<_>>();
    let residuals = evaluate_residuals(
        &input.pattern,
        &calculated_y,
        ResidualOptions {
            use_uncertainty,
            parameter_count: k - inactive_columns.len(),
        },
    )
    .map_err(err)?;
    let keys: Vec<(String, String)> = input
        .phases
        .iter()
        .flat_map(|p| p.reflection_ids.iter().map(|id| (p.id.clone(), id.clone())))
        .collect();
    let mut unobserved_reflections = Vec::new();
    let mut by_position = std::collections::BTreeMap::<u64, Vec<usize>>::new();
    for (r, key) in keys.iter().enumerate() {
        let observed = (local.offsets[r]..local.offsets[r + 1]).any(|a| {
            let i = local.starts[r] + a - local.offsets[r];
            residuals.included[i] && local.values[a * local.parameter_count] != 0.0
        });
        if observed {
            by_position
                .entry(positions[r].to_bits())
                .or_default()
                .push(r);
        } else {
            unobserved_reflections.push(key.clone());
        }
    }
    let coincident_groups = by_position
        .into_values()
        .filter(|v| v.len() > 1)
        .map(|members| PawleyCoincidentGroup {
            total_intensity: members.iter().map(|r| intensities[*r]).sum(),
            members: members.iter().map(|r| keys[*r].clone()).collect(),
        })
        .collect();
    Ok(PawleyEvaluation {
        calculated_y,
        background_y,
        intensities,
        positions,
        jacobian,
        jacobian_operator,
        residuals,
        inactive_columns,
        unobserved_reflections,
        coincident_groups,
    })
}
/// Weighted sample-major Jacobian and residual, with excluded rows exactly zero.
pub(crate) fn weighted(
    input: &PawleyInput,
    evaluation: &PawleyEvaluation,
    uncertainty: bool,
) -> (DMatrix<f64>, DVector<f64>) {
    let mut j = evaluation
        .jacobian
        .as_ref()
        .expect("dense evaluator required")
        .clone();
    for i in 0..j.nrows() {
        let w = if !evaluation.residuals.included[i] {
            0.0
        } else if uncertainty {
            input
                .pattern
                .uncertainty
                .as_ref()
                .map_or(1.0, |s| 1.0 / s[i])
        } else {
            1.0
        };
        j.row_mut(i).scale_mut(w);
    }
    (
        j,
        DVector::from_iterator(
            evaluation.residuals.weighted_residual.len(),
            evaluation
                .residuals
                .weighted_residual
                .iter()
                .zip(&evaluation.residuals.included)
                .map(|(r, included)| if *included { *r } else { 0.0 }),
        ),
    )
}
