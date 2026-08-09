//! Native fixed-reflection Le Bail integrated-intensity extraction.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

use nalgebra::{DMatrix, DVector};
use phasesmith_core::{
    Accumulation, ConstantWavelengthInstrument, CwContributionsError, GridView,
    OwnedCwContributionArrays, OwnedCwContributions, ProfileError, SupportPolicy,
    accumulate_cw_contributions_batch_with_context,
};
use phasesmith_crystallography::UnitCell;
use phasesmith_execution::{ExecutionPolicy, ExecutionPolicyError};
use phasesmith_model::{DomainError, PatternRecord};

use crate::{
    Constraint, ConstraintError, ConstraintTransform, DiagnosticValue, GeneratedLatticeDomain,
    LatticeError, LatticeReflectionDomain, ParameterBounds, ParameterError, ParameterKey,
    ParameterSet, ParameterSpec, RefinementEventKind, RefinementLimits, RefinementRuntime,
    ResidualError, ResidualEvaluation, ResidualOptions, RuntimeError, TerminationReason,
    cw_lattice_geometry, evaluate_residuals,
};

const INSTRUMENT_PARAMETER_NAMES: [&str; 5] = ["u_deg2", "v_deg2", "w_deg2", "x_deg", "y_deg"];
const LATTICE_PARAMETER_NAMES: [&str; 6] = [
    "a_angstrom",
    "b_angstrom",
    "c_angstrom",
    "alpha_deg",
    "beta_deg",
    "gamma_deg",
];

/// One fixed reflection phase whose integrated intensities are extracted.
#[derive(Clone, Debug, PartialEq)]
pub struct LeBailPhase {
    phase_id: String,
    name: String,
    reflection_ids: Vec<String>,
    hkl: Vec<[i32; 3]>,
    d_spacing_angstrom: Vec<f64>,
    two_theta_deg: Vec<f64>,
    integrated_intensity: Vec<f64>,
    scale: f64,
    preserve_unobserved: Vec<bool>,
    cell: Option<UnitCell>,
    reflection_domain: Option<LatticeReflectionDomain>,
}

impl LeBailPhase {
    /// Validate and own one ordered fixed-reflection phase.
    ///
    /// `preserve_unobserved` marks generated reflections outside the currently
    /// visible interval. Their checkpoint intensity is retained when their
    /// finite profile support has no included samples. Pass an empty vector for
    /// ordinary fixed reflection lists.
    ///
    /// # Errors
    ///
    /// Returns [`LeBailError::InvalidPhase`] for invalid identity, shape, or
    /// numerical state.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        phase_id: impl Into<String>,
        name: impl Into<String>,
        reflection_ids: Vec<String>,
        hkl: Vec<[i32; 3]>,
        d_spacing_angstrom: Vec<f64>,
        two_theta_deg: Vec<f64>,
        integrated_intensity: Vec<f64>,
        scale: f64,
        preserve_unobserved: Vec<bool>,
    ) -> Result<Self, LeBailError> {
        let phase = Self {
            phase_id: phase_id.into(),
            name: name.into(),
            reflection_ids,
            hkl,
            d_spacing_angstrom,
            two_theta_deg,
            integrated_intensity,
            scale,
            preserve_unobserved,
            cell: None,
            reflection_domain: None,
        };
        phase.validate()?;
        Ok(phase)
    }

    /// Generate and own a bounded dynamic-lattice phase.
    ///
    /// # Errors
    ///
    /// Returns [`LeBailError`] if the cell lies outside the domain, reflection
    /// generation fails, or the resulting phase state is invalid.
    pub fn from_lattice_domain(
        phase_id: impl Into<String>,
        name: impl Into<String>,
        cell: UnitCell,
        scale: f64,
        reflection_domain: LatticeReflectionDomain,
    ) -> Result<Self, LeBailError> {
        let generated = reflection_domain
            .generate(cell, None)
            .map_err(LeBailError::Lattice)?;
        let mut phase = Self::new(
            phase_id,
            name,
            generated.reflection_ids.clone(),
            generated.hkl.clone(),
            generated.d_spacing_angstrom.clone(),
            generated.two_theta_deg.clone(),
            generated.integrated_intensity.clone(),
            scale,
            generated.visible.iter().map(|visible| !visible).collect(),
        )?;
        phase.cell = Some(cell);
        phase.reflection_domain = Some(reflection_domain);
        phase.validate()?;
        Ok(phase)
    }

    fn validate(&self) -> Result<(), LeBailError> {
        validate_stable_label("phase_id", &self.phase_id)?;
        if self.name.trim().is_empty() {
            return Err(invalid_phase("phase name must be non-empty"));
        }
        let count = self.reflection_ids.len();
        if count == 0 {
            return Err(invalid_phase("at least one reflection is required"));
        }
        if self.hkl.len() != count
            || self.d_spacing_angstrom.len() != count
            || self.two_theta_deg.len() != count
            || self.integrated_intensity.len() != count
            || (!self.preserve_unobserved.is_empty() && self.preserve_unobserved.len() != count)
        {
            return Err(invalid_phase("reflection arrays must have equal lengths"));
        }
        let mut identities = std::collections::BTreeSet::new();
        for reflection_id in &self.reflection_ids {
            validate_stable_label("reflection_id", reflection_id)?;
            if !identities.insert(reflection_id) {
                return Err(invalid_phase(
                    "reflection IDs must be unique within a phase",
                ));
            }
        }
        if self
            .d_spacing_angstrom
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(invalid_phase("d-spacings must be positive and finite"));
        }
        if self
            .two_theta_deg
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0 || *value >= 180.0)
        {
            return Err(invalid_phase(
                "reflection positions must lie strictly inside (0, 180) degrees",
            ));
        }
        if self
            .integrated_intensity
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(invalid_phase(
                "integrated intensities must be non-negative and finite",
            ));
        }
        if !self.scale.is_finite() || self.scale < 0.0 {
            return Err(invalid_phase("phase scale must be non-negative and finite"));
        }
        match (&self.cell, &self.reflection_domain) {
            (None, None) => {}
            (Some(cell), Some(domain)) => {
                domain.validate_cell(*cell).map_err(LeBailError::Lattice)?;
                if self.preserve_unobserved.len() != count {
                    return Err(invalid_phase(
                        "dynamic phases require one visibility marker per reflection",
                    ));
                }
            }
            _ => {
                return Err(invalid_phase(
                    "dynamic phases require both a cell and reflection domain",
                ));
            }
        }
        Ok(())
    }

    /// Borrow the stable phase ID.
    #[must_use]
    pub fn phase_id(&self) -> &str {
        &self.phase_id
    }

    /// Borrow the display name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrow reflection IDs in calculation order.
    #[must_use]
    pub fn reflection_ids(&self) -> &[String] {
        &self.reflection_ids
    }

    /// Borrow Miller indices in reflection order.
    #[must_use]
    pub fn hkl(&self) -> &[[i32; 3]] {
        &self.hkl
    }

    /// Borrow d-spacings in ångströms.
    #[must_use]
    pub fn d_spacing_angstrom(&self) -> &[f64] {
        &self.d_spacing_angstrom
    }

    /// Borrow fixed reflection positions in degrees `2theta`.
    #[must_use]
    pub fn two_theta_deg(&self) -> &[f64] {
        &self.two_theta_deg
    }

    /// Borrow current integrated intensities.
    #[must_use]
    pub fn integrated_intensity(&self) -> &[f64] {
        &self.integrated_intensity
    }

    /// Replace integrated intensities while preserving phase identity/geometry.
    ///
    /// # Errors
    ///
    /// Returns [`LeBailError`] for a shape mismatch, negative value, or
    /// non-finite value.
    pub fn with_integrated_intensities(&self, values: &[f64]) -> Result<Self, LeBailError> {
        self.replace_intensities(values)
    }

    /// Return the phase scale.
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }

    /// Borrow the preserve-if-unobserved mask.
    #[must_use]
    pub fn preserve_unobserved(&self) -> &[bool] {
        &self.preserve_unobserved
    }

    /// Return the current cell for a dynamic-lattice phase.
    #[must_use]
    pub const fn cell(&self) -> Option<UnitCell> {
        self.cell
    }

    /// Borrow the guarded reflection domain for a dynamic-lattice phase.
    #[must_use]
    pub const fn reflection_domain(&self) -> Option<&LatticeReflectionDomain> {
        self.reflection_domain.as_ref()
    }

    /// Regenerate a dynamic phase at another accepted bounded cell.
    ///
    /// Intensities transfer by stable reflection ID and new families use the
    /// domain's declared initial intensity.
    ///
    /// # Errors
    ///
    /// Returns [`LeBailError`] for a fixed phase, out-of-bounds cell, or
    /// reflection-generation failure.
    pub fn regenerate_lattice_at_cell(&self, cell: UnitCell) -> Result<Self, LeBailError> {
        let domain = self
            .reflection_domain
            .as_ref()
            .ok_or_else(|| invalid_phase("only dynamic phases have a lattice reflection domain"))?;
        let previous = self
            .reflection_ids
            .iter()
            .cloned()
            .zip(self.integrated_intensity.iter().copied())
            .collect::<BTreeMap<_, _>>();
        let generated = domain
            .generate(cell, Some(&previous))
            .map_err(LeBailError::Lattice)?;
        self.replace_generated_domain(cell, generated)
    }

    fn replace_intensities(&self, values: &[f64]) -> Result<Self, LeBailError> {
        if values.len() != self.integrated_intensity.len()
            || values
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(invalid_phase(
                "replacement intensities must match and remain non-negative",
            ));
        }
        let mut phase = self.clone();
        phase.integrated_intensity.copy_from_slice(values);
        Ok(phase)
    }

    fn replace_scale_and_positions(
        &self,
        scale: f64,
        positions: Vec<f64>,
    ) -> Result<Self, LeBailError> {
        let mut phase = self.clone();
        phase.scale = scale;
        phase.two_theta_deg = positions;
        phase.validate()?;
        Ok(phase)
    }

    fn replace_cell_geometry(
        &self,
        cell: UnitCell,
        wavelength_angstrom: f64,
    ) -> Result<Self, LeBailError> {
        let domain = self.reflection_domain.as_ref().ok_or_else(|| {
            invalid_phase("lattice parameters require a bounded reflection domain")
        })?;
        domain.validate_cell(cell).map_err(LeBailError::Lattice)?;
        let geometry = cw_lattice_geometry(
            domain.parameterization(),
            cell,
            &self.hkl,
            wavelength_angstrom,
        )
        .map_err(LeBailError::Lattice)?;
        let mut phase = self.clone();
        phase.cell = Some(cell);
        phase.d_spacing_angstrom = geometry.d_spacing_angstrom;
        phase.two_theta_deg = geometry.two_theta_deg;
        phase.validate()?;
        Ok(phase)
    }

    fn replace_generated_domain(
        &self,
        cell: UnitCell,
        generated: GeneratedLatticeDomain,
    ) -> Result<Self, LeBailError> {
        let mut phase = self.clone();
        phase.cell = Some(cell);
        phase.reflection_ids = generated.reflection_ids;
        phase.hkl = generated.hkl;
        phase.d_spacing_angstrom = generated.d_spacing_angstrom;
        phase.two_theta_deg = generated.two_theta_deg;
        phase.integrated_intensity = generated.integrated_intensity;
        phase.preserve_unobserved = generated
            .visible
            .into_iter()
            .map(|visible| !visible)
            .collect();
        phase.validate()?;
        Ok(phase)
    }
}

/// Return the stable key for one supported CW profile coefficient.
///
/// # Errors
///
/// Returns [`LeBailError`] for an unsupported name.
pub fn lebail_instrument_parameter_key(name: &str) -> Result<ParameterKey, LeBailError> {
    if !INSTRUMENT_PARAMETER_NAMES.contains(&name) {
        return Err(LeBailError::UnsupportedParameter {
            label: format!("instrument[cw].{name}"),
        });
    }
    ParameterKey::new("instrument", "cw", name).map_err(LeBailError::Parameter)
}

/// Return the stable key for a phase scale.
///
/// # Errors
///
/// Returns [`LeBailError`] for an invalid phase ID.
pub fn lebail_phase_scale_key(phase_id: &str) -> Result<ParameterKey, LeBailError> {
    ParameterKey::new("phase", phase_id, "scale").map_err(LeBailError::Parameter)
}

/// Return the stable key for one symmetry-independent lattice variable.
///
/// # Errors
///
/// Returns [`LeBailError`] for an unsupported name or invalid phase ID.
pub fn lebail_lattice_parameter_key(
    phase_id: &str,
    name: &str,
) -> Result<ParameterKey, LeBailError> {
    if !LATTICE_PARAMETER_NAMES.contains(&name) {
        return Err(LeBailError::UnsupportedParameter {
            label: format!("lattice[{phase_id}].{name}"),
        });
    }
    ParameterKey::new("lattice", phase_id, name).map_err(LeBailError::Parameter)
}

/// Return the stable key for one independent reflection position.
///
/// # Errors
///
/// Returns [`LeBailError`] for invalid identity segments.
pub fn lebail_reflection_position_key(
    phase_id: &str,
    reflection_id: &str,
) -> Result<ParameterKey, LeBailError> {
    ParameterKey::new(
        "reflection",
        format!("{phase_id}/{reflection_id}"),
        "two_theta_deg",
    )
    .map_err(LeBailError::Parameter)
}

/// Build bounded typed specifications for selected fixed-geometry parameters.
///
/// # Errors
///
/// Returns [`LeBailError`] for unsupported names or invalid phase state.
pub fn build_lebail_parameter_set(
    instrument: ConstantWavelengthInstrument,
    phases: &[LeBailPhase],
    instrument_parameters: &[&str],
    phase_scales: bool,
    reflection_positions: bool,
) -> Result<ParameterSet, LeBailError> {
    build_lebail_parameter_set_with_lattice(
        instrument,
        phases,
        instrument_parameters,
        phase_scales,
        reflection_positions,
        false,
    )
}

/// Build bounded typed specifications including optional lattice variables.
///
/// # Errors
///
/// Returns [`LeBailError`] for unsupported selections or invalid phase state.
pub fn build_lebail_parameter_set_with_lattice(
    instrument: ConstantWavelengthInstrument,
    phases: &[LeBailPhase],
    instrument_parameters: &[&str],
    phase_scales: bool,
    reflection_positions: bool,
    lattice_parameters: bool,
) -> Result<ParameterSet, LeBailError> {
    if lattice_parameters && reflection_positions {
        return Err(invalid_phase(
            "lattice parameters and independent reflection positions are redundant",
        ));
    }
    let mut specs = Vec::new();
    for name in instrument_parameters {
        let key = lebail_instrument_parameter_key(name)?;
        let value = instrument_parameter(instrument, name)
            .ok_or_else(|| LeBailError::UnsupportedParameter { label: key.label() })?;
        specs.push(
            ParameterSpec::new(
                key,
                value,
                if name.ends_with("deg2") {
                    "degree^2"
                } else {
                    "degree"
                },
                ParameterBounds::default(),
                value.abs().max(if name.ends_with("deg2") {
                    1.0e-5
                } else {
                    1.0e-4
                }),
                true,
            )
            .map_err(LeBailError::Parameter)?,
        );
    }
    for phase in phases {
        if lattice_parameters {
            append_lattice_parameter_specs(&mut specs, phase)?;
        }
        if phase_scales {
            specs.push(
                ParameterSpec::new(
                    lebail_phase_scale_key(phase.phase_id())?,
                    phase.scale(),
                    "dimensionless",
                    ParameterBounds::new(0.0, f64::INFINITY).map_err(LeBailError::Parameter)?,
                    phase.scale().max(1.0),
                    true,
                )
                .map_err(LeBailError::Parameter)?,
            );
        }
        if reflection_positions {
            if phase.reflection_domain.is_some() {
                return Err(invalid_phase(
                    "independent reflection positions require fixed-topology phases",
                ));
            }
            for (reflection_id, position) in phase.reflection_ids.iter().zip(&phase.two_theta_deg) {
                specs.push(
                    ParameterSpec::new(
                        lebail_reflection_position_key(phase.phase_id(), reflection_id)?,
                        *position,
                        "degree_2theta",
                        ParameterBounds::new(
                            f64::from_bits(1),
                            f64::from_bits(180.0_f64.to_bits() - 1),
                        )
                        .map_err(LeBailError::Parameter)?,
                        0.01,
                        true,
                    )
                    .map_err(LeBailError::Parameter)?,
                );
            }
        }
    }
    ParameterSet::new(specs).map_err(LeBailError::Parameter)
}

fn append_lattice_parameter_specs(
    specs: &mut Vec<ParameterSpec>,
    phase: &LeBailPhase,
) -> Result<(), LeBailError> {
    let cell = phase
        .cell
        .ok_or_else(|| invalid_phase("lattice parameters require bounded dynamic phases"))?;
    let domain = phase
        .reflection_domain
        .as_ref()
        .ok_or_else(|| invalid_phase("lattice parameters require bounded dynamic phases"))?;
    let values = domain
        .parameterization()
        .values_from_cell(cell)
        .map_err(LeBailError::Lattice)?;
    for (((name, value), lower), upper) in domain
        .parameterization()
        .parameter_names()
        .iter()
        .zip(values)
        .zip(domain.bounds().lower())
        .zip(domain.bounds().upper())
    {
        specs.push(
            ParameterSpec::new(
                lebail_lattice_parameter_key(phase.phase_id(), name)?,
                value,
                if name.ends_with("_angstrom") {
                    "angstrom"
                } else {
                    "degree"
                },
                ParameterBounds::new(*lower, *upper).map_err(LeBailError::Parameter)?,
                value.abs().max(1.0),
                true,
            )
            .map_err(LeBailError::Parameter)?,
        );
    }
    Ok(())
}

/// Observations, instrument, and ordered fixed-reflection phases.
#[derive(Clone, Debug, PartialEq)]
pub struct LeBailInput {
    /// Observed pattern and fixed supplied background.
    pub pattern: PatternRecord,
    /// Constant-wavelength U/V/W/X/Y profile.
    pub instrument: ConstantWavelengthInstrument,
    /// Ordered non-empty phase list.
    pub phases: Vec<LeBailPhase>,
    /// Optional typed profile parameter set.
    pub parameters: Option<ParameterSet>,
    /// Ordered fixed/affine/linear parameter constraints.
    pub constraints: Vec<Constraint>,
}

impl LeBailInput {
    /// Validate a complete fixed-reflection Le Bail request.
    ///
    /// # Errors
    ///
    /// Returns [`LeBailError`] for invalid observations, instrument, phase
    /// state, or repeated phase IDs.
    pub fn new(
        pattern: PatternRecord,
        instrument: ConstantWavelengthInstrument,
        phases: Vec<LeBailPhase>,
    ) -> Result<Self, LeBailError> {
        pattern.validate().map_err(LeBailError::Pattern)?;
        if pattern.observed_y.is_none() {
            return Err(LeBailError::MissingObservations);
        }
        instrument
            .validate()
            .map_err(|error| LeBailError::Profile {
                message: error.to_string(),
            })?;
        if phases.is_empty() {
            return Err(invalid_phase("at least one phase is required"));
        }
        let mut phase_ids = std::collections::BTreeSet::new();
        for phase in &phases {
            phase.validate()?;
            if phase.reflection_domain.as_ref().is_some_and(|domain| {
                domain.wavelength_angstrom().to_bits() != instrument.wavelength_angstrom.to_bits()
            }) {
                return Err(invalid_phase(
                    "dynamic phase wavelength must match the Le Bail instrument",
                ));
            }
            if !phase_ids.insert(phase.phase_id()) {
                return Err(invalid_phase("phase IDs must be unique"));
            }
        }
        Ok(Self {
            pattern,
            instrument,
            phases,
            parameters: None,
            constraints: Vec::new(),
        })
    }

    /// Validate a request with optional analytical profile parameters.
    ///
    /// # Errors
    ///
    /// Returns [`LeBailError`] for unsupported keys, domain/value mismatch, or
    /// an invalid constraint graph.
    pub fn new_with_parameters(
        pattern: PatternRecord,
        instrument: ConstantWavelengthInstrument,
        phases: Vec<LeBailPhase>,
        parameters: ParameterSet,
        constraints: Vec<Constraint>,
    ) -> Result<Self, LeBailError> {
        let mut input = Self::new(pattern, instrument, phases)?;
        validate_parameter_selection(&input.phases, &parameters)?;
        domain_parameter_values(input.instrument, &input.phases, &parameters)?;
        ConstraintTransform::new(parameters.clone(), constraints.clone())
            .map_err(LeBailError::Constraint)?;
        input.parameters = Some(parameters);
        input.constraints = constraints;
        Ok(input)
    }
}

/// Deterministic controls for fixed-reflection extraction.
#[derive(Clone, Debug, PartialEq)]
pub struct LeBailOptions {
    /// Maximum accepted iterations.
    pub max_iterations: usize,
    /// Minimum accepted iterations before convergence.
    pub min_iterations: usize,
    /// Maximum relative integrated-intensity change for convergence.
    pub intensity_tolerance: f64,
    /// Absolute Rwp change for convergence.
    pub rwp_tolerance: f64,
    /// Multiplicative redistribution damping in `(0, 1]`.
    pub redistribution_damping: f64,
    /// Minimum calculated profile accepted in the observed/calculated ratio.
    pub minimum_calculated: f64,
    /// Positive starting and relative-change denominator floor.
    pub initial_intensity_floor: f64,
    /// Whether supplied one-sigma uncertainty is used.
    pub use_uncertainty: bool,
    /// Non-negative diagonal regularization for profile normal equations.
    pub profile_damping: f64,
    /// Maximum absolute free-parameter step in scaled coordinates.
    pub max_scaled_parameter_step: f64,
    /// Number of profile-step halvings after the initial candidate.
    pub max_profile_backtracks: usize,
    /// Correlation threshold used by optional unresolved-group diagnostics.
    pub unresolved_correlation: f64,
    /// Whether coincident reflection rank diagnostics are calculated.
    pub diagnose_rank_deficiency: bool,
    /// Finite profile support in FWHM units.
    pub support_fwhm: f64,
    /// Persistent bounded native worker policy.
    pub execution: ExecutionPolicy,
}

impl LeBailOptions {
    /// Validate all convergence and execution controls.
    ///
    /// # Errors
    ///
    /// Returns [`LeBailError::InvalidOptions`] for an invalid control.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_iterations: usize,
        min_iterations: usize,
        intensity_tolerance: f64,
        rwp_tolerance: f64,
        redistribution_damping: f64,
        minimum_calculated: f64,
        initial_intensity_floor: f64,
        use_uncertainty: bool,
        unresolved_correlation: f64,
        diagnose_rank_deficiency: bool,
        support_fwhm: f64,
        execution: ExecutionPolicy,
    ) -> Result<Self, LeBailError> {
        let options = Self {
            max_iterations,
            min_iterations,
            intensity_tolerance,
            rwp_tolerance,
            redistribution_damping,
            minimum_calculated,
            initial_intensity_floor,
            use_uncertainty,
            profile_damping: 1.0e-10,
            max_scaled_parameter_step: 0.25,
            max_profile_backtracks: 8,
            unresolved_correlation,
            diagnose_rank_deficiency,
            support_fwhm,
            execution,
        };
        options.validate()?;
        Ok(options)
    }

    fn validate(&self) -> Result<(), LeBailError> {
        if self.max_iterations == 0
            || self.min_iterations == 0
            || self.min_iterations > self.max_iterations
        {
            return Err(invalid_options(
                "iteration counts must be positive and minimum must not exceed maximum",
            ));
        }
        for (name, value) in [
            ("intensity_tolerance", self.intensity_tolerance),
            ("rwp_tolerance", self.rwp_tolerance),
            ("minimum_calculated", self.minimum_calculated),
            ("initial_intensity_floor", self.initial_intensity_floor),
            ("support_fwhm", self.support_fwhm),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(LeBailError::InvalidOptions {
                    message: format!("{name} must be positive and finite"),
                });
            }
        }
        if !self.redistribution_damping.is_finite()
            || self.redistribution_damping <= 0.0
            || self.redistribution_damping > 1.0
        {
            return Err(invalid_options("redistribution_damping must lie in (0, 1]"));
        }
        if !self.unresolved_correlation.is_finite()
            || !(0.0..=1.0).contains(&self.unresolved_correlation)
        {
            return Err(invalid_options("unresolved_correlation must lie in [0, 1]"));
        }
        if !self.profile_damping.is_finite() || self.profile_damping < 0.0 {
            return Err(invalid_options(
                "profile_damping must be non-negative and finite",
            ));
        }
        if !self.max_scaled_parameter_step.is_finite() || self.max_scaled_parameter_step <= 0.0 {
            return Err(invalid_options(
                "max_scaled_parameter_step must be positive and finite",
            ));
        }
        Ok(())
    }

    /// Replace the native profile-solver controls after validation.
    ///
    /// # Errors
    ///
    /// Returns [`LeBailError`] for invalid damping or step controls.
    pub fn with_profile_controls(
        mut self,
        profile_damping: f64,
        max_scaled_parameter_step: f64,
        max_profile_backtracks: usize,
    ) -> Result<Self, LeBailError> {
        self.profile_damping = profile_damping;
        self.max_scaled_parameter_step = max_scaled_parameter_step;
        self.max_profile_backtracks = max_profile_backtracks;
        self.validate()?;
        Ok(self)
    }

    /// Construct the scripting-compatible defaults with an explicit policy.
    ///
    /// # Errors
    ///
    /// Returns [`LeBailError`] if the controls cannot be constructed.
    pub fn scripting_defaults(execution: ExecutionPolicy) -> Result<Self, LeBailError> {
        Self::new(
            50,
            2,
            1.0e-6,
            1.0e-8,
            1.0,
            1.0e-15,
            1.0e-12,
            true,
            1.0 - 1.0e-10,
            false,
            20.0,
            execution,
        )
    }
}

/// One display-ready phase curve.
#[derive(Clone, Debug, PartialEq)]
pub struct PhasePatternComponent {
    /// Stable phase ID.
    pub phase_id: String,
    /// Sample-aligned phase contribution.
    pub y: Vec<f64>,
}

/// Native fixed-phase pattern result used by extraction and adapters.
#[derive(Clone, Debug, PartialEq)]
pub struct LeBailCalculation {
    /// Profile plus fixed supplied background.
    pub y: Vec<f64>,
    /// Sum of all phase profiles.
    pub profile_y: Vec<f64>,
    /// Fixed supplied background.
    pub background_y: Vec<f64>,
    /// Sparse local and dense global profile derivatives.
    pub accumulation: Accumulation,
    /// `(phase_id, reflection_id)` in local-Jacobian order.
    pub reflection_keys: Vec<(String, String)>,
    /// Prefix sum of phase reflection counts.
    pub phase_offsets: Vec<usize>,
    /// One diagnostic curve per phase.
    pub phase_components: Vec<PhasePatternComponent>,
}

/// One non-negative multiplicative redistribution result.
#[derive(Clone, Debug, PartialEq)]
pub struct IntensityExtractionResult {
    /// New integrated intensities in reflection order.
    pub intensities: Vec<f64>,
    /// Largest floored relative intensity change.
    pub maximum_relative_change: f64,
    /// Reflection keys without included finite support.
    pub unobserved_reflections: Vec<(String, String)>,
}

/// One accepted physical profile-parameter change.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterChange {
    /// Stable parameter identity.
    pub key: ParameterKey,
    /// Physical value before the step.
    pub before: f64,
    /// Physical value after the step.
    pub after: f64,
    /// Change divided by the parameter scale.
    pub scaled_change: f64,
}

/// One immutable accepted fixed-reflection iteration.
#[derive(Clone, Debug, PartialEq)]
pub struct LeBailIterationRecord {
    /// One-based attempted iteration.
    pub iteration: usize,
    /// Unweighted profile residual.
    pub rp: f64,
    /// Weighted profile residual.
    pub rwp: f64,
    /// Weighted residual sum of squares.
    pub chi_square: f64,
    /// Chi-square per positive residual degree of freedom.
    pub reduced_chi_square: f64,
    /// Largest relative integrated-intensity change.
    pub maximum_relative_intensity_change: f64,
    /// Euclidean norm of the accepted scaled profile step.
    pub scaled_profile_step_norm: f64,
    /// Accepted physical parameter changes.
    pub parameter_changes: Vec<ParameterChange>,
    /// Iteration warnings in deterministic order.
    pub warnings: Vec<String>,
}

/// Final stable reflection identity and intensity.
#[derive(Clone, Debug, PartialEq)]
pub struct ReflectionIntensity {
    /// Stable phase ID.
    pub phase_id: String,
    /// Stable reflection ID.
    pub reflection_id: String,
    /// Non-negative integrated intensity.
    pub integrated_intensity: f64,
}

/// Numerically coincident profile columns and their matrix rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoincidentReflectionGroup {
    /// Stable reflection keys.
    pub reflection_keys: Vec<(String, String)>,
    /// Numerical rank of the joined support matrix.
    pub rank: usize,
}

/// Complete immutable continuation state for the fixed-reflection workflow.
#[derive(Clone, Debug, PartialEq)]
pub struct LeBailCheckpoint {
    /// Number of accepted iterations.
    pub completed_iterations: usize,
    /// Current phase records and integrated intensities.
    pub phases: Vec<LeBailPhase>,
    /// Current instrument, including accepted profile changes.
    pub instrument: ConstantWavelengthInstrument,
    /// Flattened current integrated intensities.
    pub intensities: Vec<f64>,
    /// Current profile parameter set.
    pub parameters: Option<ParameterSet>,
    /// Rwp from the last non-converged accepted iteration.
    pub previous_rwp: f64,
    /// Complete accepted deterministic history.
    pub history: Vec<LeBailIterationRecord>,
}

impl LeBailCheckpoint {
    fn validate(&self) -> Result<(), LeBailError> {
        if self.completed_iterations != self.history.len() {
            return Err(LeBailError::InvalidCheckpoint {
                message: "checkpoint iteration count must equal its history length".to_owned(),
            });
        }
        if self.previous_rwp.is_nan() || self.previous_rwp == f64::NEG_INFINITY {
            return Err(LeBailError::InvalidCheckpoint {
                message: "checkpoint previous_rwp must be finite or positive infinity".to_owned(),
            });
        }
        for phase in &self.phases {
            phase.validate()?;
        }
        self.instrument
            .validate()
            .map_err(|error| LeBailError::Profile {
                message: error.to_string(),
            })?;
        if self.phases.iter().any(|phase| {
            phase.reflection_domain.as_ref().is_some_and(|domain| {
                domain.wavelength_angstrom().to_bits()
                    != self.instrument.wavelength_angstrom.to_bits()
            })
        }) {
            return Err(LeBailError::InvalidCheckpoint {
                message: "checkpoint dynamic phase wavelength must match its instrument".to_owned(),
            });
        }
        if let Some(parameters) = &self.parameters {
            validate_parameter_selection(&self.phases, parameters)?;
            let domain_values = domain_parameter_values(self.instrument, &self.phases, parameters)?;
            if domain_values != parameters.values() {
                return Err(LeBailError::InvalidCheckpoint {
                    message: "checkpoint parameters disagree with its live domain".to_owned(),
                });
            }
        }
        let expected = self.phases.iter().map(reflection_count).sum::<usize>();
        if self.intensities.len() != expected
            || self
                .intensities
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(LeBailError::InvalidCheckpoint {
                message: "checkpoint intensities must match its phases".to_owned(),
            });
        }
        if flatten_intensities(&self.phases) != self.intensities {
            return Err(LeBailError::InvalidCheckpoint {
                message: "checkpoint phase and flattened intensities disagree".to_owned(),
            });
        }
        if self
            .history
            .iter()
            .enumerate()
            .any(|(index, record)| record.iteration != index + 1)
        {
            return Err(LeBailError::InvalidCheckpoint {
                message: "checkpoint history iterations must be contiguous and one-based"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

/// Complete native fixed-reflection Le Bail result.
#[derive(Clone, Debug, PartialEq)]
pub struct LeBailResult {
    /// Final calculated pattern and sparse derivative storage.
    pub calculation: LeBailCalculation,
    /// Final phases.
    pub phases: Vec<LeBailPhase>,
    /// Final constant-wavelength profile.
    pub instrument: ConstantWavelengthInstrument,
    /// Final labeled integrated intensities.
    pub intensities: Vec<ReflectionIntensity>,
    /// Final residual arrays and metrics.
    pub metrics: ResidualEvaluation,
    /// Accepted deterministic iteration history.
    pub history: Vec<LeBailIterationRecord>,
    /// Stable termination category.
    pub termination_reason: TerminationReason,
    /// Optional unresolved reflection diagnostics.
    pub rank_deficient_groups: Vec<CoincidentReflectionGroup>,
    /// Final typed profile parameters.
    pub parameters: Option<ParameterSet>,
    /// Row-major free-parameter covariance, if identifiable.
    pub covariance: Option<CovarianceMatrix>,
    /// Complete restart state.
    pub checkpoint: LeBailCheckpoint,
}

/// Square row-major covariance over scaled free parameters.
///
/// The inverse weighted normal matrix is unscaled for supplied uncertainties;
/// unit-weight fits estimate their noise scale from the reduced chi-square.
#[derive(Clone, Debug, PartialEq)]
pub struct CovarianceMatrix {
    /// Matrix dimension.
    pub size: usize,
    /// Row-major values with length `size * size`.
    pub values: Vec<f64>,
}

/// Calculate all fixed phases through one native fused accumulation.
///
/// # Errors
///
/// Returns [`LeBailError`] for invalid pattern/profile state or allocation.
pub fn calculate_lebail_pattern(
    pattern: &PatternRecord,
    instrument: ConstantWavelengthInstrument,
    phases: &[LeBailPhase],
    support_fwhm: f64,
    execution: &ExecutionPolicy,
) -> Result<LeBailCalculation, LeBailError> {
    pattern.validate().map_err(LeBailError::Pattern)?;
    if phases.is_empty() {
        return Err(invalid_phase("at least one phase is required"));
    }
    if !support_fwhm.is_finite() || support_fwhm <= 0.0 {
        return Err(invalid_options("support_fwhm must be positive and finite"));
    }
    let reflection_count = phases.iter().map(reflection_count).sum::<usize>();
    let mut positions = Vec::with_capacity(reflection_count);
    let mut intensities = Vec::with_capacity(reflection_count);
    let mut multipliers = Vec::with_capacity(reflection_count);
    let mut reflection_keys = Vec::with_capacity(reflection_count);
    let mut phase_offsets = Vec::with_capacity(phases.len() + 1);
    let phase_derivative_count = phases
        .len()
        .checked_mul(reflection_count)
        .ok_or(LeBailError::SizeOverflow)?;
    let mut derivative_multipliers = vec![0.0; phase_derivative_count];
    phase_offsets.push(0);
    for (phase_index, phase) in phases.iter().enumerate() {
        phase.validate()?;
        let begin = positions.len();
        positions.extend_from_slice(&phase.two_theta_deg);
        intensities.extend_from_slice(&phase.integrated_intensity);
        multipliers.extend(std::iter::repeat_n(phase.scale, reflection_count_of(phase)));
        reflection_keys.extend(
            phase
                .reflection_ids
                .iter()
                .map(|reflection_id| (phase.phase_id.clone(), reflection_id.clone())),
        );
        let end = positions.len();
        derivative_multipliers
            [phase_index * reflection_count + begin..phase_index * reflection_count + end]
            .fill(1.0);
        phase_offsets.push(end);
    }
    let contributions = OwnedCwContributions::new(
        reflection_count,
        phases.len(),
        OwnedCwContributionArrays {
            gaussian_variance_deg2: vec![0.0; reflection_count],
            lorentzian_fwhm_deg: vec![0.0; reflection_count],
            intensity_multiplier: multipliers,
            d_gaussian_variance_d_position: vec![0.0; reflection_count],
            d_lorentzian_fwhm_d_position: vec![0.0; reflection_count],
            d_intensity_multiplier_d_position: vec![0.0; reflection_count],
            d_gaussian_variance_d_parameters: vec![0.0; phase_derivative_count],
            d_lorentzian_fwhm_d_parameters: vec![0.0; phase_derivative_count],
            d_intensity_multiplier_d_parameters: derivative_multipliers,
        },
    )
    .map_err(LeBailError::Calculation)?;
    let grid = GridView::new(&pattern.x_deg).map_err(LeBailError::Grid)?;
    let accumulation = accumulate_cw_contributions_batch_with_context(
        grid,
        &positions,
        &intensities,
        instrument,
        contributions.as_view(),
        SupportPolicy::FwhmMultiple(support_fwhm),
        execution.context(),
    )
    .map_err(LeBailError::Calculation)?;
    let profile_y = accumulation.y.clone();
    let y = profile_y
        .iter()
        .zip(&pattern.background_y)
        .map(|(profile, background)| profile + background)
        .collect::<Vec<_>>();
    let mut phase_components = Vec::with_capacity(phases.len());
    for (phase_index, phase) in phases.iter().enumerate() {
        let mut phase_y = vec![0.0; pattern.sample_count()];
        let first = phase_offsets[phase_index];
        let last = phase_offsets[phase_index + 1];
        for (reflection, intensity) in intensities.iter().enumerate().take(last).skip(first) {
            let begin = accumulation.derivatives.local.offsets[reflection];
            let end = accumulation.derivatives.local.offsets[reflection + 1];
            let start = accumulation.derivatives.local.starts[reflection];
            for active in begin..end {
                let sample = start + active - begin;
                phase_y[sample] += intensity
                    * accumulation.derivatives.local.values
                        [active * accumulation.derivatives.local.parameter_count];
            }
        }
        phase_components.push(PhasePatternComponent {
            phase_id: phase.phase_id.clone(),
            y: phase_y,
        });
    }
    Ok(LeBailCalculation {
        y,
        profile_y,
        background_y: pattern.background_y.clone(),
        accumulation,
        reflection_keys,
        phase_offsets,
        phase_components,
    })
}

/// Return deterministic positive starting intensities in phase order.
///
/// # Errors
///
/// Returns [`LeBailError`] for invalid input or option state.
pub fn initialize_lebail_intensities(
    input: &LeBailInput,
    options: &LeBailOptions,
) -> Result<Vec<f64>, LeBailError> {
    options.validate()?;
    let values = flatten_intensities(&input.phases);
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(invalid_phase(
            "starting intensities must be non-negative and finite",
        ));
    }
    if values.iter().any(|value| *value > 0.0) {
        return Ok(values
            .into_iter()
            .map(|value| value.max(options.initial_intensity_floor))
            .collect());
    }
    let observed = input
        .pattern
        .observed_y
        .as_deref()
        .ok_or(LeBailError::MissingObservations)?;
    let weights = bin_integration_weights(&input.pattern.x_deg);
    let area = observed
        .iter()
        .zip(&input.pattern.background_y)
        .zip(weights)
        .map(|((observed, background), width)| (observed - background).max(0.0) * width)
        .sum::<f64>();
    let starting = (area / count_as_f64(values.len().max(1))).max(options.initial_intensity_floor);
    Ok(vec![starting; values.len()])
}

/// Perform one non-negative multiplicative redistribution step.
///
/// # Errors
///
/// Returns [`LeBailError`] for shape, observation, or finite-state failures.
pub fn extract_lebail_intensities(
    pattern: &PatternRecord,
    calculation: &LeBailCalculation,
    current: &[f64],
    options: &LeBailOptions,
    preserve_unobserved: &[bool],
) -> Result<IntensityExtractionResult, LeBailError> {
    options.validate()?;
    let observed = pattern
        .observed_y
        .as_deref()
        .ok_or(LeBailError::MissingObservations)?;
    let reflection_count = calculation.accumulation.derivatives.local.peak_count();
    if current.len() != reflection_count
        || current
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(LeBailError::IntensityShapeMismatch);
    }
    if preserve_unobserved.len() != reflection_count {
        return Err(LeBailError::PreserveMaskLengthMismatch);
    }
    let included = pattern
        .mask
        .clone()
        .unwrap_or_else(|| vec![true; pattern.sample_count()]);
    let ratio = observed
        .iter()
        .zip(&pattern.background_y)
        .zip(&calculation.profile_y)
        .zip(&included)
        .map(|(((observed, background), calculated), included)| {
            if *included && *calculated > options.minimum_calculated {
                (observed - background).max(0.0) / calculated
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let mut weights = bin_integration_weights(&pattern.x_deg);
    if options.use_uncertainty
        && let Some(uncertainty) = &pattern.uncertainty
    {
        for (weight, uncertainty) in weights.iter_mut().zip(uncertainty) {
            *weight /= uncertainty * uncertainty;
        }
    }
    for (weight, included) in weights.iter_mut().zip(&included) {
        if !included {
            *weight = 0.0;
        }
    }
    let local = &calculation.accumulation.derivatives.local;
    let mut updated = vec![0.0; reflection_count];
    let mut unobserved_reflections = Vec::new();
    for reflection in 0..reflection_count {
        let begin = local.offsets[reflection];
        let end = local.offsets[reflection + 1];
        let start = local.starts[reflection];
        let mut denominator = 0.0;
        let mut numerator = 0.0;
        for active in begin..end {
            let sample = start + active - begin;
            let profile = local.values[active * local.parameter_count];
            let weighted_profile = weights[sample] * profile;
            denominator += weighted_profile;
            numerator += weighted_profile * ratio[sample];
        }
        if denominator <= 0.0 {
            unobserved_reflections.push(calculation.reflection_keys[reflection].clone());
            if preserve_unobserved[reflection] {
                updated[reflection] = current[reflection];
            }
            continue;
        }
        let raw = (current[reflection] * numerator / denominator).max(0.0);
        updated[reflection] =
            current[reflection] + options.redistribution_damping * (raw - current[reflection]);
    }
    let maximum_relative_change = updated
        .iter()
        .zip(current)
        .map(|(updated, current)| {
            (updated - current).abs() / current.abs().max(options.initial_intensity_floor)
        })
        .fold(0.0_f64, f64::max);
    Ok(IntensityExtractionResult {
        intensities: updated,
        maximum_relative_change,
        unobserved_reflections,
    })
}

/// Run fixed-reflection Le Bail extraction with a workflow-owned runtime.
///
/// # Errors
///
/// Returns [`LeBailError`] for invalid state, runtime construction, profile
/// evaluation, metrics, or checkpoint delivery.
pub fn refine_lebail(
    input: &LeBailInput,
    options: &LeBailOptions,
    checkpoint: Option<&LeBailCheckpoint>,
) -> Result<LeBailResult, LeBailError> {
    let evaluations_per_iteration = options
        .max_profile_backtracks
        .checked_add(2)
        .ok_or(LeBailError::SizeOverflow)?;
    let max_evaluations = options
        .max_iterations
        .checked_mul(evaluations_per_iteration)
        .and_then(|value| value.checked_add(1))
        .ok_or(LeBailError::SizeOverflow)?;
    let limits = RefinementLimits::new(options.max_iterations, max_evaluations, None, 1)
        .map_err(LeBailError::Runtime)?;
    let mut runtime = RefinementRuntime::new(limits, None).map_err(LeBailError::Runtime)?;
    refine_lebail_with_runtime(input, options, checkpoint, &mut runtime)
}

/// Advance exactly one accepted Le Bail iteration for custom orchestration.
///
/// Pass the returned checkpoint to the next call. A cancellation-aware host
/// that needs to stop before the iteration should use
/// [`refine_lebail_with_runtime`] with a one-iteration runtime budget.
///
/// # Errors
///
/// Returns [`LeBailError`] for invalid input, options, checkpoint, or numerical
/// evaluation state.
pub fn iterate_lebail_once(
    input: &LeBailInput,
    options: &LeBailOptions,
    checkpoint: Option<&LeBailCheckpoint>,
) -> Result<LeBailResult, LeBailError> {
    let completed = checkpoint.map_or(0, |value| value.completed_iterations);
    let iteration = completed.checked_add(1).ok_or(LeBailError::SizeOverflow)?;
    let mut selected = options.clone();
    selected.min_iterations = iteration;
    selected.max_iterations = iteration;
    selected.validate()?;
    refine_lebail(input, &selected, checkpoint)
}

/// Run fixed-reflection extraction with host-owned cancellation/events/checkpoints.
///
/// The runtime should be fresh for a new run. A continuation restores its
/// accepted counter from the supplied checkpoint before numerical work begins.
///
/// # Errors
///
/// Returns [`LeBailError`] for invalid input/checkpoint state, non-normal
/// runtime failures, or numerical evaluation failures.
pub fn refine_lebail_with_runtime(
    input: &LeBailInput,
    options: &LeBailOptions,
    checkpoint: Option<&LeBailCheckpoint>,
    runtime: &mut RefinementRuntime<LeBailCheckpoint>,
) -> Result<LeBailResult, LeBailError> {
    options.validate()?;
    let mut state = restore_state(input, options, checkpoint)?;
    if let Some(checkpoint) = checkpoint {
        runtime
            .resume_accepted(checkpoint.completed_iterations)
            .map_err(LeBailError::Runtime)?;
    }
    runtime
        .emit(
            RefinementEventKind::Start,
            "lebail",
            "Le Bail extraction started",
            Vec::new(),
        )
        .map_err(LeBailError::Runtime)?;
    state.calculation = Some(calculate_lebail_pattern(
        &input.pattern,
        state.instrument,
        &state.phases,
        options.support_fwhm,
        &options.execution,
    )?);
    let termination = run_lebail_iterations(input, options, &mut state, runtime)?;
    finish_result(
        input,
        options,
        state.phases,
        state.instrument,
        &state.intensities,
        state.parameters,
        state.history,
        state.previous_rwp,
        state.calculation.ok_or(LeBailError::InternalInvariant)?,
        termination,
        runtime,
    )
}

fn run_lebail_iterations(
    input: &LeBailInput,
    options: &LeBailOptions,
    state: &mut RestoredLeBailState,
    runtime: &mut RefinementRuntime<LeBailCheckpoint>,
) -> Result<TerminationReason, LeBailError> {
    if let Err(error) = runtime.begin_evaluation() {
        return stop_reason_or_error(error);
    }
    for iteration in state.first_iteration..=options.max_iterations {
        if let Err(error) = runtime.begin_iteration(iteration) {
            return stop_reason_or_error(error);
        }
        if let Err(error) = runtime.begin_evaluation() {
            return stop_reason_or_error(error);
        }
        let candidate = match evaluate_lebail_iteration(input, options, state, runtime) {
            Ok(candidate) => candidate,
            Err(LeBailError::Runtime(error)) if normal_stop_reason(&error).is_some() => {
                return stop_reason_or_error(error);
            }
            Err(error) => return Err(error),
        };
        state.history.push(LeBailIterationRecord {
            iteration,
            rp: candidate.metrics.rp,
            rwp: candidate.metrics.rwp,
            chi_square: candidate.metrics.chi_square,
            reduced_chi_square: candidate.metrics.reduced_chi_square,
            maximum_relative_intensity_change: candidate.extraction.maximum_relative_change,
            scaled_profile_step_norm: candidate.profile_step_norm,
            parameter_changes: candidate.parameter_changes,
            warnings: candidate.warnings,
        });
        state.instrument = candidate.instrument;
        state.phases = candidate.phases;
        state.parameters = candidate.parameters;
        state.intensities = candidate.extraction.intensities;
        state.calculation = Some(candidate.calculation);
        accept_lebail_iteration(runtime, state, &candidate.metrics)?;
        if iteration >= options.min_iterations
            && candidate.extraction.maximum_relative_change < options.intensity_tolerance
            && (state.previous_rwp - candidate.metrics.rwp).abs() < options.rwp_tolerance
        {
            return Ok(TerminationReason::Converged);
        }
        state.previous_rwp = candidate.metrics.rwp;
    }
    Ok(TerminationReason::MaxIterations)
}

struct EvaluatedLeBailIteration {
    extraction: IntensityExtractionResult,
    phases: Vec<LeBailPhase>,
    calculation: LeBailCalculation,
    metrics: ResidualEvaluation,
    warnings: Vec<String>,
    instrument: ConstantWavelengthInstrument,
    parameters: Option<ParameterSet>,
    profile_step_norm: f64,
    parameter_changes: Vec<ParameterChange>,
}

fn evaluate_lebail_iteration(
    input: &LeBailInput,
    options: &LeBailOptions,
    state: &RestoredLeBailState,
    runtime: &mut RefinementRuntime<LeBailCheckpoint>,
) -> Result<EvaluatedLeBailIteration, LeBailError> {
    let mut extraction = extract_lebail_intensities(
        &input.pattern,
        state.calculation()?,
        &state.intensities,
        options,
        &flatten_preserve_mask(&state.phases),
    )?;
    let phases = replace_flat_intensities(&state.phases, &extraction.intensities)?;
    let calculation = calculate_lebail_pattern(
        &input.pattern,
        state.instrument,
        &phases,
        options.support_fwhm,
        &options.execution,
    )?;
    let profile = profile_update(
        &input.pattern,
        state.instrument,
        phases,
        calculation,
        state.parameters.as_ref(),
        &input.constraints,
        options,
        runtime,
    )?;
    extraction.intensities = flatten_intensities(&profile.phases);
    let parameter_count = free_parameter_count(profile.parameters.as_ref(), &input.constraints)?;
    let metrics = evaluate_residuals(
        &input.pattern,
        &profile.calculation.y,
        ResidualOptions {
            use_uncertainty: options.use_uncertainty,
            parameter_count,
        },
    )
    .map_err(LeBailError::Residual)?;
    let warnings = if extraction.unobserved_reflections.is_empty() {
        Vec::new()
    } else {
        vec![format!(
            "{} reflections have no included support",
            extraction.unobserved_reflections.len()
        )]
    };
    Ok(EvaluatedLeBailIteration {
        extraction,
        phases: profile.phases,
        calculation: profile.calculation,
        metrics,
        warnings: [warnings, profile.warnings].concat(),
        instrument: profile.instrument,
        parameters: profile.parameters,
        profile_step_norm: profile.step_norm,
        parameter_changes: profile.parameter_changes,
    })
}

fn accept_lebail_iteration(
    runtime: &mut RefinementRuntime<LeBailCheckpoint>,
    state: &RestoredLeBailState,
    metrics: &ResidualEvaluation,
) -> Result<(), LeBailError> {
    let checkpoint = LeBailCheckpoint {
        completed_iterations: state.history.len(),
        phases: state.phases.clone(),
        instrument: state.instrument,
        intensities: state.intensities.clone(),
        parameters: state.parameters.clone(),
        previous_rwp: metrics.rwp,
        history: state.history.clone(),
    };
    runtime
        .accept_step(Some(&checkpoint))
        .map_err(LeBailError::Runtime)?;
    runtime
        .emit(
            RefinementEventKind::Iteration,
            "lebail_iteration",
            "Le Bail iteration accepted",
            vec![
                ("rwp".to_owned(), DiagnosticValue::Float(metrics.rwp)),
                (
                    "maximum_relative_intensity_change".to_owned(),
                    DiagnosticValue::Float(
                        state
                            .history
                            .last()
                            .ok_or(LeBailError::InternalInvariant)?
                            .maximum_relative_intensity_change,
                    ),
                ),
            ],
        )
        .map_err(LeBailError::Runtime)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn finish_result(
    input: &LeBailInput,
    options: &LeBailOptions,
    phases: Vec<LeBailPhase>,
    instrument: ConstantWavelengthInstrument,
    intensities: &[f64],
    parameters: Option<ParameterSet>,
    history: Vec<LeBailIterationRecord>,
    previous_rwp: f64,
    calculation: LeBailCalculation,
    termination: TerminationReason,
    runtime: &mut RefinementRuntime<LeBailCheckpoint>,
) -> Result<LeBailResult, LeBailError> {
    let metrics = evaluate_residuals(
        &input.pattern,
        &calculation.y,
        ResidualOptions {
            use_uncertainty: options.use_uncertainty,
            parameter_count: free_parameter_count(parameters.as_ref(), &input.constraints)?,
        },
    )
    .map_err(LeBailError::Residual)?;
    let checkpoint = LeBailCheckpoint {
        completed_iterations: history.len(),
        phases: phases.clone(),
        instrument,
        intensities: intensities.to_owned(),
        parameters: parameters.clone(),
        previous_rwp: if termination == TerminationReason::Cancelled {
            previous_rwp
        } else {
            metrics.rwp
        },
        history: history.clone(),
    };
    checkpoint.validate()?;
    let labeled = calculation
        .reflection_keys
        .iter()
        .zip(intensities)
        .map(
            |((phase_id, reflection_id), intensity)| ReflectionIntensity {
                phase_id: phase_id.clone(),
                reflection_id: reflection_id.clone(),
                integrated_intensity: *intensity,
            },
        )
        .collect();
    let rank_deficient_groups = if options.diagnose_rank_deficiency {
        rank_deficient_groups(&calculation, options.unresolved_correlation)
    } else {
        Vec::new()
    };
    let covariance = covariance(
        &input.pattern,
        &calculation,
        instrument,
        &phases,
        parameters.as_ref(),
        &input.constraints,
        options.use_uncertainty,
        metrics.reduced_chi_square,
    )?;
    runtime
        .emit(
            RefinementEventKind::Termination,
            "lebail",
            "Le Bail extraction terminated",
            vec![(
                "termination_reason".to_owned(),
                DiagnosticValue::String(termination.as_str().to_owned()),
            )],
        )
        .map_err(LeBailError::Runtime)?;
    Ok(LeBailResult {
        calculation,
        phases,
        instrument,
        intensities: labeled,
        metrics,
        history,
        termination_reason: termination,
        rank_deficient_groups,
        parameters,
        covariance,
        checkpoint,
    })
}

struct RestoredLeBailState {
    phases: Vec<LeBailPhase>,
    instrument: ConstantWavelengthInstrument,
    intensities: Vec<f64>,
    history: Vec<LeBailIterationRecord>,
    parameters: Option<ParameterSet>,
    previous_rwp: f64,
    first_iteration: usize,
    calculation: Option<LeBailCalculation>,
}

impl RestoredLeBailState {
    fn calculation(&self) -> Result<&LeBailCalculation, LeBailError> {
        self.calculation
            .as_ref()
            .ok_or(LeBailError::InternalInvariant)
    }
}

fn restore_state(
    input: &LeBailInput,
    options: &LeBailOptions,
    checkpoint: Option<&LeBailCheckpoint>,
) -> Result<RestoredLeBailState, LeBailError> {
    let Some(checkpoint) = checkpoint else {
        let intensities = initialize_lebail_intensities(input, options)?;
        let phases = replace_flat_intensities(&input.phases, &intensities)?;
        return Ok(RestoredLeBailState {
            phases,
            instrument: input.instrument,
            intensities,
            history: Vec::new(),
            parameters: input.parameters.clone(),
            previous_rwp: f64::INFINITY,
            first_iteration: 1,
            calculation: None,
        });
    };
    checkpoint.validate()?;
    if checkpoint.completed_iterations >= options.max_iterations {
        return Err(LeBailError::InvalidCheckpoint {
            message: "checkpoint already reached the configured maximum iteration".to_owned(),
        });
    }
    if !phases_restart_compatible(&input.phases, &checkpoint.phases) {
        return Err(LeBailError::InvalidCheckpoint {
            message: "checkpoint phase/reflection domain does not match the input".to_owned(),
        });
    }
    let input_parameter_keys = input.parameters.as_ref().map(parameter_keys);
    let checkpoint_parameter_keys = checkpoint.parameters.as_ref().map(parameter_keys);
    if input_parameter_keys != checkpoint_parameter_keys {
        return Err(LeBailError::InvalidCheckpoint {
            message: "checkpoint parameter identities do not match the input".to_owned(),
        });
    }
    Ok(RestoredLeBailState {
        phases: checkpoint.phases.clone(),
        instrument: checkpoint.instrument,
        intensities: checkpoint.intensities.clone(),
        history: checkpoint.history.clone(),
        parameters: checkpoint.parameters.clone(),
        previous_rwp: checkpoint.previous_rwp,
        first_iteration: checkpoint.completed_iterations + 1,
        calculation: None,
    })
}

struct ProfileUpdate {
    instrument: ConstantWavelengthInstrument,
    phases: Vec<LeBailPhase>,
    calculation: LeBailCalculation,
    parameters: Option<ParameterSet>,
    step_norm: f64,
    parameter_changes: Vec<ParameterChange>,
    warnings: Vec<String>,
}

#[allow(clippy::too_many_arguments)]
// The linearization, bounded solve, and backtracking order intentionally stay
// adjacent so this numerical state transition remains auditable against the
// independent Python oracle.
#[allow(clippy::too_many_lines)]
fn profile_update(
    pattern: &PatternRecord,
    instrument: ConstantWavelengthInstrument,
    phases: Vec<LeBailPhase>,
    calculation: LeBailCalculation,
    parameters: Option<&ParameterSet>,
    constraints: &[Constraint],
    options: &LeBailOptions,
    runtime: &mut RefinementRuntime<LeBailCheckpoint>,
) -> Result<ProfileUpdate, LeBailError> {
    let Some(parameters) = parameters else {
        return Ok(ProfileUpdate {
            instrument,
            phases,
            calculation,
            parameters: None,
            step_norm: 0.0,
            parameter_changes: Vec::new(),
            warnings: Vec::new(),
        });
    };
    let domain_values = domain_parameter_values(instrument, &phases, parameters)?;
    let current = parameters
        .replace_values(&domain_values)
        .map_err(LeBailError::Parameter)?;
    let transform = ConstraintTransform::new(current.clone(), constraints.to_vec())
        .map_err(LeBailError::Constraint)?;
    if transform.free_keys().is_empty() {
        return Ok(ProfileUpdate {
            instrument,
            phases,
            calculation,
            parameters: Some(current),
            step_norm: 0.0,
            parameter_changes: Vec::new(),
            warnings: Vec::new(),
        });
    }
    let physical = parameter_columns(&calculation, &current, instrument, &phases)?;
    let derivative = transform
        .derivative_matrix()
        .map_err(LeBailError::Constraint)?;
    let chain = DMatrix::from_row_slice(derivative.rows, derivative.columns, &derivative.values);
    let jacobian = physical * chain;
    let observed = pattern
        .observed_y
        .as_deref()
        .ok_or(LeBailError::MissingObservations)?;
    let included = pattern
        .mask
        .clone()
        .unwrap_or_else(|| vec![true; pattern.sample_count()]);
    let selected_count = included.iter().filter(|value| **value).count();
    let free_count = transform.free_keys().len();
    let mut selected_jacobian = DMatrix::zeros(selected_count, free_count);
    let mut selected_residual = DVector::zeros(selected_count);
    let mut selected_row = 0;
    for sample in 0..pattern.sample_count() {
        if !included[sample] {
            continue;
        }
        let weight = if options.use_uncertainty {
            pattern
                .uncertainty
                .as_ref()
                .map_or(1.0, |values| values[sample].recip())
        } else {
            1.0
        };
        selected_residual[selected_row] = (observed[sample] - calculation.y[sample]) * weight;
        for column in 0..free_count {
            selected_jacobian[(selected_row, column)] = jacobian[(sample, column)] * weight;
        }
        selected_row += 1;
    }
    let normal = selected_jacobian.transpose() * &selected_jacobian;
    let mut warnings = Vec::new();
    if matrix_rank(&normal) != free_count {
        warnings.push("profile Jacobian is rank deficient".to_owned());
    }
    if current
        .specs()
        .iter()
        .any(|spec| spec.key().module() == "phase" && spec.key().name() == "scale")
    {
        warnings.push(
            "phase scale is not identifiable independently of extracted Le Bail intensities"
                .to_owned(),
        );
    }
    let base = transform.pack().map_err(LeBailError::Constraint)?;
    let mut lower = vec![-options.max_scaled_parameter_step; free_count];
    let mut upper = vec![options.max_scaled_parameter_step; free_count];
    for (index, key) in transform.free_keys().iter().enumerate() {
        let spec = current.spec(key).ok_or(LeBailError::InternalInvariant)?;
        lower[index] = lower[index].max(spec.bounds().lower() / spec.scale() - base[index]);
        upper[index] = upper[index].min(spec.bounds().upper() / spec.scale() - base[index]);
    }
    let rhs = selected_jacobian.transpose() * &selected_residual;
    let mut regularized = normal;
    for index in 0..free_count {
        regularized[(index, index)] += options.profile_damping;
    }
    let mut step = if let Some(solution) = regularized.lu().solve(&rhs) {
        solution
    } else {
        warnings.push("profile normal equations used least-squares fallback".to_owned());
        selected_jacobian
            .clone()
            .svd(true, true)
            .solve(&selected_residual, f64::EPSILON)
            .map_err(|_| LeBailError::LinearSolve)?
    };
    for index in 0..free_count {
        step[index] = step[index].clamp(lower[index], upper[index]);
    }
    let baseline = evaluate_residuals(
        pattern,
        &calculation.y,
        ResidualOptions {
            use_uncertainty: options.use_uncertainty,
            parameter_count: free_count,
        },
    )
    .map_err(LeBailError::Residual)?;
    let mut factor = 1.0;
    for _ in 0..=options.max_profile_backtracks {
        let trial = base
            .iter()
            .zip(step.iter())
            .map(|(base, step)| base + factor * step)
            .collect::<Vec<_>>();
        let Ok(values) = transform.unpack(&trial, true) else {
            factor *= 0.5;
            continue;
        };
        let Ok((candidate_instrument, candidate_phases)) =
            apply_parameter_values(instrument, &phases, &values)
        else {
            factor *= 0.5;
            continue;
        };
        runtime.begin_evaluation().map_err(LeBailError::Runtime)?;
        let Ok(candidate_calculation) = calculate_lebail_pattern(
            pattern,
            candidate_instrument,
            &candidate_phases,
            options.support_fwhm,
            &options.execution,
        ) else {
            factor *= 0.5;
            continue;
        };
        let candidate_metrics = evaluate_residuals(
            pattern,
            &candidate_calculation.y,
            ResidualOptions {
                use_uncertainty: options.use_uncertainty,
                parameter_count: free_count,
            },
        )
        .map_err(LeBailError::Residual)?;
        if candidate_metrics.chi_square < baseline.chi_square {
            let candidate_parameters = current
                .replace_values(&values)
                .map_err(LeBailError::Parameter)?;
            let (candidate_phases, domain_warnings, topology_changed) =
                regenerate_accepted_domains(candidate_phases)?;
            let candidate_calculation = if topology_changed {
                runtime.begin_evaluation().map_err(LeBailError::Runtime)?;
                calculate_lebail_pattern(
                    pattern,
                    candidate_instrument,
                    &candidate_phases,
                    options.support_fwhm,
                    &options.execution,
                )?
            } else {
                candidate_calculation
            };
            let parameter_changes = current
                .specs()
                .iter()
                .filter_map(|spec| {
                    let after = candidate_parameters.spec(spec.key())?.value();
                    (after.to_bits() != spec.value().to_bits()).then(|| ParameterChange {
                        key: spec.key().clone(),
                        before: spec.value(),
                        after,
                        scaled_change: (after - spec.value()) / spec.scale(),
                    })
                })
                .collect();
            return Ok(ProfileUpdate {
                instrument: candidate_instrument,
                phases: candidate_phases,
                calculation: candidate_calculation,
                parameters: Some(candidate_parameters),
                step_norm: factor * step.norm(),
                parameter_changes,
                warnings: [warnings, domain_warnings].concat(),
            });
        }
        factor *= 0.5;
    }
    warnings.push("profile step rejected by backtracking".to_owned());
    Ok(ProfileUpdate {
        instrument,
        phases,
        calculation,
        parameters: Some(current),
        step_norm: 0.0,
        parameter_changes: Vec::new(),
        warnings,
    })
}

fn domain_parameter_values(
    instrument: ConstantWavelengthInstrument,
    phases: &[LeBailPhase],
    parameters: &ParameterSet,
) -> Result<BTreeMap<ParameterKey, f64>, LeBailError> {
    let mut values = BTreeMap::new();
    for spec in parameters.specs() {
        let key = spec.key();
        let value = if key.module() == "instrument" && key.owner_id() == "cw" {
            instrument_parameter(instrument, key.name())
        } else if key.module() == "phase" && key.name() == "scale" {
            phases
                .iter()
                .find(|phase| phase.phase_id() == key.owner_id())
                .map(LeBailPhase::scale)
        } else if key.module() == "lattice" {
            phases
                .iter()
                .find(|phase| phase.phase_id() == key.owner_id())
                .and_then(|phase| {
                    let cell = phase.cell?;
                    let parameterization = phase.reflection_domain.as_ref()?.parameterization();
                    let index = parameterization
                        .parameter_names()
                        .iter()
                        .position(|name| name == key.name())?;
                    parameterization
                        .values_from_cell(cell)
                        .ok()?
                        .get(index)
                        .copied()
                })
        } else if key.module() == "reflection" && key.name() == "two_theta_deg" {
            phases.iter().find_map(|phase| {
                phase
                    .reflection_ids
                    .iter()
                    .position(|reflection_id| {
                        format!("{}/{}", phase.phase_id(), reflection_id) == key.owner_id()
                    })
                    .map(|index| phase.two_theta_deg[index])
            })
        } else {
            None
        }
        .ok_or_else(|| LeBailError::UnsupportedParameter { label: key.label() })?;
        if !spec.bounds().contains(value) {
            return Err(LeBailError::ParameterDomainOutsideBounds { label: key.label() });
        }
        values.insert(key.clone(), value);
    }
    Ok(values)
}

fn parameter_columns(
    calculation: &LeBailCalculation,
    parameters: &ParameterSet,
    instrument: ConstantWavelengthInstrument,
    phases: &[LeBailPhase],
) -> Result<DMatrix<f64>, LeBailError> {
    let samples = calculation.y.len();
    let mut matrix = DMatrix::zeros(samples, parameters.specs().len());
    let global = calculation
        .accumulation
        .derivatives
        .global
        .as_ref()
        .ok_or(LeBailError::InternalInvariant)?;
    for (column, spec) in parameters.specs().iter().enumerate() {
        let key = spec.key();
        if key.module() == "instrument" {
            let row = INSTRUMENT_PARAMETER_NAMES
                .iter()
                .position(|name| *name == key.name())
                .ok_or_else(|| LeBailError::UnsupportedParameter { label: key.label() })?;
            for sample in 0..samples {
                matrix[(sample, column)] = global.values[row * samples + sample];
            }
        } else if key.module() == "phase" {
            let phase = phases
                .iter()
                .position(|phase| phase.phase_id() == key.owner_id())
                .ok_or_else(|| LeBailError::UnsupportedParameter { label: key.label() })?;
            let row = 5 + phase;
            for sample in 0..samples {
                matrix[(sample, column)] = global.values[row * samples + sample];
            }
        } else if key.module() == "reflection" {
            let reflection = calculation
                .reflection_keys
                .iter()
                .position(|(phase_id, reflection_id)| {
                    format!("{phase_id}/{reflection_id}") == key.owner_id()
                })
                .ok_or_else(|| LeBailError::UnsupportedParameter { label: key.label() })?;
            let local = &calculation.accumulation.derivatives.local;
            let begin = local.offsets[reflection];
            let end = local.offsets[reflection + 1];
            let start = local.starts[reflection];
            for active in begin..end {
                matrix[(start + active - begin, column)] =
                    local.values[active * local.parameter_count + 1];
            }
        } else if key.module() == "lattice" {
            let phase = phases
                .iter()
                .find(|phase| phase.phase_id() == key.owner_id())
                .ok_or_else(|| LeBailError::UnsupportedParameter { label: key.label() })?;
            let cell = phase
                .cell
                .ok_or_else(|| LeBailError::UnsupportedParameter { label: key.label() })?;
            let domain = phase
                .reflection_domain
                .as_ref()
                .ok_or_else(|| LeBailError::UnsupportedParameter { label: key.label() })?;
            let geometry = cw_lattice_geometry(
                domain.parameterization(),
                cell,
                &phase.hkl,
                instrument.wavelength_angstrom,
            )
            .map_err(LeBailError::Lattice)?;
            let parameter = geometry
                .parameter_names
                .iter()
                .position(|name| name == key.name())
                .ok_or_else(|| LeBailError::UnsupportedParameter { label: key.label() })?;
            let local = &calculation.accumulation.derivatives.local;
            for (phase_reflection, reflection_id) in phase.reflection_ids.iter().enumerate() {
                let reflection = calculation
                    .reflection_keys
                    .iter()
                    .position(|(phase_id, candidate_id)| {
                        phase_id == phase.phase_id() && candidate_id == reflection_id
                    })
                    .ok_or(LeBailError::InternalInvariant)?;
                let derivative = geometry.d_two_theta_d_parameters
                    [phase_reflection * geometry.parameter_names.len() + parameter];
                let begin = local.offsets[reflection];
                let end = local.offsets[reflection + 1];
                let start = local.starts[reflection];
                for active in begin..end {
                    matrix[(start + active - begin, column)] +=
                        local.values[active * local.parameter_count + 1] * derivative;
                }
            }
        } else {
            return Err(LeBailError::UnsupportedParameter { label: key.label() });
        }
    }
    Ok(matrix)
}

fn apply_parameter_values(
    instrument: ConstantWavelengthInstrument,
    phases: &[LeBailPhase],
    values: &BTreeMap<ParameterKey, f64>,
) -> Result<(ConstantWavelengthInstrument, Vec<LeBailPhase>), LeBailError> {
    let mut updated_instrument = instrument;
    for (key, value) in values {
        if key.module() == "instrument" {
            set_instrument_parameter(&mut updated_instrument, key.name(), *value)?;
        }
    }
    updated_instrument
        .validate()
        .map_err(|error| LeBailError::Profile {
            message: error.to_string(),
        })?;
    let mut updated_phases = Vec::with_capacity(phases.len());
    for phase in phases {
        let scale = values
            .get(&lebail_phase_scale_key(phase.phase_id())?)
            .copied()
            .unwrap_or(phase.scale());
        let mut positions = phase.two_theta_deg.clone();
        for (index, reflection_id) in phase.reflection_ids.iter().enumerate() {
            if let Some(value) = values.get(&lebail_reflection_position_key(
                phase.phase_id(),
                reflection_id,
            )?) {
                positions[index] = *value;
            }
        }
        let mut updated = phase.replace_scale_and_positions(scale, positions)?;
        let lattice_values = values
            .iter()
            .filter(|(key, _)| key.module() == "lattice" && key.owner_id() == phase.phase_id())
            .collect::<Vec<_>>();
        if !lattice_values.is_empty() {
            let cell = phase.cell.ok_or_else(|| {
                invalid_phase("lattice parameters require a bounded reflection domain")
            })?;
            let domain = phase.reflection_domain.as_ref().ok_or_else(|| {
                invalid_phase("lattice parameters require a bounded reflection domain")
            })?;
            let mut independent = domain
                .parameterization()
                .values_from_cell(cell)
                .map_err(LeBailError::Lattice)?;
            for (key, value) in lattice_values {
                let index = domain
                    .parameterization()
                    .parameter_names()
                    .iter()
                    .position(|name| name == key.name())
                    .ok_or_else(|| LeBailError::UnsupportedParameter { label: key.label() })?;
                independent[index] = *value;
            }
            let cell = domain
                .parameterization()
                .to_cell(&independent)
                .map_err(LeBailError::Lattice)?;
            updated =
                updated.replace_cell_geometry(cell, updated_instrument.wavelength_angstrom)?;
        }
        updated_phases.push(updated);
    }
    Ok((updated_instrument, updated_phases))
}

fn regenerate_accepted_domains(
    phases: Vec<LeBailPhase>,
) -> Result<(Vec<LeBailPhase>, Vec<String>, bool), LeBailError> {
    let mut updated = Vec::with_capacity(phases.len());
    let mut warnings = Vec::new();
    let mut topology_changed = false;
    for phase in phases {
        let Some(domain) = phase.reflection_domain.as_ref() else {
            updated.push(phase);
            continue;
        };
        let cell = phase.cell.ok_or(LeBailError::InternalInvariant)?;
        let previous = phase
            .reflection_ids
            .iter()
            .cloned()
            .zip(phase.integrated_intensity.iter().copied())
            .collect::<BTreeMap<_, _>>();
        let generated = domain
            .generate(cell, Some(&previous))
            .map_err(LeBailError::Lattice)?;
        let changed = generated.reflection_ids != phase.reflection_ids;
        topology_changed |= changed;
        if !generated.added_reflection_ids.is_empty()
            || !generated.removed_reflection_ids.is_empty()
        {
            warnings.push(format!(
                "phase {} reflection domain regenerated: {} added, {} removed",
                phase.phase_id(),
                generated.added_reflection_ids.len(),
                generated.removed_reflection_ids.len()
            ));
        }
        updated.push(phase.replace_generated_domain(cell, generated)?);
    }
    Ok((updated, warnings, topology_changed))
}

#[allow(clippy::too_many_arguments)]
fn covariance(
    pattern: &PatternRecord,
    calculation: &LeBailCalculation,
    instrument: ConstantWavelengthInstrument,
    phases: &[LeBailPhase],
    parameters: Option<&ParameterSet>,
    constraints: &[Constraint],
    use_uncertainty: bool,
    reduced_chi_square: f64,
) -> Result<Option<CovarianceMatrix>, LeBailError> {
    let Some(parameters) = parameters else {
        return Ok(None);
    };
    let transform = ConstraintTransform::new(parameters.clone(), constraints.to_vec())
        .map_err(LeBailError::Constraint)?;
    let free_count = transform.free_keys().len();
    if free_count == 0 {
        return Ok(Some(CovarianceMatrix {
            size: 0,
            values: Vec::new(),
        }));
    }
    let derivative = transform
        .derivative_matrix()
        .map_err(LeBailError::Constraint)?;
    for (row, spec) in parameters.specs().iter().enumerate() {
        if spec.key().module() == "phase"
            && spec.key().name() == "scale"
            && derivative
                .row(row)
                .is_some_and(|values| values.iter().any(|value| *value != 0.0))
        {
            return Ok(None);
        }
    }
    let physical = parameter_columns(calculation, parameters, instrument, phases)?;
    let chain = DMatrix::from_row_slice(derivative.rows, derivative.columns, &derivative.values);
    let jacobian = physical * chain;
    let included = pattern
        .mask
        .clone()
        .unwrap_or_else(|| vec![true; pattern.sample_count()]);
    let row_count = included.iter().filter(|value| **value).count();
    let mut selected = DMatrix::zeros(row_count, free_count);
    let mut row = 0;
    for sample in 0..pattern.sample_count() {
        if !included[sample] {
            continue;
        }
        let weight = if use_uncertainty {
            pattern
                .uncertainty
                .as_ref()
                .map_or(1.0, |values| values[sample].recip())
        } else {
            1.0
        };
        for column in 0..free_count {
            selected[(row, column)] = jacobian[(sample, column)] * weight;
        }
        row += 1;
    }
    let normal = selected.transpose() * selected;
    if matrix_rank(&normal) != free_count {
        return Ok(None);
    }
    let Some(mut inverse) = normal.try_inverse() else {
        return Ok(None);
    };
    let known_uncertainties = use_uncertainty && pattern.uncertainty.is_some();
    if !known_uncertainties && reduced_chi_square.is_finite() {
        inverse *= reduced_chi_square;
    }
    let mut values = Vec::with_capacity(free_count * free_count);
    for row in 0..free_count {
        for column in 0..free_count {
            values.push(inverse[(row, column)]);
        }
    }
    Ok(Some(CovarianceMatrix {
        size: free_count,
        values,
    }))
}

fn free_parameter_count(
    parameters: Option<&ParameterSet>,
    constraints: &[Constraint],
) -> Result<usize, LeBailError> {
    parameters.map_or(Ok(0), |parameters| {
        ConstraintTransform::new(parameters.clone(), constraints.to_vec())
            .map(|transform| transform.free_keys().len())
            .map_err(LeBailError::Constraint)
    })
}

fn matrix_rank(matrix: &DMatrix<f64>) -> usize {
    let singular = matrix.clone().svd(false, false).singular_values;
    let maximum = singular.iter().copied().fold(0.0_f64, f64::max);
    let tolerance = count_as_f64(matrix.nrows().max(matrix.ncols())) * f64::EPSILON * maximum;
    singular.iter().filter(|value| **value > tolerance).count()
}

fn instrument_parameter(instrument: ConstantWavelengthInstrument, name: &str) -> Option<f64> {
    match name {
        "u_deg2" => Some(instrument.u_deg2),
        "v_deg2" => Some(instrument.v_deg2),
        "w_deg2" => Some(instrument.w_deg2),
        "x_deg" => Some(instrument.x_deg),
        "y_deg" => Some(instrument.y_deg),
        _ => None,
    }
}

fn set_instrument_parameter(
    instrument: &mut ConstantWavelengthInstrument,
    name: &str,
    value: f64,
) -> Result<(), LeBailError> {
    match name {
        "u_deg2" => instrument.u_deg2 = value,
        "v_deg2" => instrument.v_deg2 = value,
        "w_deg2" => instrument.w_deg2 = value,
        "x_deg" => instrument.x_deg = value,
        "y_deg" => instrument.y_deg = value,
        _ => {
            return Err(LeBailError::UnsupportedParameter {
                label: format!("instrument[cw].{name}"),
            });
        }
    }
    Ok(())
}

fn rank_deficient_groups(
    calculation: &LeBailCalculation,
    threshold: f64,
) -> Vec<CoincidentReflectionGroup> {
    let local = &calculation.accumulation.derivatives.local;
    let count = local.peak_count();
    let mut parents = (0..count).collect::<Vec<_>>();
    let norms = (0..count)
        .map(|reflection| {
            let begin = local.offsets[reflection];
            let end = local.offsets[reflection + 1];
            (begin..end)
                .map(|active| {
                    let value = local.values[active * local.parameter_count];
                    value * value
                })
                .sum::<f64>()
                .sqrt()
        })
        .collect::<Vec<_>>();
    for left in 0..count {
        let left_begin = local.offsets[left];
        let left_end = local.offsets[left + 1];
        let left_start = local.starts[left];
        let left_stop = left_start + left_end - left_begin;
        for right in left + 1..count {
            let right_begin = local.offsets[right];
            let right_end = local.offsets[right + 1];
            let right_start = local.starts[right];
            let right_stop = right_start + right_end - right_begin;
            let start = left_start.max(right_start);
            let stop = left_stop.min(right_stop);
            if start >= stop || norms[left] == 0.0 || norms[right] == 0.0 {
                continue;
            }
            let correlation = (start..stop)
                .map(|sample| {
                    let left_active = left_begin + sample - left_start;
                    let right_active = right_begin + sample - right_start;
                    local.values[left_active * local.parameter_count]
                        * local.values[right_active * local.parameter_count]
                })
                .sum::<f64>()
                / (norms[left] * norms[right]);
            if correlation >= threshold {
                union(&mut parents, left, right);
            }
        }
    }
    let mut grouped = std::collections::BTreeMap::<usize, Vec<usize>>::new();
    for reflection in 0..count {
        let root = root(&mut parents, reflection);
        grouped.entry(root).or_default().push(reflection);
    }
    grouped
        .into_values()
        .filter(|indices| indices.len() > 1)
        .map(|indices| {
            let first = indices
                .iter()
                .map(|index| local.starts[*index])
                .min()
                .unwrap_or(0);
            let last = indices
                .iter()
                .map(|index| {
                    local.starts[*index] + local.offsets[*index + 1] - local.offsets[*index]
                })
                .max()
                .unwrap_or(first);
            let mut matrix = DMatrix::zeros(last - first, indices.len());
            for (column, reflection) in indices.iter().enumerate() {
                let begin = local.offsets[*reflection];
                let end = local.offsets[*reflection + 1];
                let start = local.starts[*reflection] - first;
                for active in begin..end {
                    matrix[(start + active - begin, column)] =
                        local.values[active * local.parameter_count];
                }
            }
            let singular_values = matrix.svd(false, false).singular_values;
            let maximum = singular_values.iter().copied().fold(0.0_f64, f64::max);
            let tolerance =
                count_as_f64((last - first).max(indices.len())) * f64::EPSILON * maximum;
            let rank = singular_values
                .iter()
                .filter(|value| **value > tolerance)
                .count();
            CoincidentReflectionGroup {
                reflection_keys: indices
                    .iter()
                    .map(|index| calculation.reflection_keys[*index].clone())
                    .collect(),
                rank,
            }
        })
        .collect()
}

fn root(parents: &mut [usize], mut index: usize) -> usize {
    while parents[index] != index {
        parents[index] = parents[parents[index]];
        index = parents[index];
    }
    index
}

fn union(parents: &mut [usize], left: usize, right: usize) {
    let left_root = root(parents, left);
    let right_root = root(parents, right);
    if left_root != right_root {
        parents[right_root] = left_root;
    }
}

fn bin_integration_weights(x: &[f64]) -> Vec<f64> {
    match x.len() {
        0 => Vec::new(),
        1 => vec![1.0],
        count => {
            let mut widths = vec![0.0; count];
            widths[0] = 0.5 * (x[1] - x[0]);
            widths[count - 1] = 0.5 * (x[count - 1] - x[count - 2]);
            for index in 1..count - 1 {
                widths[index] = 0.5 * (x[index + 1] - x[index - 1]);
            }
            widths
        }
    }
}

fn replace_flat_intensities(
    phases: &[LeBailPhase],
    intensities: &[f64],
) -> Result<Vec<LeBailPhase>, LeBailError> {
    let expected = phases.iter().map(reflection_count).sum::<usize>();
    if intensities.len() != expected {
        return Err(LeBailError::IntensityShapeMismatch);
    }
    let mut offset = 0;
    phases
        .iter()
        .map(|phase| {
            let end = offset + reflection_count(phase);
            let updated = phase.replace_intensities(&intensities[offset..end]);
            offset = end;
            updated
        })
        .collect()
}

fn flatten_intensities(phases: &[LeBailPhase]) -> Vec<f64> {
    phases
        .iter()
        .flat_map(|phase| phase.integrated_intensity.iter().copied())
        .collect()
}

fn flatten_preserve_mask(phases: &[LeBailPhase]) -> Vec<bool> {
    phases
        .iter()
        .flat_map(|phase| {
            if phase.preserve_unobserved.is_empty() {
                vec![false; reflection_count(phase)]
            } else {
                phase.preserve_unobserved.clone()
            }
        })
        .collect()
}

fn parameter_keys(parameters: &ParameterSet) -> Vec<&ParameterKey> {
    parameters.specs().iter().map(ParameterSpec::key).collect()
}

fn validate_parameter_selection(
    phases: &[LeBailPhase],
    parameters: &ParameterSet,
) -> Result<(), LeBailError> {
    for spec in parameters.specs() {
        let key = spec.key();
        if key.module() == "lattice" {
            let phase = phases
                .iter()
                .find(|phase| phase.phase_id() == key.owner_id())
                .ok_or_else(|| LeBailError::UnsupportedParameter { label: key.label() })?;
            let Some(domain) = phase.reflection_domain.as_ref() else {
                return Err(invalid_phase(
                    "lattice parameters require bounded dynamic phases",
                ));
            };
            if !domain
                .parameterization()
                .parameter_names()
                .iter()
                .any(|name| name == key.name())
            {
                return Err(LeBailError::UnsupportedParameter { label: key.label() });
            }
        } else if key.module() == "reflection" {
            let dynamic = phases.iter().any(|phase| {
                phase.reflection_domain.is_some()
                    && key
                        .owner_id()
                        .strip_prefix(phase.phase_id())
                        .is_some_and(|suffix| suffix.starts_with('/'))
            });
            if dynamic {
                return Err(invalid_phase(
                    "independent reflection positions require fixed-topology phases",
                ));
            }
        }
    }
    Ok(())
}

fn phases_restart_compatible(input: &[LeBailPhase], checkpoint: &[LeBailPhase]) -> bool {
    input.len() == checkpoint.len()
        && input.iter().zip(checkpoint).all(|(left, right)| {
            if left.phase_id() != right.phase_id() {
                return false;
            }
            match (&left.reflection_domain, &right.reflection_domain) {
                (None, None) => left.reflection_ids == right.reflection_ids,
                (Some(left_domain), Some(right_domain)) => left_domain == right_domain,
                _ => false,
            }
        })
}

fn reflection_count(phase: &LeBailPhase) -> usize {
    phase.reflection_ids.len()
}

fn reflection_count_of(phase: &LeBailPhase) -> usize {
    reflection_count(phase)
}

fn validate_stable_label(name: &'static str, value: &str) -> Result<(), LeBailError> {
    if value.is_empty()
        || value.trim() != value
        || value.chars().any(char::is_control)
        || value.contains('/')
    {
        return Err(LeBailError::InvalidPhase {
            message: format!(
                "{name} must be non-empty, trimmed, and contain neither '/' nor control characters"
            ),
        });
    }
    Ok(())
}

fn normal_stop_reason(error: &RuntimeError) -> Option<TerminationReason> {
    match error {
        RuntimeError::Stopped(stop) => Some(stop.reason),
        _ => None,
    }
}

fn stop_reason_or_error(error: RuntimeError) -> Result<TerminationReason, LeBailError> {
    normal_stop_reason(&error).ok_or(LeBailError::Runtime(error))
}

#[allow(clippy::cast_precision_loss)]
fn count_as_f64(value: usize) -> f64 {
    value as f64
}

fn invalid_phase(message: &str) -> LeBailError {
    LeBailError::InvalidPhase {
        message: message.to_owned(),
    }
}

fn invalid_options(message: &str) -> LeBailError {
    LeBailError::InvalidOptions {
        message: message.to_owned(),
    }
}

/// Invalid native fixed-reflection Le Bail state or operation.
#[derive(Debug)]
pub enum LeBailError {
    /// Pattern domain state is invalid.
    Pattern(DomainError),
    /// Observations are required.
    MissingObservations,
    /// Typed parameter construction or replacement failed.
    Parameter(ParameterError),
    /// Constraint graph or transform failed.
    Constraint(ConstraintError),
    /// Lattice parameterization, geometry, bounds, or generation failed.
    Lattice(LatticeError),
    /// A parameter key is not supported by fixed-geometry Le Bail.
    UnsupportedParameter {
        /// Stable parameter label.
        label: String,
    },
    /// A live domain value violates its declared parameter bounds.
    ParameterDomainOutsideBounds {
        /// Stable parameter label.
        label: String,
    },
    /// Native least-squares solution failed.
    LinearSolve,
    /// One phase or reflection record is invalid.
    InvalidPhase {
        /// Stable diagnostic message.
        message: String,
    },
    /// One option is invalid.
    InvalidOptions {
        /// Stable diagnostic message.
        message: String,
    },
    /// A checkpoint cannot continue this request.
    InvalidCheckpoint {
        /// Stable diagnostic message.
        message: String,
    },
    /// Current intensities do not match the calculated reflection order.
    IntensityShapeMismatch,
    /// Preserve-if-unobserved mask does not match the reflection count.
    PreserveMaskLengthMismatch,
    /// Pattern grid validation failed.
    Grid(ProfileError),
    /// Native CW accumulation failed.
    Calculation(CwContributionsError),
    /// Residual evaluation failed.
    Residual(ResidualError),
    /// Bounded runtime or host callback failed.
    Runtime(RuntimeError),
    /// Execution policy construction failed.
    Execution(ExecutionPolicyError),
    /// A workflow-owned allocation/budget count overflowed.
    SizeOverflow,
    /// A lower-level profile/instrument validation failed.
    Profile {
        /// Stable diagnostic message.
        message: String,
    },
    /// Private workflow state became inconsistent.
    InternalInvariant,
}

impl Display for LeBailError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pattern(error) => Display::fmt(error, formatter),
            Self::MissingObservations => {
                formatter.write_str("observed_y is required for Le Bail extraction")
            }
            Self::Parameter(error) => Display::fmt(error, formatter),
            Self::Constraint(error) => Display::fmt(error, formatter),
            Self::Lattice(error) => Display::fmt(error, formatter),
            Self::UnsupportedParameter { label } => {
                write!(formatter, "unsupported Le Bail parameter {label}")
            }
            Self::ParameterDomainOutsideBounds { label } => {
                write!(
                    formatter,
                    "domain value for {label} lies outside its bounds"
                )
            }
            Self::LinearSolve => formatter.write_str("profile least-squares solve failed"),
            Self::InvalidPhase { message }
            | Self::InvalidOptions { message }
            | Self::InvalidCheckpoint { message }
            | Self::Profile { message } => formatter.write_str(message),
            Self::IntensityShapeMismatch => {
                formatter.write_str("current intensities must match the reflection count")
            }
            Self::PreserveMaskLengthMismatch => {
                formatter.write_str("preserve_unobserved must match the reflection count")
            }
            Self::Grid(error) => Display::fmt(error, formatter),
            Self::Calculation(error) => Display::fmt(error, formatter),
            Self::Residual(error) => Display::fmt(error, formatter),
            Self::Runtime(error) => Display::fmt(error, formatter),
            Self::Execution(error) => Display::fmt(error, formatter),
            Self::SizeOverflow => formatter.write_str("Le Bail workflow size or budget overflowed"),
            Self::InternalInvariant => {
                formatter.write_str("internal Le Bail workflow state is inconsistent")
            }
        }
    }
}

impl Error for LeBailError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Pattern(error) => Some(error),
            Self::Parameter(error) => Some(error),
            Self::Constraint(error) => Some(error),
            Self::Lattice(error) => Some(error),
            Self::Grid(error) => Some(error),
            Self::Calculation(error) => Some(error),
            Self::Residual(error) => Some(error),
            Self::Runtime(error) => Some(error),
            Self::Execution(error) => Some(error),
            Self::MissingObservations
            | Self::UnsupportedParameter { .. }
            | Self::ParameterDomainOutsideBounds { .. }
            | Self::LinearSolve
            | Self::InvalidPhase { .. }
            | Self::InvalidOptions { .. }
            | Self::InvalidCheckpoint { .. }
            | Self::IntensityShapeMismatch
            | Self::PreserveMaskLengthMismatch
            | Self::SizeOverflow
            | Self::Profile { .. }
            | Self::InternalInvariant => None,
        }
    }
}
