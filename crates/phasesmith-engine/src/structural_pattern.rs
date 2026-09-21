//! Fused built-in scattering, structural intensity, and CW profile composition.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::{Arc, Mutex};

use phasesmith_core::{
    Accumulation, ConstantWavelengthInstrument, CwContributionsError, CwContributionsView, CwError,
    FcjGeometry, GridView, ProfileError, SupportPolicy,
};
use phasesmith_crystallography::{
    CellError, IntegratedIntensityCorrection, IntegratedIntensityCorrectionError,
    IntegratedIntensityCorrectionModel, PreparedNeutronScattering, PreparedXrayScattering,
    ScatteringBatch, ScatteringError, SpaceGroup, StructureFactorBatchError,
    StructureFactorBatchView, StructureFactorValues, UnitCell,
    calculate_powder_structure_factor_intensity_vjp_with_context,
    calculate_powder_structure_factor_jvp_with_context,
    calculate_powder_structure_factor_selected_with_context,
    calculate_powder_structure_factor_values_with_context,
};
use phasesmith_execution::ExecutionContext;

const CELL_PARAMETER_COUNT: usize = 6;
pub(crate) const CW_INSTRUMENT_PARAMETER_COUNT: usize = 5;
const DEGREES_PER_RADIAN: f64 = 180.0 / std::f64::consts::PI;

/// Monochromatic peak-position corrections evaluated with structural geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MonochromaticPositionCorrection {
    /// Constant additive shift in degrees `2theta`.
    pub zero_shift_deg: f64,
    /// Optional Bragg--Brentano `(sample displacement, goniometer radius)` in mm.
    pub bragg_brentano_mm: Option<(f64, f64)>,
    /// Optional Debye--Scherrer `(X, Y, radius)` with displacements in micrometres
    /// and the goniometer radius in millimetres.
    pub debye_scherrer_micrometre: Option<(f64, f64, f64)>,
}

/// Reflection geometry needed by built-in sample-physics providers.
#[derive(Clone, Debug, PartialEq)]
pub struct MonochromaticReflectionGeometry {
    /// Reflection d-spacings in ångströms.
    pub d_spacing_angstrom: Vec<f64>,
    /// Corrected reflection positions in degrees `2theta`.
    pub two_theta_deg: Vec<f64>,
}

/// Calculate corrected monochromatic reflection positions without profiles.
///
/// # Errors
///
/// Returns [`StructuralPatternError`] for invalid cell, wavelength, position
/// correction, or inaccessible reflections.
pub fn calculate_monochromatic_reflection_geometry(
    cell: UnitCell,
    hkl: &[[i32; 3]],
    instrument: ConstantWavelengthInstrument,
    position_correction: MonochromaticPositionCorrection,
) -> Result<MonochromaticReflectionGeometry, StructuralPatternError> {
    instrument
        .validate()
        .map_err(StructuralPatternError::InvalidInstrument)?;
    validate_position_correction(position_correction)?;
    let geometry = cell
        .geometry()
        .map_err(StructureFactorBatchError::Cell)
        .map_err(StructuralPatternError::StructureFactor)?;
    let mut d_spacing_angstrom = Vec::with_capacity(hkl.len());
    let mut two_theta_deg = Vec::with_capacity(hkl.len());
    for &reflection in hkl {
        let q_value = geometry.q_squared(reflection);
        if !q_value.is_finite() || q_value <= 0.0 {
            return Err(StructuralPatternError::ReflectionOutsideAngularDomain);
        }
        let root_q = q_value.sqrt();
        let sin_theta = 0.5 * instrument.wavelength_angstrom * root_q;
        if !(0.0..1.0).contains(&sin_theta) {
            return Err(StructuralPatternError::ReflectionOutsideAngularDomain);
        }
        d_spacing_angstrom.push(root_q.recip());
        two_theta_deg.push(
            corrected_monochromatic_position(2.0 * sin_theta.asin(), position_correction)
                .position_deg,
        );
    }
    Ok(MonochromaticReflectionGeometry {
        d_spacing_angstrom,
        two_theta_deg,
    })
}

/// Built-in native scattering model selected without a Python callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuiltInScatteringModel {
    /// Non-resonant Waasmaier--Kirfel X-ray form factors.
    XrayNonResonant,
    /// Constant bound coherent nuclear-neutron scattering lengths.
    NeutronNuclear,
}

/// Borrowed structure, reflection, profile, and sample-physics inputs.
#[derive(Clone, Copy, Debug)]
pub struct StructuralPatternInputView<'a> {
    /// Sorted pattern grid in degrees `2theta`.
    pub x_deg: &'a [f64],
    /// Canonical Miller indices.
    pub hkl: &'a [[i32; 3]],
    /// Powder multiplicity for each reflection.
    pub multiplicity: &'a [usize],
    /// Asymmetric-unit fractional coordinates.
    pub fractional_xyz: &'a [[f64; 3]],
    /// Asymmetric-site occupancies.
    pub occupancy: &'a [f64],
    /// Asymmetric-site isotropic displacement in square ångströms.
    pub u_iso_angstrom2: &'a [f64],
    /// True for asymmetric sites described by fixed CIF U tensors.
    pub anisotropic_mask: &'a [bool],
    /// CIF U tensors in component order `11,22,33,23,13,12`.
    pub u_aniso_cif_angstrom2: &'a [[f64; 6]],
    /// Exact built-in table key for every asymmetric site.
    pub scattering_species: &'a [&'a str],
    /// Fixed real X-ray dispersion offset for every site, or empty when absent.
    pub scattering_real_offset: &'a [f64],
    /// Fixed imaginary X-ray dispersion offset for every site, or empty when absent.
    pub scattering_imag_offset: &'a [f64],
    /// Structural phase scale.
    pub scale: f64,
    /// Fixed symmetry-expansion deduplication tolerance.
    pub coordinate_tolerance: f64,
    /// Monochromatic CW instrument/profile parameters.
    pub instrument: ConstantWavelengthInstrument,
    /// Optional Finger--Cox--Jephcoat axial-divergence geometry.
    pub axial_geometry: Option<FcjGeometry>,
    /// Explicit zero/sample-displacement position correction.
    pub position_correction: MonochromaticPositionCorrection,
    /// Explicit integrated-intensity correction model.
    pub correction_model: IntegratedIntensityCorrectionModel,
    /// Built-in native scattering selection.
    pub scattering_model: BuiltInScatteringModel,
    /// Vectorized sample-physics contribution batch.
    pub contributions: CwContributionsView<'a>,
    /// Exact finite profile-support policy.
    pub support: SupportPolicy,
    /// Explicit CW profile accuracy controls.
    pub profile_accuracy: phasesmith_core::ProfileAccuracy,
    /// Whether calculation includes axial derivative rows; omitted rows are zero.
    pub calculate_axial_derivatives: bool,
}

/// Structural reflection intermediates and fused profile result.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralPatternResult {
    /// Structure factors and integrated intensities before sample physics.
    pub structure_factors: StructureFactorValues,
    /// Reflection d-spacings in ångströms.
    pub d_spacing_angstrom: Vec<f64>,
    /// Monochromatic peak positions in degrees `2theta`.
    pub two_theta_deg: Vec<f64>,
    /// Support-limited profile values and local/global derivatives.
    pub accumulation: Accumulation,
}

/// Fused values and one structural forward derivative product.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralPatternJvpResult {
    /// Calculated structural pattern.
    pub result: StructuralPatternResult,
    /// Structural directional derivative of the pattern samples.
    pub d_y: Vec<f64>,
    /// Directional derivative of integrated reflection intensities.
    pub d_integrated_intensity: Vec<f64>,
    /// Directional derivative of reflection positions in degrees.
    pub d_two_theta_deg: Vec<f64>,
}

/// Fused values and a reusable parameter-major structural pattern Jacobian.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralPatternDenseResult {
    /// Calculated structural pattern.
    pub result: StructuralPatternResult,
    /// Pattern Jacobian with shape parameter count by sample count.
    pub d_y: Vec<f64>,
    /// Number of rows in the pattern Jacobian.
    pub parameter_count: usize,
}

/// Fused values and one reverse product from pattern sample weights.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralPatternVjpResult {
    /// Calculated structural pattern.
    pub result: StructuralPatternResult,
    /// Pattern-Jacobian transpose product in structural parameter order.
    pub gradient: Vec<f64>,
}

/// Invalid fused structural-pattern request.
#[derive(Debug)]
pub enum StructuralPatternError {
    /// The owned unit cell is invalid.
    InvalidCell(CellError),
    /// Reflection indices and multiplicities have different lengths.
    ReflectionLengthMismatch,
    /// Owned asymmetric-site arrays have different lengths.
    SiteLengthMismatch,
    /// Scattering species count does not match the asymmetric-site count.
    SpeciesLengthMismatch,
    /// Offset vectors are neither both empty nor matched to the asymmetric sites.
    ScatteringOffsetLengthMismatch,
    /// A fixed scattering offset is non-finite.
    NonFiniteScatteringOffset,
    /// Fixed dispersion offsets were supplied to a non-X-ray model.
    UnsupportedScatteringOffset,
    /// Built-in scattering preparation or evaluation failed.
    Scattering(ScatteringError),
    /// Integrated-intensity correction evaluation failed.
    Correction(IntegratedIntensityCorrectionError),
    /// A TOF-only intensity correction was supplied to the CW pattern engine.
    TimeOfFlightCorrectionInConstantWavelengthPattern,
    /// General-symmetry structural intensity failed.
    StructureFactor(StructureFactorBatchError),
    /// Grid validation failed.
    Profile(ProfileError),
    /// CW/sample-physics accumulation failed.
    Contributions(CwContributionsError),
    /// A reflection is inaccessible for the monochromatic wavelength.
    ReflectionOutsideAngularDomain,
    /// The CW instrument model is invalid.
    InvalidInstrument(CwError),
    /// A position-correction parameter is invalid.
    InvalidPositionCorrection,
    /// Pattern reverse weights do not match the sample count.
    PatternWeightLengthMismatch,
    /// A pattern reverse weight is non-finite.
    NonFinitePatternWeight,
}

impl Display for StructuralPatternError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCell(error) => Display::fmt(error, formatter),
            Self::ReflectionLengthMismatch => {
                formatter.write_str("hkl and multiplicity must have the same reflection count")
            }
            Self::SiteLengthMismatch => {
                formatter.write_str("all structural site arrays must have the same site count")
            }
            Self::SpeciesLengthMismatch => {
                formatter.write_str("scattering species must contain one key per asymmetric site")
            }
            Self::ScatteringOffsetLengthMismatch => formatter
                .write_str("scattering offset vectors must both be empty or match the site count"),
            Self::NonFiniteScatteringOffset => {
                formatter.write_str("scattering offsets must be finite")
            }
            Self::UnsupportedScatteringOffset => formatter
                .write_str("fixed scattering offsets are supported only for X-ray scattering"),
            Self::Scattering(error) => Display::fmt(error, formatter),
            Self::Correction(error) => Display::fmt(error, formatter),
            Self::TimeOfFlightCorrectionInConstantWavelengthPattern => formatter.write_str(
                "TOF neutron Lorentz correction is not valid for a constant-wavelength pattern",
            ),
            Self::StructureFactor(error) => Display::fmt(error, formatter),
            Self::Profile(error) => Display::fmt(error, formatter),
            Self::Contributions(error) => Display::fmt(error, formatter),
            Self::ReflectionOutsideAngularDomain => formatter.write_str(
                "structural CW reflections must lie strictly within 0 < 2theta < 180 degrees",
            ),
            Self::InvalidInstrument(error) => Display::fmt(error, formatter),
            Self::InvalidPositionCorrection => formatter.write_str(
                "position corrections must be finite and goniometer radius must be positive",
            ),
            Self::PatternWeightLengthMismatch => {
                formatter.write_str("pattern reverse weights must match the sample count")
            }
            Self::NonFinitePatternWeight => {
                formatter.write_str("pattern reverse weights must be finite")
            }
        }
    }
}

impl Error for StructuralPatternError {}

/// Bounded per-refinement cache of invariant reciprocal geometry and scattering.
/// At most 32 exact keys are retained. No process-global state or tolerance
/// comparisons are used; changing cell, HKLs, species, dispersion or correction
/// invalidates the entry. Widths, scales, sites and zero shifts do not affect it.
#[derive(Default)]
pub struct StructuralPreparationCache {
    entries: Mutex<Vec<(PreparationKey, Arc<PreparedTables>)>>,
}

#[derive(Clone, PartialEq)]
struct PreparationKey {
    cell: UnitCell,
    hkl: Vec<[i32; 3]>,
    species: Vec<String>,
    real: Vec<f64>,
    imag: Vec<f64>,
    scattering: BuiltInScatteringModel,
    correction: IntegratedIntensityCorrectionModel,
}

struct PreparedTables {
    geometry: Vec<(f64, [f64; CELL_PARAMETER_COUNT])>,
    scattering: ScatteringBatch,
    correction: IntegratedIntensityCorrection,
}

struct PreparedNumerics {
    scattering: ScatteringBatch,
    correction: IntegratedIntensityCorrection,
    d_spacing: Vec<f64>,
    two_theta_deg: Vec<f64>,
    d_two_theta_d_cell: Vec<[f64; CELL_PARAMETER_COUNT]>,
    d_two_theta_d_wavelength: Vec<f64>,
    d_two_theta_d_sample_displacement: Option<Vec<f64>>,
    d_two_theta_d_displace_x: Option<Vec<f64>>,
    d_two_theta_d_displace_y: Option<Vec<f64>>,
}

#[derive(Clone, Copy, Debug)]
struct CorrectedPosition {
    position_deg: f64,
    d_position_d_base: f64,
    d_position_d_sample_displacement: Option<f64>,
    d_position_d_displace_x: Option<f64>,
    d_position_d_displace_y: Option<f64>,
}

fn corrected_monochromatic_position(
    base_position_radians: f64,
    correction: MonochromaticPositionCorrection,
) -> CorrectedPosition {
    let mut result = CorrectedPosition {
        position_deg: base_position_radians.to_degrees() + correction.zero_shift_deg,
        d_position_d_base: 1.0,
        d_position_d_sample_displacement: None,
        d_position_d_displace_x: None,
        d_position_d_displace_y: None,
    };
    if let Some((displacement, radius)) = correction.bragg_brentano_mm {
        let theta = 0.5 * base_position_radians;
        result.position_deg -= 2.0 * displacement / radius * theta.cos() * DEGREES_PER_RADIAN;
        result.d_position_d_base += displacement / radius * theta.sin();
        result.d_position_d_sample_displacement =
            Some(-2.0 / radius * theta.cos() * DEGREES_PER_RADIAN);
    }
    if let Some((displace_x, displace_y, radius)) = correction.debye_scherrer_micrometre {
        let (sin_position, cos_position) = base_position_radians.sin_cos();
        let displacement_scale = 0.18 / (std::f64::consts::PI * radius);
        result.position_deg -=
            displacement_scale * (displace_x * cos_position + displace_y * sin_position);
        result.d_position_d_base += displacement_scale.to_radians()
            * (displace_x * sin_position - displace_y * cos_position);
        result.d_position_d_displace_x = Some(-displacement_scale * cos_position);
        result.d_position_d_displace_y = Some(-displacement_scale * sin_position);
    }
    result
}

fn validate_position_correction(
    correction: MonochromaticPositionCorrection,
) -> Result<(), StructuralPatternError> {
    let invalid = !correction.zero_shift_deg.is_finite()
        || (correction.bragg_brentano_mm.is_some()
            && correction.debye_scherrer_micrometre.is_some())
        || correction
            .bragg_brentano_mm
            .is_some_and(|(displacement, radius)| {
                !displacement.is_finite() || !radius.is_finite() || radius <= 0.0
            })
        || correction
            .debye_scherrer_micrometre
            .is_some_and(|(displace_x, displace_y, radius)| {
                !displace_x.is_finite()
                    || !displace_y.is_finite()
                    || !radius.is_finite()
                    || radius <= 0.0
            });
    if invalid {
        return Err(StructuralPatternError::InvalidPositionCorrection);
    }
    Ok(())
}

fn prepared_tables(
    cell: UnitCell,
    input: &StructuralPatternInputView<'_>,
    cache: Option<&StructuralPreparationCache>,
) -> Result<Arc<PreparedTables>, StructuralPatternError> {
    let key = PreparationKey {
        cell,
        hkl: input.hkl.to_vec(),
        species: input
            .scattering_species
            .iter()
            .map(|s| (*s).to_owned())
            .collect(),
        real: input.scattering_real_offset.to_vec(),
        imag: input.scattering_imag_offset.to_vec(),
        scattering: input.scattering_model,
        correction: input.correction_model,
    };
    if let Some(cache) = cache {
        if let Some((_, tables)) = cache
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .find(|(candidate, _)| candidate == &key)
        {
            return Ok(Arc::clone(tables));
        }
    }
    let geometry = cell
        .geometry()
        .map_err(StructuralPatternError::InvalidCell)?;
    let geometry = input
        .hkl
        .iter()
        .map(|&hkl| geometry.q_squared_and_derivatives(hkl))
        .collect::<Vec<_>>();
    if geometry.iter().any(|(q, _)| !q.is_finite() || *q <= 0.0) {
        return Err(StructuralPatternError::ReflectionOutsideAngularDomain);
    }
    let q_squared = geometry.iter().map(|(q, _)| *q).collect::<Vec<_>>();
    let s = q_squared.iter().map(|q| 0.5 * q.sqrt()).collect::<Vec<_>>();
    let mut scattering = match input.scattering_model {
        BuiltInScatteringModel::XrayNonResonant => {
            PreparedXrayScattering::new(input.scattering_species.iter().copied())
                .and_then(|model| model.evaluate(&s))
        }
        BuiltInScatteringModel::NeutronNuclear => {
            PreparedNeutronScattering::new(input.scattering_species.iter().copied())
                .and_then(|model| model.evaluate(&s))
        }
    }
    .map_err(StructuralPatternError::Scattering)?;
    apply_scattering_offsets(&mut scattering, input);
    let correction = input
        .correction_model
        .evaluate(&q_squared)
        .map_err(StructuralPatternError::Correction)?;
    let tables = Arc::new(PreparedTables {
        geometry,
        scattering,
        correction,
    });
    if let Some(cache) = cache {
        let mut entries = cache
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if entries.len() == 32 {
            entries.remove(0);
        }
        entries.push((key, Arc::clone(&tables)));
    }
    Ok(tables)
}

fn validate_scattering_offsets(
    input: &StructuralPatternInputView<'_>,
) -> Result<(), StructuralPatternError> {
    if input.scattering_real_offset.is_empty() && input.scattering_imag_offset.is_empty() {
        return Ok(());
    }
    if input.scattering_real_offset.len() != input.fractional_xyz.len()
        || input.scattering_imag_offset.len() != input.fractional_xyz.len()
    {
        return Err(StructuralPatternError::ScatteringOffsetLengthMismatch);
    }
    if input
        .scattering_real_offset
        .iter()
        .chain(input.scattering_imag_offset)
        .any(|value| !value.is_finite())
    {
        return Err(StructuralPatternError::NonFiniteScatteringOffset);
    }
    if input.scattering_model == BuiltInScatteringModel::NeutronNuclear
        && input
            .scattering_real_offset
            .iter()
            .chain(input.scattering_imag_offset)
            .any(|value| *value != 0.0)
    {
        return Err(StructuralPatternError::UnsupportedScatteringOffset);
    }
    Ok(())
}

fn apply_scattering_offsets(
    scattering: &mut ScatteringBatch,
    input: &StructuralPatternInputView<'_>,
) {
    if input.scattering_real_offset.is_empty() {
        return;
    }
    for reflection in 0..scattering.reflection_count {
        for site in 0..scattering.site_count {
            let index = reflection * scattering.site_count + site;
            scattering.real[index] += input.scattering_real_offset[site];
            scattering.imag[index] += input.scattering_imag_offset[site];
        }
    }
}

impl PreparedNumerics {
    fn structure_batch<'a>(
        &'a self,
        input: &StructuralPatternInputView<'a>,
    ) -> StructureFactorBatchView<'a> {
        StructureFactorBatchView {
            hkl: input.hkl,
            multiplicity: input.multiplicity,
            fractional_xyz: input.fractional_xyz,
            occupancy: input.occupancy,
            u_iso_angstrom2: input.u_iso_angstrom2,
            anisotropic_mask: input.anisotropic_mask,
            u_aniso_cif_angstrom2: input.u_aniso_cif_angstrom2,
            scattering_real: &self.scattering.real,
            scattering_imag: &self.scattering.imag,
            d_scattering_real_d_s: &self.scattering.d_real_d_s,
            d_scattering_imag_d_s: &self.scattering.d_imag_d_s,
            correction: &self.correction.values,
            d_correction_d_q_squared: &self.correction.d_values_d_q_squared,
            scale: input.scale,
            coordinate_tolerance: input.coordinate_tolerance,
        }
    }
}

/// Calculate built-in scattering, structural intensities, and one CW profile.
///
/// # Errors
///
/// Returns [`StructuralPatternError`] for invalid structural, scattering,
/// correction, instrument, contribution, grid, or support inputs.
pub fn calculate_structural_pattern(
    cell: UnitCell,
    space_group: &SpaceGroup,
    input: &StructuralPatternInputView<'_>,
) -> Result<StructuralPatternResult, StructuralPatternError> {
    calculate_structural_pattern_with_context(cell, space_group, input, &ExecutionContext::serial())
}

/// Calculate a structural pattern with an explicit bounded execution context.
///
/// # Errors
///
/// Returns [`StructuralPatternError`] for invalid inputs.
pub fn calculate_structural_pattern_with_context(
    cell: UnitCell,
    space_group: &SpaceGroup,
    input: &StructuralPatternInputView<'_>,
    execution: &ExecutionContext,
) -> Result<StructuralPatternResult, StructuralPatternError> {
    let prepared = prepare(cell, input)?;
    calculate_values(cell, space_group, input, &prepared, execution)
}

/// Calculate values and a reusable dense structural pattern linearization.
///
/// # Errors
///
/// Returns an error for invalid inputs or allocation overflow.
pub fn calculate_structural_pattern_dense(
    cell: UnitCell,
    space_group: &SpaceGroup,
    input: &StructuralPatternInputView<'_>,
) -> Result<StructuralPatternDenseResult, StructuralPatternError> {
    calculate_structural_pattern_dense_with_context(
        cell,
        space_group,
        input,
        &ExecutionContext::serial(),
    )
}

/// Calculate a dense structural linearization with a bounded context.
///
/// # Errors
///
/// Returns an error for invalid inputs or allocation overflow.
pub fn calculate_structural_pattern_dense_with_context(
    cell: UnitCell,
    space_group: &SpaceGroup,
    input: &StructuralPatternInputView<'_>,
    execution: &ExecutionContext,
) -> Result<StructuralPatternDenseResult, StructuralPatternError> {
    calculate_structural_pattern_selected_with_context(
        cell,
        space_group,
        input,
        execution,
        None,
        None,
    )
}

/// Evaluate selected native structural rows, retaining zeros for omitted rows.
/// `None` requests every structural row. A mask skips fixed structural chains;
/// axial derivatives follow the input's `calculate_axial_derivatives` flag.
/// Values and requested derivatives share the peak/sample pass.
/// # Errors
/// Returns an error for invalid inputs or a mask of the wrong length.
#[allow(clippy::too_many_lines)]
pub fn calculate_structural_pattern_selected_with_context(
    cell: UnitCell,
    space_group: &SpaceGroup,
    input: &StructuralPatternInputView<'_>,
    execution: &ExecutionContext,
    selected: Option<&[bool]>,
    cache: Option<&StructuralPreparationCache>,
) -> Result<StructuralPatternDenseResult, StructuralPatternError> {
    if selected.is_some_and(|mask| mask.len() != 7 + 5 * input.fractional_xyz.len()) {
        return Err(StructuralPatternError::StructureFactor(
            StructureFactorBatchError::TangentLengthMismatch,
        ));
    }
    let prepared = prepare_cached(cell, input, cache)?;
    let structural = calculate_powder_structure_factor_selected_with_context(
        cell,
        space_group,
        prepared.structure_batch(input),
        execution,
        selected,
    )
    .map_err(StructuralPatternError::StructureFactor)?;
    let parameter_count = structural.layout.parameter_count();
    let sample_count = input.x_deg.len();
    let element_count =
        parameter_count
            .checked_mul(sample_count)
            .ok_or(StructuralPatternError::Contributions(
                CwContributionsError::AllocationOverflow,
            ))?;
    let mut accumulation = accumulate_selected(
        input,
        &prepared.two_theta_deg,
        &structural.values.intensity,
        execution,
        true,
    )?;
    append_instrument_derivatives(&mut accumulation, &structural.values, input, &prepared)?;
    let reflection_count = input.hkl.len();
    let local = &accumulation.derivatives.local;
    let d_y = if execution.threads() == 1 || parameter_count < 2 {
        let mut values = zeroed_values(element_count)?;
        for parameter in 0..parameter_count {
            if selected.is_some_and(|mask| !mask[parameter]) {
                continue;
            }
            let row = &mut values[parameter * sample_count..(parameter + 1) * sample_count];
            for reflection in 0..reflection_count {
                let structural_index = parameter * reflection_count + reflection;
                let position_derivative = if parameter < CELL_PARAMETER_COUNT {
                    prepared.d_two_theta_d_cell[reflection][parameter]
                } else {
                    0.0
                };
                let begin = local.offsets[reflection];
                let end = local.offsets[reflection + 1];
                for active in begin..end {
                    let sample = local.starts[reflection] + active - begin;
                    let local_base = 2 * active;
                    row[sample] += local.values[local_base]
                        * structural.d_intensity[structural_index]
                        + local.values[local_base + 1] * position_derivative;
                }
            }
        }
        values
    } else {
        let rows = execution.map_ordered(parameter_count, 2, |parameter| {
            let mut row = zeroed_values(sample_count)?;
            if selected.is_some_and(|mask| !mask[parameter]) {
                return Ok::<_, StructuralPatternError>(row);
            }
            for reflection in 0..reflection_count {
                let begin = local.offsets[reflection];
                let end = local.offsets[reflection + 1];
                for active in begin..end {
                    let sample = local.starts[reflection] + active - begin;
                    let local_base = 2 * active;
                    let structural_index = parameter * reflection_count + reflection;
                    let position_derivative = if parameter < CELL_PARAMETER_COUNT {
                        prepared.d_two_theta_d_cell[reflection][parameter]
                    } else {
                        0.0
                    };
                    row[sample] += local.values[local_base]
                        * structural.d_intensity[structural_index]
                        + local.values[local_base + 1] * position_derivative;
                }
            }
            Ok::<_, StructuralPatternError>(row)
        });
        let mut values = Vec::new();
        values.try_reserve_exact(element_count).map_err(|_| {
            StructuralPatternError::Contributions(CwContributionsError::AllocationOverflow)
        })?;
        for row in rows {
            values.extend(row?);
        }
        values
    };
    Ok(StructuralPatternDenseResult {
        result: StructuralPatternResult {
            structure_factors: structural.values,
            d_spacing_angstrom: prepared.d_spacing,
            two_theta_deg: prepared.two_theta_deg,
            accumulation,
        },
        d_y,
        parameter_count,
    })
}

fn zeroed_values(count: usize) -> Result<Vec<f64>, StructuralPatternError> {
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| {
        StructuralPatternError::Contributions(CwContributionsError::AllocationOverflow)
    })?;
    values.resize(count, 0.0);
    Ok(values)
}

/// Calculate a full structural-pattern JVP without a dense pattern Jacobian.
///
/// # Errors
///
/// Returns [`StructuralPatternError`] for invalid inputs or structural tangent.
pub fn calculate_structural_pattern_jvp(
    cell: UnitCell,
    space_group: &SpaceGroup,
    input: &StructuralPatternInputView<'_>,
    tangent: &[f64],
) -> Result<StructuralPatternJvpResult, StructuralPatternError> {
    calculate_structural_pattern_jvp_with_context(
        cell,
        space_group,
        input,
        tangent,
        &ExecutionContext::serial(),
    )
}

/// Calculate a structural JVP with a bounded execution context.
///
/// # Errors
///
/// Returns [`StructuralPatternError`] for invalid inputs or tangent shape.
pub fn calculate_structural_pattern_jvp_with_context(
    cell: UnitCell,
    space_group: &SpaceGroup,
    input: &StructuralPatternInputView<'_>,
    tangent: &[f64],
    execution: &ExecutionContext,
) -> Result<StructuralPatternJvpResult, StructuralPatternError> {
    let prepared = prepare(cell, input)?;
    let structural = calculate_powder_structure_factor_jvp_with_context(
        cell,
        space_group,
        prepared.structure_batch(input),
        tangent,
        execution,
    )
    .map_err(StructuralPatternError::StructureFactor)?;
    let d_two_theta_deg = prepared
        .d_two_theta_d_cell
        .iter()
        .map(|derivatives| {
            derivatives
                .iter()
                .zip(&tangent[..CELL_PARAMETER_COUNT])
                .map(|(derivative, direction)| derivative * direction)
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let mut accumulation = accumulate(
        input,
        &prepared.two_theta_deg,
        &structural.values.intensity,
        execution,
    )?;
    append_instrument_derivatives(&mut accumulation, &structural.values, input, &prepared)?;
    let d_y = chain_pattern_jvp(&accumulation, &structural.d_intensity, &d_two_theta_deg);
    Ok(StructuralPatternJvpResult {
        result: StructuralPatternResult {
            structure_factors: structural.values,
            d_spacing_angstrom: prepared.d_spacing,
            two_theta_deg: prepared.two_theta_deg,
            accumulation,
        },
        d_y,
        d_integrated_intensity: structural.d_intensity,
        d_two_theta_deg,
    })
}

/// Calculate a full structural-pattern transpose product from sample weights.
///
/// # Errors
///
/// Returns [`StructuralPatternError`] for invalid inputs or sample weights.
pub fn calculate_structural_pattern_vjp(
    cell: UnitCell,
    space_group: &SpaceGroup,
    input: &StructuralPatternInputView<'_>,
    sample_weights: &[f64],
) -> Result<StructuralPatternVjpResult, StructuralPatternError> {
    calculate_structural_pattern_vjp_with_context(
        cell,
        space_group,
        input,
        sample_weights,
        &ExecutionContext::serial(),
    )
}

/// Calculate a structural transpose product with a bounded context.
///
/// # Errors
///
/// Returns [`StructuralPatternError`] for invalid inputs or sample weights.
pub fn calculate_structural_pattern_vjp_with_context(
    cell: UnitCell,
    space_group: &SpaceGroup,
    input: &StructuralPatternInputView<'_>,
    sample_weights: &[f64],
    execution: &ExecutionContext,
) -> Result<StructuralPatternVjpResult, StructuralPatternError> {
    if sample_weights.len() != input.x_deg.len() {
        return Err(StructuralPatternError::PatternWeightLengthMismatch);
    }
    if sample_weights.iter().any(|value| !value.is_finite()) {
        return Err(StructuralPatternError::NonFinitePatternWeight);
    }
    let prepared = prepare(cell, input)?;
    let values = calculate_powder_structure_factor_values_with_context(
        cell,
        space_group,
        prepared.structure_batch(input),
        execution,
    )
    .map_err(StructuralPatternError::StructureFactor)?;
    let mut accumulation =
        accumulate(input, &prepared.two_theta_deg, &values.intensity, execution)?;
    append_instrument_derivatives(&mut accumulation, &values, input, &prepared)?;
    let (intensity_weights, position_weights) =
        local_transpose_weights(&accumulation, sample_weights);
    let mut structural = calculate_powder_structure_factor_intensity_vjp_with_context(
        cell,
        space_group,
        prepared.structure_batch(input),
        &intensity_weights,
        execution,
    )
    .map_err(StructuralPatternError::StructureFactor)?;
    for (reflection, weight) in position_weights.into_iter().enumerate() {
        for parameter in 0..CELL_PARAMETER_COUNT {
            structural.gradient[parameter] +=
                weight * prepared.d_two_theta_d_cell[reflection][parameter];
        }
    }
    Ok(StructuralPatternVjpResult {
        result: StructuralPatternResult {
            structure_factors: values,
            d_spacing_angstrom: prepared.d_spacing,
            two_theta_deg: prepared.two_theta_deg,
            accumulation,
        },
        gradient: structural.gradient,
    })
}

fn prepare(
    cell: UnitCell,
    input: &StructuralPatternInputView<'_>,
) -> Result<PreparedNumerics, StructuralPatternError> {
    prepare_cached(cell, input, None)
}

fn prepare_cached(
    cell: UnitCell,
    input: &StructuralPatternInputView<'_>,
    cache: Option<&StructuralPreparationCache>,
) -> Result<PreparedNumerics, StructuralPatternError> {
    validate_constant_wavelength_correction(input.correction_model)?;
    if input.scattering_species.len() != input.fractional_xyz.len() {
        return Err(StructuralPatternError::SpeciesLengthMismatch);
    }
    validate_scattering_offsets(input)?;
    input
        .instrument
        .validate()
        .map_err(StructuralPatternError::InvalidInstrument)?;
    validate_position_correction(input.position_correction)?;
    let tables = prepared_tables(cell, input, cache)?;
    let mut q_squared = Vec::with_capacity(input.hkl.len());
    let mut d_spacing = Vec::with_capacity(input.hkl.len());
    let mut two_theta_deg = Vec::with_capacity(input.hkl.len());
    let mut d_two_theta_d_cell = Vec::with_capacity(input.hkl.len());
    let mut d_two_theta_d_wavelength = Vec::with_capacity(input.hkl.len());
    let mut d_two_theta_d_sample_displacement = input
        .position_correction
        .bragg_brentano_mm
        .map(|_| Vec::with_capacity(input.hkl.len()));
    let mut d_two_theta_d_displace_x = input
        .position_correction
        .debye_scherrer_micrometre
        .map(|_| Vec::with_capacity(input.hkl.len()));
    let mut d_two_theta_d_displace_y = input
        .position_correction
        .debye_scherrer_micrometre
        .map(|_| Vec::with_capacity(input.hkl.len()));
    for &(q_value, d_q) in &tables.geometry {
        if !q_value.is_finite() || q_value <= 0.0 {
            return Err(StructuralPatternError::ReflectionOutsideAngularDomain);
        }
        let root_q = q_value.sqrt();
        let sin_theta = 0.5 * input.instrument.wavelength_angstrom * root_q;
        if !(0.0..1.0).contains(&sin_theta) {
            return Err(StructuralPatternError::ReflectionOutsideAngularDomain);
        }
        let theta = sin_theta.asin();
        let corrected = corrected_monochromatic_position(2.0 * theta, input.position_correction);
        let d_position_factor =
            corrected.d_position_d_base * input.instrument.wavelength_angstrom * DEGREES_PER_RADIAN
                / (2.0 * root_q * theta.cos());
        let d_position_d_wavelength =
            corrected.d_position_d_base * DEGREES_PER_RADIAN * root_q / theta.cos();
        q_squared.push(q_value);
        d_spacing.push(root_q.recip());
        two_theta_deg.push(corrected.position_deg);
        d_two_theta_d_cell.push(d_q.map(|derivative| d_position_factor * derivative));
        d_two_theta_d_wavelength.push(d_position_d_wavelength);
        if let (Some(values), Some(derivative)) = (
            d_two_theta_d_sample_displacement.as_mut(),
            corrected.d_position_d_sample_displacement,
        ) {
            values.push(derivative);
        }
        if let (Some(values), Some(derivative)) = (
            d_two_theta_d_displace_x.as_mut(),
            corrected.d_position_d_displace_x,
        ) {
            values.push(derivative);
        }
        if let (Some(values), Some(derivative)) = (
            d_two_theta_d_displace_y.as_mut(),
            corrected.d_position_d_displace_y,
        ) {
            values.push(derivative);
        }
    }
    Ok(PreparedNumerics {
        scattering: tables.scattering.clone(),
        correction: tables.correction.clone(),
        d_spacing,
        two_theta_deg,
        d_two_theta_d_cell,
        d_two_theta_d_wavelength,
        d_two_theta_d_sample_displacement,
        d_two_theta_d_displace_x,
        d_two_theta_d_displace_y,
    })
}

fn validate_constant_wavelength_correction(
    correction: IntegratedIntensityCorrectionModel,
) -> Result<(), StructuralPatternError> {
    if matches!(
        correction,
        IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz { .. }
    ) {
        Err(StructuralPatternError::TimeOfFlightCorrectionInConstantWavelengthPattern)
    } else {
        Ok(())
    }
}

fn calculate_values(
    cell: UnitCell,
    space_group: &SpaceGroup,
    input: &StructuralPatternInputView<'_>,
    prepared: &PreparedNumerics,
    execution: &ExecutionContext,
) -> Result<StructuralPatternResult, StructuralPatternError> {
    let structure_factors = calculate_powder_structure_factor_values_with_context(
        cell,
        space_group,
        prepared.structure_batch(input),
        execution,
    )
    .map_err(StructuralPatternError::StructureFactor)?;
    let mut accumulation = accumulate(
        input,
        &prepared.two_theta_deg,
        &structure_factors.intensity,
        execution,
    )?;
    append_instrument_derivatives(&mut accumulation, &structure_factors, input, prepared)?;
    Ok(StructuralPatternResult {
        structure_factors,
        d_spacing_angstrom: prepared.d_spacing.clone(),
        two_theta_deg: prepared.two_theta_deg.clone(),
        accumulation,
    })
}

fn append_instrument_derivatives(
    accumulation: &mut Accumulation,
    structure_factors: &StructureFactorValues,
    input: &StructuralPatternInputView<'_>,
    prepared: &PreparedNumerics,
) -> Result<(), StructuralPatternError> {
    let sample_count = accumulation.sample_count;
    let extra_count = 2
        + usize::from(prepared.d_two_theta_d_sample_displacement.is_some())
        + usize::from(prepared.d_two_theta_d_displace_x.is_some())
        + usize::from(prepared.d_two_theta_d_displace_y.is_some());
    let global = accumulation
        .derivatives
        .global
        .as_mut()
        .expect("CW contribution accumulation always has global derivatives");
    let old_values = std::mem::take(&mut global.values);
    let mut combined = Vec::new();
    combined
        .try_reserve(old_values.len() + extra_count * sample_count)
        .map_err(|_| {
            StructuralPatternError::Contributions(CwContributionsError::AllocationOverflow)
        })?;
    let mut wavelength = vec![0.0; sample_count];
    let mut zero_shift = vec![0.0; sample_count];
    let mut sample_displacement = prepared
        .d_two_theta_d_sample_displacement
        .as_ref()
        .map(|_| vec![0.0; sample_count]);
    let mut displace_x = prepared
        .d_two_theta_d_displace_x
        .as_ref()
        .map(|_| vec![0.0; sample_count]);
    let mut displace_y = prepared
        .d_two_theta_d_displace_y
        .as_ref()
        .map(|_| vec![0.0; sample_count]);
    let local = &accumulation.derivatives.local;
    for reflection in 0..local.peak_count() {
        #[allow(clippy::cast_precision_loss)]
        let multiplicity = input.multiplicity[reflection] as f64;
        let d_intensity_d_wavelength = input.scale
            * multiplicity
            * structure_factors.f_squared[reflection]
            * prepared.correction.d_values_d_wavelength[reflection];
        let begin = local.offsets[reflection];
        let end = local.offsets[reflection + 1];
        for active in begin..end {
            let sample = local.starts[reflection] + active - begin;
            let base = 2 * active;
            let d_intensity = local.values[base];
            let d_position = local.values[base + 1];
            wavelength[sample] += d_intensity * d_intensity_d_wavelength
                + d_position * prepared.d_two_theta_d_wavelength[reflection];
            zero_shift[sample] += d_position;
            if let (Some(values), Some(derivatives)) = (
                sample_displacement.as_mut(),
                prepared.d_two_theta_d_sample_displacement.as_ref(),
            ) {
                values[sample] += d_position * derivatives[reflection];
            }
            if let (Some(values), Some(derivatives)) = (
                displace_x.as_mut(),
                prepared.d_two_theta_d_displace_x.as_ref(),
            ) {
                values[sample] += d_position * derivatives[reflection];
            }
            if let (Some(values), Some(derivatives)) = (
                displace_y.as_mut(),
                prepared.d_two_theta_d_displace_y.as_ref(),
            ) {
                values[sample] += d_position * derivatives[reflection];
            }
        }
    }
    let instrument_end = CW_INSTRUMENT_PARAMETER_COUNT * sample_count;
    combined.extend_from_slice(&old_values[..instrument_end]);
    combined.extend(wavelength);
    combined.extend(zero_shift);
    if let Some(values) = sample_displacement {
        combined.extend(values);
    }
    if let Some(values) = displace_x {
        combined.extend(values);
    }
    if let Some(values) = displace_y {
        combined.extend(values);
    }
    combined.extend_from_slice(&old_values[instrument_end..]);
    global.values = combined;
    global.parameter_count += extra_count;
    Ok(())
}

fn accumulate(
    input: &StructuralPatternInputView<'_>,
    two_theta_deg: &[f64],
    intensities: &[f64],
    execution: &ExecutionContext,
) -> Result<Accumulation, StructuralPatternError> {
    accumulate_selected(input, two_theta_deg, intensities, execution, true)
}

fn accumulate_selected(
    input: &StructuralPatternInputView<'_>,
    two_theta_deg: &[f64],
    intensities: &[f64],
    execution: &ExecutionContext,
    axial_derivatives: bool,
) -> Result<Accumulation, StructuralPatternError> {
    let grid = GridView::new(input.x_deg).map_err(StructuralPatternError::Profile)?;
    let result = phasesmith_core::cw_contributions::accumulate_cw_contributions_with_accuracy(
        grid,
        two_theta_deg,
        intensities,
        input.instrument,
        input.contributions,
        input.axial_geometry,
        input.support,
        input.profile_accuracy,
        axial_derivatives && input.calculate_axial_derivatives,
        execution,
    );
    result.map_err(StructuralPatternError::Contributions)
}

fn chain_pattern_jvp(
    accumulation: &Accumulation,
    d_intensity: &[f64],
    d_position: &[f64],
) -> Vec<f64> {
    let local = &accumulation.derivatives.local;
    let mut result = vec![0.0; accumulation.sample_count];
    for reflection in 0..local.peak_count() {
        let begin = local.offsets[reflection];
        let end = local.offsets[reflection + 1];
        for active in begin..end {
            let sample = local.starts[reflection] + active - begin;
            let base = 2 * active;
            result[sample] += local.values[base] * d_intensity[reflection]
                + local.values[base + 1] * d_position[reflection];
        }
    }
    result
}

fn local_transpose_weights(
    accumulation: &Accumulation,
    sample_weights: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let local = &accumulation.derivatives.local;
    let mut intensity = vec![0.0; local.peak_count()];
    let mut position = vec![0.0; local.peak_count()];
    for reflection in 0..local.peak_count() {
        let begin = local.offsets[reflection];
        let end = local.offsets[reflection + 1];
        for active in begin..end {
            let sample = local.starts[reflection] + active - begin;
            let base = 2 * active;
            intensity[reflection] += local.values[base] * sample_weights[sample];
            position[reflection] += local.values[base + 1] * sample_weights[sample];
        }
    }
    (intensity, position)
}

#[cfg(test)]
mod tests {
    use super::*;
    use phasesmith_core::CwContributionArrays;
    use phasesmith_crystallography::{P1ParameterLayout, Rational, SymmetryOperation};

    fn cell() -> UnitCell {
        UnitCell {
            a_angstrom: 4.7,
            b_angstrom: 5.1,
            c_angstrom: 6.2,
            alpha_deg: 82.0,
            beta_deg: 87.0,
            gamma_deg: 74.0,
        }
    }

    fn group() -> SpaceGroup {
        SpaceGroup::new(vec![
            SymmetryOperation::identity(),
            SymmetryOperation::new([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], [Rational::zero(); 3])
                .expect("inversion"),
        ])
        .expect("P-1")
    }

    fn instrument() -> ConstantWavelengthInstrument {
        ConstantWavelengthInstrument {
            wavelength_angstrom: 1.5406,
            u_deg2: 2.0e-4,
            v_deg2: -1.0e-4,
            w_deg2: 1.2e-4,
            x_deg: 1.5e-3,
            y_deg: 3.0e-3,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn calculate_case(
        selected_cell: UnitCell,
        x: &[f64],
        hkl: &[[i32; 3]],
        multiplicity: &[usize],
        xyz: &[[f64; 3]],
        occupancy: &[f64],
        u_iso: &[f64],
        scale: f64,
        multiplier: &[f64],
    ) -> StructuralPatternResult {
        calculate_case_with_correction(
            selected_cell,
            x,
            hkl,
            multiplicity,
            xyz,
            occupancy,
            u_iso,
            scale,
            multiplier,
            IntegratedIntensityCorrectionModel::Neutral,
        )
        .expect("structural pattern")
    }

    #[allow(clippy::too_many_arguments)]
    fn calculate_case_with_correction(
        selected_cell: UnitCell,
        x: &[f64],
        hkl: &[[i32; 3]],
        multiplicity: &[usize],
        xyz: &[[f64; 3]],
        occupancy: &[f64],
        u_iso: &[f64],
        scale: f64,
        multiplier: &[f64],
        correction_model: IntegratedIntensityCorrectionModel,
    ) -> Result<StructuralPatternResult, StructuralPatternError> {
        let zeros = vec![0.0; hkl.len()];
        let contributions = CwContributionsView::new(
            hkl.len(),
            0,
            CwContributionArrays {
                gaussian_variance_deg2: &zeros,
                lorentzian_fwhm_deg: &zeros,
                intensity_multiplier: multiplier,
                d_gaussian_variance_d_position: &zeros,
                d_lorentzian_fwhm_d_position: &zeros,
                d_intensity_multiplier_d_position: &zeros,
                d_gaussian_variance_d_parameters: &[],
                d_lorentzian_fwhm_d_parameters: &[],
                d_intensity_multiplier_d_parameters: &[],
            },
        )
        .expect("contributions");
        calculate_structural_pattern(
            selected_cell,
            &group(),
            &StructuralPatternInputView {
                x_deg: x,
                hkl,
                multiplicity,
                fractional_xyz: xyz,
                occupancy,
                u_iso_angstrom2: u_iso,
                anisotropic_mask: &[false, false],
                u_aniso_cif_angstrom2: &[[0.0; 6]; 2],
                scattering_species: &["Si", "O"],
                scattering_real_offset: &[],
                scattering_imag_offset: &[],
                scale,
                coordinate_tolerance: 1.0e-10,
                instrument: instrument(),
                axial_geometry: None,
                position_correction: MonochromaticPositionCorrection {
                    zero_shift_deg: 0.0,
                    bragg_brentano_mm: None,
                    debye_scherrer_micrometre: None,
                },
                correction_model,
                scattering_model: BuiltInScatteringModel::XrayNonResonant,
                contributions,
                support: SupportPolicy::FwhmMultiple(20.0),
                profile_accuracy: phasesmith_core::ProfileAccuracy::default(),
                calculate_axial_derivatives: true,
            },
        )
    }

    #[test]
    fn constant_wavelength_pattern_rejects_tof_only_correction() {
        let error = calculate_case_with_correction(
            cell(),
            &[10.0, 20.0],
            &[[1, 0, 1]],
            &[2],
            &[[0.17, 0.23, 0.31], [0.37, 0.11, 0.19]],
            &[0.82, 0.55],
            &[0.012, 0.018],
            1.0,
            &[1.0],
            IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz {
                two_theta_deg: 90.0,
            },
        )
        .expect_err("TOF correction must not enter CW accumulation");
        assert!(matches!(
            error,
            StructuralPatternError::TimeOfFlightCorrectionInConstantWavelengthPattern
        ));
    }

    #[test]
    fn fused_values_apply_sample_intensity_multiplier_exactly_once() {
        let x: Vec<f64> = (0..9_001)
            .map(|index| 10.0 + f64::from(index) * 0.01)
            .collect();
        let hkl = [[1, 0, 1], [2, 1, 1], [1, 2, 3]];
        let multiplicity = [2, 4, 2];
        let xyz = [[0.17, 0.23, 0.31], [0.37, 0.11, 0.19]];
        let occupancy = [0.82, 0.55];
        let u_iso = [0.012, 0.018];
        let neutral = calculate_case(
            cell(),
            &x,
            &hkl,
            &multiplicity,
            &xyz,
            &occupancy,
            &u_iso,
            1.4,
            &[1.0; 3],
        );
        let doubled = calculate_case(
            cell(),
            &x,
            &hkl,
            &multiplicity,
            &xyz,
            &occupancy,
            &u_iso,
            1.4,
            &[2.0; 3],
        );
        assert_eq!(
            neutral.structure_factors.intensity,
            doubled.structure_factors.intensity
        );
        for (left, right) in neutral.accumulation.y.iter().zip(&doubled.accumulation.y) {
            assert!((2.0 * left - right).abs() < 2.0e-15 * right.abs().max(1.0));
        }
    }

    #[test]
    fn reflection_geometry_helper_matches_the_fused_structural_positions() {
        let x = (0..9_001)
            .map(|index| 10.0 + f64::from(index) * 0.01)
            .collect::<Vec<_>>();
        let hkl = [[1, 0, 1], [2, 1, 1], [1, 2, 3]];
        let correction = MonochromaticPositionCorrection {
            zero_shift_deg: 0.0,
            bragg_brentano_mm: None,
            debye_scherrer_micrometre: None,
        };
        let fused = calculate_case(
            cell(),
            &x,
            &hkl,
            &[2, 4, 2],
            &[[0.17, 0.23, 0.31], [0.37, 0.11, 0.19]],
            &[0.82, 0.55],
            &[0.012, 0.018],
            1.4,
            &[1.0; 3],
        );
        let geometry =
            calculate_monochromatic_reflection_geometry(cell(), &hkl, instrument(), correction)
                .unwrap();
        assert_eq!(geometry.d_spacing_angstrom, fused.d_spacing_angstrom);
        assert_eq!(geometry.two_theta_deg, fused.two_theta_deg);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn fused_jvp_vjp_match_pattern_finite_differences_and_adjoint_identity() {
        let x: Vec<f64> = (0..9_001)
            .map(|index| 10.0 + f64::from(index) * 0.01)
            .collect();
        let hkl = [[1, 0, 1], [2, 1, 1], [1, 2, 3]];
        let multiplicity = [2, 4, 2];
        let xyz = [[0.17, 0.23, 0.31], [0.37, 0.11, 0.19]];
        let occupancy = [0.82, 0.55];
        let u_iso = [0.012, 0.018];
        let scale = 1.4;
        let zeros = [0.0; 3];
        let ones = [1.0; 3];
        let contributions = CwContributionsView::new(
            hkl.len(),
            0,
            CwContributionArrays {
                gaussian_variance_deg2: &zeros,
                lorentzian_fwhm_deg: &zeros,
                intensity_multiplier: &ones,
                d_gaussian_variance_d_position: &zeros,
                d_lorentzian_fwhm_d_position: &zeros,
                d_intensity_multiplier_d_position: &zeros,
                d_gaussian_variance_d_parameters: &[],
                d_lorentzian_fwhm_d_parameters: &[],
                d_intensity_multiplier_d_parameters: &[],
            },
        )
        .expect("contributions");
        let input = StructuralPatternInputView {
            x_deg: &x,
            hkl: &hkl,
            multiplicity: &multiplicity,
            fractional_xyz: &xyz,
            occupancy: &occupancy,
            u_iso_angstrom2: &u_iso,
            anisotropic_mask: &[false, false],
            u_aniso_cif_angstrom2: &[[0.0; 6]; 2],
            scattering_species: &["Si", "O"],
            scattering_real_offset: &[],
            scattering_imag_offset: &[],
            scale,
            coordinate_tolerance: 1.0e-10,
            instrument: instrument(),
            axial_geometry: None,
            position_correction: MonochromaticPositionCorrection {
                zero_shift_deg: 0.0,
                bragg_brentano_mm: None,
                debye_scherrer_micrometre: None,
            },
            correction_model: IntegratedIntensityCorrectionModel::Neutral,
            scattering_model: BuiltInScatteringModel::XrayNonResonant,
            contributions,
            support: SupportPolicy::FwhmMultiple(20.0),
            profile_accuracy: phasesmith_core::ProfileAccuracy::default(),
            calculate_axial_derivatives: true,
        };
        let layout = P1ParameterLayout { site_count: 2 };
        let tangent: Vec<f64> = (0..layout.parameter_count())
            .map(|index| f64::from(u32::try_from(index + 1).expect("small index")) * 2.0e-5)
            .collect();
        let jvp = calculate_structural_pattern_jvp(cell(), &group(), &input, &tangent)
            .expect("structural JVP");
        let dense = calculate_structural_pattern_dense(cell(), &group(), &input)
            .expect("structural dense linearization");
        assert_eq!(dense.parameter_count, tangent.len());
        for sample in 0..x.len() {
            let product = tangent
                .iter()
                .enumerate()
                .map(|(parameter, direction)| direction * dense.d_y[parameter * x.len() + sample])
                .sum::<f64>();
            assert!((product - jvp.d_y[sample]).abs() < 2.0e-11 * product.abs().max(1.0));
        }
        let step = 1.0e-5;
        let mut plus_cell = cell();
        let mut minus_cell = cell();
        for (parameter, direction) in tangent
            .iter()
            .copied()
            .take(CELL_PARAMETER_COUNT)
            .enumerate()
        {
            perturb_cell(&mut plus_cell, parameter, step * direction);
            perturb_cell(&mut minus_cell, parameter, -step * direction);
        }
        let mut plus_xyz = xyz;
        let mut minus_xyz = xyz;
        for site in 0..2 {
            for component in 0..3 {
                let direction = tangent[layout.coordinate(site, component)];
                plus_xyz[site][component] += step * direction;
                minus_xyz[site][component] -= step * direction;
            }
        }
        let mut plus_occupancy = occupancy;
        let mut minus_occupancy = occupancy;
        let mut plus_u = u_iso;
        let mut minus_u = u_iso;
        for site in 0..2 {
            plus_occupancy[site] += step * tangent[layout.occupancy(site)];
            minus_occupancy[site] -= step * tangent[layout.occupancy(site)];
            plus_u[site] += step * tangent[layout.u_iso(site)];
            minus_u[site] -= step * tangent[layout.u_iso(site)];
        }
        let plus = calculate_case(
            plus_cell,
            &x,
            &hkl,
            &multiplicity,
            &plus_xyz,
            &plus_occupancy,
            &plus_u,
            scale + step * tangent[layout.scale()],
            &ones,
        );
        let minus = calculate_case(
            minus_cell,
            &x,
            &hkl,
            &multiplicity,
            &minus_xyz,
            &minus_occupancy,
            &minus_u,
            scale - step * tangent[layout.scale()],
            &ones,
        );
        for ((actual, plus_value), minus_value) in jvp
            .d_y
            .iter()
            .zip(&plus.accumulation.y)
            .zip(&minus.accumulation.y)
        {
            let finite_difference = (plus_value - minus_value) / (2.0 * step);
            assert!((actual - finite_difference).abs() < 3.0e-5 * finite_difference.abs().max(1.0));
        }
        let sample_weights: Vec<f64> = x.iter().map(|value| (0.17 * value).sin()).collect();
        let vjp = calculate_structural_pattern_vjp(cell(), &group(), &input, &sample_weights)
            .expect("structural VJP");
        let forward = jvp
            .d_y
            .iter()
            .zip(&sample_weights)
            .map(|(derivative, weight)| derivative * weight)
            .sum::<f64>();
        let reverse = tangent
            .iter()
            .zip(&vjp.gradient)
            .map(|(direction, gradient)| direction * gradient)
            .sum::<f64>();
        assert!((forward - reverse).abs() < 2.0e-10 * forward.abs().max(1.0));
        for (parameter, actual) in vjp.gradient.iter().copied().enumerate() {
            let expected = dense.d_y[parameter * x.len()..(parameter + 1) * x.len()]
                .iter()
                .zip(&sample_weights)
                .map(|(derivative, weight)| derivative * weight)
                .sum::<f64>();
            assert!((actual - expected).abs() < 2.0e-10 * expected.abs().max(1.0));
        }
    }

    fn perturb_cell(cell: &mut UnitCell, parameter: usize, change: f64) {
        let value = match parameter {
            0 => &mut cell.a_angstrom,
            1 => &mut cell.b_angstrom,
            2 => &mut cell.c_angstrom,
            3 => &mut cell.alpha_deg,
            4 => &mut cell.beta_deg,
            5 => &mut cell.gamma_deg,
            _ => panic!("invalid cell parameter"),
        };
        *value += change;
    }
}
