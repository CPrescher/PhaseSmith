//! Python bindings for the Rietveld Engine numerical core.

#![allow(clippy::needless_pass_by_value)] // PyO3 extracts owned argument guards.

use npy::ndarray::Array2;
use npy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rietveld_core::{
    Accumulation, ConstantWavelengthInstrument, CwContributionArrays, CwContributionsView,
    CwProfileParameters, CwReflectionBatchView, FcjGeometry, FcjProfile, GridView, PeakBatchView,
    SupportPolicy, TchPeakBatchView, TchShape, TchWidths, TofInstrument, TofProfile,
    TofProfileParameters, WavelengthComponentsView, accumulate_batch, accumulate_cw_batch,
    accumulate_cw_components_batch, accumulate_cw_contributions_batch, accumulate_cw_fcj_batch,
    accumulate_cw_fcj_components_batch, accumulate_tch_batch, accumulate_tof_batch,
    accumulate_values_batch, smooth_bruckner as native_smooth_bruckner, symmetric_pseudo_voigt,
};
use rietveld_engine::crystallography::{
    IntegratedIntensityCorrectionModel, NEUTRON_TABLE_PROVENANCE, P1BatchView,
    PreparedNeutronScattering, PreparedReflectionGenerator, PreparedXrayScattering, Rational,
    ReflectionRange, ScatteringBatch, SpaceGroup, StructureFactorBatchView,
    StructureFactorDenseResult, StructureFactorValues, SymmetryOperation, UnitCell,
    XRAY_TABLE_PROVENANCE, calculate_p1_dense, calculate_p1_intensity_vjp, calculate_p1_jvp,
    calculate_structure_factor_dense, calculate_structure_factor_values, neutron_species_metadata,
    xray_species_metadata,
};
use rietveld_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPatternError,
    StructuralPatternInputView, StructuralPatternJvpResult, StructuralPatternResult,
    StructuralPatternVjpResult, calculate_structural_pattern, calculate_structural_pattern_jvp,
    calculate_structural_pattern_vjp,
};

type ProfileArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

type AccumulationArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
);

type StructuralReflectionArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

type StructuralPatternArrays<'py> = (AccumulationArrays<'py>, StructuralReflectionArrays<'py>);

type StructuralPatternJvpArrays<'py> = (
    StructuralPatternArrays<'py>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

type StructuralPatternVjpArrays<'py> = (StructuralPatternArrays<'py>, Bound<'py, PyArray1<f64>>);

type TchShapeValues = (f64, f64, f64, f64, f64, f64);

type CwProfileArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
);

type FcjProfileArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

type TofProfileArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

type TofParameterArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

type CellGeometryArrays<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    f64,
    Bound<'py, PyArray1<f64>>,
);

type CellSpacingArrays<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>);

type P1DenseArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
);

type P1JvpArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

type P1VjpArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

type SymmetryTopology = (
    Vec<i64>,
    Vec<i64>,
    Vec<i64>,
    String,
    Vec<i64>,
    Vec<i64>,
    usize,
);

type ExpandedSiteArrays<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>);

type ReflectionFamilyArrays<'py> = (
    Vec<String>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<i64>>,
);

type GeneratedReflectionArrays<'py> = (
    Vec<String>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
);

type ScatteringArrays<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
);

type StructureFactorDenseArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
);

type StructureFactorValueArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

type CorrectionArrays<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

type TableProvenanceRecord = (&'static str, &'static str, &'static str, u64, usize);
type TableProvenanceRecords = (TableProvenanceRecord, TableProvenanceRecord);
type NeutronMetadataRecord = (
    &'static str,
    u8,
    Option<u16>,
    f64,
    Option<f64>,
    bool,
    Option<&'static str>,
);

/// Cached native Waasmaier--Kirfel species rows.
#[pyclass(name = "_PreparedXrayScattering")]
struct NativePreparedXrayScattering {
    model: PreparedXrayScattering,
}

#[pymethods]
impl NativePreparedXrayScattering {
    #[new]
    fn new(species: Vec<String>) -> PyResult<Self> {
        let model = PreparedXrayScattering::new(&species)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { model })
    }

    #[getter]
    fn site_count(&self) -> usize {
        self.model.site_count()
    }

    #[getter]
    fn unique_species_count(&self) -> usize {
        self.model.unique_species_count()
    }

    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        s_inverse_angstrom: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<ScatteringArrays<'py>> {
        let values = contiguous_slice(&s_inverse_angstrom, "s_inverse_angstrom")?;
        let result = self
            .model
            .evaluate(values)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        scattering_to_numpy(py, result)
    }
}

/// Cached native constant coherent neutron species rows.
#[pyclass(name = "_PreparedNeutronScattering")]
struct NativePreparedNeutronScattering {
    model: PreparedNeutronScattering,
}

#[pymethods]
impl NativePreparedNeutronScattering {
    #[new]
    fn new(species: Vec<String>) -> PyResult<Self> {
        let model = PreparedNeutronScattering::new(&species)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { model })
    }

    #[getter]
    fn site_count(&self) -> usize {
        self.model.site_count()
    }

    #[getter]
    fn unique_species_count(&self) -> usize {
        self.model.unique_species_count()
    }

    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        s_inverse_angstrom: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<ScatteringArrays<'py>> {
        let values = contiguous_slice(&s_inverse_angstrom, "s_inverse_angstrom")?;
        let result = self
            .model
            .evaluate(values)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        scattering_to_numpy(py, result)
    }
}

/// Cached native group topology and bounded reflection generator.
#[pyclass(name = "_PreparedReflectionGenerator")]
struct NativePreparedReflectionGenerator {
    generator: PreparedReflectionGenerator,
}

#[pymethods]
impl NativePreparedReflectionGenerator {
    #[new]
    fn new(
        rotations_flat: PyReadonlyArray1<'_, i64>,
        translation_numerators_flat: PyReadonlyArray1<'_, i64>,
        translation_denominators_flat: PyReadonlyArray1<'_, i64>,
        merge_friedel: bool,
        max_candidates: usize,
    ) -> PyResult<Self> {
        let operations = symmetry_operations(
            &rotations_flat,
            &translation_numerators_flat,
            &translation_denominators_flat,
        )?;
        let space_group = SpaceGroup::new(operations)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let generator =
            PreparedReflectionGenerator::new(space_group, merge_friedel, max_candidates)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { generator })
    }

    fn topology(&self) -> SymmetryTopology {
        let group = self.generator.space_group();
        let mut rotations = Vec::with_capacity(group.operations().len() * 9);
        let mut numerators = Vec::with_capacity(group.operations().len() * 3);
        let mut denominators = Vec::with_capacity(group.operations().len() * 3);
        for operation in group.operations() {
            rotations.extend(operation.rotation().into_iter().flatten().map(i64::from));
            for translation in operation.translation() {
                numerators.push(translation.numerator());
                denominators.push(translation.denominator());
            }
        }
        let equations = group
            .metric_constraints()
            .equations
            .iter()
            .flatten()
            .copied()
            .collect();
        let parameterization_basis = group
            .metric_constraints()
            .parameterization_basis
            .iter()
            .flatten()
            .copied()
            .collect();
        (
            rotations,
            numerators,
            denominators,
            format!("{:?}", group.crystal_system()).to_lowercase(),
            equations,
            parameterization_basis,
            group.metric_constraints().independent_parameter_count,
        )
    }

    fn expand_sites<'py>(
        &self,
        py: Python<'py>,
        fractional_xyz_flat: PyReadonlyArray1<'py, f64>,
        tolerance: f64,
    ) -> PyResult<ExpandedSiteArrays<'py>> {
        let xyz = xyz_rows(&fractional_xyz_flat)?;
        let expanded = self
            .generator
            .space_group()
            .expand_sites(&xyz, tolerance)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let positions = Array2::from_shape_vec(
            (expanded.fractional_xyz.len(), 3),
            expanded.fractional_xyz.into_iter().flatten().collect(),
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?
        .into_pyarray(py);
        let source = expanded
            .source_site
            .into_iter()
            .map(|value| {
                i64::try_from(value)
                    .map_err(|_| PyValueError::new_err("expanded source index overflow"))
            })
            .collect::<PyResult<Vec<_>>>()?
            .into_pyarray(py);
        Ok((positions, source))
    }

    #[allow(clippy::too_many_arguments)]
    fn structure_factor_values<'py>(
        &self,
        py: Python<'py>,
        hkl_flat: PyReadonlyArray1<'py, i64>,
        multiplicity: PyReadonlyArray1<'py, i64>,
        fractional_xyz_flat: PyReadonlyArray1<'py, f64>,
        occupancy: PyReadonlyArray1<'py, f64>,
        u_iso_angstrom2: PyReadonlyArray1<'py, f64>,
        scattering_real: PyReadonlyArray1<'py, f64>,
        scattering_imag: PyReadonlyArray1<'py, f64>,
        d_scattering_real_d_s: PyReadonlyArray1<'py, f64>,
        d_scattering_imag_d_s: PyReadonlyArray1<'py, f64>,
        correction: PyReadonlyArray1<'py, f64>,
        d_correction_d_q_squared: PyReadonlyArray1<'py, f64>,
        a_angstrom: f64,
        b_angstrom: f64,
        c_angstrom: f64,
        alpha_deg: f64,
        beta_deg: f64,
        gamma_deg: f64,
        scale: f64,
        coordinate_tolerance: f64,
    ) -> PyResult<StructureFactorValueArrays<'py>> {
        let hkl = hkl_rows(&hkl_flat)?;
        let multiplicity = multiplicity_rows(&multiplicity)?;
        let xyz = xyz_rows(&fractional_xyz_flat)?;
        let result = calculate_structure_factor_values(
            crystallographic_cell(
                a_angstrom, b_angstrom, c_angstrom, alpha_deg, beta_deg, gamma_deg,
            ),
            self.generator.space_group(),
            StructureFactorBatchView {
                hkl: &hkl,
                multiplicity: &multiplicity,
                fractional_xyz: &xyz,
                occupancy: contiguous_slice(&occupancy, "occupancy")?,
                u_iso_angstrom2: contiguous_slice(&u_iso_angstrom2, "u_iso_angstrom2")?,
                scattering_real: contiguous_slice(&scattering_real, "scattering_real")?,
                scattering_imag: contiguous_slice(&scattering_imag, "scattering_imag")?,
                d_scattering_real_d_s: contiguous_slice(
                    &d_scattering_real_d_s,
                    "d_scattering_real_d_s",
                )?,
                d_scattering_imag_d_s: contiguous_slice(
                    &d_scattering_imag_d_s,
                    "d_scattering_imag_d_s",
                )?,
                correction: contiguous_slice(&correction, "correction")?,
                d_correction_d_q_squared: contiguous_slice(
                    &d_correction_d_q_squared,
                    "d_correction_d_q_squared",
                )?,
                scale,
                coordinate_tolerance,
            },
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(structure_factor_values_to_numpy(py, result))
    }

    #[allow(clippy::too_many_arguments)]
    fn structure_factor_dense<'py>(
        &self,
        py: Python<'py>,
        hkl_flat: PyReadonlyArray1<'py, i64>,
        multiplicity: PyReadonlyArray1<'py, i64>,
        fractional_xyz_flat: PyReadonlyArray1<'py, f64>,
        occupancy: PyReadonlyArray1<'py, f64>,
        u_iso_angstrom2: PyReadonlyArray1<'py, f64>,
        scattering_real: PyReadonlyArray1<'py, f64>,
        scattering_imag: PyReadonlyArray1<'py, f64>,
        d_scattering_real_d_s: PyReadonlyArray1<'py, f64>,
        d_scattering_imag_d_s: PyReadonlyArray1<'py, f64>,
        correction: PyReadonlyArray1<'py, f64>,
        d_correction_d_q_squared: PyReadonlyArray1<'py, f64>,
        a_angstrom: f64,
        b_angstrom: f64,
        c_angstrom: f64,
        alpha_deg: f64,
        beta_deg: f64,
        gamma_deg: f64,
        scale: f64,
        coordinate_tolerance: f64,
    ) -> PyResult<StructureFactorDenseArrays<'py>> {
        let hkl = hkl_rows(&hkl_flat)?;
        let multiplicity = multiplicity_rows(&multiplicity)?;
        let xyz = xyz_rows(&fractional_xyz_flat)?;
        let result = calculate_structure_factor_dense(
            crystallographic_cell(
                a_angstrom, b_angstrom, c_angstrom, alpha_deg, beta_deg, gamma_deg,
            ),
            self.generator.space_group(),
            StructureFactorBatchView {
                hkl: &hkl,
                multiplicity: &multiplicity,
                fractional_xyz: &xyz,
                occupancy: contiguous_slice(&occupancy, "occupancy")?,
                u_iso_angstrom2: contiguous_slice(&u_iso_angstrom2, "u_iso_angstrom2")?,
                scattering_real: contiguous_slice(&scattering_real, "scattering_real")?,
                scattering_imag: contiguous_slice(&scattering_imag, "scattering_imag")?,
                d_scattering_real_d_s: contiguous_slice(
                    &d_scattering_real_d_s,
                    "d_scattering_real_d_s",
                )?,
                d_scattering_imag_d_s: contiguous_slice(
                    &d_scattering_imag_d_s,
                    "d_scattering_imag_d_s",
                )?,
                correction: contiguous_slice(&correction, "correction")?,
                d_correction_d_q_squared: contiguous_slice(
                    &d_correction_d_q_squared,
                    "d_correction_d_q_squared",
                )?,
                scale,
                coordinate_tolerance,
            },
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        structure_factor_dense_to_numpy(py, result)
    }

    fn systematic_absences<'py>(
        &self,
        py: Python<'py>,
        hkl_flat: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let hkl = hkl_rows(&hkl_flat)?;
        hkl.into_iter()
            .map(|reflection| {
                self.generator
                    .space_group()
                    .is_systematically_absent(reflection)
                    .map_err(|error| PyValueError::new_err(error.to_string()))
            })
            .collect::<PyResult<Vec<_>>>()
            .map(|values| values.into_pyarray(py))
    }

    fn reflection_families<'py>(
        &self,
        py: Python<'py>,
        hkl_flat: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<ReflectionFamilyArrays<'py>> {
        let hkl = hkl_rows(&hkl_flat)?;
        let mut ids = Vec::with_capacity(hkl.len());
        let mut canonical = Vec::with_capacity(3 * hkl.len());
        let mut multiplicity = Vec::with_capacity(hkl.len());
        for reflection in hkl {
            let family = self
                .generator
                .space_group()
                .reflection_family(reflection, self.generator.merge_friedel())
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            ids.push(family.reflection_id);
            canonical.extend(family.canonical_hkl.map(i64::from));
            multiplicity.push(
                i64::try_from(family.multiplicity)
                    .map_err(|_| PyValueError::new_err("reflection multiplicity overflow"))?,
            );
        }
        Ok((
            ids,
            Array2::from_shape_vec((multiplicity.len(), 3), canonical)
                .map_err(|error| PyValueError::new_err(error.to_string()))?
                .into_pyarray(py),
            multiplicity.into_pyarray(py),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn generate<'py>(
        &self,
        py: Python<'py>,
        a_angstrom: f64,
        b_angstrom: f64,
        c_angstrom: f64,
        alpha_deg: f64,
        beta_deg: f64,
        gamma_deg: f64,
        range_kind: &str,
        range_parameters: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<GeneratedReflectionArrays<'py>> {
        let parameters = contiguous_slice(&range_parameters, "range_parameters")?;
        let range = reflection_range(range_kind, parameters)?;
        let reflections = self
            .generator
            .generate(
                crystallographic_cell(
                    a_angstrom, b_angstrom, c_angstrom, alpha_deg, beta_deg, gamma_deg,
                ),
                range,
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let count = reflections.len();
        let mut ids = Vec::with_capacity(count);
        let mut hkl = Vec::with_capacity(3 * count);
        let mut multiplicity = Vec::with_capacity(count);
        let mut d_spacing = Vec::with_capacity(count);
        let mut reciprocal_length = Vec::with_capacity(count);
        let mut derivatives = Vec::with_capacity(6 * count);
        for reflection in reflections {
            ids.push(reflection.reflection_id);
            hkl.extend(reflection.hkl.map(i64::from));
            multiplicity.push(
                i64::try_from(reflection.multiplicity)
                    .map_err(|_| PyValueError::new_err("reflection multiplicity overflow"))?,
            );
            d_spacing.push(reflection.d_spacing_angstrom);
            reciprocal_length.push(reflection.reciprocal_length_inverse_angstrom);
            derivatives.extend(reflection.d_spacing_derivatives);
        }
        Ok((
            ids,
            Array2::from_shape_vec((count, 3), hkl)
                .map_err(|error| PyValueError::new_err(error.to_string()))?
                .into_pyarray(py),
            multiplicity.into_pyarray(py),
            d_spacing.into_pyarray(py),
            reciprocal_length.into_pyarray(py),
            Array2::from_shape_vec((count, 6), derivatives)
                .map_err(|error| PyValueError::new_err(error.to_string()))?
                .into_pyarray(py),
        ))
    }
}

/// Immutable native structural phase used by values and derivative products.
#[pyclass(name = "_StructuralPhase")]
struct NativeStructuralPhase {
    space_group: SpaceGroup,
    cell: UnitCell,
    hkl: Vec<[i32; 3]>,
    multiplicity: Vec<usize>,
    fractional_xyz: Vec<[f64; 3]>,
    occupancy: Vec<f64>,
    u_iso_angstrom2: Vec<f64>,
    scattering_species: Vec<String>,
    scale: f64,
    coordinate_tolerance: f64,
    scattering_model: BuiltInScatteringModel,
    correction_model: IntegratedIntensityCorrectionModel,
}

impl NativeStructuralPhase {
    #[allow(clippy::too_many_arguments)]
    fn contribution_view<'a>(
        reflection_count: usize,
        parameter_count: usize,
        gaussian_variance_deg2: &'a PyReadonlyArray1<'_, f64>,
        lorentzian_fwhm_deg: &'a PyReadonlyArray1<'_, f64>,
        intensity_multiplier: &'a PyReadonlyArray1<'_, f64>,
        d_gaussian_variance_d_position: &'a PyReadonlyArray1<'_, f64>,
        d_lorentzian_fwhm_d_position: &'a PyReadonlyArray1<'_, f64>,
        d_intensity_multiplier_d_position: &'a PyReadonlyArray1<'_, f64>,
        d_gaussian_variance_d_parameters: &'a PyReadonlyArray1<'_, f64>,
        d_lorentzian_fwhm_d_parameters: &'a PyReadonlyArray1<'_, f64>,
        d_intensity_multiplier_d_parameters: &'a PyReadonlyArray1<'_, f64>,
    ) -> PyResult<CwContributionsView<'a>> {
        CwContributionsView::new(
            reflection_count,
            parameter_count,
            CwContributionArrays {
                gaussian_variance_deg2: contiguous_slice(
                    gaussian_variance_deg2,
                    "gaussian_variance_deg2",
                )?,
                lorentzian_fwhm_deg: contiguous_slice(lorentzian_fwhm_deg, "lorentzian_fwhm_deg")?,
                intensity_multiplier: contiguous_slice(
                    intensity_multiplier,
                    "intensity_multiplier",
                )?,
                d_gaussian_variance_d_position: contiguous_slice(
                    d_gaussian_variance_d_position,
                    "d_gaussian_variance_d_position",
                )?,
                d_lorentzian_fwhm_d_position: contiguous_slice(
                    d_lorentzian_fwhm_d_position,
                    "d_lorentzian_fwhm_d_position",
                )?,
                d_intensity_multiplier_d_position: contiguous_slice(
                    d_intensity_multiplier_d_position,
                    "d_intensity_multiplier_d_position",
                )?,
                d_gaussian_variance_d_parameters: contiguous_slice(
                    d_gaussian_variance_d_parameters,
                    "d_gaussian_variance_d_parameters",
                )?,
                d_lorentzian_fwhm_d_parameters: contiguous_slice(
                    d_lorentzian_fwhm_d_parameters,
                    "d_lorentzian_fwhm_d_parameters",
                )?,
                d_intensity_multiplier_d_parameters: contiguous_slice(
                    d_intensity_multiplier_d_parameters,
                    "d_intensity_multiplier_d_parameters",
                )?,
            },
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn with_input<R>(
        &self,
        x_deg: &[f64],
        instrument: ConstantWavelengthInstrument,
        position_correction: MonochromaticPositionCorrection,
        contributions: CwContributionsView<'_>,
        support_fwhm: f64,
        operation: impl FnOnce(
            UnitCell,
            &SpaceGroup,
            &StructuralPatternInputView<'_>,
        ) -> Result<R, StructuralPatternError>,
    ) -> PyResult<R> {
        let species = self
            .scattering_species
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let correction_model = match self.correction_model {
            IntegratedIntensityCorrectionModel::Neutral => {
                IntegratedIntensityCorrectionModel::Neutral
            }
            IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp { .. } => {
                IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp {
                    wavelength_angstrom: instrument.wavelength_angstrom,
                }
            }
        };
        let input = StructuralPatternInputView {
            x_deg,
            hkl: &self.hkl,
            multiplicity: &self.multiplicity,
            fractional_xyz: &self.fractional_xyz,
            occupancy: &self.occupancy,
            u_iso_angstrom2: &self.u_iso_angstrom2,
            scattering_species: &species,
            scale: self.scale,
            coordinate_tolerance: self.coordinate_tolerance,
            instrument,
            position_correction,
            correction_model,
            scattering_model: self.scattering_model,
            contributions,
            support: SupportPolicy::FwhmMultiple(support_fwhm),
        };
        operation(self.cell, &self.space_group, &input)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

#[pymethods]
impl NativeStructuralPhase {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        generator: PyRef<'_, NativePreparedReflectionGenerator>,
        hkl_flat: PyReadonlyArray1<'_, i64>,
        multiplicity: PyReadonlyArray1<'_, i64>,
        fractional_xyz_flat: PyReadonlyArray1<'_, f64>,
        occupancy: PyReadonlyArray1<'_, f64>,
        u_iso_angstrom2: PyReadonlyArray1<'_, f64>,
        scattering_species: Vec<String>,
        a_angstrom: f64,
        b_angstrom: f64,
        c_angstrom: f64,
        alpha_deg: f64,
        beta_deg: f64,
        gamma_deg: f64,
        scale: f64,
        coordinate_tolerance: f64,
        scattering_model: &str,
        correction_model: &str,
        correction_wavelength_angstrom: Option<f64>,
    ) -> PyResult<Self> {
        let hkl = hkl_rows(&hkl_flat)?;
        let multiplicity = multiplicity_rows(&multiplicity)?;
        let fractional_xyz = xyz_rows(&fractional_xyz_flat)?;
        let occupancy = contiguous_slice(&occupancy, "occupancy")?.to_vec();
        let u_iso_angstrom2 = contiguous_slice(&u_iso_angstrom2, "u_iso_angstrom2")?.to_vec();
        if hkl.len() != multiplicity.len() {
            return Err(PyValueError::new_err(
                "hkl and multiplicity must have the same reflection count",
            ));
        }
        if fractional_xyz.len() != occupancy.len()
            || fractional_xyz.len() != u_iso_angstrom2.len()
            || fractional_xyz.len() != scattering_species.len()
        {
            return Err(PyValueError::new_err(
                "all structural site arrays must have the same site count",
            ));
        }
        let scattering_model = parse_built_in_scattering_model(scattering_model)?;
        match scattering_model {
            BuiltInScatteringModel::XrayNonResonant => {
                PreparedXrayScattering::new(&scattering_species).map(|_| ())
            }
            BuiltInScatteringModel::NeutronNuclear => {
                PreparedNeutronScattering::new(&scattering_species).map(|_| ())
            }
        }
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            space_group: generator.generator.space_group().clone(),
            cell: crystallographic_cell(
                a_angstrom, b_angstrom, c_angstrom, alpha_deg, beta_deg, gamma_deg,
            ),
            hkl,
            multiplicity,
            fractional_xyz,
            occupancy,
            u_iso_angstrom2,
            scattering_species,
            scale,
            coordinate_tolerance,
            scattering_model,
            correction_model: parse_correction_model(
                correction_model,
                correction_wavelength_angstrom,
            )?,
        })
    }

    #[getter]
    fn reflection_count(&self) -> usize {
        self.hkl.len()
    }

    #[getter]
    fn structural_parameter_count(&self) -> usize {
        6 + 5 * self.fractional_xyz.len() + 1
    }

    #[allow(clippy::too_many_arguments)]
    fn calculate<'py>(
        &self,
        py: Python<'py>,
        x_deg: PyReadonlyArray1<'py, f64>,
        wavelength_angstrom: f64,
        zero_shift_deg: f64,
        sample_displacement_mm: Option<f64>,
        goniometer_radius_mm: Option<f64>,
        u_deg2: f64,
        v_deg2: f64,
        w_deg2: f64,
        x_width_deg: f64,
        y_width_deg: f64,
        gaussian_variance_deg2: PyReadonlyArray1<'py, f64>,
        lorentzian_fwhm_deg: PyReadonlyArray1<'py, f64>,
        intensity_multiplier: PyReadonlyArray1<'py, f64>,
        d_gaussian_variance_d_position: PyReadonlyArray1<'py, f64>,
        d_lorentzian_fwhm_d_position: PyReadonlyArray1<'py, f64>,
        d_intensity_multiplier_d_position: PyReadonlyArray1<'py, f64>,
        d_gaussian_variance_d_parameters: PyReadonlyArray1<'py, f64>,
        d_lorentzian_fwhm_d_parameters: PyReadonlyArray1<'py, f64>,
        d_intensity_multiplier_d_parameters: PyReadonlyArray1<'py, f64>,
        parameter_count: usize,
        support_fwhm: f64,
    ) -> PyResult<StructuralPatternArrays<'py>> {
        let x_deg_values = contiguous_slice(&x_deg, "x_deg")?;
        let contributions = Self::contribution_view(
            self.hkl.len(),
            parameter_count,
            &gaussian_variance_deg2,
            &lorentzian_fwhm_deg,
            &intensity_multiplier,
            &d_gaussian_variance_d_position,
            &d_lorentzian_fwhm_d_position,
            &d_intensity_multiplier_d_position,
            &d_gaussian_variance_d_parameters,
            &d_lorentzian_fwhm_d_parameters,
            &d_intensity_multiplier_d_parameters,
        )?;
        let result = self.with_input(
            x_deg_values,
            cw_instrument(
                wavelength_angstrom,
                u_deg2,
                v_deg2,
                w_deg2,
                x_width_deg,
                y_width_deg,
            ),
            position_correction(zero_shift_deg, sample_displacement_mm, goniometer_radius_mm)?,
            contributions,
            support_fwhm,
            calculate_structural_pattern,
        )?;
        structural_pattern_to_numpy(py, result)
    }

    #[allow(clippy::too_many_arguments)]
    fn jvp<'py>(
        &self,
        py: Python<'py>,
        tangent: PyReadonlyArray1<'py, f64>,
        x_deg: PyReadonlyArray1<'py, f64>,
        wavelength_angstrom: f64,
        zero_shift_deg: f64,
        sample_displacement_mm: Option<f64>,
        goniometer_radius_mm: Option<f64>,
        u_deg2: f64,
        v_deg2: f64,
        w_deg2: f64,
        x_width_deg: f64,
        y_width_deg: f64,
        gaussian_variance_deg2: PyReadonlyArray1<'py, f64>,
        lorentzian_fwhm_deg: PyReadonlyArray1<'py, f64>,
        intensity_multiplier: PyReadonlyArray1<'py, f64>,
        d_gaussian_variance_d_position: PyReadonlyArray1<'py, f64>,
        d_lorentzian_fwhm_d_position: PyReadonlyArray1<'py, f64>,
        d_intensity_multiplier_d_position: PyReadonlyArray1<'py, f64>,
        d_gaussian_variance_d_parameters: PyReadonlyArray1<'py, f64>,
        d_lorentzian_fwhm_d_parameters: PyReadonlyArray1<'py, f64>,
        d_intensity_multiplier_d_parameters: PyReadonlyArray1<'py, f64>,
        parameter_count: usize,
        support_fwhm: f64,
    ) -> PyResult<StructuralPatternJvpArrays<'py>> {
        let tangent = contiguous_slice(&tangent, "tangent")?;
        let x_deg_values = contiguous_slice(&x_deg, "x_deg")?;
        let contributions = Self::contribution_view(
            self.hkl.len(),
            parameter_count,
            &gaussian_variance_deg2,
            &lorentzian_fwhm_deg,
            &intensity_multiplier,
            &d_gaussian_variance_d_position,
            &d_lorentzian_fwhm_d_position,
            &d_intensity_multiplier_d_position,
            &d_gaussian_variance_d_parameters,
            &d_lorentzian_fwhm_d_parameters,
            &d_intensity_multiplier_d_parameters,
        )?;
        let result = self.with_input(
            x_deg_values,
            cw_instrument(
                wavelength_angstrom,
                u_deg2,
                v_deg2,
                w_deg2,
                x_width_deg,
                y_width_deg,
            ),
            position_correction(zero_shift_deg, sample_displacement_mm, goniometer_radius_mm)?,
            contributions,
            support_fwhm,
            |cell, group, input| calculate_structural_pattern_jvp(cell, group, input, tangent),
        )?;
        structural_pattern_jvp_to_numpy(py, result)
    }

    #[allow(clippy::too_many_arguments)]
    fn vjp<'py>(
        &self,
        py: Python<'py>,
        sample_weights: PyReadonlyArray1<'py, f64>,
        x_deg: PyReadonlyArray1<'py, f64>,
        wavelength_angstrom: f64,
        zero_shift_deg: f64,
        sample_displacement_mm: Option<f64>,
        goniometer_radius_mm: Option<f64>,
        u_deg2: f64,
        v_deg2: f64,
        w_deg2: f64,
        x_width_deg: f64,
        y_width_deg: f64,
        gaussian_variance_deg2: PyReadonlyArray1<'py, f64>,
        lorentzian_fwhm_deg: PyReadonlyArray1<'py, f64>,
        intensity_multiplier: PyReadonlyArray1<'py, f64>,
        d_gaussian_variance_d_position: PyReadonlyArray1<'py, f64>,
        d_lorentzian_fwhm_d_position: PyReadonlyArray1<'py, f64>,
        d_intensity_multiplier_d_position: PyReadonlyArray1<'py, f64>,
        d_gaussian_variance_d_parameters: PyReadonlyArray1<'py, f64>,
        d_lorentzian_fwhm_d_parameters: PyReadonlyArray1<'py, f64>,
        d_intensity_multiplier_d_parameters: PyReadonlyArray1<'py, f64>,
        parameter_count: usize,
        support_fwhm: f64,
    ) -> PyResult<StructuralPatternVjpArrays<'py>> {
        let sample_weights = contiguous_slice(&sample_weights, "sample_weights")?;
        let x_deg_values = contiguous_slice(&x_deg, "x_deg")?;
        let contributions = Self::contribution_view(
            self.hkl.len(),
            parameter_count,
            &gaussian_variance_deg2,
            &lorentzian_fwhm_deg,
            &intensity_multiplier,
            &d_gaussian_variance_d_position,
            &d_lorentzian_fwhm_d_position,
            &d_intensity_multiplier_d_position,
            &d_gaussian_variance_d_parameters,
            &d_lorentzian_fwhm_d_parameters,
            &d_intensity_multiplier_d_parameters,
        )?;
        let result = self.with_input(
            x_deg_values,
            cw_instrument(
                wavelength_angstrom,
                u_deg2,
                v_deg2,
                w_deg2,
                x_width_deg,
                y_width_deg,
            ),
            position_correction(zero_shift_deg, sample_displacement_mm, goniometer_radius_mm)?,
            contributions,
            support_fwhm,
            |cell, group, input| {
                calculate_structural_pattern_vjp(cell, group, input, sample_weights)
            },
        )?;
        structural_pattern_vjp_to_numpy(py, result)
    }
}

/// Derive direct and reciprocal cell geometry plus volume derivatives.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn unit_cell_geometry(
    py: Python<'_>,
    a_angstrom: f64,
    b_angstrom: f64,
    c_angstrom: f64,
    alpha_deg: f64,
    beta_deg: f64,
    gamma_deg: f64,
) -> PyResult<CellGeometryArrays<'_>> {
    let geometry = crystallographic_cell(
        a_angstrom, b_angstrom, c_angstrom, alpha_deg, beta_deg, gamma_deg,
    )
    .geometry()
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        matrix3_to_numpy(py, geometry.direct_basis)?,
        matrix3_to_numpy(py, geometry.reciprocal_basis)?,
        matrix3_to_numpy(py, geometry.direct_metric)?,
        matrix3_to_numpy(py, geometry.reciprocal_metric)?,
        geometry.volume_angstrom3,
        geometry.volume_derivatives().to_vec().into_pyarray(py),
    ))
}

/// Evaluate d-spacings and six direct-cell derivatives for a flat hkl array.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn unit_cell_d_spacings<'py>(
    py: Python<'py>,
    hkl_flat: PyReadonlyArray1<'py, i64>,
    a_angstrom: f64,
    b_angstrom: f64,
    c_angstrom: f64,
    alpha_deg: f64,
    beta_deg: f64,
    gamma_deg: f64,
) -> PyResult<CellSpacingArrays<'py>> {
    let hkl = hkl_rows(&hkl_flat)?;
    let geometry = crystallographic_cell(
        a_angstrom, b_angstrom, c_angstrom, alpha_deg, beta_deg, gamma_deg,
    )
    .geometry()
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mut spacings = Vec::with_capacity(hkl.len());
    let mut derivatives = Vec::with_capacity(hkl.len() * 6);
    for reflection in hkl {
        let (spacing, derivative) = geometry
            .d_spacing_and_derivatives(reflection)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        spacings.push(spacing);
        derivatives.extend_from_slice(&derivative);
    }
    Ok((
        spacings.into_pyarray(py),
        Array2::from_shape_vec((derivatives.len() / 6, 6), derivatives)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .into_pyarray(py),
    ))
}

/// Calculate P1 structure factors and a parameter-major dense Jacobian.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn p1_structure_factors_dense<'py>(
    py: Python<'py>,
    hkl_flat: PyReadonlyArray1<'py, i64>,
    fractional_xyz_flat: PyReadonlyArray1<'py, f64>,
    occupancy: PyReadonlyArray1<'py, f64>,
    u_iso_angstrom2: PyReadonlyArray1<'py, f64>,
    scattering_real_flat: PyReadonlyArray1<'py, f64>,
    scattering_imag_flat: PyReadonlyArray1<'py, f64>,
    a_angstrom: f64,
    b_angstrom: f64,
    c_angstrom: f64,
    alpha_deg: f64,
    beta_deg: f64,
    gamma_deg: f64,
    scale: f64,
) -> PyResult<P1DenseArrays<'py>> {
    let hkl = hkl_rows(&hkl_flat)?;
    let xyz = xyz_rows(&fractional_xyz_flat)?;
    let occupancy = contiguous_slice(&occupancy, "occupancy")?;
    let u_iso = contiguous_slice(&u_iso_angstrom2, "u_iso_angstrom2")?;
    let scattering_real = contiguous_slice(&scattering_real_flat, "scattering_real")?;
    let scattering_imag = contiguous_slice(&scattering_imag_flat, "scattering_imag")?;
    let result = calculate_p1_dense(
        crystallographic_cell(
            a_angstrom, b_angstrom, c_angstrom, alpha_deg, beta_deg, gamma_deg,
        ),
        P1BatchView {
            hkl: &hkl,
            fractional_xyz: &xyz,
            occupancy,
            u_iso_angstrom2: u_iso,
            scattering_real,
            scattering_imag,
            scale,
        },
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let reflection_count = result.values.intensity.len();
    let parameter_count = result.layout.parameter_count();
    Ok((
        result.values.f_real.into_pyarray(py),
        result.values.f_imag.into_pyarray(py),
        result.values.intensity.into_pyarray(py),
        derivative_matrix(py, parameter_count, reflection_count, result.d_f_real)?,
        derivative_matrix(py, parameter_count, reflection_count, result.d_f_imag)?,
        derivative_matrix(py, parameter_count, reflection_count, result.d_intensity)?,
    ))
}

/// Calculate P1 values and one structural forward derivative product.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn p1_structure_factors_jvp<'py>(
    py: Python<'py>,
    hkl_flat: PyReadonlyArray1<'py, i64>,
    fractional_xyz_flat: PyReadonlyArray1<'py, f64>,
    occupancy: PyReadonlyArray1<'py, f64>,
    u_iso_angstrom2: PyReadonlyArray1<'py, f64>,
    scattering_real_flat: PyReadonlyArray1<'py, f64>,
    scattering_imag_flat: PyReadonlyArray1<'py, f64>,
    a_angstrom: f64,
    b_angstrom: f64,
    c_angstrom: f64,
    alpha_deg: f64,
    beta_deg: f64,
    gamma_deg: f64,
    scale: f64,
    tangent: PyReadonlyArray1<'py, f64>,
) -> PyResult<P1JvpArrays<'py>> {
    let hkl = hkl_rows(&hkl_flat)?;
    let xyz = xyz_rows(&fractional_xyz_flat)?;
    let occupancy = contiguous_slice(&occupancy, "occupancy")?;
    let u_iso = contiguous_slice(&u_iso_angstrom2, "u_iso_angstrom2")?;
    let scattering_real = contiguous_slice(&scattering_real_flat, "scattering_real")?;
    let scattering_imag = contiguous_slice(&scattering_imag_flat, "scattering_imag")?;
    let tangent = contiguous_slice(&tangent, "tangent")?;
    let result = calculate_p1_jvp(
        crystallographic_cell(
            a_angstrom, b_angstrom, c_angstrom, alpha_deg, beta_deg, gamma_deg,
        ),
        P1BatchView {
            hkl: &hkl,
            fractional_xyz: &xyz,
            occupancy,
            u_iso_angstrom2: u_iso,
            scattering_real,
            scattering_imag,
            scale,
        },
        tangent,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        result.values.f_real.into_pyarray(py),
        result.values.f_imag.into_pyarray(py),
        result.values.intensity.into_pyarray(py),
        result.d_f_real.into_pyarray(py),
        result.d_f_imag.into_pyarray(py),
        result.d_intensity.into_pyarray(py),
    ))
}

/// Calculate P1 values and an intensity reverse derivative product.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn p1_structure_factors_vjp<'py>(
    py: Python<'py>,
    hkl_flat: PyReadonlyArray1<'py, i64>,
    fractional_xyz_flat: PyReadonlyArray1<'py, f64>,
    occupancy: PyReadonlyArray1<'py, f64>,
    u_iso_angstrom2: PyReadonlyArray1<'py, f64>,
    scattering_real_flat: PyReadonlyArray1<'py, f64>,
    scattering_imag_flat: PyReadonlyArray1<'py, f64>,
    a_angstrom: f64,
    b_angstrom: f64,
    c_angstrom: f64,
    alpha_deg: f64,
    beta_deg: f64,
    gamma_deg: f64,
    scale: f64,
    weights: PyReadonlyArray1<'py, f64>,
) -> PyResult<P1VjpArrays<'py>> {
    let hkl = hkl_rows(&hkl_flat)?;
    let xyz = xyz_rows(&fractional_xyz_flat)?;
    let occupancy = contiguous_slice(&occupancy, "occupancy")?;
    let u_iso = contiguous_slice(&u_iso_angstrom2, "u_iso_angstrom2")?;
    let scattering_real = contiguous_slice(&scattering_real_flat, "scattering_real")?;
    let scattering_imag = contiguous_slice(&scattering_imag_flat, "scattering_imag")?;
    let weights = contiguous_slice(&weights, "weights")?;
    let result = calculate_p1_intensity_vjp(
        crystallographic_cell(
            a_angstrom, b_angstrom, c_angstrom, alpha_deg, beta_deg, gamma_deg,
        ),
        P1BatchView {
            hkl: &hkl,
            fractional_xyz: &xyz,
            occupancy,
            u_iso_angstrom2: u_iso,
            scattering_real,
            scattering_imag,
            scale,
        },
        weights,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        result.values.f_real.into_pyarray(py),
        result.values.f_imag.into_pyarray(py),
        result.values.intensity.into_pyarray(py),
        result.gradient.into_pyarray(py),
    ))
}

/// Vectorized scalar profile evaluation used by the public Python wrapper.
#[pyfunction]
fn profile<'py>(
    py: Python<'py>,
    delta: PyReadonlyArray1<'py, f64>,
    fwhm: f64,
    eta: f64,
) -> PyResult<ProfileArrays<'py>> {
    if !fwhm.is_finite() || fwhm <= 0.0 {
        return Err(PyValueError::new_err("fwhm must be positive and finite"));
    }
    if !eta.is_finite() || !(0.0..=1.0).contains(&eta) {
        return Err(PyValueError::new_err(
            "eta must be finite and within [0, 1]",
        ));
    }
    let delta = delta
        .as_slice()
        .map_err(|_| PyValueError::new_err("delta must be a contiguous one-dimensional array"))?;
    if delta.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "delta must contain only finite values",
        ));
    }

    let mut value = Vec::with_capacity(delta.len());
    let mut d_delta = Vec::with_capacity(delta.len());
    let mut d_fwhm = Vec::with_capacity(delta.len());
    let mut d_eta = Vec::with_capacity(delta.len());
    for coordinate in delta.iter().copied() {
        let point = symmetric_pseudo_voigt(coordinate, fwhm, eta);
        value.push(point.value);
        d_delta.push(point.d_delta);
        d_fwhm.push(point.d_fwhm);
        d_eta.push(point.d_eta);
    }
    Ok((
        value.into_pyarray(py),
        d_delta.into_pyarray(py),
        d_fwhm.into_pyarray(py),
        d_eta.into_pyarray(py),
    ))
}

/// Transform component FWHMs into TCH total width, eta, and derivatives.
#[pyfunction]
fn tch_shape_from_fwhm(gaussian_fwhm: f64, lorentzian_fwhm: f64) -> PyResult<TchShapeValues> {
    let shape = TchShape::from_component_fwhm(TchWidths {
        gaussian_fwhm,
        lorentzian_fwhm,
    })
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        shape.total_fwhm,
        shape.eta,
        shape.d_total_fwhm_d_gaussian_fwhm,
        shape.d_total_fwhm_d_lorentzian_fwhm,
        shape.d_eta_d_gaussian_fwhm,
        shape.d_eta_d_lorentzian_fwhm,
    ))
}

/// Vectorized TCH profile evaluation with component-width derivatives.
#[pyfunction]
fn profile_tch<'py>(
    py: Python<'py>,
    delta: PyReadonlyArray1<'py, f64>,
    gaussian_fwhm: f64,
    lorentzian_fwhm: f64,
) -> PyResult<ProfileArrays<'py>> {
    let delta = contiguous_slice(&delta, "delta")?;
    if delta.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "delta must contain only finite values",
        ));
    }
    let widths = TchWidths {
        gaussian_fwhm,
        lorentzian_fwhm,
    };
    let shape = TchShape::from_component_fwhm(widths)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mut value = Vec::with_capacity(delta.len());
    let mut d_delta = Vec::with_capacity(delta.len());
    let mut d_gaussian = Vec::with_capacity(delta.len());
    let mut d_lorentzian = Vec::with_capacity(delta.len());
    for coordinate in delta.iter().copied() {
        let point = shape.evaluate(coordinate);
        value.push(point.value);
        d_delta.push(point.d_delta);
        d_gaussian.push(point.d_gaussian_fwhm);
        d_lorentzian.push(point.d_lorentzian_fwhm);
    }
    Ok((
        value.into_pyarray(py),
        d_delta.into_pyarray(py),
        d_gaussian.into_pyarray(py),
        d_lorentzian.into_pyarray(py),
    ))
}

/// Vectorized FCJ-convolved TCH evaluation with direct-input derivatives.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn profile_fcj<'py>(
    py: Python<'py>,
    x_deg: PyReadonlyArray1<'py, f64>,
    position_deg: f64,
    gaussian_fwhm_deg: f64,
    lorentzian_fwhm_deg: f64,
    sample_over_radius: f64,
    detector_over_radius: f64,
) -> PyResult<FcjProfileArrays<'py>> {
    let x_deg = contiguous_slice(&x_deg, "x_deg")?;
    if x_deg.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "x_deg must contain only finite values",
        ));
    }
    let profile = FcjProfile::new(
        position_deg,
        TchWidths {
            gaussian_fwhm: gaussian_fwhm_deg,
            lorentzian_fwhm: lorentzian_fwhm_deg,
        },
        FcjGeometry {
            sample_over_radius,
            detector_over_radius,
        },
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mut value = Vec::with_capacity(x_deg.len());
    let mut d_position = Vec::with_capacity(x_deg.len());
    let mut d_gaussian = Vec::with_capacity(x_deg.len());
    let mut d_lorentzian = Vec::with_capacity(x_deg.len());
    let mut d_sample = Vec::with_capacity(x_deg.len());
    let mut d_detector = Vec::with_capacity(x_deg.len());
    for coordinate in x_deg.iter().copied() {
        let point = profile.evaluate(coordinate);
        value.push(point.value);
        d_position.push(point.d_position);
        d_gaussian.push(point.d_gaussian_fwhm);
        d_lorentzian.push(point.d_lorentzian_fwhm);
        d_sample.push(point.d_sample_over_radius);
        d_detector.push(point.d_detector_over_radius);
    }
    Ok((
        value.into_pyarray(py),
        d_position.into_pyarray(py),
        d_gaussian.into_pyarray(py),
        d_lorentzian.into_pyarray(py),
        d_sample.into_pyarray(py),
        d_detector.into_pyarray(py),
    ))
}

/// Vectorized truncated double-exponential TCH profile and derivatives.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn profile_tof<'py>(
    py: Python<'py>,
    x_us: PyReadonlyArray1<'py, f64>,
    position_us: f64,
    alpha_per_us: f64,
    beta_per_us: f64,
    gaussian_fwhm_us: f64,
    lorentzian_fwhm_us: f64,
    tail_log: f64,
) -> PyResult<TofProfileArrays<'py>> {
    let x_us = contiguous_slice(&x_us, "x_us")?;
    if x_us.iter().any(|value| !value.is_finite()) || !position_us.is_finite() {
        return Err(PyValueError::new_err(
            "TOF coordinates and position must be finite",
        ));
    }
    let profile = TofProfile::new(
        alpha_per_us,
        beta_per_us,
        TchWidths {
            gaussian_fwhm: gaussian_fwhm_us,
            lorentzian_fwhm: lorentzian_fwhm_us,
        },
        tail_log,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mut value = Vec::with_capacity(x_us.len());
    let mut d_position = Vec::with_capacity(x_us.len());
    let mut d_alpha = Vec::with_capacity(x_us.len());
    let mut d_beta = Vec::with_capacity(x_us.len());
    let mut d_gaussian = Vec::with_capacity(x_us.len());
    let mut d_lorentzian = Vec::with_capacity(x_us.len());
    for coordinate in x_us.iter().copied() {
        let point = profile.evaluate(coordinate - position_us);
        value.push(point.value);
        d_position.push(point.d_position);
        d_alpha.push(point.d_alpha);
        d_beta.push(point.d_beta);
        d_gaussian.push(point.d_gaussian_fwhm);
        d_lorentzian.push(point.d_lorentzian_fwhm);
    }
    Ok((
        value.into_pyarray(py),
        d_position.into_pyarray(py),
        d_alpha.into_pyarray(py),
        d_beta.into_pyarray(py),
        d_gaussian.into_pyarray(py),
        d_lorentzian.into_pyarray(py),
    ))
}

/// Fused peak accumulation returning dense values and support-sparse derivatives.
#[pyfunction]
fn accumulate<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    positions: PyReadonlyArray1<'py, f64>,
    intensities: PyReadonlyArray1<'py, f64>,
    fwhms: PyReadonlyArray1<'py, f64>,
    etas: PyReadonlyArray1<'py, f64>,
    support_fwhm: f64,
) -> PyResult<AccumulationArrays<'py>> {
    let x = contiguous_slice(&x, "x")?;
    let positions = contiguous_slice(&positions, "positions")?;
    let intensities = contiguous_slice(&intensities, "intensities")?;
    let fwhms = contiguous_slice(&fwhms, "fwhms")?;
    let etas = contiguous_slice(&etas, "etas")?;

    let grid = GridView::new(x).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let peaks = PeakBatchView::new(positions, intensities, fwhms, etas)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let accumulation = accumulate_batch(grid, peaks, SupportPolicy::FwhmMultiple(support_fwhm))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    accumulation_to_numpy(py, accumulation)
}

/// Pinned xypattern-compatible Smooth Bruckner background estimator.
#[pyfunction]
fn smooth_bruckner<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    smooth_points: usize,
    iterations: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let y = contiguous_slice(&y, "y")?;
    let background = native_smooth_bruckner(y, smooth_points, iterations)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(background.into_pyarray(py))
}

/// Fused TCH peak accumulation with support-sparse direct-input derivatives.
#[pyfunction]
fn accumulate_tch<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    positions: PyReadonlyArray1<'py, f64>,
    intensities: PyReadonlyArray1<'py, f64>,
    gaussian_fwhms: PyReadonlyArray1<'py, f64>,
    lorentzian_fwhms: PyReadonlyArray1<'py, f64>,
    support_fwhm: f64,
) -> PyResult<AccumulationArrays<'py>> {
    let x = contiguous_slice(&x, "x")?;
    let positions = contiguous_slice(&positions, "positions")?;
    let intensities = contiguous_slice(&intensities, "intensities")?;
    let gaussian_fwhms = contiguous_slice(&gaussian_fwhms, "gaussian_fwhms")?;
    let lorentzian_fwhms = contiguous_slice(&lorentzian_fwhms, "lorentzian_fwhms")?;
    let grid = GridView::new(x).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let peaks = TchPeakBatchView::new(positions, intensities, gaussian_fwhms, lorentzian_fwhms)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let accumulation = accumulate_tch_batch(grid, peaks, SupportPolicy::FwhmMultiple(support_fwhm))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    accumulation_to_numpy(py, accumulation)
}

fn accumulation_to_numpy(
    py: Python<'_>,
    accumulation: Accumulation,
) -> PyResult<AccumulationArrays<'_>> {
    let local = accumulation.derivatives.local;
    let value_rows = local.active_sample_count();
    let local_parameter_count = local.parameter_count;
    let starts = indices_to_i64(local.starts, "support starts")?;
    let offsets = indices_to_i64(local.offsets, "support offsets")?;
    let values = Array2::from_shape_vec((value_rows, local_parameter_count), local.values)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let global = if let Some(global) = accumulation.derivatives.global {
        if global.sample_count != accumulation.sample_count {
            return Err(PyValueError::new_err(
                "global Jacobian sample count does not match accumulation",
            ));
        }
        Array2::from_shape_vec((global.parameter_count, global.sample_count), global.values)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
    } else {
        Array2::zeros((0, accumulation.sample_count))
    };
    Ok((
        accumulation.y.into_pyarray(py),
        starts.into_pyarray(py),
        offsets.into_pyarray(py),
        values.into_pyarray(py),
        global.into_pyarray(py),
    ))
}

fn structural_pattern_to_numpy(
    py: Python<'_>,
    result: StructuralPatternResult,
) -> PyResult<StructuralPatternArrays<'_>> {
    let StructuralPatternResult {
        structure_factors,
        d_spacing_angstrom,
        two_theta_deg,
        accumulation,
    } = result;
    let reflections = (
        structure_factors.f_real.into_pyarray(py),
        structure_factors.f_imag.into_pyarray(py),
        structure_factors.f_squared.into_pyarray(py),
        structure_factors.intensity.into_pyarray(py),
        structure_factors
            .q_squared_inverse_angstrom2
            .into_pyarray(py),
        structure_factors.s_inverse_angstrom.into_pyarray(py),
        d_spacing_angstrom.into_pyarray(py),
        two_theta_deg.into_pyarray(py),
    );
    Ok((accumulation_to_numpy(py, accumulation)?, reflections))
}

fn structural_pattern_jvp_to_numpy(
    py: Python<'_>,
    result: StructuralPatternJvpResult,
) -> PyResult<StructuralPatternJvpArrays<'_>> {
    Ok((
        structural_pattern_to_numpy(py, result.result)?,
        result.d_y.into_pyarray(py),
        result.d_integrated_intensity.into_pyarray(py),
        result.d_two_theta_deg.into_pyarray(py),
    ))
}

fn structural_pattern_vjp_to_numpy(
    py: Python<'_>,
    result: StructuralPatternVjpResult,
) -> PyResult<StructuralPatternVjpArrays<'_>> {
    Ok((
        structural_pattern_to_numpy(py, result.result)?,
        result.gradient.into_pyarray(py),
    ))
}

/// Derive CW component widths, TCH shape, and width derivatives for reflections.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn cw_profile_parameters<'py>(
    py: Python<'py>,
    two_theta_deg: PyReadonlyArray1<'py, f64>,
    reference_wavelength_angstrom: f64,
    u_deg2: f64,
    v_deg2: f64,
    w_deg2: f64,
    x_deg: f64,
    y_deg: f64,
) -> PyResult<CwProfileArrays<'py>> {
    let two_theta_deg = contiguous_slice(&two_theta_deg, "two_theta_deg")?;
    let instrument = cw_instrument(
        reference_wavelength_angstrom,
        u_deg2,
        v_deg2,
        w_deg2,
        x_deg,
        y_deg,
    );
    instrument
        .validate()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mut variance = Vec::with_capacity(two_theta_deg.len());
    let mut gaussian = Vec::with_capacity(two_theta_deg.len());
    let mut lorentzian = Vec::with_capacity(two_theta_deg.len());
    let mut total = Vec::with_capacity(two_theta_deg.len());
    let mut eta = Vec::with_capacity(two_theta_deg.len());
    let mut d_gaussian = Vec::with_capacity(two_theta_deg.len() * 5);
    let mut d_lorentzian = Vec::with_capacity(two_theta_deg.len() * 5);
    let mut d_position = Vec::with_capacity(two_theta_deg.len() * 2);
    for position in two_theta_deg.iter().copied() {
        let profile = CwProfileParameters::from_instrument(position, instrument)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        variance.push(profile.gaussian_variance_deg2);
        gaussian.push(profile.gaussian_fwhm_deg);
        lorentzian.push(profile.lorentzian_fwhm_deg);
        total.push(profile.tch.total_fwhm);
        eta.push(profile.tch.eta);
        d_gaussian.extend_from_slice(&profile.d_gaussian_fwhm_d_instrument);
        d_lorentzian.extend_from_slice(&profile.d_lorentzian_fwhm_d_instrument);
        d_position.push(profile.d_gaussian_fwhm_d_two_theta);
        d_position.push(profile.d_lorentzian_fwhm_d_two_theta);
    }
    let count = two_theta_deg.len();
    Ok((
        variance.into_pyarray(py),
        gaussian.into_pyarray(py),
        lorentzian.into_pyarray(py),
        total.into_pyarray(py),
        eta.into_pyarray(py),
        Array2::from_shape_vec((count, 5), d_gaussian)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .into_pyarray(py),
        Array2::from_shape_vec((count, 5), d_lorentzian)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .into_pyarray(py),
        Array2::from_shape_vec((count, 2), d_position)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .into_pyarray(py),
    ))
}

/// Derive TOF calibration, rates, widths, and TCH shape for d-spacings.
#[pyfunction]
#[allow(clippy::similar_names, clippy::too_many_arguments)]
fn tof_profile_parameters<'py>(
    py: Python<'py>,
    d_spacing_angstrom: PyReadonlyArray1<'py, f64>,
    zero_us: f64,
    difc_us_per_angstrom: f64,
    difa_us_per_angstrom2: f64,
    difb_us_angstrom: f64,
    alpha_coefficient: f64,
    beta0_per_us: f64,
    beta1_angstrom4_per_us: f64,
    betaq_angstrom2_per_us: f64,
    sigma0_us2: f64,
    sigma1_us2_per_angstrom2: f64,
    sigma2_us2_per_angstrom4: f64,
    sigmaq_us2_per_angstrom: f64,
    x_us_per_angstrom: f64,
    y_us_per_angstrom2: f64,
    z_us: f64,
) -> PyResult<TofParameterArrays<'py>> {
    let d_spacing = contiguous_slice(&d_spacing_angstrom, "d_spacing_angstrom")?;
    let instrument = tof_instrument(
        zero_us,
        difc_us_per_angstrom,
        difa_us_per_angstrom2,
        difb_us_angstrom,
        alpha_coefficient,
        beta0_per_us,
        beta1_angstrom4_per_us,
        betaq_angstrom2_per_us,
        sigma0_us2,
        sigma1_us2_per_angstrom2,
        sigma2_us2_per_angstrom4,
        sigmaq_us2_per_angstrom,
        x_us_per_angstrom,
        y_us_per_angstrom2,
        z_us,
    );
    let mut position = Vec::with_capacity(d_spacing.len());
    let mut alpha = Vec::with_capacity(d_spacing.len());
    let mut beta = Vec::with_capacity(d_spacing.len());
    let mut variance = Vec::with_capacity(d_spacing.len());
    let mut gaussian = Vec::with_capacity(d_spacing.len());
    let mut lorentzian = Vec::with_capacity(d_spacing.len());
    let mut total = Vec::with_capacity(d_spacing.len());
    let mut eta = Vec::with_capacity(d_spacing.len());
    for d in d_spacing.iter().copied() {
        let parameters = TofProfileParameters::from_instrument(d, instrument)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        position.push(parameters.position_us);
        alpha.push(parameters.alpha_per_us);
        beta.push(parameters.beta_per_us);
        variance.push(parameters.gaussian_variance_us2);
        gaussian.push(parameters.gaussian_fwhm_us);
        lorentzian.push(parameters.lorentzian_fwhm_us);
        total.push(parameters.tch.total_fwhm);
        eta.push(parameters.tch.eta);
    }
    Ok((
        position.into_pyarray(py),
        alpha.into_pyarray(py),
        beta.into_pyarray(py),
        variance.into_pyarray(py),
        gaussian.into_pyarray(py),
        lorentzian.into_pyarray(py),
        total.into_pyarray(py),
        eta.into_pyarray(py),
    ))
}

/// Fused TOF reflection accumulation with local and global derivatives.
#[pyfunction]
#[allow(clippy::similar_names, clippy::too_many_arguments)]
fn accumulate_tof<'py>(
    py: Python<'py>,
    x_us: PyReadonlyArray1<'py, f64>,
    d_spacing_angstrom: PyReadonlyArray1<'py, f64>,
    integrated_intensities: PyReadonlyArray1<'py, f64>,
    zero_us: f64,
    difc_us_per_angstrom: f64,
    difa_us_per_angstrom2: f64,
    difb_us_angstrom: f64,
    alpha_coefficient: f64,
    beta0_per_us: f64,
    beta1_angstrom4_per_us: f64,
    betaq_angstrom2_per_us: f64,
    sigma0_us2: f64,
    sigma1_us2_per_angstrom2: f64,
    sigma2_us2_per_angstrom4: f64,
    sigmaq_us2_per_angstrom: f64,
    x_us_per_angstrom: f64,
    y_us_per_angstrom2: f64,
    z_us: f64,
    support_fwhm: f64,
    tail_log: f64,
) -> PyResult<AccumulationArrays<'py>> {
    let x = contiguous_slice(&x_us, "x_us")?;
    let d_spacing = contiguous_slice(&d_spacing_angstrom, "d_spacing_angstrom")?;
    let intensities = contiguous_slice(&integrated_intensities, "integrated_intensities")?;
    let grid = GridView::new(x).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let accumulation = accumulate_tof_batch(
        grid,
        d_spacing,
        intensities,
        tof_instrument(
            zero_us,
            difc_us_per_angstrom,
            difa_us_per_angstrom2,
            difb_us_angstrom,
            alpha_coefficient,
            beta0_per_us,
            beta1_angstrom4_per_us,
            betaq_angstrom2_per_us,
            sigma0_us2,
            sigma1_us2_per_angstrom2,
            sigma2_us2_per_angstrom4,
            sigmaq_us2_per_angstrom,
            x_us_per_angstrom,
            y_us_per_angstrom2,
            z_us,
        ),
        support_fwhm,
        tail_log,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    accumulation_to_numpy(py, accumulation)
}

/// Accumulate a CW reflection batch with local and global derivatives.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn accumulate_cw<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    two_theta_deg: PyReadonlyArray1<'py, f64>,
    intensities: PyReadonlyArray1<'py, f64>,
    wavelength_angstrom: f64,
    u_deg2: f64,
    v_deg2: f64,
    w_deg2: f64,
    x_deg: f64,
    y_deg: f64,
    support_fwhm: f64,
) -> PyResult<AccumulationArrays<'py>> {
    let x = contiguous_slice(&x, "x")?;
    let two_theta_deg = contiguous_slice(&two_theta_deg, "two_theta_deg")?;
    let intensities = contiguous_slice(&intensities, "intensities")?;
    let grid = GridView::new(x).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let reflections = CwReflectionBatchView::new(two_theta_deg, intensities)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let accumulation = accumulate_cw_batch(
        grid,
        reflections,
        cw_instrument(wavelength_angstrom, u_deg2, v_deg2, w_deg2, x_deg, y_deg),
        SupportPolicy::FwhmMultiple(support_fwhm),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    accumulation_to_numpy(py, accumulation)
}

/// Accumulate a CW batch with vectorized sample-physics contributions.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn accumulate_cw_contributions<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    two_theta_deg: PyReadonlyArray1<'py, f64>,
    base_intensities: PyReadonlyArray1<'py, f64>,
    wavelength_angstrom: f64,
    u_deg2: f64,
    v_deg2: f64,
    w_deg2: f64,
    x_deg: f64,
    y_deg: f64,
    gaussian_variance_deg2: PyReadonlyArray1<'py, f64>,
    lorentzian_fwhm_deg: PyReadonlyArray1<'py, f64>,
    intensity_multiplier: PyReadonlyArray1<'py, f64>,
    d_gaussian_variance_d_position: PyReadonlyArray1<'py, f64>,
    d_lorentzian_fwhm_d_position: PyReadonlyArray1<'py, f64>,
    d_intensity_multiplier_d_position: PyReadonlyArray1<'py, f64>,
    d_gaussian_variance_d_parameters: PyReadonlyArray1<'py, f64>,
    d_lorentzian_fwhm_d_parameters: PyReadonlyArray1<'py, f64>,
    d_intensity_multiplier_d_parameters: PyReadonlyArray1<'py, f64>,
    parameter_count: usize,
    support_fwhm: f64,
) -> PyResult<AccumulationArrays<'py>> {
    let x = contiguous_slice(&x, "x")?;
    let two_theta_deg = contiguous_slice(&two_theta_deg, "two_theta_deg")?;
    let base_intensities = contiguous_slice(&base_intensities, "base_intensities")?;
    let gaussian_variance_deg2 =
        contiguous_slice(&gaussian_variance_deg2, "gaussian_variance_deg2")?;
    let lorentzian_fwhm_deg = contiguous_slice(&lorentzian_fwhm_deg, "lorentzian_fwhm_deg")?;
    let intensity_multiplier = contiguous_slice(&intensity_multiplier, "intensity_multiplier")?;
    let d_gaussian_variance_d_position = contiguous_slice(
        &d_gaussian_variance_d_position,
        "d_gaussian_variance_d_position",
    )?;
    let d_lorentzian_fwhm_d_position = contiguous_slice(
        &d_lorentzian_fwhm_d_position,
        "d_lorentzian_fwhm_d_position",
    )?;
    let d_intensity_multiplier_d_position = contiguous_slice(
        &d_intensity_multiplier_d_position,
        "d_intensity_multiplier_d_position",
    )?;
    let d_gaussian_variance_d_parameters = contiguous_slice(
        &d_gaussian_variance_d_parameters,
        "d_gaussian_variance_d_parameters",
    )?;
    let d_lorentzian_fwhm_d_parameters = contiguous_slice(
        &d_lorentzian_fwhm_d_parameters,
        "d_lorentzian_fwhm_d_parameters",
    )?;
    let d_intensity_multiplier_d_parameters = contiguous_slice(
        &d_intensity_multiplier_d_parameters,
        "d_intensity_multiplier_d_parameters",
    )?;
    let grid = GridView::new(x).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let contributions = CwContributionsView::new(
        two_theta_deg.len(),
        parameter_count,
        CwContributionArrays {
            gaussian_variance_deg2,
            lorentzian_fwhm_deg,
            intensity_multiplier,
            d_gaussian_variance_d_position,
            d_lorentzian_fwhm_d_position,
            d_intensity_multiplier_d_position,
            d_gaussian_variance_d_parameters,
            d_lorentzian_fwhm_d_parameters,
            d_intensity_multiplier_d_parameters,
        },
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let accumulation = accumulate_cw_contributions_batch(
        grid,
        two_theta_deg,
        base_intensities,
        cw_instrument(wavelength_angstrom, u_deg2, v_deg2, w_deg2, x_deg, y_deg),
        contributions,
        SupportPolicy::FwhmMultiple(support_fwhm),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    accumulation_to_numpy(py, accumulation)
}

/// Accumulate an FCJ-asymmetric CW reflection batch with derivatives.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn accumulate_cw_fcj<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    two_theta_deg: PyReadonlyArray1<'py, f64>,
    intensities: PyReadonlyArray1<'py, f64>,
    wavelength_angstrom: f64,
    u_deg2: f64,
    v_deg2: f64,
    w_deg2: f64,
    x_deg: f64,
    y_deg: f64,
    sample_over_radius: f64,
    detector_over_radius: f64,
    support_fwhm: f64,
) -> PyResult<AccumulationArrays<'py>> {
    let x = contiguous_slice(&x, "x")?;
    let two_theta_deg = contiguous_slice(&two_theta_deg, "two_theta_deg")?;
    let intensities = contiguous_slice(&intensities, "intensities")?;
    let grid = GridView::new(x).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let reflections = CwReflectionBatchView::new(two_theta_deg, intensities)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let accumulation = accumulate_cw_fcj_batch(
        grid,
        reflections,
        cw_instrument(wavelength_angstrom, u_deg2, v_deg2, w_deg2, x_deg, y_deg),
        FcjGeometry {
            sample_over_radius,
            detector_over_radius,
        },
        SupportPolicy::FwhmMultiple(support_fwhm),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    accumulation_to_numpy(py, accumulation)
}

/// Accumulate an optional-FCJ CW wavelength-component reflection batch.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn accumulate_cw_components<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    two_theta_deg: PyReadonlyArray1<'py, f64>,
    intensities: PyReadonlyArray1<'py, f64>,
    wavelengths_angstrom: PyReadonlyArray1<'py, f64>,
    relative_component_intensities: PyReadonlyArray1<'py, f64>,
    reference_wavelength_angstrom: f64,
    u_deg2: f64,
    v_deg2: f64,
    w_deg2: f64,
    x_deg: f64,
    y_deg: f64,
    use_fcj: bool,
    sample_over_radius: f64,
    detector_over_radius: f64,
    support_fwhm: f64,
) -> PyResult<AccumulationArrays<'py>> {
    let x = contiguous_slice(&x, "x")?;
    let two_theta_deg = contiguous_slice(&two_theta_deg, "two_theta_deg")?;
    let intensities = contiguous_slice(&intensities, "intensities")?;
    let wavelengths_angstrom = contiguous_slice(&wavelengths_angstrom, "wavelengths_angstrom")?;
    let relative_component_intensities = contiguous_slice(
        &relative_component_intensities,
        "relative_component_intensities",
    )?;
    let grid = GridView::new(x).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let reflections = CwReflectionBatchView::new(two_theta_deg, intensities)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let components =
        WavelengthComponentsView::new(wavelengths_angstrom, relative_component_intensities)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let instrument = cw_instrument(
        reference_wavelength_angstrom,
        u_deg2,
        v_deg2,
        w_deg2,
        x_deg,
        y_deg,
    );
    let support = SupportPolicy::FwhmMultiple(support_fwhm);
    let accumulation = if use_fcj {
        accumulate_cw_fcj_components_batch(
            grid,
            reflections,
            instrument,
            components,
            FcjGeometry {
                sample_over_radius,
                detector_over_radius,
            },
            support,
        )
    } else {
        accumulate_cw_components_batch(grid, reflections, instrument, components, support)
    }
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    accumulation_to_numpy(py, accumulation)
}

const fn cw_instrument(
    wavelength_angstrom: f64,
    u_deg2: f64,
    v_deg2: f64,
    w_deg2: f64,
    x_deg: f64,
    y_deg: f64,
) -> ConstantWavelengthInstrument {
    ConstantWavelengthInstrument {
        wavelength_angstrom,
        u_deg2,
        v_deg2,
        w_deg2,
        x_deg,
        y_deg,
    }
}

fn position_correction(
    zero_shift_deg: f64,
    sample_displacement_mm: Option<f64>,
    goniometer_radius_mm: Option<f64>,
) -> PyResult<MonochromaticPositionCorrection> {
    let bragg_brentano_mm = match (sample_displacement_mm, goniometer_radius_mm) {
        (None, None) => None,
        (Some(displacement), Some(radius)) => Some((displacement, radius)),
        _ => {
            return Err(PyValueError::new_err(
                "sample displacement and goniometer radius must be provided together",
            ));
        }
    };
    Ok(MonochromaticPositionCorrection {
        zero_shift_deg,
        bragg_brentano_mm,
    })
}

#[allow(clippy::similar_names, clippy::too_many_arguments)]
const fn tof_instrument(
    zero_us: f64,
    difc_us_per_angstrom: f64,
    difa_us_per_angstrom2: f64,
    difb_us_angstrom: f64,
    alpha_coefficient: f64,
    beta0_per_us: f64,
    beta1_angstrom4_per_us: f64,
    betaq_angstrom2_per_us: f64,
    sigma0_us2: f64,
    sigma1_us2_per_angstrom2: f64,
    sigma2_us2_per_angstrom4: f64,
    sigmaq_us2_per_angstrom: f64,
    x_us_per_angstrom: f64,
    y_us_per_angstrom2: f64,
    z_us: f64,
) -> TofInstrument {
    TofInstrument {
        zero_us,
        difc_us_per_angstrom,
        difa_us_per_angstrom2,
        difb_us_angstrom,
        alpha_coefficient,
        beta0_per_us,
        beta1_angstrom4_per_us,
        betaq_angstrom2_per_us,
        sigma0_us2,
        sigma1_us2_per_angstrom2,
        sigma2_us2_per_angstrom4,
        sigmaq_us2_per_angstrom,
        x_us_per_angstrom,
        y_us_per_angstrom2,
        z_us,
    }
}

/// Fused peak accumulation returning calculated values without derivatives.
#[pyfunction]
fn accumulate_values<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    positions: PyReadonlyArray1<'py, f64>,
    intensities: PyReadonlyArray1<'py, f64>,
    fwhms: PyReadonlyArray1<'py, f64>,
    etas: PyReadonlyArray1<'py, f64>,
    support_fwhm: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = contiguous_slice(&x, "x")?;
    let positions = contiguous_slice(&positions, "positions")?;
    let intensities = contiguous_slice(&intensities, "intensities")?;
    let fwhms = contiguous_slice(&fwhms, "fwhms")?;
    let etas = contiguous_slice(&etas, "etas")?;
    let grid = GridView::new(x).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let peaks = PeakBatchView::new(positions, intensities, fwhms, etas)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let y = accumulate_values_batch(grid, peaks, SupportPolicy::FwhmMultiple(support_fwhm))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(y.into_pyarray(py))
}

fn indices_to_i64(indices: Vec<usize>, name: &str) -> PyResult<Vec<i64>> {
    indices
        .into_iter()
        .map(|index| {
            i64::try_from(index)
                .map_err(|_| PyValueError::new_err(format!("{name} exceed NumPy int64 range")))
        })
        .collect()
}

fn contiguous_slice<'array>(
    array: &'array PyReadonlyArray1<'_, f64>,
    name: &str,
) -> PyResult<&'array [f64]> {
    array.as_slice().map_err(|_| {
        PyValueError::new_err(format!("{name} must be a contiguous one-dimensional array"))
    })
}

fn crystallographic_cell(
    a_angstrom: f64,
    b_angstrom: f64,
    c_angstrom: f64,
    alpha_deg: f64,
    beta_deg: f64,
    gamma_deg: f64,
) -> UnitCell {
    UnitCell {
        a_angstrom,
        b_angstrom,
        c_angstrom,
        alpha_deg,
        beta_deg,
        gamma_deg,
    }
}

fn hkl_rows(array: &PyReadonlyArray1<'_, i64>) -> PyResult<Vec<[i32; 3]>> {
    let values = array
        .as_slice()
        .map_err(|_| PyValueError::new_err("hkl must be a contiguous flattened array"))?;
    if values.len() % 3 != 0 {
        return Err(PyValueError::new_err(
            "flattened hkl length must be divisible by three",
        ));
    }
    values
        .chunks_exact(3)
        .map(|row| {
            Ok([
                i32::try_from(row[0])
                    .map_err(|_| PyValueError::new_err("hkl values must fit signed 32-bit"))?,
                i32::try_from(row[1])
                    .map_err(|_| PyValueError::new_err("hkl values must fit signed 32-bit"))?,
                i32::try_from(row[2])
                    .map_err(|_| PyValueError::new_err("hkl values must fit signed 32-bit"))?,
            ])
        })
        .collect()
}

fn multiplicity_rows(array: &PyReadonlyArray1<'_, i64>) -> PyResult<Vec<usize>> {
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err("multiplicity must be a contiguous array"))?
        .iter()
        .map(|&value| {
            usize::try_from(value)
                .map_err(|_| PyValueError::new_err("multiplicity must be non-negative"))
        })
        .collect()
}

fn symmetry_operations(
    rotations_flat: &PyReadonlyArray1<'_, i64>,
    translation_numerators_flat: &PyReadonlyArray1<'_, i64>,
    translation_denominators_flat: &PyReadonlyArray1<'_, i64>,
) -> PyResult<Vec<SymmetryOperation>> {
    let rotations = rotations_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("symmetry rotations must be contiguous"))?;
    let numerators = translation_numerators_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("translation numerators must be contiguous"))?;
    let denominators = translation_denominators_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("translation denominators must be contiguous"))?;
    if rotations.len() % 9 != 0 {
        return Err(PyValueError::new_err(
            "flattened symmetry rotation length must be divisible by nine",
        ));
    }
    let operation_count = rotations.len() / 9;
    if numerators.len() != 3 * operation_count || denominators.len() != 3 * operation_count {
        return Err(PyValueError::new_err(
            "symmetry translations must have three entries per operation",
        ));
    }
    (0..operation_count)
        .map(|operation| {
            let mut rotation = [[0_i32; 3]; 3];
            for (index, value) in rotation.iter_mut().flatten().enumerate() {
                *value = i32::try_from(rotations[9 * operation + index]).map_err(|_| {
                    PyValueError::new_err("symmetry rotations must fit signed 32-bit")
                })?;
            }
            let mut translation = [Rational::zero(); 3];
            for (component, value) in translation.iter_mut().enumerate() {
                let index = 3 * operation + component;
                *value = Rational::new(numerators[index], denominators[index])
                    .map_err(|error| PyValueError::new_err(error.to_string()))?;
            }
            SymmetryOperation::new(rotation, translation)
                .map_err(|error| PyValueError::new_err(error.to_string()))
        })
        .collect()
}

fn reflection_range(kind: &str, parameters: &[f64]) -> PyResult<ReflectionRange> {
    match (kind, parameters) {
        ("d", [min_angstrom, max_angstrom]) => Ok(ReflectionRange::DSpacing {
            min_angstrom: *min_angstrom,
            max_angstrom: *max_angstrom,
        }),
        ("q", [min_inverse_angstrom, max_inverse_angstrom]) => {
            Ok(ReflectionRange::ScatteringVector {
                min_inverse_angstrom: *min_inverse_angstrom,
                max_inverse_angstrom: *max_inverse_angstrom,
            })
        }
        ("cw", [min_deg, max_deg, wavelength_angstrom]) => Ok(ReflectionRange::CwTwoTheta {
            min_deg: *min_deg,
            max_deg: *max_deg,
            wavelength_angstrom: *wavelength_angstrom,
        }),
        (
            "tof",
            [
                min_us,
                max_us,
                search_min_d_angstrom,
                search_max_d_angstrom,
                zero_us,
                difc_us_per_angstrom,
                difa_us_per_angstrom2,
                difb_us_angstrom,
            ],
        ) => Ok(ReflectionRange::Tof {
            min_us: *min_us,
            max_us: *max_us,
            search_min_d_angstrom: *search_min_d_angstrom,
            search_max_d_angstrom: *search_max_d_angstrom,
            zero_us: *zero_us,
            difc_us_per_angstrom: *difc_us_per_angstrom,
            difa_us_per_angstrom2: *difa_us_per_angstrom2,
            difb_us_angstrom: *difb_us_angstrom,
        }),
        _ => Err(PyValueError::new_err(
            "range must be d(2), q(2), cw(3), or tof(8) parameters",
        )),
    }
}

fn xyz_rows(array: &PyReadonlyArray1<'_, f64>) -> PyResult<Vec<[f64; 3]>> {
    let values = contiguous_slice(array, "fractional_xyz")?;
    if values.len() % 3 != 0 {
        return Err(PyValueError::new_err(
            "flattened fractional_xyz length must be divisible by three",
        ));
    }
    Ok(values
        .chunks_exact(3)
        .map(|row| [row[0], row[1], row[2]])
        .collect())
}

fn matrix3_to_numpy(py: Python<'_>, matrix: [[f64; 3]; 3]) -> PyResult<Bound<'_, PyArray2<f64>>> {
    Array2::from_shape_vec((3, 3), matrix.into_iter().flatten().collect())
        .map_err(|error| PyValueError::new_err(error.to_string()))
        .map(|array| array.into_pyarray(py))
}

fn derivative_matrix(
    py: Python<'_>,
    parameter_count: usize,
    reflection_count: usize,
    values: Vec<f64>,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    Array2::from_shape_vec((parameter_count, reflection_count), values)
        .map_err(|error| PyValueError::new_err(error.to_string()))
        .map(|array| array.into_pyarray(py))
}

fn scattering_to_numpy(py: Python<'_>, values: ScatteringBatch) -> PyResult<ScatteringArrays<'_>> {
    let shape = (values.reflection_count, values.site_count);
    let matrix = |data| {
        Array2::from_shape_vec(shape, data)
            .map_err(|error| PyValueError::new_err(error.to_string()))
            .map(|array| array.into_pyarray(py))
    };
    Ok((
        matrix(values.real)?,
        matrix(values.imag)?,
        matrix(values.d_real_d_s)?,
        matrix(values.d_imag_d_s)?,
    ))
}

fn structure_factor_dense_to_numpy(
    py: Python<'_>,
    result: StructureFactorDenseResult,
) -> PyResult<StructureFactorDenseArrays<'_>> {
    let reflection_count = result.values.f_real.len();
    let parameter_count = result.layout.parameter_count();
    Ok((
        result.values.f_real.into_pyarray(py),
        result.values.f_imag.into_pyarray(py),
        result.values.f_squared.into_pyarray(py),
        result.values.intensity.into_pyarray(py),
        result.values.q_squared_inverse_angstrom2.into_pyarray(py),
        result.values.s_inverse_angstrom.into_pyarray(py),
        derivative_matrix(py, parameter_count, reflection_count, result.d_f_real)?,
        derivative_matrix(py, parameter_count, reflection_count, result.d_f_imag)?,
        derivative_matrix(py, parameter_count, reflection_count, result.d_intensity)?,
    ))
}

fn structure_factor_values_to_numpy(
    py: Python<'_>,
    values: StructureFactorValues,
) -> StructureFactorValueArrays<'_> {
    (
        values.f_real.into_pyarray(py),
        values.f_imag.into_pyarray(py),
        values.f_squared.into_pyarray(py),
        values.intensity.into_pyarray(py),
        values.q_squared_inverse_angstrom2.into_pyarray(py),
        values.s_inverse_angstrom.into_pyarray(py),
    )
}

fn parse_built_in_scattering_model(model: &str) -> PyResult<BuiltInScatteringModel> {
    match model {
        "xray_non_resonant" => Ok(BuiltInScatteringModel::XrayNonResonant),
        "neutron_nuclear" => Ok(BuiltInScatteringModel::NeutronNuclear),
        _ => Err(PyValueError::new_err("unknown built-in scattering model")),
    }
}

fn parse_correction_model(
    model: &str,
    wavelength_angstrom: Option<f64>,
) -> PyResult<IntegratedIntensityCorrectionModel> {
    let selected = match (model, wavelength_angstrom) {
        ("neutral", None) => IntegratedIntensityCorrectionModel::Neutral,
        ("bragg_brentano_unpolarized_lp", Some(wavelength_angstrom)) => {
            IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp {
                wavelength_angstrom,
            }
        }
        ("neutral", Some(_)) => {
            return Err(PyValueError::new_err(
                "neutral correction does not accept a wavelength",
            ));
        }
        ("bragg_brentano_unpolarized_lp", None) => {
            return Err(PyValueError::new_err(
                "Bragg-Brentano LP correction requires a wavelength",
            ));
        }
        _ => {
            return Err(PyValueError::new_err(
                "unknown integrated-intensity correction model",
            ));
        }
    };
    Ok(selected)
}

/// Evaluate one explicit integrated-intensity correction model.
#[pyfunction]
fn integrated_intensity_correction<'py>(
    py: Python<'py>,
    q_squared_inverse_angstrom2: PyReadonlyArray1<'py, f64>,
    model: &str,
    wavelength_angstrom: Option<f64>,
) -> PyResult<CorrectionArrays<'py>> {
    let selected = parse_correction_model(model, wavelength_angstrom)?;
    let result = selected
        .evaluate(contiguous_slice(
            &q_squared_inverse_angstrom2,
            "q_squared_inverse_angstrom2",
        )?)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        result.values.into_pyarray(py),
        result.d_values_d_q_squared.into_pyarray(py),
    ))
}

/// Return generated-table provenance without importing any source project.
#[pyfunction]
fn scattering_table_provenance() -> TableProvenanceRecords {
    let record = |value: rietveld_engine::crystallography::ScatteringTableProvenance| {
        (
            value.name,
            value.upstream_commit,
            value.source_sha256,
            value.table_fnv64,
            value.row_count,
        )
    };
    (
        record(XRAY_TABLE_PROVENANCE),
        record(NEUTRON_TABLE_PROVENANCE),
    )
}

/// Return exact X-ray state metadata when a source key exists.
#[pyfunction]
fn xray_scattering_species_metadata(key: &str) -> Option<(&'static str, u8)> {
    xray_species_metadata(key).map(|value| (value.key, value.atomic_number))
}

/// Return exact neutron identity metadata when a source key exists.
#[pyfunction]
fn neutron_scattering_species_metadata(key: &str) -> Option<NeutronMetadataRecord> {
    neutron_species_metadata(key).map(|value| {
        (
            value.key,
            value.atomic_number,
            value.isotope,
            value.b_c_fm,
            value.uncertainty_fm,
            value.energy_dependent,
            value.derived_alias_of,
        )
    })
}

/// Native Python module.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativePreparedXrayScattering>()?;
    module.add_class::<NativePreparedNeutronScattering>()?;
    module.add_class::<NativePreparedReflectionGenerator>()?;
    module.add_class::<NativeStructuralPhase>()?;
    module.add_function(wrap_pyfunction!(unit_cell_geometry, module)?)?;
    module.add_function(wrap_pyfunction!(unit_cell_d_spacings, module)?)?;
    module.add_function(wrap_pyfunction!(p1_structure_factors_dense, module)?)?;
    module.add_function(wrap_pyfunction!(p1_structure_factors_jvp, module)?)?;
    module.add_function(wrap_pyfunction!(p1_structure_factors_vjp, module)?)?;
    module.add_function(wrap_pyfunction!(scattering_table_provenance, module)?)?;
    module.add_function(wrap_pyfunction!(integrated_intensity_correction, module)?)?;
    module.add_function(wrap_pyfunction!(xray_scattering_species_metadata, module)?)?;
    module.add_function(wrap_pyfunction!(
        neutron_scattering_species_metadata,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(profile, module)?)?;
    module.add_function(wrap_pyfunction!(tch_shape_from_fwhm, module)?)?;
    module.add_function(wrap_pyfunction!(profile_tch, module)?)?;
    module.add_function(wrap_pyfunction!(profile_fcj, module)?)?;
    module.add_function(wrap_pyfunction!(profile_tof, module)?)?;
    module.add_function(wrap_pyfunction!(smooth_bruckner, module)?)?;
    module.add_function(wrap_pyfunction!(accumulate, module)?)?;
    module.add_function(wrap_pyfunction!(accumulate_tch, module)?)?;
    module.add_function(wrap_pyfunction!(accumulate_values, module)?)?;
    module.add_function(wrap_pyfunction!(cw_profile_parameters, module)?)?;
    module.add_function(wrap_pyfunction!(tof_profile_parameters, module)?)?;
    module.add_function(wrap_pyfunction!(accumulate_tof, module)?)?;
    module.add_function(wrap_pyfunction!(accumulate_cw, module)?)?;
    module.add_function(wrap_pyfunction!(accumulate_cw_contributions, module)?)?;
    module.add_function(wrap_pyfunction!(accumulate_cw_fcj, module)?)?;
    module.add_function(wrap_pyfunction!(accumulate_cw_components, module)?)?;
    module.add("PARAMETER_ORDER", ("intensity", "position", "fwhm", "eta"))?;
    module.add(
        "TCH_PARAMETER_ORDER",
        ("intensity", "position", "gaussian_fwhm", "lorentzian_fwhm"),
    )?;
    module.add("CW_LOCAL_PARAMETER_ORDER", ("intensity", "position"))?;
    module.add("TOF_LOCAL_PARAMETER_ORDER", ("intensity", "d_spacing"))?;
    module.add(
        "TOF_GLOBAL_PARAMETER_ORDER",
        PyTuple::new(
            module.py(),
            [
                "zero", "difc", "difa", "difb", "alpha", "beta0", "beta1", "betaq", "sigma0",
                "sigma1", "sigma2", "sigmaq", "x", "y", "z",
            ],
        )?,
    )?;
    module.add("CW_GLOBAL_PARAMETER_ORDER", ("u", "v", "w", "x", "y"))?;
    module.add(
        "CW_FCJ_GLOBAL_PARAMETER_ORDER",
        (
            "u",
            "v",
            "w",
            "x",
            "y",
            "sample_over_radius",
            "detector_over_radius",
        ),
    )?;
    module.add(
        "BUILD_MODE",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;
    Ok(())
}
