//! Python bindings for the `PhaseSmith` numerical core.

#![allow(clippy::needless_pass_by_value)] // PyO3 extracts owned argument guards.

use std::path::PathBuf;
use std::sync::Arc;

use npy::ndarray::Array2;
use npy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use phasesmith_core::{
    Accumulation, ConstantWavelengthInstrument, CwContributionArrays, CwContributionsView, CwError,
    CwProfileParameters, CwReflectionBatchView, FcjGeometry, FcjProfile, GridView,
    OwnedCwContributionArrays, OwnedCwContributions, PeakBatchView, SupportPolicy,
    TOF_GLOBAL_PARAMETER_NAMES, TchPeakBatchView, TchShape, TchWidths, TofError, TofInstrument,
    TofProfile, TofProfileParameters, WavelengthComponentsView, accumulate_batch,
    accumulate_cw_batch, accumulate_cw_components_batch, accumulate_cw_contributions_batch,
    accumulate_cw_fcj_batch, accumulate_cw_fcj_components_batch,
    accumulate_cw_fcj_contributions_batch, accumulate_tch_batch, accumulate_tof_batch_with_context,
    accumulate_values_batch, smooth_bruckner as native_smooth_bruckner, symmetric_pseudo_voigt,
};
use phasesmith_engine::crystallography::{
    CellError, IntegratedIntensityCorrectionModel, NEUTRON_TABLE_PROVENANCE, P1BatchView,
    PreparedNeutronScattering, PreparedReflectionGenerator, PreparedXrayScattering, Rational,
    ReflectionRange, ScatteringBatch, SpaceGroup, StructureFactorBatchView,
    StructureFactorDenseResult, StructureFactorValues, SymmetryOperation, UnitCell,
    XRAY_TABLE_PROVENANCE, calculate_p1_dense, calculate_p1_intensity_vjp, calculate_p1_jvp,
    calculate_structure_factor_dense, calculate_structure_factor_values, neutron_species_metadata,
    xray_species_metadata,
};
use phasesmith_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, PreparedStructuralModel,
    PreparedStructuralModelInputView, PreparedStructuralMultiphase,
    PreparedStructuralPatternInputView, PreparedStructuralPhase, PreparedStructuralSpectrum,
    PreparedStructuralSpectrumInputView, StructuralMultiphaseResult, StructuralPatternDenseResult,
    StructuralPatternJvpResult, StructuralPatternResult, StructuralPatternVjpResult,
    StructuralPhaseDefinition,
};
use phasesmith_execution::ExecutionPolicy as NativeExecutionPolicyModel;
use phasesmith_io::{
    CifDiagnosticSeverity, CifIoError, CifReadLimits as NativeCifReadLimits,
    CifReadResult as NativeCifReadResult, DisplacementConvention,
    GsasTofInstrumentData as NativeGsasTofInstrumentData, GsasTofInstrumentIoError,
    GsasTofInstrumentReadLimits as NativeGsasTofInstrumentReadLimits, NATIVE_CIF_BACKEND,
    NATIVE_CIF_BACKEND_VERSION, PowderData as NativePowderData, PowderFormat as NativePowderFormat,
    PowderIoError, PowderReadLimits as NativePowderReadLimits,
    SpaceGroupInfo as NativeSpaceGroupInfo, TofPowderData as NativeTofPowderData,
    TofPowderFormat as NativeTofPowderFormat, parse_cif_text as parse_native_cif_text,
    parse_gsas_tof_instrument_text as parse_native_gsas_tof_instrument_text,
    parse_powder_text as parse_native_powder_text,
    parse_tof_powder_text_as as parse_native_tof_powder_text,
    read_gsas_tof_instrument_file as read_native_gsas_tof_instrument_file,
    read_powder_file as read_native_powder_file,
    read_tof_powder_file_as as read_native_tof_powder_file,
    space_group_by_number as native_space_group_by_number,
    space_group_by_symbol as native_space_group_by_symbol,
};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

mod profile_estimation;
mod rietveld;
mod tof_lebail;
mod tof_multibank;

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
type StructuralPatternDenseArrays<'py> = (StructuralPatternArrays<'py>, Bound<'py, PyArray2<f64>>);

type PowderDataArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Option<Bound<'py, PyArray1<f64>>>,
    &'static str,
    Option<String>,
    Option<usize>,
    Option<Bound<'py, PyArray1<bool>>>,
);

type TofPowderDataArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Option<Bound<'py, PyArray1<f64>>>,
    &'static str,
    Option<String>,
    Option<usize>,
    bool,
    Option<Bound<'py, PyArray1<bool>>>,
);

type GsasTofInstrumentRecord = (Vec<f64>, usize, usize, Option<String>);

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

/// Persistent native execution policy shared by prepared Python operations.
#[pyclass(name = "_ExecutionPolicy", frozen)]
struct NativeExecutionPolicy {
    policy: NativeExecutionPolicyModel,
}

#[pymethods]
impl NativeExecutionPolicy {
    #[new]
    fn new(threads: Option<usize>, minimum_parallel_tasks: usize) -> PyResult<Self> {
        Ok(Self {
            policy: NativeExecutionPolicyModel::new(threads, minimum_parallel_tasks)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[getter]
    fn resolved_budget(&self) -> usize {
        self.policy.resolved_budget()
    }

    fn worker_count(&self, task_count: usize) -> usize {
        self.policy.worker_count(task_count)
    }
}

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
        let result = py
            .detach(|| self.model.evaluate(values))
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
        let result = py
            .detach(|| self.model.evaluate(values))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        scattering_to_numpy(py, result)
    }
}

/// Cached native group topology and bounded reflection generator.
#[pyclass(name = "_PreparedReflectionGenerator")]
pub(crate) struct NativePreparedReflectionGenerator {
    generator: PreparedReflectionGenerator,
}

impl NativePreparedReflectionGenerator {
    pub(crate) fn space_group(&self) -> &SpaceGroup {
        self.generator.space_group()
    }
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
        let expanded = py
            .detach(|| self.generator.space_group().expand_sites(&xyz, tolerance))
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
        anisotropic_mask: PyReadonlyArray1<'py, bool>,
        u_aniso_cif_angstrom2_flat: PyReadonlyArray1<'py, f64>,
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
        let tensors = tensor_rows(&u_aniso_cif_angstrom2_flat)?;
        let cell = crystallographic_cell(
            a_angstrom, b_angstrom, c_angstrom, alpha_deg, beta_deg, gamma_deg,
        );
        let batch = StructureFactorBatchView {
            hkl: &hkl,
            multiplicity: &multiplicity,
            fractional_xyz: &xyz,
            occupancy: contiguous_slice(&occupancy, "occupancy")?,
            u_iso_angstrom2: contiguous_slice(&u_iso_angstrom2, "u_iso_angstrom2")?,
            anisotropic_mask: bool_slice(&anisotropic_mask, "anisotropic_mask")?,
            u_aniso_cif_angstrom2: &tensors,
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
        };
        let result = py
            .detach(|| calculate_structure_factor_values(cell, self.generator.space_group(), batch))
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
        anisotropic_mask: PyReadonlyArray1<'py, bool>,
        u_aniso_cif_angstrom2_flat: PyReadonlyArray1<'py, f64>,
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
        let tensors = tensor_rows(&u_aniso_cif_angstrom2_flat)?;
        let cell = crystallographic_cell(
            a_angstrom, b_angstrom, c_angstrom, alpha_deg, beta_deg, gamma_deg,
        );
        let batch = StructureFactorBatchView {
            hkl: &hkl,
            multiplicity: &multiplicity,
            fractional_xyz: &xyz,
            occupancy: contiguous_slice(&occupancy, "occupancy")?,
            u_iso_angstrom2: contiguous_slice(&u_iso_angstrom2, "u_iso_angstrom2")?,
            anisotropic_mask: bool_slice(&anisotropic_mask, "anisotropic_mask")?,
            u_aniso_cif_angstrom2: &tensors,
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
        };
        let result = py
            .detach(|| calculate_structure_factor_dense(cell, self.generator.space_group(), batch))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        structure_factor_dense_to_numpy(py, result)
    }

    fn systematic_absences<'py>(
        &self,
        py: Python<'py>,
        hkl_flat: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let hkl = hkl_rows(&hkl_flat)?;
        let values = py
            .detach(|| {
                hkl.into_iter()
                    .map(|reflection| {
                        self.generator
                            .space_group()
                            .is_systematically_absent(reflection)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(values.into_pyarray(py))
    }

    fn reflection_families<'py>(
        &self,
        py: Python<'py>,
        hkl_flat: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<ReflectionFamilyArrays<'py>> {
        let hkl = hkl_rows(&hkl_flat)?;
        let families = py
            .detach(|| {
                hkl.into_iter()
                    .map(|reflection| {
                        self.generator
                            .space_group()
                            .reflection_family(reflection, self.generator.merge_friedel())
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let mut ids = Vec::with_capacity(families.len());
        let mut canonical = Vec::with_capacity(3 * families.len());
        let mut multiplicity = Vec::with_capacity(families.len());
        for family in families {
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
        let reflections = py
            .detach(|| {
                self.generator.generate(
                    crystallographic_cell(
                        a_angstrom, b_angstrom, c_angstrom, alpha_deg, beta_deg, gamma_deg,
                    ),
                    range,
                )
            })
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

#[derive(Clone)]
struct NativeComponentContributions {
    components: Arc<[OwnedCwContributions]>,
}

impl NativeComponentContributions {
    fn new(
        component_count: usize,
        reflection_count: usize,
        parameter_count: usize,
        arrays: OwnedCwContributionArrays,
    ) -> PyResult<Self> {
        let reflection_stride = reflection_count;
        let parameter_stride = reflection_count
            .checked_mul(parameter_count)
            .ok_or_else(|| PyValueError::new_err("component contribution size overflow"))?;
        let reflection_values = component_count
            .checked_mul(reflection_stride)
            .ok_or_else(|| PyValueError::new_err("component contribution size overflow"))?;
        let parameter_values = component_count
            .checked_mul(parameter_stride)
            .ok_or_else(|| PyValueError::new_err("component contribution size overflow"))?;
        for (name, values) in [
            ("gaussian_variance_deg2", &arrays.gaussian_variance_deg2),
            ("lorentzian_fwhm_deg", &arrays.lorentzian_fwhm_deg),
            ("intensity_multiplier", &arrays.intensity_multiplier),
            (
                "d_gaussian_variance_d_position",
                &arrays.d_gaussian_variance_d_position,
            ),
            (
                "d_lorentzian_fwhm_d_position",
                &arrays.d_lorentzian_fwhm_d_position,
            ),
            (
                "d_intensity_multiplier_d_position",
                &arrays.d_intensity_multiplier_d_position,
            ),
        ] {
            if values.len() != reflection_values {
                return Err(PyValueError::new_err(format!(
                    "{name} must contain {reflection_values} flattened component values"
                )));
            }
        }
        for (name, values) in [
            (
                "d_gaussian_variance_d_parameters",
                &arrays.d_gaussian_variance_d_parameters,
            ),
            (
                "d_lorentzian_fwhm_d_parameters",
                &arrays.d_lorentzian_fwhm_d_parameters,
            ),
            (
                "d_intensity_multiplier_d_parameters",
                &arrays.d_intensity_multiplier_d_parameters,
            ),
        ] {
            if values.len() != parameter_values {
                return Err(PyValueError::new_err(format!(
                    "{name} must contain {parameter_values} flattened component values"
                )));
            }
        }
        let components = (0..component_count)
            .map(|component| {
                let reflection_begin = component * reflection_stride;
                let reflection_end = reflection_begin + reflection_stride;
                let parameter_begin = component * parameter_stride;
                let parameter_end = parameter_begin + parameter_stride;
                OwnedCwContributions::new(
                    reflection_count,
                    parameter_count,
                    OwnedCwContributionArrays {
                        gaussian_variance_deg2: arrays.gaussian_variance_deg2
                            [reflection_begin..reflection_end]
                            .to_vec(),
                        lorentzian_fwhm_deg: arrays.lorentzian_fwhm_deg
                            [reflection_begin..reflection_end]
                            .to_vec(),
                        intensity_multiplier: arrays.intensity_multiplier
                            [reflection_begin..reflection_end]
                            .to_vec(),
                        d_gaussian_variance_d_position: arrays.d_gaussian_variance_d_position
                            [reflection_begin..reflection_end]
                            .to_vec(),
                        d_lorentzian_fwhm_d_position: arrays.d_lorentzian_fwhm_d_position
                            [reflection_begin..reflection_end]
                            .to_vec(),
                        d_intensity_multiplier_d_position: arrays.d_intensity_multiplier_d_position
                            [reflection_begin..reflection_end]
                            .to_vec(),
                        d_gaussian_variance_d_parameters: arrays.d_gaussian_variance_d_parameters
                            [parameter_begin..parameter_end]
                            .to_vec(),
                        d_lorentzian_fwhm_d_parameters: arrays.d_lorentzian_fwhm_d_parameters
                            [parameter_begin..parameter_end]
                            .to_vec(),
                        d_intensity_multiplier_d_parameters: arrays
                            .d_intensity_multiplier_d_parameters
                            [parameter_begin..parameter_end]
                            .to_vec(),
                    },
                )
                .map_err(|error| PyValueError::new_err(error.to_string()))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            components: components.into(),
        })
    }

    fn views(&self) -> Vec<CwContributionsView<'_>> {
        self.components
            .iter()
            .map(OwnedCwContributions::as_view)
            .collect()
    }
}

fn copied_exact_array(
    array: &PyReadonlyArray1<'_, f64>,
    name: &'static str,
    expected: usize,
) -> PyResult<Vec<f64>> {
    let values = contiguous_slice(array, name)?;
    if values.len() != expected {
        return Err(PyValueError::new_err(format!(
            "{name} must contain {expected} flattened values"
        )));
    }
    Ok(values.to_vec())
}

/// Immutable native structural phase used by values and derivative products.
#[pyclass(name = "_StructuralPhase")]
struct NativeStructuralPhase {
    phase: PreparedStructuralPhase,
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
        anisotropic_mask: PyReadonlyArray1<'_, bool>,
        u_aniso_cif_angstrom2_flat: PyReadonlyArray1<'_, f64>,
        scattering_species: Vec<String>,
        scattering_real_offset: PyReadonlyArray1<'_, f64>,
        scattering_imag_offset: PyReadonlyArray1<'_, f64>,
        a_angstrom: f64,
        b_angstrom: f64,
        c_angstrom: f64,
        alpha_deg: f64,
        beta_deg: f64,
        gamma_deg: f64,
        scale: f64,
        coordinate_tolerance: f64,
        execution: PyRef<'_, NativeExecutionPolicy>,
        scattering_model: &str,
        correction_model: &str,
        correction_wavelength_angstrom: Option<f64>,
        correction_polarization: Option<f64>,
    ) -> PyResult<Self> {
        let hkl = hkl_rows(&hkl_flat)?;
        let multiplicity = multiplicity_rows(&multiplicity)?;
        let fractional_xyz = xyz_rows(&fractional_xyz_flat)?;
        let occupancy = contiguous_slice(&occupancy, "occupancy")?.to_vec();
        let u_iso_angstrom2 = contiguous_slice(&u_iso_angstrom2, "u_iso_angstrom2")?.to_vec();
        let anisotropic_mask = bool_slice(&anisotropic_mask, "anisotropic_mask")?.to_vec();
        let u_aniso_cif_angstrom2 = tensor_rows(&u_aniso_cif_angstrom2_flat)?;
        let scattering_real_offset =
            contiguous_slice(&scattering_real_offset, "scattering_real_offset")?.to_vec();
        let scattering_imag_offset =
            contiguous_slice(&scattering_imag_offset, "scattering_imag_offset")?.to_vec();
        let scattering_model = parse_built_in_scattering_model(scattering_model)?;
        let execution_context = execution.policy.context().clone();
        Ok(Self {
            phase: PreparedStructuralPhase::new(
                StructuralPhaseDefinition {
                    cell: crystallographic_cell(
                        a_angstrom, b_angstrom, c_angstrom, alpha_deg, beta_deg, gamma_deg,
                    ),
                    space_group: generator.generator.space_group().clone(),
                    hkl,
                    multiplicity,
                    fractional_xyz,
                    occupancy,
                    u_iso_angstrom2,
                    anisotropic_mask,
                    u_aniso_cif_angstrom2,
                    scattering_species,
                    scattering_real_offset,
                    scattering_imag_offset,
                    scale,
                    coordinate_tolerance,
                    scattering_model,
                    correction_model: parse_correction_model(
                        correction_model,
                        correction_wavelength_angstrom,
                        correction_polarization,
                    )?,
                },
                execution_context,
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[getter]
    fn reflection_count(&self) -> usize {
        self.phase.reflection_count()
    }

    #[getter]
    fn structural_parameter_count(&self) -> usize {
        self.phase.structural_parameter_count()
    }

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn calculate<'py>(
        &self,
        py: Python<'py>,
        x_deg: PyReadonlyArray1<'py, f64>,
        wavelength_angstrom: f64,
        zero_shift_deg: f64,
        sample_displacement_mm: Option<f64>,
        displace_x_micrometre: Option<f64>,
        displace_y_micrometre: Option<f64>,
        goniometer_radius_mm: Option<f64>,
        fcj_sample_over_radius: Option<f64>,
        fcj_detector_over_radius: Option<f64>,
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
            self.phase.reflection_count(),
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
        let correction = position_correction(
            zero_shift_deg,
            sample_displacement_mm,
            displace_x_micrometre,
            displace_y_micrometre,
            goniometer_radius_mm,
        )?;
        let axial = axial_geometry(fcj_sample_over_radius, fcj_detector_over_radius)?;
        let input = PreparedStructuralPatternInputView {
            x_deg: x_deg_values,
            instrument: cw_instrument(
                wavelength_angstrom,
                u_deg2,
                v_deg2,
                w_deg2,
                x_width_deg,
                y_width_deg,
            ),
            position_correction: correction,
            axial_geometry: axial,
            contributions,
            support: SupportPolicy::FwhmMultiple(support_fwhm),
        };
        let result = py
            .detach(|| self.phase.calculate(&input))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        structural_pattern_to_numpy(py, result)
    }

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn linearize<'py>(
        &self,
        py: Python<'py>,
        x_deg: PyReadonlyArray1<'py, f64>,
        wavelength_angstrom: f64,
        zero_shift_deg: f64,
        sample_displacement_mm: Option<f64>,
        displace_x_micrometre: Option<f64>,
        displace_y_micrometre: Option<f64>,
        goniometer_radius_mm: Option<f64>,
        fcj_sample_over_radius: Option<f64>,
        fcj_detector_over_radius: Option<f64>,
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
    ) -> PyResult<StructuralPatternDenseArrays<'py>> {
        let x_deg_values = contiguous_slice(&x_deg, "x_deg")?;
        let contributions = Self::contribution_view(
            self.phase.reflection_count(),
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
        let correction = position_correction(
            zero_shift_deg,
            sample_displacement_mm,
            displace_x_micrometre,
            displace_y_micrometre,
            goniometer_radius_mm,
        )?;
        let axial = axial_geometry(fcj_sample_over_radius, fcj_detector_over_radius)?;
        let input = PreparedStructuralPatternInputView {
            x_deg: x_deg_values,
            instrument: cw_instrument(
                wavelength_angstrom,
                u_deg2,
                v_deg2,
                w_deg2,
                x_width_deg,
                y_width_deg,
            ),
            position_correction: correction,
            axial_geometry: axial,
            contributions,
            support: SupportPolicy::FwhmMultiple(support_fwhm),
        };
        let result = py
            .detach(|| self.phase.linearize(&input))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        structural_pattern_dense_to_numpy(py, result)
    }

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn jvp<'py>(
        &self,
        py: Python<'py>,
        tangent: PyReadonlyArray1<'py, f64>,
        x_deg: PyReadonlyArray1<'py, f64>,
        wavelength_angstrom: f64,
        zero_shift_deg: f64,
        sample_displacement_mm: Option<f64>,
        displace_x_micrometre: Option<f64>,
        displace_y_micrometre: Option<f64>,
        goniometer_radius_mm: Option<f64>,
        fcj_sample_over_radius: Option<f64>,
        fcj_detector_over_radius: Option<f64>,
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
            self.phase.reflection_count(),
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
        let correction = position_correction(
            zero_shift_deg,
            sample_displacement_mm,
            displace_x_micrometre,
            displace_y_micrometre,
            goniometer_radius_mm,
        )?;
        let axial = axial_geometry(fcj_sample_over_radius, fcj_detector_over_radius)?;
        let input = PreparedStructuralPatternInputView {
            x_deg: x_deg_values,
            instrument: cw_instrument(
                wavelength_angstrom,
                u_deg2,
                v_deg2,
                w_deg2,
                x_width_deg,
                y_width_deg,
            ),
            position_correction: correction,
            axial_geometry: axial,
            contributions,
            support: SupportPolicy::FwhmMultiple(support_fwhm),
        };
        let result = py
            .detach(|| self.phase.jvp(&input, tangent))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        structural_pattern_jvp_to_numpy(py, result)
    }

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn vjp<'py>(
        &self,
        py: Python<'py>,
        sample_weights: PyReadonlyArray1<'py, f64>,
        x_deg: PyReadonlyArray1<'py, f64>,
        wavelength_angstrom: f64,
        zero_shift_deg: f64,
        sample_displacement_mm: Option<f64>,
        displace_x_micrometre: Option<f64>,
        displace_y_micrometre: Option<f64>,
        goniometer_radius_mm: Option<f64>,
        fcj_sample_over_radius: Option<f64>,
        fcj_detector_over_radius: Option<f64>,
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
            self.phase.reflection_count(),
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
        let correction = position_correction(
            zero_shift_deg,
            sample_displacement_mm,
            displace_x_micrometre,
            displace_y_micrometre,
            goniometer_radius_mm,
        )?;
        let axial = axial_geometry(fcj_sample_over_radius, fcj_detector_over_radius)?;
        let input = PreparedStructuralPatternInputView {
            x_deg: x_deg_values,
            instrument: cw_instrument(
                wavelength_angstrom,
                u_deg2,
                v_deg2,
                w_deg2,
                x_width_deg,
                y_width_deg,
            ),
            position_correction: correction,
            axial_geometry: axial,
            contributions,
            support: SupportPolicy::FwhmMultiple(support_fwhm),
        };
        let result = py
            .detach(|| self.phase.vjp(&input, sample_weights))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        structural_pattern_vjp_to_numpy(py, result)
    }
}

/// Native fixed-wavelength structural spectrum with owned contribution arrays.
#[pyclass(name = "_StructuralSpectrum")]
struct NativeStructuralSpectrum {
    spectrum: PreparedStructuralSpectrum,
    contributions: NativeComponentContributions,
}

impl NativeStructuralSpectrum {
    fn with_input<R>(
        &self,
        x_deg: &[f64],
        instrument: ConstantWavelengthInstrument,
        position_correction: MonochromaticPositionCorrection,
        axial_geometry: Option<FcjGeometry>,
        support_fwhm: f64,
        operation: impl FnOnce(
            &PreparedStructuralSpectrum,
            &PreparedStructuralSpectrumInputView<'_>,
        ) -> Result<R, phasesmith_engine::StructuralSpectrumError>,
    ) -> PyResult<R> {
        let contributions = self.contributions.views();
        let input = PreparedStructuralSpectrumInputView {
            x_deg,
            instrument,
            axial_geometry,
            position_correction,
            contributions: &contributions,
            support: SupportPolicy::FwhmMultiple(support_fwhm),
        };
        operation(&self.spectrum, &input).map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

#[pymethods]
impl NativeStructuralSpectrum {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        base: PyRef<'_, NativeStructuralPhase>,
        wavelengths_angstrom: PyReadonlyArray1<'_, f64>,
        relative_intensities: PyReadonlyArray1<'_, f64>,
        gaussian_variance_deg2: PyReadonlyArray1<'_, f64>,
        lorentzian_fwhm_deg: PyReadonlyArray1<'_, f64>,
        intensity_multiplier: PyReadonlyArray1<'_, f64>,
        d_gaussian_variance_d_position: PyReadonlyArray1<'_, f64>,
        d_lorentzian_fwhm_d_position: PyReadonlyArray1<'_, f64>,
        d_intensity_multiplier_d_position: PyReadonlyArray1<'_, f64>,
        d_gaussian_variance_d_parameters: PyReadonlyArray1<'_, f64>,
        d_lorentzian_fwhm_d_parameters: PyReadonlyArray1<'_, f64>,
        d_intensity_multiplier_d_parameters: PyReadonlyArray1<'_, f64>,
        parameter_count: usize,
        execution: PyRef<'_, NativeExecutionPolicy>,
    ) -> PyResult<Self> {
        let wavelengths = contiguous_slice(&wavelengths_angstrom, "wavelengths_angstrom")?.to_vec();
        let relative_intensities =
            contiguous_slice(&relative_intensities, "relative_intensities")?.to_vec();
        let component_count = wavelengths.len();
        let reflection_count = base.phase.reflection_count();
        let reflection_values = component_count
            .checked_mul(reflection_count)
            .ok_or_else(|| PyValueError::new_err("component contribution size overflow"))?;
        let parameter_values = reflection_values
            .checked_mul(parameter_count)
            .ok_or_else(|| PyValueError::new_err("component contribution size overflow"))?;
        let copy_array = |array: &PyReadonlyArray1<'_, f64>,
                          name: &'static str,
                          expected: usize|
         -> PyResult<Vec<f64>> {
            let values = contiguous_slice(array, name)?;
            if values.len() != expected {
                return Err(PyValueError::new_err(format!(
                    "{name} must contain {expected} flattened component values"
                )));
            }
            Ok(values.to_vec())
        };
        let contributions = NativeComponentContributions::new(
            component_count,
            reflection_count,
            parameter_count,
            OwnedCwContributionArrays {
                gaussian_variance_deg2: copy_array(
                    &gaussian_variance_deg2,
                    "gaussian_variance_deg2",
                    reflection_values,
                )?,
                lorentzian_fwhm_deg: copy_array(
                    &lorentzian_fwhm_deg,
                    "lorentzian_fwhm_deg",
                    reflection_values,
                )?,
                intensity_multiplier: copy_array(
                    &intensity_multiplier,
                    "intensity_multiplier",
                    reflection_values,
                )?,
                d_gaussian_variance_d_position: copy_array(
                    &d_gaussian_variance_d_position,
                    "d_gaussian_variance_d_position",
                    reflection_values,
                )?,
                d_lorentzian_fwhm_d_position: copy_array(
                    &d_lorentzian_fwhm_d_position,
                    "d_lorentzian_fwhm_d_position",
                    reflection_values,
                )?,
                d_intensity_multiplier_d_position: copy_array(
                    &d_intensity_multiplier_d_position,
                    "d_intensity_multiplier_d_position",
                    reflection_values,
                )?,
                d_gaussian_variance_d_parameters: copy_array(
                    &d_gaussian_variance_d_parameters,
                    "d_gaussian_variance_d_parameters",
                    parameter_values,
                )?,
                d_lorentzian_fwhm_d_parameters: copy_array(
                    &d_lorentzian_fwhm_d_parameters,
                    "d_lorentzian_fwhm_d_parameters",
                    parameter_values,
                )?,
                d_intensity_multiplier_d_parameters: copy_array(
                    &d_intensity_multiplier_d_parameters,
                    "d_intensity_multiplier_d_parameters",
                    parameter_values,
                )?,
            },
        )?;
        Ok(Self {
            spectrum: PreparedStructuralSpectrum::new(
                base.phase.definition(),
                wavelengths,
                &relative_intensities,
                execution.policy.clone(),
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
            contributions,
        })
    }

    #[getter]
    fn component_count(&self) -> usize {
        self.spectrum.component_count()
    }

    #[getter]
    fn reflection_count(&self) -> usize {
        self.contributions
            .components
            .iter()
            .map(OwnedCwContributions::reflection_count)
            .sum()
    }

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn calculate<'py>(
        &self,
        py: Python<'py>,
        x_deg: PyReadonlyArray1<'py, f64>,
        zero_shift_deg: f64,
        sample_displacement_mm: Option<f64>,
        displace_x_micrometre: Option<f64>,
        displace_y_micrometre: Option<f64>,
        goniometer_radius_mm: Option<f64>,
        fcj_sample_over_radius: Option<f64>,
        fcj_detector_over_radius: Option<f64>,
        wavelength_angstrom: f64,
        u_deg2: f64,
        v_deg2: f64,
        w_deg2: f64,
        x_width_deg: f64,
        y_width_deg: f64,
        support_fwhm: f64,
    ) -> PyResult<StructuralPatternArrays<'py>> {
        let x = contiguous_slice(&x_deg, "x_deg")?;
        let correction = position_correction(
            zero_shift_deg,
            sample_displacement_mm,
            displace_x_micrometre,
            displace_y_micrometre,
            goniometer_radius_mm,
        )?;
        let axial = axial_geometry(fcj_sample_over_radius, fcj_detector_over_radius)?;
        let result = py.detach(|| {
            self.with_input(
                x,
                cw_instrument(
                    wavelength_angstrom,
                    u_deg2,
                    v_deg2,
                    w_deg2,
                    x_width_deg,
                    y_width_deg,
                ),
                correction,
                axial,
                support_fwhm,
                PreparedStructuralSpectrum::calculate,
            )
        })?;
        structural_pattern_to_numpy(py, result)
    }

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn linearize<'py>(
        &self,
        py: Python<'py>,
        x_deg: PyReadonlyArray1<'py, f64>,
        zero_shift_deg: f64,
        sample_displacement_mm: Option<f64>,
        displace_x_micrometre: Option<f64>,
        displace_y_micrometre: Option<f64>,
        goniometer_radius_mm: Option<f64>,
        fcj_sample_over_radius: Option<f64>,
        fcj_detector_over_radius: Option<f64>,
        wavelength_angstrom: f64,
        u_deg2: f64,
        v_deg2: f64,
        w_deg2: f64,
        x_width_deg: f64,
        y_width_deg: f64,
        support_fwhm: f64,
    ) -> PyResult<StructuralPatternDenseArrays<'py>> {
        let x = contiguous_slice(&x_deg, "x_deg")?;
        let correction = position_correction(
            zero_shift_deg,
            sample_displacement_mm,
            displace_x_micrometre,
            displace_y_micrometre,
            goniometer_radius_mm,
        )?;
        let axial = axial_geometry(fcj_sample_over_radius, fcj_detector_over_radius)?;
        let result = py.detach(|| {
            self.with_input(
                x,
                cw_instrument(
                    wavelength_angstrom,
                    u_deg2,
                    v_deg2,
                    w_deg2,
                    x_width_deg,
                    y_width_deg,
                ),
                correction,
                axial,
                support_fwhm,
                PreparedStructuralSpectrum::linearize,
            )
        })?;
        structural_pattern_dense_to_numpy(py, result)
    }

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn jvp<'py>(
        &self,
        py: Python<'py>,
        tangent: PyReadonlyArray1<'py, f64>,
        x_deg: PyReadonlyArray1<'py, f64>,
        zero_shift_deg: f64,
        sample_displacement_mm: Option<f64>,
        displace_x_micrometre: Option<f64>,
        displace_y_micrometre: Option<f64>,
        goniometer_radius_mm: Option<f64>,
        fcj_sample_over_radius: Option<f64>,
        fcj_detector_over_radius: Option<f64>,
        wavelength_angstrom: f64,
        u_deg2: f64,
        v_deg2: f64,
        w_deg2: f64,
        x_width_deg: f64,
        y_width_deg: f64,
        support_fwhm: f64,
    ) -> PyResult<StructuralPatternJvpArrays<'py>> {
        let tangent = contiguous_slice(&tangent, "tangent")?;
        let x = contiguous_slice(&x_deg, "x_deg")?;
        let correction = position_correction(
            zero_shift_deg,
            sample_displacement_mm,
            displace_x_micrometre,
            displace_y_micrometre,
            goniometer_radius_mm,
        )?;
        let axial = axial_geometry(fcj_sample_over_radius, fcj_detector_over_radius)?;
        let result = py.detach(|| {
            self.with_input(
                x,
                cw_instrument(
                    wavelength_angstrom,
                    u_deg2,
                    v_deg2,
                    w_deg2,
                    x_width_deg,
                    y_width_deg,
                ),
                correction,
                axial,
                support_fwhm,
                |spectrum, input| spectrum.jvp(input, tangent),
            )
        })?;
        structural_pattern_jvp_to_numpy(py, result)
    }

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn vjp<'py>(
        &self,
        py: Python<'py>,
        sample_weights: PyReadonlyArray1<'py, f64>,
        x_deg: PyReadonlyArray1<'py, f64>,
        zero_shift_deg: f64,
        sample_displacement_mm: Option<f64>,
        displace_x_micrometre: Option<f64>,
        displace_y_micrometre: Option<f64>,
        goniometer_radius_mm: Option<f64>,
        fcj_sample_over_radius: Option<f64>,
        fcj_detector_over_radius: Option<f64>,
        wavelength_angstrom: f64,
        u_deg2: f64,
        v_deg2: f64,
        w_deg2: f64,
        x_width_deg: f64,
        y_width_deg: f64,
        support_fwhm: f64,
    ) -> PyResult<StructuralPatternVjpArrays<'py>> {
        let sample_weights = contiguous_slice(&sample_weights, "sample_weights")?;
        let x = contiguous_slice(&x_deg, "x_deg")?;
        let correction = position_correction(
            zero_shift_deg,
            sample_displacement_mm,
            displace_x_micrometre,
            displace_y_micrometre,
            goniometer_radius_mm,
        )?;
        let axial = axial_geometry(fcj_sample_over_radius, fcj_detector_over_radius)?;
        let result = py.detach(|| {
            self.with_input(
                x,
                cw_instrument(
                    wavelength_angstrom,
                    u_deg2,
                    v_deg2,
                    w_deg2,
                    x_width_deg,
                    y_width_deg,
                ),
                correction,
                axial,
                support_fwhm,
                |spectrum, input| spectrum.vjp(input, sample_weights),
            )
        })?;
        structural_pattern_vjp_to_numpy(py, result)
    }
}

/// One native structural phase model plus its owned sample-physics inputs.
#[pyclass(name = "_PreparedStructuralModel")]
struct NativePreparedStructuralModel {
    model: PreparedStructuralModel,
    contributions: NativeComponentContributions,
}

#[pymethods]
impl NativePreparedStructuralModel {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn monochromatic(
        phase: PyRef<'_, NativeStructuralPhase>,
        gaussian_variance_deg2: PyReadonlyArray1<'_, f64>,
        lorentzian_fwhm_deg: PyReadonlyArray1<'_, f64>,
        intensity_multiplier: PyReadonlyArray1<'_, f64>,
        d_gaussian_variance_d_position: PyReadonlyArray1<'_, f64>,
        d_lorentzian_fwhm_d_position: PyReadonlyArray1<'_, f64>,
        d_intensity_multiplier_d_position: PyReadonlyArray1<'_, f64>,
        d_gaussian_variance_d_parameters: PyReadonlyArray1<'_, f64>,
        d_lorentzian_fwhm_d_parameters: PyReadonlyArray1<'_, f64>,
        d_intensity_multiplier_d_parameters: PyReadonlyArray1<'_, f64>,
        parameter_count: usize,
    ) -> PyResult<Self> {
        let reflection_count = phase.phase.reflection_count();
        let parameter_values = reflection_count
            .checked_mul(parameter_count)
            .ok_or_else(|| PyValueError::new_err("contribution size overflow"))?;
        let contributions = NativeComponentContributions::new(
            1,
            reflection_count,
            parameter_count,
            OwnedCwContributionArrays {
                gaussian_variance_deg2: copied_exact_array(
                    &gaussian_variance_deg2,
                    "gaussian_variance_deg2",
                    reflection_count,
                )?,
                lorentzian_fwhm_deg: copied_exact_array(
                    &lorentzian_fwhm_deg,
                    "lorentzian_fwhm_deg",
                    reflection_count,
                )?,
                intensity_multiplier: copied_exact_array(
                    &intensity_multiplier,
                    "intensity_multiplier",
                    reflection_count,
                )?,
                d_gaussian_variance_d_position: copied_exact_array(
                    &d_gaussian_variance_d_position,
                    "d_gaussian_variance_d_position",
                    reflection_count,
                )?,
                d_lorentzian_fwhm_d_position: copied_exact_array(
                    &d_lorentzian_fwhm_d_position,
                    "d_lorentzian_fwhm_d_position",
                    reflection_count,
                )?,
                d_intensity_multiplier_d_position: copied_exact_array(
                    &d_intensity_multiplier_d_position,
                    "d_intensity_multiplier_d_position",
                    reflection_count,
                )?,
                d_gaussian_variance_d_parameters: copied_exact_array(
                    &d_gaussian_variance_d_parameters,
                    "d_gaussian_variance_d_parameters",
                    parameter_values,
                )?,
                d_lorentzian_fwhm_d_parameters: copied_exact_array(
                    &d_lorentzian_fwhm_d_parameters,
                    "d_lorentzian_fwhm_d_parameters",
                    parameter_values,
                )?,
                d_intensity_multiplier_d_parameters: copied_exact_array(
                    &d_intensity_multiplier_d_parameters,
                    "d_intensity_multiplier_d_parameters",
                    parameter_values,
                )?,
            },
        )?;
        Ok(Self {
            model: PreparedStructuralModel::monochromatic(phase.phase.clone()),
            contributions,
        })
    }

    #[staticmethod]
    fn fixed_spectrum(spectrum: PyRef<'_, NativeStructuralSpectrum>) -> Self {
        Self {
            model: PreparedStructuralModel::fixed_spectrum(spectrum.spectrum.clone()),
            contributions: spectrum.contributions.clone(),
        }
    }
}

/// Native scheduler for a non-empty ordered list of structural phase models.
#[pyclass(name = "_StructuralMultiphase")]
struct NativeStructuralMultiphase {
    prepared: PreparedStructuralMultiphase,
    contributions: Vec<NativeComponentContributions>,
}

impl NativeStructuralMultiphase {
    fn with_inputs<R>(
        &self,
        x_deg: &[f64],
        instrument: ConstantWavelengthInstrument,
        position_correction: MonochromaticPositionCorrection,
        axial_geometry: Option<FcjGeometry>,
        support_fwhm: f64,
        operation: impl FnOnce(
            &PreparedStructuralMultiphase,
            &[PreparedStructuralModelInputView<'_>],
        ) -> Result<R, phasesmith_engine::StructuralMultiphaseError>,
    ) -> PyResult<R> {
        let contribution_views = self
            .contributions
            .iter()
            .map(NativeComponentContributions::views)
            .collect::<Vec<_>>();
        let inputs = contribution_views
            .iter()
            .map(|contributions| PreparedStructuralModelInputView {
                x_deg,
                instrument,
                axial_geometry,
                position_correction,
                contributions,
                support: SupportPolicy::FwhmMultiple(support_fwhm),
            })
            .collect::<Vec<_>>();
        operation(&self.prepared, &inputs).map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

#[pymethods]
impl NativeStructuralMultiphase {
    #[new]
    fn new(
        models: &Bound<'_, PyList>,
        execution: PyRef<'_, NativeExecutionPolicy>,
    ) -> PyResult<Self> {
        let mut prepared_models = Vec::with_capacity(models.len());
        let mut contributions = Vec::with_capacity(models.len());
        for item in models.iter() {
            let model = item.extract::<PyRef<'_, NativePreparedStructuralModel>>()?;
            prepared_models.push(model.model.clone());
            contributions.push(model.contributions.clone());
        }
        Ok(Self {
            prepared: PreparedStructuralMultiphase::new(prepared_models, execution.policy.clone())
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
            contributions,
        })
    }

    #[getter]
    fn phase_count(&self) -> usize {
        self.prepared.phase_count()
    }

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn calculate<'py>(
        &self,
        py: Python<'py>,
        x_deg: PyReadonlyArray1<'py, f64>,
        zero_shift_deg: f64,
        sample_displacement_mm: Option<f64>,
        displace_x_micrometre: Option<f64>,
        displace_y_micrometre: Option<f64>,
        goniometer_radius_mm: Option<f64>,
        fcj_sample_over_radius: Option<f64>,
        fcj_detector_over_radius: Option<f64>,
        wavelength_angstrom: f64,
        u_deg2: f64,
        v_deg2: f64,
        w_deg2: f64,
        x_width_deg: f64,
        y_width_deg: f64,
        support_fwhm: f64,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Vec<StructuralPatternArrays<'py>>)> {
        let x = contiguous_slice(&x_deg, "x_deg")?;
        let correction = position_correction(
            zero_shift_deg,
            sample_displacement_mm,
            displace_x_micrometre,
            displace_y_micrometre,
            goniometer_radius_mm,
        )?;
        let axial = axial_geometry(fcj_sample_over_radius, fcj_detector_over_radius)?;
        let result = py.detach(|| {
            self.with_inputs(
                x,
                cw_instrument(
                    wavelength_angstrom,
                    u_deg2,
                    v_deg2,
                    w_deg2,
                    x_width_deg,
                    y_width_deg,
                ),
                correction,
                axial,
                support_fwhm,
                PreparedStructuralMultiphase::calculate,
            )
        })?;
        multiphase_result_to_numpy(py, result)
    }

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn linearize<'py>(
        &self,
        py: Python<'py>,
        x_deg: PyReadonlyArray1<'py, f64>,
        zero_shift_deg: f64,
        sample_displacement_mm: Option<f64>,
        displace_x_micrometre: Option<f64>,
        displace_y_micrometre: Option<f64>,
        goniometer_radius_mm: Option<f64>,
        fcj_sample_over_radius: Option<f64>,
        fcj_detector_over_radius: Option<f64>,
        wavelength_angstrom: f64,
        u_deg2: f64,
        v_deg2: f64,
        w_deg2: f64,
        x_width_deg: f64,
        y_width_deg: f64,
        support_fwhm: f64,
    ) -> PyResult<Vec<StructuralPatternDenseArrays<'py>>> {
        let x = contiguous_slice(&x_deg, "x_deg")?;
        let correction = position_correction(
            zero_shift_deg,
            sample_displacement_mm,
            displace_x_micrometre,
            displace_y_micrometre,
            goniometer_radius_mm,
        )?;
        let axial = axial_geometry(fcj_sample_over_radius, fcj_detector_over_radius)?;
        let results = py.detach(|| {
            self.with_inputs(
                x,
                cw_instrument(
                    wavelength_angstrom,
                    u_deg2,
                    v_deg2,
                    w_deg2,
                    x_width_deg,
                    y_width_deg,
                ),
                correction,
                axial,
                support_fwhm,
                PreparedStructuralMultiphase::linearize,
            )
        })?;
        results
            .into_iter()
            .map(|result| structural_pattern_dense_to_numpy(py, result))
            .collect()
    }

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn jvp<'py>(
        &self,
        py: Python<'py>,
        tangent: PyReadonlyArray1<'py, f64>,
        x_deg: PyReadonlyArray1<'py, f64>,
        zero_shift_deg: f64,
        sample_displacement_mm: Option<f64>,
        displace_x_micrometre: Option<f64>,
        displace_y_micrometre: Option<f64>,
        goniometer_radius_mm: Option<f64>,
        fcj_sample_over_radius: Option<f64>,
        fcj_detector_over_radius: Option<f64>,
        wavelength_angstrom: f64,
        u_deg2: f64,
        v_deg2: f64,
        w_deg2: f64,
        x_width_deg: f64,
        y_width_deg: f64,
        support_fwhm: f64,
    ) -> PyResult<Vec<StructuralPatternJvpArrays<'py>>> {
        let tangent = contiguous_slice(&tangent, "tangent")?;
        let counts = self.prepared.structural_parameter_counts();
        let expected = counts.iter().sum::<usize>();
        if tangent.len() != expected {
            return Err(PyValueError::new_err(
                "flattened tangent must match all structural phase parameters",
            ));
        }
        let mut cursor = 0;
        let tangents = counts
            .iter()
            .map(|count| {
                let selected = &tangent[cursor..cursor + count];
                cursor += count;
                selected
            })
            .collect::<Vec<_>>();
        let x = contiguous_slice(&x_deg, "x_deg")?;
        let correction = position_correction(
            zero_shift_deg,
            sample_displacement_mm,
            displace_x_micrometre,
            displace_y_micrometre,
            goniometer_radius_mm,
        )?;
        let axial = axial_geometry(fcj_sample_over_radius, fcj_detector_over_radius)?;
        let results = py.detach(|| {
            self.with_inputs(
                x,
                cw_instrument(
                    wavelength_angstrom,
                    u_deg2,
                    v_deg2,
                    w_deg2,
                    x_width_deg,
                    y_width_deg,
                ),
                correction,
                axial,
                support_fwhm,
                |prepared, inputs| prepared.jvp(inputs, &tangents),
            )
        })?;
        results
            .into_iter()
            .map(|result| structural_pattern_jvp_to_numpy(py, result))
            .collect()
    }

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn vjp<'py>(
        &self,
        py: Python<'py>,
        sample_weights: PyReadonlyArray1<'py, f64>,
        x_deg: PyReadonlyArray1<'py, f64>,
        zero_shift_deg: f64,
        sample_displacement_mm: Option<f64>,
        displace_x_micrometre: Option<f64>,
        displace_y_micrometre: Option<f64>,
        goniometer_radius_mm: Option<f64>,
        fcj_sample_over_radius: Option<f64>,
        fcj_detector_over_radius: Option<f64>,
        wavelength_angstrom: f64,
        u_deg2: f64,
        v_deg2: f64,
        w_deg2: f64,
        x_width_deg: f64,
        y_width_deg: f64,
        support_fwhm: f64,
    ) -> PyResult<Vec<StructuralPatternVjpArrays<'py>>> {
        let weights = contiguous_slice(&sample_weights, "sample_weights")?;
        let x = contiguous_slice(&x_deg, "x_deg")?;
        let correction = position_correction(
            zero_shift_deg,
            sample_displacement_mm,
            displace_x_micrometre,
            displace_y_micrometre,
            goniometer_radius_mm,
        )?;
        let axial = axial_geometry(fcj_sample_over_radius, fcj_detector_over_radius)?;
        let results = py.detach(|| {
            self.with_inputs(
                x,
                cw_instrument(
                    wavelength_angstrom,
                    u_deg2,
                    v_deg2,
                    w_deg2,
                    x_width_deg,
                    y_width_deg,
                ),
                correction,
                axial,
                support_fwhm,
                |prepared, inputs| prepared.vjp(inputs, weights),
            )
        })?;
        results
            .into_iter()
            .map(|result| structural_pattern_vjp_to_numpy(py, result))
            .collect()
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
    let (spacings, derivatives) = py
        .detach(|| {
            let mut spacings = Vec::with_capacity(hkl.len());
            let mut derivatives = Vec::with_capacity(hkl.len() * 6);
            for reflection in hkl {
                let (spacing, derivative) = geometry.d_spacing_and_derivatives(reflection)?;
                spacings.push(spacing);
                derivatives.extend_from_slice(&derivative);
            }
            Ok((spacings, derivatives))
        })
        .map_err(|error: CellError| PyValueError::new_err(error.to_string()))?;
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
    let cell = crystallographic_cell(
        a_angstrom, b_angstrom, c_angstrom, alpha_deg, beta_deg, gamma_deg,
    );
    let batch = P1BatchView {
        hkl: &hkl,
        fractional_xyz: &xyz,
        occupancy,
        u_iso_angstrom2: u_iso,
        scattering_real,
        scattering_imag,
        scale,
    };
    let result = py
        .detach(|| calculate_p1_dense(cell, batch))
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
    let cell = crystallographic_cell(
        a_angstrom, b_angstrom, c_angstrom, alpha_deg, beta_deg, gamma_deg,
    );
    let batch = P1BatchView {
        hkl: &hkl,
        fractional_xyz: &xyz,
        occupancy,
        u_iso_angstrom2: u_iso,
        scattering_real,
        scattering_imag,
        scale,
    };
    let result = py
        .detach(|| calculate_p1_jvp(cell, batch, tangent))
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
    let cell = crystallographic_cell(
        a_angstrom, b_angstrom, c_angstrom, alpha_deg, beta_deg, gamma_deg,
    );
    let batch = P1BatchView {
        hkl: &hkl,
        fractional_xyz: &xyz,
        occupancy,
        u_iso_angstrom2: u_iso,
        scattering_real,
        scattering_imag,
        scale,
    };
    let result = py
        .detach(|| calculate_p1_intensity_vjp(cell, batch, weights))
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

    let (value, d_delta, d_fwhm, d_eta) = py.detach(|| {
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
        (value, d_delta, d_fwhm, d_eta)
    });
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
    let (value, d_delta, d_gaussian, d_lorentzian) = py.detach(|| {
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
        (value, d_delta, d_gaussian, d_lorentzian)
    });
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
    let (value, d_position, d_gaussian, d_lorentzian, d_sample, d_detector) = py.detach(|| {
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
        (
            value,
            d_position,
            d_gaussian,
            d_lorentzian,
            d_sample,
            d_detector,
        )
    });
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
    let (value, d_position, d_alpha, d_beta, d_gaussian, d_lorentzian) = py.detach(|| {
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
        (value, d_position, d_alpha, d_beta, d_gaussian, d_lorentzian)
    });
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
    let accumulation = py
        .detach(|| accumulate_batch(grid, peaks, SupportPolicy::FwhmMultiple(support_fwhm)))
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
    let background = py
        .detach(|| native_smooth_bruckner(y, smooth_points, iterations))
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
    let accumulation = py
        .detach(|| accumulate_tch_batch(grid, peaks, SupportPolicy::FwhmMultiple(support_fwhm)))
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

fn multiphase_result_to_numpy(
    py: Python<'_>,
    result: StructuralMultiphaseResult,
) -> PyResult<(Bound<'_, PyArray1<f64>>, Vec<StructuralPatternArrays<'_>>)> {
    Ok((
        result.profile_y.into_pyarray(py),
        result
            .phases
            .into_iter()
            .map(|phase| structural_pattern_to_numpy(py, phase))
            .collect::<PyResult<Vec<_>>>()?,
    ))
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

fn structural_pattern_dense_to_numpy(
    py: Python<'_>,
    result: StructuralPatternDenseResult,
) -> PyResult<StructuralPatternDenseArrays<'_>> {
    let StructuralPatternDenseResult {
        result,
        d_y,
        parameter_count,
    } = result;
    let sample_count = result.accumulation.sample_count;
    let jacobian = Array2::from_shape_vec((parameter_count, sample_count), d_y)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        structural_pattern_to_numpy(py, result)?,
        jacobian.into_pyarray(py),
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
    let (variance, gaussian, lorentzian, total, eta, d_gaussian, d_lorentzian, d_position) = py
        .detach(|| {
            let mut variance = Vec::with_capacity(two_theta_deg.len());
            let mut gaussian = Vec::with_capacity(two_theta_deg.len());
            let mut lorentzian = Vec::with_capacity(two_theta_deg.len());
            let mut total = Vec::with_capacity(two_theta_deg.len());
            let mut eta = Vec::with_capacity(two_theta_deg.len());
            let mut d_gaussian = Vec::with_capacity(two_theta_deg.len() * 5);
            let mut d_lorentzian = Vec::with_capacity(two_theta_deg.len() * 5);
            let mut d_position = Vec::with_capacity(two_theta_deg.len() * 2);
            for position in two_theta_deg.iter().copied() {
                let profile = CwProfileParameters::from_instrument(position, instrument)?;
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
            Ok((
                variance,
                gaussian,
                lorentzian,
                total,
                eta,
                d_gaussian,
                d_lorentzian,
                d_position,
            ))
        })
        .map_err(|error: CwError| PyValueError::new_err(error.to_string()))?;
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
    let (position, alpha, beta, variance, gaussian, lorentzian, total, eta) = py
        .detach(|| {
            let mut position = Vec::with_capacity(d_spacing.len());
            let mut alpha = Vec::with_capacity(d_spacing.len());
            let mut beta = Vec::with_capacity(d_spacing.len());
            let mut variance = Vec::with_capacity(d_spacing.len());
            let mut gaussian = Vec::with_capacity(d_spacing.len());
            let mut lorentzian = Vec::with_capacity(d_spacing.len());
            let mut total = Vec::with_capacity(d_spacing.len());
            let mut eta = Vec::with_capacity(d_spacing.len());
            for d in d_spacing.iter().copied() {
                let parameters = TofProfileParameters::from_instrument(d, instrument)?;
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
                position, alpha, beta, variance, gaussian, lorentzian, total, eta,
            ))
        })
        .map_err(|error: TofError| PyValueError::new_err(error.to_string()))?;
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
    execution: PyRef<'_, NativeExecutionPolicy>,
) -> PyResult<AccumulationArrays<'py>> {
    let x = contiguous_slice(&x_us, "x_us")?;
    let d_spacing = contiguous_slice(&d_spacing_angstrom, "d_spacing_angstrom")?;
    let intensities = contiguous_slice(&integrated_intensities, "integrated_intensities")?;
    let grid = GridView::new(x).map_err(|error| PyValueError::new_err(error.to_string()))?;
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
    let execution_context = execution.policy.context().clone();
    let accumulation = py
        .detach(|| {
            accumulate_tof_batch_with_context(
                grid,
                d_spacing,
                intensities,
                instrument,
                support_fwhm,
                tail_log,
                &execution_context,
            )
        })
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
    let instrument = cw_instrument(wavelength_angstrom, u_deg2, v_deg2, w_deg2, x_deg, y_deg);
    let support = SupportPolicy::FwhmMultiple(support_fwhm);
    let accumulation = py
        .detach(|| accumulate_cw_batch(grid, reflections, instrument, support))
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
    fcj_sample_over_radius: Option<f64>,
    fcj_detector_over_radius: Option<f64>,
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
    let instrument = cw_instrument(wavelength_angstrom, u_deg2, v_deg2, w_deg2, x_deg, y_deg);
    let support = SupportPolicy::FwhmMultiple(support_fwhm);
    let geometry = axial_geometry(fcj_sample_over_radius, fcj_detector_over_radius)?;
    let accumulation = py
        .detach(|| match geometry {
            Some(geometry) => accumulate_cw_fcj_contributions_batch(
                grid,
                two_theta_deg,
                base_intensities,
                instrument,
                contributions,
                geometry,
                support,
            ),
            None => accumulate_cw_contributions_batch(
                grid,
                two_theta_deg,
                base_intensities,
                instrument,
                contributions,
                support,
            ),
        })
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
    let instrument = cw_instrument(wavelength_angstrom, u_deg2, v_deg2, w_deg2, x_deg, y_deg);
    let geometry = FcjGeometry {
        sample_over_radius,
        detector_over_radius,
    };
    let support = SupportPolicy::FwhmMultiple(support_fwhm);
    let accumulation = py
        .detach(|| accumulate_cw_fcj_batch(grid, reflections, instrument, geometry, support))
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
    let geometry = FcjGeometry {
        sample_over_radius,
        detector_over_radius,
    };
    let accumulation = py
        .detach(|| {
            if use_fcj {
                accumulate_cw_fcj_components_batch(
                    grid,
                    reflections,
                    instrument,
                    components,
                    geometry,
                    support,
                )
            } else {
                accumulate_cw_components_batch(grid, reflections, instrument, components, support)
            }
        })
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

#[allow(clippy::similar_names)]
fn position_correction(
    zero_shift_deg: f64,
    sample_displacement_mm: Option<f64>,
    displace_x_micrometre: Option<f64>,
    displace_y_micrometre: Option<f64>,
    goniometer_radius_mm: Option<f64>,
) -> PyResult<MonochromaticPositionCorrection> {
    let bragg_brentano_mm = match (
        sample_displacement_mm,
        displace_x_micrometre,
        displace_y_micrometre,
        goniometer_radius_mm,
    ) {
        (Some(displacement), None, None, Some(radius)) => Some((displacement, radius)),
        (None, None, None, None) | (None, Some(_), Some(_), Some(_)) => None,
        _ => {
            return Err(PyValueError::new_err(
                "one complete Bragg-Brentano or Debye-Scherrer geometry is required",
            ));
        }
    };
    let debye_scherrer_micrometre = match (
        sample_displacement_mm,
        displace_x_micrometre,
        displace_y_micrometre,
        goniometer_radius_mm,
    ) {
        (None, Some(displace_x), Some(displace_y), Some(radius)) => {
            Some((displace_x, displace_y, radius))
        }
        _ => None,
    };
    Ok(MonochromaticPositionCorrection {
        zero_shift_deg,
        bragg_brentano_mm,
        debye_scherrer_micrometre,
    })
}

fn axial_geometry(
    sample_over_radius: Option<f64>,
    detector_over_radius: Option<f64>,
) -> PyResult<Option<FcjGeometry>> {
    match (sample_over_radius, detector_over_radius) {
        (None, None) => Ok(None),
        (Some(sample_over_radius), Some(detector_over_radius)) => Ok(Some(FcjGeometry {
            sample_over_radius,
            detector_over_radius,
        })),
        _ => Err(PyValueError::new_err(
            "FCJ sample and detector ratios must be provided together",
        )),
    }
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
    let y = py
        .detach(|| accumulate_values_batch(grid, peaks, SupportPolicy::FwhmMultiple(support_fwhm)))
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

fn bool_slice<'array>(
    array: &'array PyReadonlyArray1<'_, bool>,
    name: &str,
) -> PyResult<&'array [bool]> {
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

fn tensor_rows(array: &PyReadonlyArray1<'_, f64>) -> PyResult<Vec<[f64; 6]>> {
    let values = contiguous_slice(array, "u_aniso_cif_angstrom2")?;
    if values.len() % 6 != 0 {
        return Err(PyValueError::new_err(
            "flattened u_aniso_cif_angstrom2 length must be divisible by six",
        ));
    }
    Ok(values
        .chunks_exact(6)
        .map(|row| [row[0], row[1], row[2], row[3], row[4], row[5]])
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
    polarization: Option<f64>,
) -> PyResult<IntegratedIntensityCorrectionModel> {
    let selected = match (model, wavelength_angstrom, polarization) {
        ("neutral", None, None) => IntegratedIntensityCorrectionModel::Neutral,
        ("bragg_brentano_unpolarized_lp", Some(wavelength_angstrom), None) => {
            IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp {
                wavelength_angstrom,
            }
        }
        ("bragg_brentano_polarized_lp", Some(wavelength_angstrom), Some(polarization)) => {
            IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
                wavelength_angstrom,
                polarization,
            }
        }
        ("constant_wavelength_neutron_lorentz", Some(wavelength_angstrom), None) => {
            IntegratedIntensityCorrectionModel::ConstantWavelengthNeutronLorentz {
                wavelength_angstrom,
            }
        }
        ("neutral", Some(_), _) | ("neutral", None, Some(_)) => {
            return Err(PyValueError::new_err(
                "neutral correction does not accept wavelength or polarization",
            ));
        }
        ("bragg_brentano_unpolarized_lp" | "constant_wavelength_neutron_lorentz", None, None)
        | ("bragg_brentano_polarized_lp", None, _) => {
            return Err(PyValueError::new_err(
                "angular intensity correction requires a wavelength",
            ));
        }
        ("bragg_brentano_unpolarized_lp", _, Some(_)) => {
            return Err(PyValueError::new_err(
                "unpolarized Bragg-Brentano LP does not accept polarization",
            ));
        }
        ("bragg_brentano_polarized_lp", Some(_), None) => {
            return Err(PyValueError::new_err(
                "polarized Bragg-Brentano LP requires polarization",
            ));
        }
        ("constant_wavelength_neutron_lorentz", _, Some(_)) => {
            return Err(PyValueError::new_err(
                "neutron Lorentz correction does not accept polarization",
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
#[pyfunction(signature = (
    q_squared_inverse_angstrom2,
    model,
    wavelength_angstrom = None,
    polarization = None
))]
fn integrated_intensity_correction<'py>(
    py: Python<'py>,
    q_squared_inverse_angstrom2: PyReadonlyArray1<'py, f64>,
    model: &str,
    wavelength_angstrom: Option<f64>,
    polarization: Option<f64>,
) -> PyResult<CorrectionArrays<'py>> {
    let selected = parse_correction_model(model, wavelength_angstrom, polarization)?;
    let q_squared = contiguous_slice(&q_squared_inverse_angstrom2, "q_squared_inverse_angstrom2")?;
    let result = py
        .detach(|| selected.evaluate(q_squared))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        result.values.into_pyarray(py),
        result.d_values_d_q_squared.into_pyarray(py),
    ))
}

/// Return generated-table provenance without importing any source project.
#[pyfunction]
fn scattering_table_provenance() -> TableProvenanceRecords {
    let record = |value: phasesmith_engine::crystallography::ScatteringTableProvenance| {
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

fn cif_error(error: CifIoError) -> PyErr {
    match error {
        CifIoError::Io(error) => error.into(),
        CifIoError::Unsupported { .. } => PyNotImplementedError::new_err(error.to_string()),
        error => PyValueError::new_err(error.to_string()),
    }
}

fn symmetry_operations_to_python<'py>(
    py: Python<'py>,
    operations: &[SymmetryOperation],
) -> PyResult<Bound<'py, PyList>> {
    let records = PyList::empty(py);
    for operation in operations {
        let record = PyDict::new(py);
        record.set_item("rotation", operation.rotation())?;
        record.set_item(
            "translation",
            operation
                .translation()
                .map(|value| (value.numerator(), value.denominator())),
        )?;
        records.append(record)?;
    }
    Ok(records)
}

fn native_space_group_record(
    py: Python<'_>,
    info: NativeSpaceGroupInfo,
) -> PyResult<Bound<'_, PyDict>> {
    let record = PyDict::new(py);
    record.set_item("number", info.number)?;
    record.set_item("hm_symbol", info.hm_symbol)?;
    record.set_item("hall_symbol", info.hall_symbol)?;
    record.set_item("setting", info.setting)?;
    record.set_item(
        "operations",
        symmetry_operations_to_python(py, info.space_group.operations())?,
    )?;
    Ok(record)
}

/// Resolve one conventional native space group by International number.
#[pyfunction(name = "_space_group_by_number")]
fn space_group_by_number_for_python(py: Python<'_>, number: i32) -> PyResult<Bound<'_, PyDict>> {
    native_space_group_by_number(number)
        .map_err(|error| PyValueError::new_err(error.to_string()))
        .and_then(|info| native_space_group_record(py, info))
}

/// Resolve one conventional native space group by Hermann--Mauguin or Hall symbol.
#[pyfunction(name = "_space_group_by_symbol")]
fn space_group_by_symbol_for_python<'py>(
    py: Python<'py>,
    symbol: &str,
) -> PyResult<Bound<'py, PyDict>> {
    native_space_group_by_symbol(symbol)
        .map_err(|error| PyValueError::new_err(error.to_string()))
        .and_then(|info| native_space_group_record(py, info))
}

// The record mirrors `structure_to_record` so Python reconstruction uses the
// same stable parser-independent boundary as JSON persistence.
#[allow(clippy::too_many_lines)]
fn native_cif_record(
    py: Python<'_>,
    result: NativeCifReadResult,
) -> PyResult<(Bound<'_, PyDict>, String, Vec<String>)> {
    let structure = &result.structure;
    let record = PyDict::new(py);
    record.set_item("format_version", 1)?;
    record.set_item("structure_id", &structure.structure_id)?;
    record.set_item("name", &structure.name)?;
    record.set_item(
        "cell",
        [
            structure.cell.a_angstrom,
            structure.cell.b_angstrom,
            structure.cell.c_angstrom,
            structure.cell.alpha_deg,
            structure.cell.beta_deg,
            structure.cell.gamma_deg,
        ],
    )?;
    record.set_item(
        "cell_standard_uncertainties",
        structure.cell_standard_uncertainties,
    )?;
    record.set_item(
        "operations",
        symmetry_operations_to_python(py, structure.space_group.operations())?,
    )?;

    let sites = PyList::empty(py);
    for site in &structure.sites {
        let site_record = PyDict::new(py);
        site_record.set_item("site_id", &site.site_id)?;
        site_record.set_item("source_label", &site.source_label)?;
        site_record.set_item("type_symbol", &site.type_symbol)?;
        site_record.set_item("element_symbol", &site.element_symbol)?;
        site_record.set_item("fractional_xyz", site.fractional_xyz)?;
        site_record.set_item("occupancy", site.occupancy)?;
        site_record.set_item("u_iso_angstrom2", site.u_iso_angstrom2)?;
        if let Some(anisotropic) = &site.anisotropic_displacement {
            let anisotropic_record = PyDict::new(py);
            anisotropic_record.set_item("u_cif_angstrom2", anisotropic.u_cif_angstrom2)?;
            anisotropic_record.set_item(
                "source_convention",
                match anisotropic.source_convention {
                    DisplacementConvention::CifU => "U_cif",
                    DisplacementConvention::CifB => "B_cif",
                },
            )?;
            anisotropic_record
                .set_item("standard_uncertainty", anisotropic.standard_uncertainty)?;
            site_record.set_item("anisotropic_displacement", anisotropic_record)?;
        } else {
            site_record.set_item("anisotropic_displacement", py.None())?;
        }
        site_record.set_item("charge", site.charge)?;
        site_record.set_item("isotope", site.isotope)?;
        site_record.set_item("disorder_group", &site.disorder_group)?;
        site_record.set_item(
            "fractional_xyz_standard_uncertainty",
            site.fractional_xyz_standard_uncertainty,
        )?;
        site_record.set_item(
            "occupancy_standard_uncertainty",
            site.occupancy_standard_uncertainty,
        )?;
        site_record.set_item(
            "u_iso_standard_uncertainty",
            site.u_iso_standard_uncertainty,
        )?;
        sites.append(site_record)?;
    }
    record.set_item("sites", sites)?;

    let source = PyDict::new(py);
    source.set_item("format", &structure.source.format)?;
    source.set_item("block_name", &structure.source.block_name)?;
    source.set_item("backend", &structure.source.backend)?;
    source.set_item("backend_version", &structure.source.backend_version)?;
    source.set_item(
        "source_name",
        structure
            .source
            .source_path
            .as_ref()
            .map(|path| path.to_string_lossy().into_owned()),
    )?;
    record.set_item("source", source)?;

    let diagnostics = PyList::empty(py);
    for diagnostic in &structure.diagnostics {
        let diagnostic_record = PyDict::new(py);
        diagnostic_record.set_item(
            "severity",
            match diagnostic.severity {
                CifDiagnosticSeverity::Warning => "warning",
                CifDiagnosticSeverity::Error => "error",
            },
        )?;
        diagnostic_record.set_item("code", &diagnostic.code)?;
        diagnostic_record.set_item("message", &diagnostic.message)?;
        diagnostic_record.set_item("tag", &diagnostic.tag)?;
        diagnostic_record.set_item("row", diagnostic.row)?;
        diagnostics.append(diagnostic_record)?;
    }
    record.set_item("diagnostics", diagnostics)?;
    record.set_item("metadata", &structure.metadata)?;
    Ok((record, result.selected_block, result.available_blocks))
}

/// Parse CIF text through the shared native adapter.
#[pyfunction(name = "_parse_cif_text")]
#[allow(clippy::too_many_arguments)]
fn parse_cif_text_for_python(
    py: Python<'_>,
    text: String,
    source_name: Option<String>,
    block: Option<String>,
    strict: bool,
    max_bytes: usize,
    max_blocks: usize,
    max_loop_rows: usize,
    max_atom_sites: usize,
) -> PyResult<(Bound<'_, PyDict>, String, Vec<String>)> {
    let limits = NativeCifReadLimits {
        max_bytes,
        max_blocks,
        max_loop_rows,
        max_atom_sites,
    };
    let mut result = py
        .detach(move || parse_native_cif_text(&text, block.as_deref(), strict, limits))
        .map_err(cif_error)?;
    result.structure.source.source_path = source_name.map(Into::into);
    native_cif_record(py, result)
}

fn parse_powder_format(value: &str) -> PyResult<NativePowderFormat> {
    match value {
        "auto" => Ok(NativePowderFormat::Auto),
        "columns" => Ok(NativePowderFormat::Columns),
        "gsas_fxye" => Ok(NativePowderFormat::GsasFxye),
        "gsas_std" => Ok(NativePowderFormat::GsasStd),
        _ => Err(PyValueError::new_err(
            "format must be 'auto', 'columns', 'gsas_fxye', or 'gsas_std'",
        )),
    }
}

fn parse_tof_powder_format(value: &str) -> PyResult<NativeTofPowderFormat> {
    match value {
        "auto" => Ok(NativeTofPowderFormat::Auto),
        "columns" => Ok(NativeTofPowderFormat::Columns),
        "gsas_slog_fxye" => Ok(NativeTofPowderFormat::GsasSlogFxye),
        "gsas_const_std" => Ok(NativeTofPowderFormat::GsasConstStd),
        _ => Err(PyValueError::new_err(
            "format must be 'auto', 'columns', 'gsas_slog_fxye', or 'gsas_const_std'",
        )),
    }
}

fn powder_error(error: PowderIoError) -> PyErr {
    match error {
        PowderIoError::Io(error) => error.into(),
        error => PyValueError::new_err(error.to_string()),
    }
}

fn powder_data_to_numpy(py: Python<'_>, data: NativePowderData) -> PyResult<PowderDataArrays<'_>> {
    let format = match data.format {
        NativePowderFormat::Columns => "columns",
        NativePowderFormat::GsasFxye => "gsas_fxye",
        NativePowderFormat::GsasStd => "gsas_std",
        NativePowderFormat::Auto => {
            return Err(PyValueError::new_err(
                "native powder reader returned unresolved auto format",
            ));
        }
    };
    let observed_y = data
        .pattern
        .observed_y
        .ok_or_else(|| PyValueError::new_err("native powder reader returned no observations"))?;
    Ok((
        data.pattern.x_deg.into_pyarray(py),
        observed_y.into_pyarray(py),
        data.pattern
            .uncertainty
            .map(|values| values.into_pyarray(py)),
        format,
        data.source_path
            .map(|path| path.to_string_lossy().into_owned()),
        data.bank,
        data.pattern.mask.map(|values| values.into_pyarray(py)),
    ))
}

fn tof_powder_data_to_numpy(
    py: Python<'_>,
    data: NativeTofPowderData,
) -> PyResult<TofPowderDataArrays<'_>> {
    let format = match data.format {
        NativeTofPowderFormat::Columns => "columns",
        NativeTofPowderFormat::GsasSlogFxye => "gsas_slog_fxye",
        NativeTofPowderFormat::GsasConstStd => "gsas_const_std",
        NativeTofPowderFormat::Auto => {
            return Err(PyValueError::new_err(
                "native TOF powder reader returned unresolved auto format",
            ));
        }
    };
    let observed_y = data.pattern.observed_y.ok_or_else(|| {
        PyValueError::new_err("native TOF powder reader returned no observations")
    })?;
    Ok((
        data.pattern.tof_us.into_pyarray(py),
        observed_y.into_pyarray(py),
        data.pattern
            .uncertainty
            .map(|values| values.into_pyarray(py)),
        format,
        data.source_path
            .map(|path| path.to_string_lossy().into_owned()),
        data.bank,
        data.logarithmic_grid,
        data.pattern.mask.map(|values| values.into_pyarray(py)),
    ))
}

fn gsas_tof_instrument_record(data: NativeGsasTofInstrumentData) -> GsasTofInstrumentRecord {
    let instrument = data.instrument;
    (
        vec![
            instrument.zero_us,
            instrument.difc_us_per_angstrom,
            instrument.difa_us_per_angstrom2,
            instrument.difb_us_angstrom,
            instrument.alpha_coefficient,
            instrument.beta0_per_us,
            instrument.beta1_angstrom4_per_us,
            instrument.betaq_angstrom2_per_us,
            instrument.sigma0_us2,
            instrument.sigma1_us2_per_angstrom2,
            instrument.sigma2_us2_per_angstrom4,
            instrument.sigmaq_us2_per_angstrom,
            instrument.x_us_per_angstrom,
            instrument.y_us_per_angstrom2,
            instrument.z_us,
        ],
        data.bank,
        data.profile_function,
        data.source_path
            .map(|path| path.to_string_lossy().into_owned()),
    )
}

fn gsas_tof_instrument_error(error: GsasTofInstrumentIoError) -> PyErr {
    match error {
        GsasTofInstrumentIoError::Io(error) => error.into(),
        error => PyValueError::new_err(error.to_string()),
    }
}

/// Parse powder text through the shared native adapter.
#[pyfunction(name = "_parse_powder_text")]
fn parse_powder_text_for_python<'py>(
    py: Python<'py>,
    text: String,
    format: &str,
    bank: usize,
    max_bytes: usize,
    max_rows: usize,
) -> PyResult<PowderDataArrays<'py>> {
    let format = parse_powder_format(format)?;
    let limits = NativePowderReadLimits {
        max_bytes,
        max_rows,
    };
    let data = py
        .detach(move || parse_native_powder_text(&text, format, bank, limits))
        .map_err(powder_error)?;
    powder_data_to_numpy(py, data)
}

/// Read a powder file through the shared native adapter.
#[pyfunction(name = "_read_powder_file")]
fn read_powder_file_for_python<'py>(
    py: Python<'py>,
    path: String,
    format: &str,
    bank: usize,
    max_bytes: usize,
    max_rows: usize,
) -> PyResult<PowderDataArrays<'py>> {
    let format = parse_powder_format(format)?;
    let limits = NativePowderReadLimits {
        max_bytes,
        max_rows,
    };
    let data = py
        .detach(move || read_native_powder_file(path, format, bank, limits))
        .map_err(powder_error)?;
    powder_data_to_numpy(py, data)
}

/// Parse one GSAS TOF bank through the dedicated microsecond-domain adapter.
#[pyfunction(name = "_parse_tof_powder_text")]
fn parse_tof_powder_text_for_python<'py>(
    py: Python<'py>,
    text: String,
    format: &str,
    bank: usize,
    max_bytes: usize,
    max_rows: usize,
) -> PyResult<TofPowderDataArrays<'py>> {
    let format = parse_tof_powder_format(format)?;
    let limits = NativePowderReadLimits {
        max_bytes,
        max_rows,
    };
    let data = py
        .detach(move || parse_native_tof_powder_text(&text, format, bank, limits))
        .map_err(powder_error)?;
    tof_powder_data_to_numpy(py, data)
}

/// Read one GSAS TOF bank through the dedicated microsecond-domain adapter.
#[pyfunction(name = "_read_tof_powder_file")]
fn read_tof_powder_file_for_python<'py>(
    py: Python<'py>,
    path: String,
    format: &str,
    bank: usize,
    max_bytes: usize,
    max_rows: usize,
) -> PyResult<TofPowderDataArrays<'py>> {
    let format = parse_tof_powder_format(format)?;
    let limits = NativePowderReadLimits {
        max_bytes,
        max_rows,
    };
    let data = py
        .detach(move || read_native_tof_powder_file(path, format, bank, limits))
        .map_err(powder_error)?;
    tof_powder_data_to_numpy(py, data)
}

/// Parse one bounded legacy GSAS TOF instrument bank.
#[pyfunction(name = "_parse_gsas_tof_instrument_text")]
fn parse_gsas_tof_instrument_text_for_python(
    py: Python<'_>,
    text: String,
    bank: usize,
    max_bytes: usize,
) -> PyResult<GsasTofInstrumentRecord> {
    let limits = NativeGsasTofInstrumentReadLimits { max_bytes };
    py.detach(move || parse_native_gsas_tof_instrument_text(&text, bank, limits))
        .map(gsas_tof_instrument_record)
        .map_err(gsas_tof_instrument_error)
}

/// Read one bounded legacy GSAS TOF instrument bank.
#[pyfunction(name = "_read_gsas_tof_instrument_file")]
fn read_gsas_tof_instrument_file_for_python(
    py: Python<'_>,
    path: String,
    bank: usize,
    max_bytes: usize,
) -> PyResult<GsasTofInstrumentRecord> {
    let limits = NativeGsasTofInstrumentReadLimits { max_bytes };
    py.detach(move || read_native_gsas_tof_instrument_file(path, bank, limits))
        .map(gsas_tof_instrument_record)
        .map_err(gsas_tof_instrument_error)
}

/// Run one Python-free validation workflow and return its stable JSON report.
#[pyfunction(name = "_run_native_validation")]
fn run_native_validation(py: Python<'_>, runner: String, directory: String) -> PyResult<String> {
    py.detach(move || {
        let directory = PathBuf::from(directory);
        let report = match runner.as_str() {
            "ansto-echidna-lab6-cw-neutron" => {
                phasesmith_validation::run_echidna_lab6_validation(&directory)
                    .map_err(|error| error.to_string())?
            }
            "aps-sucrose-11bmb" => phasesmith_validation::run_sucrose_lebail_validation(&directory)
                .map_err(|error| error.to_string())?,
            "iucr-qarr-1g" => phasesmith_validation::run_qarr_1g_validation(&directory)
                .map_err(|error| error.to_string())?,
            "iucr-qarr-1h" => phasesmith_validation::run_qarr_1h_validation(&directory)
                .map_err(|error| error.to_string())?,
            "nist-srm660c-lab6-xray" => {
                phasesmith_validation::run_nist_srm660c_validation(&directory)
                    .map_err(|error| error.to_string())?
            }
            "lanl-nickel-tof" => phasesmith_validation::run_nickel_tof_validation(&directory)
                .map_err(|error| error.to_string())?,
            "powgen-lab6-tof-calibration" => {
                phasesmith_validation::run_powgen_tof_readiness(&directory)
                    .map_err(|error| error.to_string())?
            }
            "gsasii-pbso4-cw-neutron" => {
                phasesmith_validation::run_pbso4_neutron_validation(&directory)
                    .map_err(|error| error.to_string())?
            }
            "gsasii-pbso4-cw-x-ray" => phasesmith_validation::run_pbso4_xray_validation(&directory)
                .map_err(|error| error.to_string())?,
            _ => return Err(format!("unknown native validation runner {runner:?}")),
        };
        report.to_json().map_err(|error| error.to_string())
    })
    .map_err(PyValueError::new_err)
}

/// Native Python module.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeExecutionPolicy>()?;
    module.add_class::<NativePreparedXrayScattering>()?;
    module.add_class::<NativePreparedNeutronScattering>()?;
    module.add_class::<NativePreparedReflectionGenerator>()?;
    module.add_class::<NativeStructuralPhase>()?;
    module.add_class::<NativeStructuralSpectrum>()?;
    module.add_class::<NativePreparedStructuralModel>()?;
    module.add_class::<NativeStructuralMultiphase>()?;
    rietveld::register(module)?;
    profile_estimation::register(module)?;
    tof_lebail::register(module)?;
    tof_multibank::register(module)?;
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
    module.add_function(wrap_pyfunction!(space_group_by_number_for_python, module)?)?;
    module.add_function(wrap_pyfunction!(space_group_by_symbol_for_python, module)?)?;
    module.add_function(wrap_pyfunction!(parse_cif_text_for_python, module)?)?;
    module.add_function(wrap_pyfunction!(parse_powder_text_for_python, module)?)?;
    module.add_function(wrap_pyfunction!(read_powder_file_for_python, module)?)?;
    module.add_function(wrap_pyfunction!(parse_tof_powder_text_for_python, module)?)?;
    module.add_function(wrap_pyfunction!(read_tof_powder_file_for_python, module)?)?;
    module.add_function(wrap_pyfunction!(
        parse_gsas_tof_instrument_text_for_python,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        read_gsas_tof_instrument_file_for_python,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(run_native_validation, module)?)?;
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
        PyTuple::new(module.py(), TOF_GLOBAL_PARAMETER_NAMES)?,
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
    module.add("NATIVE_CIF_BACKEND", NATIVE_CIF_BACKEND)?;
    module.add("NATIVE_CIF_BACKEND_VERSION", NATIVE_CIF_BACKEND_VERSION)?;
    Ok(())
}
