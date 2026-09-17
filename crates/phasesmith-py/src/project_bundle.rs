//! Opaque Python ownership for lossless mixed native bundles.
use phasesmith_engine::MonochromaticPositionCorrection;
use phasesmith_model::{
    ExperimentRecord, HistogramRecord, ProjectRecord, RadiationDefinition, RadiationProbe, RecordId,
};
use phasesmith_persistence::{
    PawleyProject, ProjectBundle, ProjectReadLimits, ProjectSaveOptions, decode_pawley_project,
    encode_pawley_project, load_project_bundle, save_project_bundle,
};
use phasesmith_workflows::PawleyAnalysis;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeMap;
fn error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}
const MAX_BYTES: usize = 256 * 1024 * 1024;
#[pyclass(name = "_ProjectBundle", frozen)]
struct NativeProjectBundle {
    bundle: ProjectBundle,
}
#[pymethods]
impl NativeProjectBundle {
    #[staticmethod]
    fn load(py: Python<'_>, path: String) -> PyResult<Self> {
        py.detach(|| load_project_bundle(path, ProjectReadLimits::default()))
            .map(|bundle| Self { bundle })
            .map_err(error)
    }
    fn save(&self, py: Python<'_>, path: String, overwrite: bool) -> PyResult<()> {
        py.detach(|| save_project_bundle(path, &self.bundle, ProjectSaveOptions { overwrite }))
            .map(|_| ())
            .map_err(error)
    }
    #[staticmethod]
    fn from_pawley(
        record: &str,
        probe: &str,
        project_id: &str,
        histogram_id: &str,
        name: &str,
    ) -> PyResult<Self> {
        let p = decode_pawley_project(record, MAX_BYTES).map_err(error)?;
        let probe = match probe {
            "x-ray" => RadiationProbe::Xray,
            "neutron" => RadiationProbe::Neutron,
            _ => return Err(error("probe must be x-ray or neutron")),
        };
        let histogram_id = RecordId::new(histogram_id).map_err(error)?;
        let experiment = ExperimentRecord::new(
            p.input.instrument,
            match &p.input.fixed_spectrum {
                Some(spectrum) => RadiationDefinition::FixedSpectrum {
                    probe,
                    spectrum: spectrum.clone(),
                },
                None => RadiationDefinition::Monochromatic {
                    probe,
                    wavelength_angstrom: p.input.instrument.wavelength_angstrom,
                },
            },
            p.input.axial,
            MonochromaticPositionCorrection {
                zero_shift_deg: 0.0,
                bragg_brentano_mm: None,
                debye_scherrer_micrometre: None,
            },
        )
        .map_err(error)?;
        let project = ProjectRecord {
            project_id: RecordId::new(project_id).map_err(error)?,
            revision: 0,
            name: name.into(),
            histograms: vec![HistogramRecord {
                histogram_id: histogram_id.clone(),
                name: name.into(),
                pattern: p.input.pattern.clone(),
                experiment,
                phase_ids: Vec::new(),
            }],
            tof_histograms: Vec::new(),
            phases: Vec::new(),
            metadata: BTreeMap::new(),
        };
        let mut bundle = ProjectBundle::new(project);
        bundle.pawley_analyses.push(PawleyAnalysis {
            histogram_id,
            input: p.input,
            options: p.options,
            checkpoint: p.checkpoint,
        });
        bundle.validate().map_err(error)?;
        Ok(Self { bundle })
    }
    fn with_pawley(&self, histogram_id: &str, record: &str) -> PyResult<Self> {
        let p = decode_pawley_project(record, MAX_BYTES).map_err(error)?;
        let histogram_id = RecordId::new(histogram_id).map_err(error)?;
        let analysis = PawleyAnalysis {
            histogram_id: histogram_id.clone(),
            input: p.input,
            options: p.options,
            checkpoint: p.checkpoint,
        };
        let mut bundle = self.bundle.clone();
        if let Some(old) = bundle
            .pawley_analyses
            .iter_mut()
            .find(|a| a.histogram_id == histogram_id)
        {
            *old = analysis;
        } else {
            bundle.pawley_analyses.push(analysis);
        }
        bundle.validate().map_err(error)?;
        Ok(Self { bundle })
    }
    fn pawley(&self, histogram_id: &str) -> PyResult<String> {
        let a = self
            .bundle
            .pawley_analyses
            .iter()
            .find(|a| a.histogram_id.as_str() == histogram_id)
            .ok_or_else(|| error("unknown Pawley histogram analysis"))?;
        encode_pawley_project(&PawleyProject {
            input: a.input.clone(),
            options: a.options.clone(),
            checkpoint: a.checkpoint.clone(),
        })
        .map_err(error)
    }
    #[getter]
    fn pawley_histograms(&self) -> Vec<String> {
        self.bundle
            .pawley_analyses
            .iter()
            .map(|a| a.histogram_id.as_str().into())
            .collect()
    }
    #[getter]
    fn analysis_counts(&self) -> BTreeMap<String, usize> {
        BTreeMap::from([
            ("rietveld".into(), self.bundle.rietveld_analyses.len()),
            ("tof_lebail".into(), self.bundle.tof_lebail_analyses.len()),
            (
                "tof_geometry".into(),
                self.bundle.tof_multibank_geometry_analyses.len(),
            ),
            (
                "structural_tof".into(),
                self.bundle.structural_tof_multibank_analyses.len(),
            ),
            ("pawley".into(), self.bundle.pawley_analyses.len()),
        ])
    }
}
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeProjectBundle>()
}
