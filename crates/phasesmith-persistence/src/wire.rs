//! Explicit project wire records and live-domain conversion.

use std::collections::BTreeMap;

use phasesmith_engine::crystallography::{
    IntegratedIntensityCorrectionModel, Rational, SpaceGroup, SymmetryOperation, UnitCell,
};
use phasesmith_engine::profile::{ConstantWavelengthInstrument, FcjGeometry};
use phasesmith_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPhaseDefinition,
};
use phasesmith_model::{
    ExperimentRecord, FixedWavelengthSpectrum, HistogramRecord, PatternRecord, ProjectRecord,
    ProviderRequirement, RadiationDefinition, RadiationProbe, RecordId, StructuralPhaseRecord,
};
use serde::{Deserialize, Serialize};

use crate::arrays::ArrayData;
use crate::{PersistenceError, ProjectReadLimits};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireProject {
    project_id: String,
    revision: u64,
    name: String,
    histograms: Vec<WireHistogram>,
    phases: Vec<WirePhase>,
    metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireHistogram {
    histogram_id: String,
    name: String,
    pattern: WirePattern,
    experiment: WireExperiment,
    phase_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePattern {
    x_deg: ArrayReference,
    observed_y: Option<ArrayReference>,
    uncertainty: Option<ArrayReference>,
    mask: Option<ArrayReference>,
    background_y: ArrayReference,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireExperiment {
    instrument: WireInstrument,
    radiation: WireRadiation,
    axial_geometry: Option<WireAxialGeometry>,
    position_correction: WirePositionCorrection,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireInstrument {
    wavelength_angstrom: f64,
    u_deg2: f64,
    v_deg2: f64,
    w_deg2: f64,
    x_deg: f64,
    y_deg: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum WireRadiation {
    Monochromatic {
        probe: WireProbe,
        wavelength_angstrom: f64,
    },
    FixedSpectrum {
        probe: WireProbe,
        wavelengths_angstrom: Vec<f64>,
        relative_intensities: Vec<f64>,
    },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WireProbe {
    Xray,
    Neutron,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAxialGeometry {
    sample_over_radius: f64,
    detector_over_radius: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePositionCorrection {
    zero_shift_deg: f64,
    bragg_brentano_mm: Option<[f64; 2]>,
    debye_scherrer_micrometre: Option<[f64; 3]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePhase {
    phase_id: String,
    name: String,
    definition: WireStructuralDefinition,
    required_providers: Vec<WireProviderRequirement>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireProviderRequirement {
    provider_id: String,
    provider_version: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireStructuralDefinition {
    cell: [f64; 6],
    operations: Vec<WireSymmetryOperation>,
    hkl: ArrayReference,
    multiplicity: ArrayReference,
    fractional_xyz: ArrayReference,
    occupancy: ArrayReference,
    u_iso_angstrom2: ArrayReference,
    anisotropic_mask: ArrayReference,
    u_aniso_cif_angstrom2: ArrayReference,
    scattering_species: Vec<String>,
    scattering_real_offset: Option<ArrayReference>,
    scattering_imag_offset: Option<ArrayReference>,
    scale: f64,
    coordinate_tolerance: f64,
    scattering_model: WireScatteringModel,
    correction_model: WireCorrectionModel,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSymmetryOperation {
    rotation: [[i32; 3]; 3],
    translation: [[i64; 2]; 3],
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WireScatteringModel {
    XrayNonResonant,
    NeutronNuclear,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum WireCorrectionModel {
    Neutral,
    BraggBrentanoUnpolarizedLp {
        wavelength_angstrom: f64,
    },
    BraggBrentanoPolarizedLp {
        wavelength_angstrom: f64,
        polarization: f64,
    },
    ConstantWavelengthNeutronLorentz {
        wavelength_angstrom: f64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArrayReference {
    array: String,
}

#[derive(Default)]
struct ArrayWriter {
    arrays: BTreeMap<String, ArrayData>,
}

impl ArrayWriter {
    fn add(&mut self, name: String, value: ArrayData) -> Result<ArrayReference, PersistenceError> {
        if self.arrays.insert(name.clone(), value).is_some() {
            return Err(invalid_record(format!(
                "duplicate wire array name {name:?}"
            )));
        }
        Ok(ArrayReference { array: name })
    }
}

pub(crate) fn encode_project(
    project: &ProjectRecord,
) -> Result<(WireProject, BTreeMap<String, ArrayData>), PersistenceError> {
    let mut writer = ArrayWriter::default();
    let histograms = project
        .histograms
        .iter()
        .map(|histogram| encode_histogram(histogram, &mut writer))
        .collect::<Result<Vec<_>, _>>()?;
    let phases = project
        .phases
        .iter()
        .map(|phase| encode_phase(phase, &mut writer))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((
        WireProject {
            project_id: project.project_id.as_str().to_owned(),
            revision: project.revision,
            name: project.name.clone(),
            histograms,
            phases,
            metadata: project.metadata.clone(),
        },
        writer.arrays,
    ))
}

fn encode_histogram(
    histogram: &HistogramRecord,
    writer: &mut ArrayWriter,
) -> Result<WireHistogram, PersistenceError> {
    let prefix = format!("histogram.{}", histogram.histogram_id.as_str());
    let count = histogram.pattern.sample_count();
    let x_deg = writer.add(
        format!("{prefix}.x_deg"),
        ArrayData::f64(histogram.pattern.x_deg.clone(), vec![count])?,
    )?;
    let observed_y = histogram
        .pattern
        .observed_y
        .as_ref()
        .map(|values| {
            writer.add(
                format!("{prefix}.observed_y"),
                ArrayData::f64(values.clone(), vec![count])?,
            )
        })
        .transpose()?;
    let uncertainty = histogram
        .pattern
        .uncertainty
        .as_ref()
        .map(|values| {
            writer.add(
                format!("{prefix}.uncertainty"),
                ArrayData::f64(values.clone(), vec![count])?,
            )
        })
        .transpose()?;
    let mask = histogram
        .pattern
        .mask
        .as_ref()
        .map(|values| {
            writer.add(
                format!("{prefix}.mask"),
                ArrayData::bool(values.clone(), vec![count])?,
            )
        })
        .transpose()?;
    let background_y = writer.add(
        format!("{prefix}.background_y"),
        ArrayData::f64(histogram.pattern.background_y.clone(), vec![count])?,
    )?;
    Ok(WireHistogram {
        histogram_id: histogram.histogram_id.as_str().to_owned(),
        name: histogram.name.clone(),
        pattern: WirePattern {
            x_deg,
            observed_y,
            uncertainty,
            mask,
            background_y,
        },
        experiment: encode_experiment(&histogram.experiment),
        phase_ids: histogram
            .phase_ids
            .iter()
            .map(|value| value.as_str().to_owned())
            .collect(),
    })
}

fn encode_experiment(experiment: &ExperimentRecord) -> WireExperiment {
    let instrument = experiment.instrument;
    let radiation = match &experiment.radiation {
        RadiationDefinition::Monochromatic {
            probe,
            wavelength_angstrom,
        } => WireRadiation::Monochromatic {
            probe: encode_probe(*probe),
            wavelength_angstrom: *wavelength_angstrom,
        },
        RadiationDefinition::FixedSpectrum { probe, spectrum } => WireRadiation::FixedSpectrum {
            probe: encode_probe(*probe),
            wavelengths_angstrom: spectrum.wavelengths_angstrom().to_vec(),
            relative_intensities: spectrum.relative_intensities().to_vec(),
        },
    };
    WireExperiment {
        instrument: WireInstrument {
            wavelength_angstrom: instrument.wavelength_angstrom,
            u_deg2: instrument.u_deg2,
            v_deg2: instrument.v_deg2,
            w_deg2: instrument.w_deg2,
            x_deg: instrument.x_deg,
            y_deg: instrument.y_deg,
        },
        radiation,
        axial_geometry: experiment.axial_geometry.map(|geometry| WireAxialGeometry {
            sample_over_radius: geometry.sample_over_radius,
            detector_over_radius: geometry.detector_over_radius,
        }),
        position_correction: WirePositionCorrection {
            zero_shift_deg: experiment.position_correction.zero_shift_deg,
            bragg_brentano_mm: experiment
                .position_correction
                .bragg_brentano_mm
                .map(|(displacement, radius)| [displacement, radius]),
            debye_scherrer_micrometre: experiment
                .position_correction
                .debye_scherrer_micrometre
                .map(|(x, y, radius)| [x, y, radius]),
        },
    }
}

// Phase arrays and their references are emitted together to keep the wire
// names, shapes, and optional offset pairing auditable in one place.
#[allow(clippy::too_many_lines)]
fn encode_phase(
    phase: &StructuralPhaseRecord,
    writer: &mut ArrayWriter,
) -> Result<WirePhase, PersistenceError> {
    let definition = &phase.definition;
    let prefix = format!("phase.{}", phase.phase_id.as_str());
    let reflection_count = definition.hkl.len();
    let site_count = definition.fractional_xyz.len();
    let hkl = writer.add(
        format!("{prefix}.hkl"),
        ArrayData::i32(
            definition.hkl.iter().flatten().copied().collect(),
            vec![reflection_count, 3],
        )?,
    )?;
    let multiplicity = writer.add(
        format!("{prefix}.multiplicity"),
        ArrayData::u64(
            definition
                .multiplicity
                .iter()
                .map(|value| {
                    u64::try_from(*value).map_err(|_| {
                        invalid_record("reflection multiplicity does not fit uint64".to_owned())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            vec![reflection_count],
        )?,
    )?;
    let fractional_xyz = writer.add(
        format!("{prefix}.fractional_xyz"),
        ArrayData::f64(
            definition
                .fractional_xyz
                .iter()
                .flatten()
                .copied()
                .collect(),
            vec![site_count, 3],
        )?,
    )?;
    let occupancy = add_site_f64(
        writer,
        &prefix,
        "occupancy",
        &definition.occupancy,
        site_count,
    )?;
    let u_iso_angstrom2 = add_site_f64(
        writer,
        &prefix,
        "u_iso_angstrom2",
        &definition.u_iso_angstrom2,
        site_count,
    )?;
    let anisotropic_mask = writer.add(
        format!("{prefix}.anisotropic_mask"),
        ArrayData::bool(definition.anisotropic_mask.clone(), vec![site_count])?,
    )?;
    let u_aniso_cif_angstrom2 = writer.add(
        format!("{prefix}.u_aniso_cif_angstrom2"),
        ArrayData::f64(
            definition
                .u_aniso_cif_angstrom2
                .iter()
                .flatten()
                .copied()
                .collect(),
            vec![site_count, 6],
        )?,
    )?;
    let scattering_real_offset = add_optional_f64(
        writer,
        &prefix,
        "scattering_real_offset",
        &definition.scattering_real_offset,
        site_count,
    )?;
    let scattering_imag_offset = add_optional_f64(
        writer,
        &prefix,
        "scattering_imag_offset",
        &definition.scattering_imag_offset,
        site_count,
    )?;
    Ok(WirePhase {
        phase_id: phase.phase_id.as_str().to_owned(),
        name: phase.name.clone(),
        definition: WireStructuralDefinition {
            cell: [
                definition.cell.a_angstrom,
                definition.cell.b_angstrom,
                definition.cell.c_angstrom,
                definition.cell.alpha_deg,
                definition.cell.beta_deg,
                definition.cell.gamma_deg,
            ],
            operations: definition
                .space_group
                .operations()
                .iter()
                .map(|operation| WireSymmetryOperation {
                    rotation: operation.rotation(),
                    translation: operation
                        .translation()
                        .map(|value| [value.numerator(), value.denominator()]),
                })
                .collect(),
            hkl,
            multiplicity,
            fractional_xyz,
            occupancy,
            u_iso_angstrom2,
            anisotropic_mask,
            u_aniso_cif_angstrom2,
            scattering_species: definition.scattering_species.clone(),
            scattering_real_offset,
            scattering_imag_offset,
            scale: definition.scale,
            coordinate_tolerance: definition.coordinate_tolerance,
            scattering_model: match definition.scattering_model {
                BuiltInScatteringModel::XrayNonResonant => WireScatteringModel::XrayNonResonant,
                BuiltInScatteringModel::NeutronNuclear => WireScatteringModel::NeutronNuclear,
            },
            correction_model: encode_correction(definition.correction_model),
        },
        required_providers: phase
            .required_providers
            .iter()
            .map(|requirement| WireProviderRequirement {
                provider_id: requirement.provider_id.clone(),
                provider_version: requirement.provider_version.clone(),
            })
            .collect(),
    })
}

fn add_site_f64(
    writer: &mut ArrayWriter,
    prefix: &str,
    name: &str,
    values: &[f64],
    site_count: usize,
) -> Result<ArrayReference, PersistenceError> {
    writer.add(
        format!("{prefix}.{name}"),
        ArrayData::f64(values.to_vec(), vec![site_count])?,
    )
}

fn add_optional_f64(
    writer: &mut ArrayWriter,
    prefix: &str,
    name: &str,
    values: &[f64],
    site_count: usize,
) -> Result<Option<ArrayReference>, PersistenceError> {
    if values.is_empty() {
        Ok(None)
    } else {
        add_site_f64(writer, prefix, name, values, site_count).map(Some)
    }
}

fn encode_probe(probe: RadiationProbe) -> WireProbe {
    match probe {
        RadiationProbe::Xray => WireProbe::Xray,
        RadiationProbe::Neutron => WireProbe::Neutron,
    }
}

fn encode_correction(model: IntegratedIntensityCorrectionModel) -> WireCorrectionModel {
    match model {
        IntegratedIntensityCorrectionModel::Neutral => WireCorrectionModel::Neutral,
        IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp {
            wavelength_angstrom,
        } => WireCorrectionModel::BraggBrentanoUnpolarizedLp {
            wavelength_angstrom,
        },
        IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
            wavelength_angstrom,
            polarization,
        } => WireCorrectionModel::BraggBrentanoPolarizedLp {
            wavelength_angstrom,
            polarization,
        },
        IntegratedIntensityCorrectionModel::ConstantWavelengthNeutronLorentz {
            wavelength_angstrom,
        } => WireCorrectionModel::ConstantWavelengthNeutronLorentz {
            wavelength_angstrom,
        },
    }
}

pub(crate) fn decode_project(
    wire: WireProject,
    mut arrays: BTreeMap<String, ArrayData>,
    limits: ProjectReadLimits,
) -> Result<ProjectRecord, PersistenceError> {
    if wire.histograms.len() > limits.max_histograms {
        return Err(PersistenceError::LimitExceeded {
            message: "project exceeds max_histograms".to_owned(),
        });
    }
    if wire.phases.len() > limits.max_phases {
        return Err(PersistenceError::LimitExceeded {
            message: "project exceeds max_phases".to_owned(),
        });
    }
    let histograms = wire
        .histograms
        .into_iter()
        .map(|histogram| decode_histogram(histogram, &mut arrays))
        .collect::<Result<Vec<_>, _>>()?;
    let phases = wire
        .phases
        .into_iter()
        .map(|phase| decode_phase(phase, &mut arrays))
        .collect::<Result<Vec<_>, _>>()?;
    if !arrays.is_empty() {
        return Err(invalid_record(
            "manifest contains arrays that are not referenced by the project".to_owned(),
        ));
    }
    let project = ProjectRecord {
        project_id: RecordId::new(wire.project_id).map_err(PersistenceError::Domain)?,
        revision: wire.revision,
        name: wire.name,
        histograms,
        phases,
        metadata: wire.metadata,
    };
    project.validate().map_err(PersistenceError::Domain)?;
    Ok(project)
}

fn decode_histogram(
    wire: WireHistogram,
    arrays: &mut BTreeMap<String, ArrayData>,
) -> Result<HistogramRecord, PersistenceError> {
    let x_deg = take_f64(arrays, &wire.pattern.x_deg, None)?;
    let count = x_deg.len();
    let observed_y = wire
        .pattern
        .observed_y
        .map(|reference| take_f64(arrays, &reference, Some(&[count])))
        .transpose()?;
    let uncertainty = wire
        .pattern
        .uncertainty
        .map(|reference| take_f64(arrays, &reference, Some(&[count])))
        .transpose()?;
    let mask = wire
        .pattern
        .mask
        .map(|reference| take_bool(arrays, &reference, &[count]))
        .transpose()?;
    let background_y = take_f64(arrays, &wire.pattern.background_y, Some(&[count]))?;
    Ok(HistogramRecord {
        histogram_id: RecordId::new(wire.histogram_id).map_err(PersistenceError::Domain)?,
        name: wire.name,
        pattern: PatternRecord::new(x_deg, observed_y, uncertainty, mask, Some(background_y))
            .map_err(PersistenceError::Domain)?,
        experiment: decode_experiment(wire.experiment)?,
        phase_ids: wire
            .phase_ids
            .into_iter()
            .map(|value| RecordId::new(value).map_err(PersistenceError::Domain))
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn decode_experiment(wire: WireExperiment) -> Result<ExperimentRecord, PersistenceError> {
    let instrument = ConstantWavelengthInstrument {
        wavelength_angstrom: wire.instrument.wavelength_angstrom,
        u_deg2: wire.instrument.u_deg2,
        v_deg2: wire.instrument.v_deg2,
        w_deg2: wire.instrument.w_deg2,
        x_deg: wire.instrument.x_deg,
        y_deg: wire.instrument.y_deg,
    };
    let radiation = match wire.radiation {
        WireRadiation::Monochromatic {
            probe,
            wavelength_angstrom,
        } => RadiationDefinition::Monochromatic {
            probe: decode_probe(probe),
            wavelength_angstrom,
        },
        WireRadiation::FixedSpectrum {
            probe,
            wavelengths_angstrom,
            relative_intensities,
        } => RadiationDefinition::FixedSpectrum {
            probe: decode_probe(probe),
            spectrum: FixedWavelengthSpectrum::new(wavelengths_angstrom, relative_intensities)
                .map_err(PersistenceError::Domain)?,
        },
    };
    let position = wire.position_correction;
    ExperimentRecord::new(
        instrument,
        radiation,
        wire.axial_geometry.map(|geometry| FcjGeometry {
            sample_over_radius: geometry.sample_over_radius,
            detector_over_radius: geometry.detector_over_radius,
        }),
        MonochromaticPositionCorrection {
            zero_shift_deg: position.zero_shift_deg,
            bragg_brentano_mm: position.bragg_brentano_mm.map(|value| (value[0], value[1])),
            debye_scherrer_micrometre: position
                .debye_scherrer_micrometre
                .map(|value| (value[0], value[1], value[2])),
        },
    )
    .map_err(PersistenceError::Domain)
}

// Reconstruction mirrors the encoded structural definition and validates the
// completed live phase through `ProjectRecord::validate` before returning.
#[allow(clippy::too_many_lines)]
fn decode_phase(
    wire: WirePhase,
    arrays: &mut BTreeMap<String, ArrayData>,
) -> Result<StructuralPhaseRecord, PersistenceError> {
    let definition = wire.definition;
    let hkl_values = take_i32(arrays, &definition.hkl, None)?;
    if hkl_values.len() % 3 != 0 {
        return Err(invalid_record("hkl array shape is invalid".to_owned()));
    }
    let reflection_count = hkl_values.len() / 3;
    let hkl = hkl_values
        .chunks_exact(3)
        .map(|value| [value[0], value[1], value[2]])
        .collect();
    let multiplicity = take_u64(arrays, &definition.multiplicity, &[reflection_count])?
        .into_iter()
        .map(|value| {
            usize::try_from(value)
                .map_err(|_| invalid_record("multiplicity does not fit this platform".to_owned()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let site_count = definition.scattering_species.len();
    let fractional = take_f64(arrays, &definition.fractional_xyz, Some(&[site_count, 3]))?;
    let fractional_xyz = fractional
        .chunks_exact(3)
        .map(|value| [value[0], value[1], value[2]])
        .collect();
    let occupancy = take_f64(arrays, &definition.occupancy, Some(&[site_count]))?;
    let u_iso_angstrom2 = take_f64(arrays, &definition.u_iso_angstrom2, Some(&[site_count]))?;
    let anisotropic_mask = take_bool(arrays, &definition.anisotropic_mask, &[site_count])?;
    let u_aniso = take_f64(
        arrays,
        &definition.u_aniso_cif_angstrom2,
        Some(&[site_count, 6]),
    )?;
    let u_aniso_cif_angstrom2 = u_aniso
        .chunks_exact(6)
        .map(|value| [value[0], value[1], value[2], value[3], value[4], value[5]])
        .collect();
    let scattering_real_offset = definition
        .scattering_real_offset
        .map(|reference| take_f64(arrays, &reference, Some(&[site_count])))
        .transpose()?
        .unwrap_or_default();
    let scattering_imag_offset = definition
        .scattering_imag_offset
        .map(|reference| take_f64(arrays, &reference, Some(&[site_count])))
        .transpose()?
        .unwrap_or_default();
    let operations = definition
        .operations
        .into_iter()
        .map(|operation| {
            let translation = operation
                .translation
                .map(|value| Rational::new(value[0], value[1]))
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;
            let translation: [Rational; 3] = translation.try_into().map_err(|_| {
                phasesmith_engine::crystallography::SymmetryError::ArithmeticOverflow
            })?;
            SymmetryOperation::new(operation.rotation, translation)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| invalid_record(format!("invalid symmetry operation: {error}")))?;
    let space_group = SpaceGroup::new(operations)
        .map_err(|error| invalid_record(format!("invalid space group: {error}")))?;
    let cell = definition.cell;
    let native_definition = StructuralPhaseDefinition {
        cell: UnitCell {
            a_angstrom: cell[0],
            b_angstrom: cell[1],
            c_angstrom: cell[2],
            alpha_deg: cell[3],
            beta_deg: cell[4],
            gamma_deg: cell[5],
        },
        space_group,
        hkl,
        multiplicity,
        fractional_xyz,
        occupancy,
        u_iso_angstrom2,
        anisotropic_mask,
        u_aniso_cif_angstrom2,
        scattering_species: definition.scattering_species,
        scattering_real_offset,
        scattering_imag_offset,
        scale: definition.scale,
        coordinate_tolerance: definition.coordinate_tolerance,
        scattering_model: match definition.scattering_model {
            WireScatteringModel::XrayNonResonant => BuiltInScatteringModel::XrayNonResonant,
            WireScatteringModel::NeutronNuclear => BuiltInScatteringModel::NeutronNuclear,
        },
        correction_model: decode_correction(definition.correction_model),
    };
    Ok(StructuralPhaseRecord {
        phase_id: RecordId::new(wire.phase_id).map_err(PersistenceError::Domain)?,
        name: wire.name,
        definition: native_definition,
        required_providers: wire
            .required_providers
            .into_iter()
            .map(|requirement| {
                ProviderRequirement::new(requirement.provider_id, requirement.provider_version)
                    .map_err(PersistenceError::Domain)
            })
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn decode_probe(probe: WireProbe) -> RadiationProbe {
    match probe {
        WireProbe::Xray => RadiationProbe::Xray,
        WireProbe::Neutron => RadiationProbe::Neutron,
    }
}

fn decode_correction(model: WireCorrectionModel) -> IntegratedIntensityCorrectionModel {
    match model {
        WireCorrectionModel::Neutral => IntegratedIntensityCorrectionModel::Neutral,
        WireCorrectionModel::BraggBrentanoUnpolarizedLp {
            wavelength_angstrom,
        } => IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp {
            wavelength_angstrom,
        },
        WireCorrectionModel::BraggBrentanoPolarizedLp {
            wavelength_angstrom,
            polarization,
        } => IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
            wavelength_angstrom,
            polarization,
        },
        WireCorrectionModel::ConstantWavelengthNeutronLorentz {
            wavelength_angstrom,
        } => IntegratedIntensityCorrectionModel::ConstantWavelengthNeutronLorentz {
            wavelength_angstrom,
        },
    }
}

fn take_array(
    arrays: &mut BTreeMap<String, ArrayData>,
    reference: &ArrayReference,
    expected_shape: Option<&[usize]>,
) -> Result<ArrayData, PersistenceError> {
    let array = arrays.remove(&reference.array).ok_or_else(|| {
        invalid_record(format!("referenced array {:?} is missing", reference.array))
    })?;
    if expected_shape.is_some_and(|shape| array.shape() != shape) {
        return Err(invalid_record(format!(
            "array {:?} has an unexpected shape",
            reference.array
        )));
    }
    Ok(array)
}

fn take_f64(
    arrays: &mut BTreeMap<String, ArrayData>,
    reference: &ArrayReference,
    expected_shape: Option<&[usize]>,
) -> Result<Vec<f64>, PersistenceError> {
    take_array(arrays, reference, expected_shape)?.into_f64()
}

fn take_i32(
    arrays: &mut BTreeMap<String, ArrayData>,
    reference: &ArrayReference,
    expected_shape: Option<&[usize]>,
) -> Result<Vec<i32>, PersistenceError> {
    take_array(arrays, reference, expected_shape)?.into_i32()
}

fn take_u64(
    arrays: &mut BTreeMap<String, ArrayData>,
    reference: &ArrayReference,
    expected_shape: &[usize],
) -> Result<Vec<u64>, PersistenceError> {
    take_array(arrays, reference, Some(expected_shape))?.into_u64()
}

fn take_bool(
    arrays: &mut BTreeMap<String, ArrayData>,
    reference: &ArrayReference,
    expected_shape: &[usize],
) -> Result<Vec<bool>, PersistenceError> {
    take_array(arrays, reference, Some(expected_shape))?.into_bool()
}

fn invalid_record(message: String) -> PersistenceError {
    PersistenceError::InvalidRecord { message }
}
