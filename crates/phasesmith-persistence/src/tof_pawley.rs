//! Versioned native TOF Pawley records, shared verbatim by Rust and Python.
use crate::pawley::{
    WireCheckpoint, WireConstraint, WireOperation, WireOptions, WireSpec, decode_constraints,
    decode_options, decode_parameters, encode_constraints, encode_options, encode_parameters,
};
use phasesmith_core::TofInstrument;
use phasesmith_crystallography::{Rational, SpaceGroup, SymmetryOperation, UnitCell};
use phasesmith_model::{RecordId, TofPatternRecord};
use phasesmith_workflows::{
    ConstraintTransform, LatticeBounds, LatticeParameterization, PawleyError, PawleyOptions,
    TofChebyshevBackground, TofPawleyBank, TofPawleyCheckpoint, TofPawleyInput, TofPawleyPhase,
    TofSharedLatticePhase, tof_pawley_parameters,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
fn err(e: impl std::fmt::Display) -> PawleyError {
    PawleyError(e.to_string())
}
/// Complete multi-bank scientific state and its optional atomic accepted checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct TofPawleyProject {
    /// Original request, including every bank's density normalization provenance.
    pub input: TofPawleyInput,
    /// Shared numerical controls.
    pub options: PawleyOptions,
    /// Last jointly accepted state.
    pub checkpoint: Option<TofPawleyCheckpoint>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireProject {
    format: String,
    version: u32,
    input: WireInput,
    options: WireOptions,
    checkpoint: Option<WireCheckpoint>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireInput {
    banks: Vec<WireBank>,
    shared_lattice: Vec<WireCell>,
    signed_intensities: bool,
    tail_log: f64,
    parameters: Option<Vec<WireSpec>>,
    constraints: Vec<WireConstraint>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireBank {
    id: String,
    tof_us: Vec<f64>,
    observed_y: Option<Vec<f64>>,
    uncertainty: Option<Vec<f64>>,
    mask: Option<Vec<bool>>,
    background_y: Vec<f64>,
    instrument: [f64; 15],
    phases: Vec<WirePhase>,
    background: Option<WireBackground>,
    normalization: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePhase {
    id: String,
    reflection_ids: Vec<String>,
    hkl: Vec<[i32; 3]>,
    d_spacing_angstrom: Vec<f64>,
    intensities: Vec<f64>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireBackground {
    id: String,
    coefficients: Vec<f64>,
    domain_us: [f64; 2],
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCell {
    id: String,
    cell: [f64; 6],
    reference_cell: [f64; 6],
    operations: Vec<WireOperation>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}
fn encode_input(i: &TofPawleyInput) -> Result<WireInput, PawleyError> {
    i.validate()?;
    Ok(WireInput {
        banks: i
            .banks
            .iter()
            .map(|b| WireBank {
                id: b.id.clone(),
                tof_us: b.pattern.tof_us.clone(),
                observed_y: b.pattern.observed_y.clone(),
                uncertainty: b.pattern.uncertainty.clone(),
                mask: b.pattern.mask.clone(),
                background_y: b.pattern.background_y.clone(),
                instrument: b.instrument.values(),
                phases: b
                    .phases
                    .iter()
                    .map(|p| WirePhase {
                        id: p.id.clone(),
                        reflection_ids: p.reflection_ids.clone(),
                        hkl: p.hkl.clone(),
                        d_spacing_angstrom: p.d_spacing_angstrom.clone(),
                        intensities: p.intensities.clone(),
                    })
                    .collect(),
                background: b.background.as_ref().map(|bg| WireBackground {
                    id: bg.background_id().as_str().into(),
                    coefficients: bg.coefficients().to_vec(),
                    domain_us: bg.domain_us(),
                }),
                normalization: b.normalization.clone(),
            })
            .collect(),
        shared_lattice: i
            .shared_lattice
            .iter()
            .map(|c| WireCell {
                id: c.phase_id().as_str().into(),
                reference_cell: {
                    let v = c.parameterization().reference_cell();
                    [
                        v.a_angstrom,
                        v.b_angstrom,
                        v.c_angstrom,
                        v.alpha_deg,
                        v.beta_deg,
                        v.gamma_deg,
                    ]
                },
                cell: {
                    let v = c.initial_cell();
                    [
                        v.a_angstrom,
                        v.b_angstrom,
                        v.c_angstrom,
                        v.alpha_deg,
                        v.beta_deg,
                        v.gamma_deg,
                    ]
                },
                operations: c
                    .parameterization()
                    .space_group()
                    .operations()
                    .iter()
                    .map(|op| WireOperation {
                        rotation: op.rotation(),
                        translation: op.translation().map(|r| [r.numerator(), r.denominator()]),
                    })
                    .collect(),
                lower: c.bounds().lower().to_vec(),
                upper: c.bounds().upper().to_vec(),
            })
            .collect(),
        signed_intensities: i.signed_intensities,
        tail_log: i.tail_log,
        parameters: Some(encode_parameters(&i.parameters)),
        constraints: encode_constraints(&i.constraints),
    })
}
fn decode_input(i: WireInput) -> Result<TofPawleyInput, PawleyError> {
    let banks = i
        .banks
        .into_iter()
        .map(|b| {
            Ok(TofPawleyBank {
                id: b.id,
                pattern: TofPatternRecord::new(
                    b.tof_us,
                    b.observed_y,
                    b.uncertainty,
                    b.mask,
                    Some(b.background_y),
                )
                .map_err(err)?,
                instrument: TofInstrument::from_values(b.instrument).map_err(err)?,
                phases: b
                    .phases
                    .into_iter()
                    .map(|p| TofPawleyPhase {
                        id: p.id,
                        reflection_ids: p.reflection_ids,
                        hkl: p.hkl,
                        d_spacing_angstrom: p.d_spacing_angstrom,
                        intensities: p.intensities,
                    })
                    .collect(),
                background: b
                    .background
                    .map(|bg| {
                        TofChebyshevBackground::new(
                            RecordId::new(bg.id).map_err(err)?,
                            bg.coefficients,
                            bg.domain_us,
                        )
                        .map_err(err)
                    })
                    .transpose()?,
                normalization: b.normalization,
            })
        })
        .collect::<Result<Vec<_>, PawleyError>>()?;
    let cells = i
        .shared_lattice
        .into_iter()
        .map(|c| {
            let v = c.cell;
            let cell = UnitCell {
                a_angstrom: v[0],
                b_angstrom: v[1],
                c_angstrom: v[2],
                alpha_deg: v[3],
                beta_deg: v[4],
                gamma_deg: v[5],
            };
            let ops = c
                .operations
                .into_iter()
                .map(|op| {
                    let mut tr = [Rational::zero(); 3];
                    for (j, v) in op.translation.iter().enumerate() {
                        tr[j] = Rational::new(v[0], v[1]).map_err(err)?;
                    }
                    SymmetryOperation::new(op.rotation, tr).map_err(err)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let v = c.reference_cell;
            let reference = UnitCell {
                a_angstrom: v[0],
                b_angstrom: v[1],
                c_angstrom: v[2],
                alpha_deg: v[3],
                beta_deg: v[4],
                gamma_deg: v[5],
            };
            let par = LatticeParameterization::new(SpaceGroup::new(ops).map_err(err)?, reference)
                .map_err(err)?;
            let bounds = LatticeBounds::new(&par, c.lower, c.upper).map_err(err)?;
            TofSharedLatticePhase::new(RecordId::new(c.id).map_err(err)?, par, bounds, cell)
                .map_err(err)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let parameters = i
        .parameters
        .map(decode_parameters)
        .transpose()?
        .map_or_else(
            || tof_pawley_parameters(&banks, &cells, i.signed_intensities),
            Ok,
        )?;
    let input = TofPawleyInput {
        banks,
        shared_lattice: cells,
        signed_intensities: i.signed_intensities,
        tail_log: i.tail_log,
        parameters,
        constraints: decode_constraints(i.constraints)?,
    };
    input.validate()?;
    Ok(input)
}
fn digest(input: &WireInput, options: &WireOptions) -> Result<String, PawleyError> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&(input, options)).map_err(err)?)
    ))
}
/// Encode deterministic finite JSON and verify the complete checkpoint binding.
/// # Errors
/// Rejects corrupt, stale or infeasible restart records.
pub fn encode_tof_pawley_project(project: &TofPawleyProject) -> Result<String, PawleyError> {
    project.options.validate()?;
    let input = encode_input(&project.input)?;
    let options = encode_options(&project.options);
    let checkpoint = project
        .checkpoint
        .as_ref()
        .map(|c| {
            if c.input != project.input
                || c.options != project.options
                || c.support_local
                || !c.damping.is_finite()
                || c.damping <= 0.0
                || c.chi_square_history.is_empty()
                || c.chi_square_history
                    .iter()
                    .any(|v| !v.is_finite() || *v < 0.0)
                || c.chi_square_history.windows(2).any(|v| v[1] >= v[0])
            {
                return Err(err("stale or corrupt TOF Pawley checkpoint"));
            }
            ConstraintTransform::new(
                project.input.parameters.clone(),
                project.input.constraints.clone(),
            )
            .map_err(err)?
            .unpack(&c.free, false)
            .map_err(err)?;
            Ok(WireCheckpoint {
                support_local: false,
                request_sha256: digest(&input, &options)?,
                linear_initialized: c.linear_initialized,
                free: c.free.clone(),
                chi_square_history: c.chi_square_history.clone(),
                damping: c.damping,
            })
        })
        .transpose()?;
    serde_json::to_string(&WireProject {
        format: "phasesmith-tof-pawley".into(),
        version: 1,
        input,
        options,
        checkpoint,
    })
    .map_err(err)
}
/// Decode a bounded native record and validate the identity of all banks atomically.
/// # Errors
/// Rejects future versions, unknown fields, corrupt digests and invalid scientific data.
pub fn decode_tof_pawley_project(
    text: &str,
    max_bytes: usize,
) -> Result<TofPawleyProject, PawleyError> {
    if text.len() > max_bytes {
        return Err(err("TOF Pawley project byte limit exceeded"));
    }
    let w: WireProject = serde_json::from_str(text).map_err(err)?;
    if w.format != "phasesmith-tof-pawley" || w.version != 1 {
        return Err(err("unsupported TOF Pawley format/version"));
    }
    if let Some(cp) = &w.checkpoint {
        if cp.request_sha256 != digest(&w.input, &w.options)? {
            return Err(err("TOF Pawley checkpoint request digest mismatch"));
        }
    }
    let input = decode_input(w.input)?;
    let options = decode_options(&w.options)?;
    let checkpoint = w.checkpoint.map(|c| TofPawleyCheckpoint {
        input: input.clone(),
        options: options.clone(),
        free: c.free,
        chi_square_history: c.chi_square_history,
        damping: c.damping,
        linear_initialized: c.linear_initialized,
        support_local: c.support_local,
    });
    let project = TofPawleyProject {
        input,
        options,
        checkpoint,
    };
    encode_tof_pawley_project(&project)?;
    Ok(project)
}
/// Save a complete joint record with an atomic, exclusive link into its destination.
/// # Errors
/// Rejects existing paths, malformed state and filesystem failures.
pub fn save_tof_pawley_project(
    path: impl AsRef<std::path::Path>,
    project: &TofPawleyProject,
) -> Result<(), PawleyError> {
    use std::io::Write;
    let text = encode_tof_pawley_project(project)?;
    let path = path.as_ref();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(err)?
        .as_nanos();
    let temporary = path.with_extension(format!("tof-pawley-{}-{nonce}.tmp", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(err)?;
    let result = (|| {
        file.write_all(text.as_bytes()).map_err(err)?;
        file.sync_all().map_err(err)?;
        std::fs::hard_link(&temporary, path).map_err(err)
    })();
    let _ = std::fs::remove_file(temporary);
    result
}
/// Load a joint record through a bounded reader.
/// # Errors
/// Rejects oversized files, invalid UTF-8 and invalid scientific state.
pub fn load_tof_pawley_project(
    path: impl AsRef<std::path::Path>,
    max_bytes: usize,
) -> Result<TofPawleyProject, PawleyError> {
    use std::io::Read;
    let file = std::fs::File::open(path).map_err(err)?;
    if file.metadata().map_err(err)?.len() > max_bytes as u64 {
        return Err(err("TOF Pawley project byte limit exceeded"));
    }
    let mut text = String::new();
    file.take((max_bytes as u64).saturating_add(1))
        .read_to_string(&mut text)
        .map_err(err)?;
    decode_tof_pawley_project(&text, max_bytes)
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireTofPawleyAnalysis {
    analysis_id: String,
    banks: Vec<WireBankAnalysis>,
    shared_lattice: Vec<WireCell>,
    signed_intensities: bool,
    tail_log: f64,
    parameters: Option<Vec<WireSpec>>,
    constraints: Vec<WireConstraint>,
    options: WireOptions,
    checkpoint: Option<WireCheckpoint>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireBankAnalysis {
    id: String,
    phases: Vec<WirePhase>,
    background: Option<WireBackground>,
    normalization: String,
}
fn bundle_error(e: impl std::fmt::Display) -> crate::PersistenceError {
    crate::PersistenceError::InvalidRecord {
        message: e.to_string(),
    }
}
pub(crate) fn encode_analyses(
    state: &phasesmith_workflows::TofPawleyProjectState,
) -> Result<Vec<WireTofPawleyAnalysis>, crate::PersistenceError> {
    state.validate().map_err(bundle_error)?;
    state
        .analyses
        .iter()
        .map(|a| {
            let text = encode_tof_pawley_project(&TofPawleyProject {
                input: a.input.clone(),
                options: a.options.clone(),
                checkpoint: a.checkpoint.clone(),
            })
            .map_err(bundle_error)?;
            let w: WireProject = serde_json::from_str(&text)?;
            Ok(WireTofPawleyAnalysis {
                analysis_id: a.analysis_id.as_str().into(),
                banks: w
                    .input
                    .banks
                    .into_iter()
                    .map(|b| WireBankAnalysis {
                        id: b.id,
                        phases: b.phases,
                        background: b.background,
                        normalization: b.normalization,
                    })
                    .collect(),
                shared_lattice: w.input.shared_lattice,
                signed_intensities: w.input.signed_intensities,
                tail_log: w.input.tail_log,
                parameters: w.input.parameters,
                constraints: w.input.constraints,
                options: w.options,
                checkpoint: w.checkpoint,
            })
        })
        .collect()
}
pub(crate) fn decode_state(
    project: phasesmith_model::ProjectRecord,
    records: Vec<WireTofPawleyAnalysis>,
    limits: crate::ProjectReadLimits,
) -> Result<phasesmith_workflows::TofPawleyProjectState, crate::PersistenceError> {
    if records.len() > limits.max_histograms {
        return Err(bundle_error("too many TOF Pawley analyses"));
    }
    let analyses = records
        .into_iter()
        .map(|a| {
            if a.banks.len() > limits.max_histograms {
                return Err(bundle_error("too many TOF Pawley banks"));
            }
            let banks = a
                .banks
                .into_iter()
                .map(|b| {
                    let h = project
                        .tof_histograms
                        .iter()
                        .find(|h| h.histogram_id.as_str() == b.id)
                        .ok_or_else(|| bundle_error("unknown TOF Pawley histogram owner"))?;
                    Ok(WireBank {
                        id: b.id,
                        tof_us: h.pattern.tof_us.clone(),
                        observed_y: h.pattern.observed_y.clone(),
                        uncertainty: h.pattern.uncertainty.clone(),
                        mask: h.pattern.mask.clone(),
                        background_y: h.pattern.background_y.clone(),
                        instrument: h.experiment.instrument.values(),
                        phases: b.phases,
                        background: b.background,
                        normalization: b.normalization,
                    })
                })
                .collect::<Result<Vec<_>, crate::PersistenceError>>()?;
            let text = serde_json::to_string(&WireProject {
                format: "phasesmith-tof-pawley".into(),
                version: 1,
                input: WireInput {
                    banks,
                    shared_lattice: a.shared_lattice,
                    signed_intensities: a.signed_intensities,
                    tail_log: a.tail_log,
                    parameters: a.parameters,
                    constraints: a.constraints,
                },
                options: a.options,
                checkpoint: a.checkpoint,
            })?;
            let p = decode_tof_pawley_project(&text, text.len()).map_err(bundle_error)?;
            Ok(phasesmith_workflows::TofPawleyAnalysis {
                analysis_id: RecordId::new(a.analysis_id).map_err(bundle_error)?,
                input: p.input,
                options: p.options,
                checkpoint: p.checkpoint,
            })
        })
        .collect::<Result<Vec<_>, crate::PersistenceError>>()?;
    let state = phasesmith_workflows::TofPawleyProjectState { project, analyses };
    state.validate().map_err(bundle_error)?;
    Ok(state)
}
