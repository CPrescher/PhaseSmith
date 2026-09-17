//! Versioned standalone Pawley project codec, shared by Rust and Python.
use phasesmith_core::{ConstantWavelengthInstrument, FcjGeometry};
use phasesmith_crystallography::{Rational, SpaceGroup, SymmetryOperation, UnitCell};
use phasesmith_model::PatternRecord;
use phasesmith_workflows::{
    AffineConstraint, BackgroundModel, ChebyshevBackground, CompositeBackground, Constraint,
    DifferentiableBackground, FixedConstraint, LatticeBounds, LatticeParameterization,
    LatticeReflectionDomain, LinearConstraint, LinearTerm, ParameterBounds, ParameterKey,
    ParameterSet, ParameterSpec, PawleyCheckpoint, PawleyError, PawleyInput, PawleyOptions,
    PawleyPhase, PointBackground, PolynomialBackground, pawley_parameters,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
fn err(e: impl std::fmt::Display) -> PawleyError {
    PawleyError(e.to_string())
}
/// A complete native Pawley analysis and optional accepted restart state.
#[derive(Clone, Debug, PartialEq)]
pub struct PawleyProject {
    /// Scientific input, including full observations and reflection topology.
    pub input: PawleyInput,
    /// Numerical controls.
    pub options: PawleyOptions,
    /// Accepted restart state.
    pub checkpoint: Option<PawleyCheckpoint>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireProject {
    format: String,
    version: u32,
    input: WireInput,
    options: WireOptions,
    checkpoint: Option<WireCheckpoint>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireInput {
    x_deg: Vec<f64>,
    observed_y: Option<Vec<f64>>,
    uncertainty: Option<Vec<f64>>,
    mask: Option<Vec<bool>>,
    background_y: Vec<f64>,
    instrument: [f64; 6],
    axial: Option<[f64; 2]>,
    phases: Vec<WirePhase>,
    background: Option<WireBackground>,
    signed_intensities: bool,
    parameters: Option<Vec<WireSpec>>,
    constraints: Vec<WireConstraint>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePhase {
    id: String,
    reflection_ids: Vec<String>,
    two_theta_deg: Vec<f64>,
    intensities: Vec<f64>,
    hkl: Vec<[i32; 3]>,
    lattice: Option<WireDomain>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDomain {
    cell: [f64; 6],
    operations: Vec<WireOperation>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    wavelength_angstrom: f64,
    visible_two_theta_deg: [f64; 2],
    initial_intensity: f64,
    merge_friedel: bool,
    max_candidates: usize,
    guard_scale: f64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireOperation {
    rotation: [[i32; 3]; 3],
    translation: [[i64; 2]; 3],
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSpec {
    key: [String; 3],
    value: f64,
    unit: String,
    lower: Option<f64>,
    upper: Option<f64>,
    scale: f64,
    refine: bool,
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum WireConstraint {
    Fixed {
        target: [String; 3],
        value: f64,
    },
    Affine {
        target: [String; 3],
        source: [String; 3],
        multiplier: f64,
        offset: f64,
    },
    Linear {
        target: [String; 3],
        terms: Vec<([String; 3], f64)>,
        offset: f64,
    },
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum WireBackground {
    Polynomial {
        id: String,
        coefficients: Vec<f64>,
    },
    Chebyshev {
        id: String,
        coefficients: Vec<f64>,
        domain_deg: [f64; 2],
    },
    Point {
        id: String,
        knot_x: Vec<f64>,
        values: Vec<f64>,
    },
    Composite {
        id: String,
        components: Vec<Self>,
    },
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireOptions {
    #[serde(default = "dense_solver", skip_serializing_if = "is_dense")]
    solver: String,
    #[serde(
        default = "linear_tolerance",
        skip_serializing_if = "default_linear_tolerance"
    )]
    linear_tolerance: f64,
    #[serde(
        default = "linear_iterations",
        skip_serializing_if = "default_linear_iterations"
    )]
    max_linear_iterations: usize,
    support_fwhm: f64,
    use_uncertainty: bool,
    max_elements: usize,
    rank_tolerance: f64,
    tolerance: f64,
    damping: f64,
    max_active_iterations: usize,
}
#[allow(clippy::trivially_copy_pass_by_ref)] // Serde predicate signature.
fn is_false(value: &bool) -> bool {
    !value
}
fn dense_solver() -> String {
    "dense".into()
}
fn is_dense(v: &str) -> bool {
    v == "dense"
}
const fn linear_tolerance() -> f64 {
    1e-11
}
#[allow(clippy::trivially_copy_pass_by_ref)] // Serde predicate signature.
fn default_linear_tolerance(v: &f64) -> bool {
    v.to_bits() == linear_tolerance().to_bits()
}
const fn linear_iterations() -> usize {
    4000
}
#[allow(clippy::trivially_copy_pass_by_ref)] // Serde predicate signature.
fn default_linear_iterations(v: &usize) -> bool {
    *v == linear_iterations()
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCheckpoint {
    #[serde(default, skip_serializing_if = "is_false")]
    support_local: bool,
    request_sha256: String,
    linear_initialized: bool,
    free: Vec<f64>,
    chi_square_history: Vec<f64>,
    damping: f64,
}
fn key(k: &ParameterKey) -> [String; 3] {
    [k.module().into(), k.owner_id().into(), k.name().into()]
}
fn decode_key(k: [String; 3]) -> Result<ParameterKey, PawleyError> {
    {
        let [module, owner, name] = k;
        ParameterKey::new(module, owner, name).map_err(err)
    }
}
fn encode_background(b: &BackgroundModel) -> Result<WireBackground, PawleyError> {
    let id = b.background_id().to_owned();
    let coefficients = b.coefficients();
    Ok(match b {
        BackgroundModel::Polynomial(_) => WireBackground::Polynomial { id, coefficients },
        BackgroundModel::Chebyshev(v) => WireBackground::Chebyshev {
            id,
            coefficients,
            domain_deg: v.domain_deg(),
        },
        BackgroundModel::Point(v) => WireBackground::Point {
            id,
            knot_x: v.knot_x().to_vec(),
            values: coefficients,
        },
        BackgroundModel::Composite(v) => WireBackground::Composite {
            id,
            components: v
                .components()
                .iter()
                .map(encode_background)
                .collect::<Result<_, _>>()?,
        },
        BackgroundModel::Amorphous(_) => {
            return Err(err("nonlinear Pawley background is unsupported"));
        }
    })
}
fn decode_background(b: WireBackground) -> Result<BackgroundModel, PawleyError> {
    Ok(match b {
        WireBackground::Polynomial { id, coefficients } => {
            BackgroundModel::Polynomial(PolynomialBackground::new(id, coefficients).map_err(err)?)
        }
        WireBackground::Chebyshev {
            id,
            coefficients,
            domain_deg,
        } => BackgroundModel::Chebyshev(
            ChebyshevBackground::new(id, coefficients, domain_deg).map_err(err)?,
        ),
        WireBackground::Point { id, knot_x, values } => {
            BackgroundModel::Point(PointBackground::new(id, knot_x, values).map_err(err)?)
        }
        WireBackground::Composite { id, components } => BackgroundModel::Composite(
            CompositeBackground::new(
                id,
                components
                    .into_iter()
                    .map(decode_background)
                    .collect::<Result<_, _>>()?,
            )
            .map_err(err)?,
        ),
    })
}
fn encode_domain(d: &LatticeReflectionDomain) -> WireDomain {
    let p = d.parameterization();
    let c = p.reference_cell();
    WireDomain {
        cell: [
            c.a_angstrom,
            c.b_angstrom,
            c.c_angstrom,
            c.alpha_deg,
            c.beta_deg,
            c.gamma_deg,
        ],
        operations: p
            .space_group()
            .operations()
            .iter()
            .map(|s| WireOperation {
                rotation: s.rotation(),
                translation: s.translation().map(|v| [v.numerator(), v.denominator()]),
            })
            .collect(),
        lower: d.bounds().lower().to_vec(),
        upper: d.bounds().upper().to_vec(),
        wavelength_angstrom: d.wavelength_angstrom(),
        visible_two_theta_deg: d.visible_two_theta_deg(),
        initial_intensity: d.initial_intensity(),
        merge_friedel: d.merge_friedel(),
        max_candidates: d.max_candidates(),
        guard_scale: d.guard_scale(),
    }
}
fn decode_domain(d: WireDomain) -> Result<LatticeReflectionDomain, PawleyError> {
    let c = d.cell;
    let cell = UnitCell {
        a_angstrom: c[0],
        b_angstrom: c[1],
        c_angstrom: c[2],
        alpha_deg: c[3],
        beta_deg: c[4],
        gamma_deg: c[5],
    };
    let ops = d
        .operations
        .into_iter()
        .map(|op| {
            let mut tr = [Rational::zero(); 3];
            for (i, v) in op.translation.iter().enumerate() {
                tr[i] = Rational::new(v[0], v[1]).map_err(err)?;
            }
            SymmetryOperation::new(op.rotation, tr).map_err(err)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let par =
        LatticeParameterization::new(SpaceGroup::new(ops).map_err(err)?, cell).map_err(err)?;
    let bounds = LatticeBounds::new(&par, d.lower, d.upper).map_err(err)?;
    LatticeReflectionDomain::new(
        par,
        bounds,
        d.wavelength_angstrom,
        d.visible_two_theta_deg,
        d.initial_intensity,
        d.merge_friedel,
        d.max_candidates,
        d.guard_scale,
    )
    .map_err(err)
}
fn encode_input(i: &PawleyInput) -> Result<WireInput, PawleyError> {
    i.validate()?;
    let v = i.instrument;
    Ok(WireInput {
        x_deg: i.pattern.x_deg.clone(),
        observed_y: i.pattern.observed_y.clone(),
        uncertainty: i.pattern.uncertainty.clone(),
        mask: i.pattern.mask.clone(),
        background_y: i.pattern.background_y.clone(),
        instrument: [
            v.wavelength_angstrom,
            v.u_deg2,
            v.v_deg2,
            v.w_deg2,
            v.x_deg,
            v.y_deg,
        ],
        axial: i
            .axial
            .map(|g| [g.sample_over_radius, g.detector_over_radius]),
        phases: i
            .phases
            .iter()
            .map(|p| WirePhase {
                id: p.id.clone(),
                reflection_ids: p.reflection_ids.clone(),
                two_theta_deg: p.two_theta_deg.clone(),
                intensities: p.intensities.clone(),
                hkl: p.hkl.clone(),
                lattice: p.lattice.as_ref().map(encode_domain),
            })
            .collect(),
        background: i.background.as_ref().map(encode_background).transpose()?,
        signed_intensities: i.signed_intensities,
        parameters: Some(
            i.parameters
                .specs()
                .iter()
                .map(|s| WireSpec {
                    key: key(s.key()),
                    value: s.value(),
                    unit: s.unit().into(),
                    lower: s.bounds().lower().is_finite().then_some(s.bounds().lower()),
                    upper: s.bounds().upper().is_finite().then_some(s.bounds().upper()),
                    scale: s.scale(),
                    refine: s.refine(),
                })
                .collect(),
        ),
        constraints: i
            .constraints
            .iter()
            .map(|c| match c {
                Constraint::Fixed(v) => WireConstraint::Fixed {
                    target: key(v.target()),
                    value: v.value(),
                },
                Constraint::Affine(v) => WireConstraint::Affine {
                    target: key(v.target()),
                    source: key(v.source()),
                    multiplier: v.multiplier(),
                    offset: v.offset(),
                },
                Constraint::Linear(v) => WireConstraint::Linear {
                    target: key(v.target()),
                    terms: v
                        .terms()
                        .iter()
                        .map(|t| (key(t.source()), t.coefficient()))
                        .collect(),
                    offset: v.offset(),
                },
            })
            .collect(),
    })
}
#[allow(clippy::too_many_lines)]
fn decode_input(i: WireInput) -> Result<PawleyInput, PawleyError> {
    let v = i.instrument;
    let instrument = ConstantWavelengthInstrument {
        wavelength_angstrom: v[0],
        u_deg2: v[1],
        v_deg2: v[2],
        w_deg2: v[3],
        x_deg: v[4],
        y_deg: v[5],
    };
    let phases = i
        .phases
        .into_iter()
        .map(|p| {
            let lattice = p.lattice.map(decode_domain).transpose()?;
            if p.reflection_ids.is_empty()
                && p.two_theta_deg.is_empty()
                && p.intensities.is_empty()
                && p.hkl.is_empty()
            {
                if let Some(domain) = lattice {
                    return PawleyPhase::from_domain(p.id, domain);
                }
            }
            Ok(PawleyPhase {
                id: p.id,
                reflection_ids: p.reflection_ids,
                two_theta_deg: p.two_theta_deg,
                intensities: p.intensities,
                hkl: p.hkl,
                lattice,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let background = i.background.map(decode_background).transpose()?;
    let parameters = if let Some(specs) = i.parameters {
        ParameterSet::new(
            specs
                .into_iter()
                .map(|s| {
                    ParameterSpec::new(
                        decode_key(s.key)?,
                        s.value,
                        s.unit,
                        ParameterBounds::new(
                            s.lower.unwrap_or(f64::NEG_INFINITY),
                            s.upper.unwrap_or(f64::INFINITY),
                        )
                        .map_err(err)?,
                        s.scale,
                        s.refine,
                    )
                    .map_err(err)
                })
                .collect::<Result<Vec<_>, PawleyError>>()?,
        )
        .map_err(err)?
    } else {
        pawley_parameters(
            &phases,
            instrument,
            background.as_ref(),
            i.signed_intensities,
        )?
    };
    let constraints = i
        .constraints
        .into_iter()
        .map(|c| {
            Ok(match c {
                WireConstraint::Fixed { target, value } => Constraint::Fixed(
                    FixedConstraint::new(decode_key(target)?, value).map_err(err)?,
                ),
                WireConstraint::Affine {
                    target,
                    source,
                    multiplier,
                    offset,
                } => Constraint::Affine(
                    AffineConstraint::new(
                        decode_key(target)?,
                        decode_key(source)?,
                        multiplier,
                        offset,
                    )
                    .map_err(err)?,
                ),
                WireConstraint::Linear {
                    target,
                    terms,
                    offset,
                } => Constraint::Linear(
                    LinearConstraint::new(
                        decode_key(target)?,
                        terms
                            .into_iter()
                            .map(|(k, c)| LinearTerm::new(decode_key(k)?, c).map_err(err))
                            .collect::<Result<_, _>>()?,
                        offset,
                    )
                    .map_err(err)?,
                ),
            })
        })
        .collect::<Result<Vec<_>, PawleyError>>()?;
    let input = PawleyInput {
        pattern: PatternRecord::new(
            i.x_deg,
            i.observed_y,
            i.uncertainty,
            i.mask,
            Some(i.background_y),
        )
        .map_err(err)?,
        instrument,
        axial: i.axial.map(|g| FcjGeometry {
            sample_over_radius: g[0],
            detector_over_radius: g[1],
        }),
        phases,
        background,
        signed_intensities: i.signed_intensities,
        parameters,
        constraints,
    };
    input.validate()?;
    Ok(input)
}
fn encode_options(o: &PawleyOptions) -> WireOptions {
    WireOptions {
        solver: match o.solver {
            phasesmith_workflows::PawleySolver::Dense => "dense",
            phasesmith_workflows::PawleySolver::MatrixFree => "matrix_free",
        }
        .into(),
        linear_tolerance: o.linear_tolerance,
        max_linear_iterations: o.max_linear_iterations,
        support_fwhm: o.support_fwhm,
        use_uncertainty: o.use_uncertainty,
        max_elements: o.max_elements,
        rank_tolerance: o.rank_tolerance,
        tolerance: o.tolerance,
        damping: o.damping,
        max_active_iterations: o.max_active_iterations,
    }
}
fn digest(i: &WireInput, o: &WireOptions) -> Result<String, PawleyError> {
    let bytes = serde_json::to_vec(&(i, o)).map_err(err)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
/// Encode a validated standalone project using deterministic finite JSON.
///
/// # Errors
/// Rejects inconsistent checkpoints or invalid records.
pub fn encode_pawley_project(project: &PawleyProject) -> Result<String, PawleyError> {
    project.options.validate()?;
    let input = encode_input(&project.input)?;
    let options = encode_options(&project.options);
    let checkpoint = project
        .checkpoint
        .as_ref()
        .map(|c| {
            if c.input != project.input
                || c.options != project.options
                || c.free.iter().any(|v| !v.is_finite())
                || c.chi_square_history.is_empty()
                || c.chi_square_history
                    .iter()
                    .any(|v| !v.is_finite() || *v < 0.0)
                || !c.damping.is_finite()
                || c.damping <= 0.0
            {
                return Err(err("invalid Pawley checkpoint"));
            }
            let transform = phasesmith_workflows::ConstraintTransform::new(
                project.input.parameters.clone(),
                project.input.constraints.clone(),
            )
            .map_err(err)?;
            transform.unpack(&c.free, false).map_err(err)?;
            if c.chi_square_history.windows(2).any(|v| v[1] >= v[0]) {
                return Err(err("Pawley accepted history must decrease"));
            }
            Ok(WireCheckpoint {
                support_local: c.support_local,
                request_sha256: digest(&input, &options)?,
                linear_initialized: c.linear_initialized,
                free: c.free.clone(),
                chi_square_history: c.chi_square_history.clone(),
                damping: c.damping,
            })
        })
        .transpose()?;
    serde_json::to_string(&WireProject {
        format: "phasesmith-pawley".into(),
        version: 2,
        input,
        options,
        checkpoint,
    })
    .map_err(err)
}
/// Decode a bounded project, reject unknown versions and verify checkpoint binding.
///
/// # Errors
/// Returns malformed, unsupported, oversized, or scientifically invalid record errors.
pub fn decode_pawley_project(text: &str, max_bytes: usize) -> Result<PawleyProject, PawleyError> {
    if text.len() > max_bytes {
        return Err(err("Pawley project byte limit exceeded"));
    }
    let w: WireProject = serde_json::from_str(text).map_err(err)?;
    if w.format != "phasesmith-pawley" || !matches!(w.version, 1 | 2) {
        return Err(err("unsupported Pawley project format/version"));
    }
    if w.version == 1
        && (!is_dense(&w.options.solver)
            || !default_linear_tolerance(&w.options.linear_tolerance)
            || !default_linear_iterations(&w.options.max_linear_iterations))
    {
        return Err(err(
            "version-1 Pawley projects require the original dense controls",
        ));
    }
    if let Some(cp) = &w.checkpoint {
        if w.version == 1 && cp.support_local {
            return Err(err(
                "version-1 checkpoints cannot select support-local optimization",
            ));
        }
        if cp.request_sha256 != digest(&w.input, &w.options)? {
            return Err(err("Pawley checkpoint request digest mismatch"));
        }
    }
    let input = decode_input(w.input)?;
    let o = w.options;
    let options = PawleyOptions {
        solver: match o.solver.as_str() {
            "dense" => phasesmith_workflows::PawleySolver::Dense,
            "matrix_free" => phasesmith_workflows::PawleySolver::MatrixFree,
            _ => return Err(err("unknown Pawley solver")),
        },
        linear_tolerance: o.linear_tolerance,
        max_linear_iterations: o.max_linear_iterations,
        support_fwhm: o.support_fwhm,
        use_uncertainty: o.use_uncertainty,
        max_elements: o.max_elements,
        rank_tolerance: o.rank_tolerance,
        tolerance: o.tolerance,
        damping: o.damping,
        max_active_iterations: o.max_active_iterations,
    };
    options.validate()?;
    let checkpoint = w.checkpoint.map(|c| PawleyCheckpoint {
        input: input.clone(),
        options: options.clone(),
        free: c.free,
        chi_square_history: c.chi_square_history,
        damping: c.damping,
        linear_initialized: c.linear_initialized,
        support_local: c.support_local,
    });
    let project = PawleyProject {
        input,
        options,
        checkpoint,
    };
    // Revalidation also forbids invalid finite state before it reaches callers.
    encode_pawley_project(&project)?;
    Ok(project)
}
/// Save a new project file without replacing an existing file.
///
/// # Errors
/// Rejects existing destinations, invalid state or filesystem errors.
pub fn save_pawley_project(
    path: impl AsRef<std::path::Path>,
    project: &PawleyProject,
) -> Result<(), PawleyError> {
    use std::io::Write;
    let bytes = encode_pawley_project(project)?;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(err)?;
    f.write_all(bytes.as_bytes()).map_err(err)?;
    f.sync_all().map_err(err)
}
/// Read and validate a project with a caller-selected byte ceiling.
///
/// # Errors
/// Rejects oversized files, invalid UTF-8 and invalid project content.
pub fn load_pawley_project(
    path: impl AsRef<std::path::Path>,
    max_bytes: usize,
) -> Result<PawleyProject, PawleyError> {
    use std::io::Read;
    let f = std::fs::File::open(path).map_err(err)?;
    if f.metadata().map_err(err)?.len() > max_bytes as u64 {
        return Err(err("Pawley project byte limit exceeded"));
    }
    let mut text = String::new();
    f.take((max_bytes as u64).saturating_add(1))
        .read_to_string(&mut text)
        .map_err(err)?;
    decode_pawley_project(&text, max_bytes)
}
