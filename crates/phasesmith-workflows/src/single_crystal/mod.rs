//! Typed interpreter-free coordination of original detector counts through structure search.
//! Kernels retain their own finite supports and analytical derivatives. Source/observer
//! traits carry no Python objects; file formats and optional grading stay in adapters.
#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
mod grains;
mod math;
mod measurement;
mod positions;
mod scaling;
pub use grains::{Grain, Indexed};
pub use math::{M3, V3};
pub use measurement::{Measurements, ProfileData};
use phasesmith_crystallography::{SpaceGroup, UnitCell};
use phasesmith_spots::{frame_input, frame_validity, reciprocal::Geometry};
pub use positions::Positions;
pub use scaling::Reduced;
use std::{fmt, time::Instant};
/// Workflow validation, numerical, I/O-adapter, cancellation or observer failure.
#[derive(Debug)]
pub struct Error(pub String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
impl From<phasesmith_spots::SpotError> for Error {
    fn from(e: phasesmith_spots::SpotError) -> Self {
        Self(e.to_string())
    }
}
impl From<frame_validity::Error> for Error {
    fn from(e: frame_validity::Error) -> Self {
        Self(e.to_string())
    }
}
impl From<phasesmith_solution::SolutionError> for Error {
    fn from(e: phasesmith_solution::SolutionError) -> Self {
        Self(e.to_string())
    }
}
/// Workflow result type.
pub type Result<T> = std::result::Result<T, Error>;
/// Read original oriented-independent u32 counts. Implementations must detect changed sources.
pub trait FrameSource {
    /// Original source shape: frames, rows, columns.
    fn shape(&self) -> [usize; 3];
    /// A new owned original frame; no masking or intensity correction.
    fn read(&mut self, index: usize) -> Result<Vec<u32>>;
}
/// Declared acquisition and independent source identity.
pub struct Acquisition {
    pub first: usize,
    pub last: usize,
    pub omega_zero: f64,
    pub omega_step: f64,
    pub exposure: f64,
    pub vertical_flip: bool,
    pub reserved_minimum: u32,
}
/// Supplied scientific priors; no source HKLs, atomic counts or atomic model.
pub struct Priors {
    pub cell: UnitCell,
    pub group: SpaceGroup,
    pub elements: Vec<String>,
    pub provenance: String,
}
/// Explicit resource limits and algorithm controls for the supported native workflow.
pub struct Options {
    pub acquisition: Acquisition,
    pub geometry: Geometry,
    pub priors: Priors,
    pub null_rotations: Vec<M3>,
    pub cancellation: phasesmith_solution::Cancellation,
    pub detection_workers: usize,
    pub profile_workers: usize,
    pub learning_workers: usize,
    pub frame_workers: usize,
    pub geometry_cache_bytes: usize,
    pub frame_payload_bytes: usize,
    pub max_samples: usize,
    pub search_seconds: f64,
    pub dense_memory_bytes: usize,
    pub search_trials: usize,
    pub search_cycles: usize,
    pub seed: u64,
}
/// Ordered stage observer; errors/cancellation stop before the next stage. Owned products
/// may be archived here without allowing an observer to alter numerical decisions.
pub trait Observer {
    fn stage(&mut self, _name: &str, _seconds: f64) -> Result<()> {
        Ok(())
    }
    fn masks(&mut self, _plan: &frame_input::Plan, _scales: &[f64]) -> Result<()> {
        Ok(())
    }
    fn positions(&mut self, _positions: &Positions) -> Result<()> {
        Ok(())
    }
    fn indexed(&mut self, _indexed: &Indexed) -> Result<()> {
        Ok(())
    }
    fn profiles(&mut self, _data: &Measurements) -> Result<()> {
        Ok(())
    }
    fn reduced(&mut self, _data: &Reduced) -> Result<()> {
        Ok(())
    }
    fn solution(
        &mut self,
        _solution: &phasesmith_solution::Solution,
        _search: &phasesmith_solution::SharedSearchSummary,
    ) -> Result<()> {
        Ok(())
    }
}
/// No-output observer for library consumers.
pub struct NoObserver;
impl Observer for NoObserver {}
/// Complete native products and explicit source/mask provenance.
pub struct Output {
    pub positions: Positions,
    pub indexed: Indexed,
    pub measurements: Measurements,
    pub reduced: Reduced,
    pub solution: phasesmith_solution::Solution,
    pub search: phasesmith_solution::SharedSearchSummary,
    pub seconds: f64,
}
impl Options {
    fn check(&self, shape: [usize; 3]) -> Result<()> {
        let a = &self.acquisition;
        if shape.contains(&0)
            || a.first > a.last
            || a.last >= shape[0]
            || a.last - a.first + 1 > 1_000_000
            || !a.omega_zero.is_finite()
            || !a.omega_step.is_finite()
            || a.omega_step <= 0.
            || !a.exposure.is_finite()
            || a.exposure <= 0.
            || a.exposure > a.omega_step + 1e-8
        {
            return Err(Error(
                "invalid native scan shape, selection or angular intervals".into(),
            ));
        }
        if [
            self.detection_workers,
            self.profile_workers,
            self.learning_workers,
            self.frame_workers,
        ]
        .iter()
        .any(|w| !(1..=32).contains(w))
            || self.max_samples == 0
            || self.max_samples > 20_000_000
            || self.frame_payload_bytes > 1024 * 1024 * 1024
            || self.geometry_cache_bytes > 1024 * 1024 * 1024
        {
            return Err(Error("invalid workflow resource limits".into()));
        }
        if self.priors.provenance.trim().is_empty()
            || self.null_rotations.len() != 64
            || self
                .null_rotations
                .iter()
                .flatten()
                .flatten()
                .any(|x| !x.is_finite())
        {
            return Err(Error(
                "explicit prior provenance and 64 predeclared null rotations required".into(),
            ));
        }
        self.geometry.convert(&[[0., 0., 0.]])?;
        Ok(())
    }
    fn checkpoint(&self) -> Result<()> {
        if self.cancellation.is_cancelled() {
            Err(Error("single-crystal workflow cancelled".into()))
        } else {
            Ok(())
        }
    }
}
pub(super) fn bounds(o: &Options) -> Vec<[f64; 2]> {
    let a = &o.acquisition;
    (a.first..=a.last)
        .map(|i| {
            let low = a.omega_zero + i as f64 * a.omega_step;
            [low, low + a.exposure]
        })
        .collect()
}
pub(super) fn image(p: &frame_input::Prepared, shape: [usize; 2]) -> Vec<f64> {
    let _ = shape;
    p.pixels.iter().map(|&v| f64::from(v)).collect()
}
pub(super) fn prepared(
    s: &mut impl FrameSource,
    plan: &frame_input::Plan,
    first: usize,
    i: usize,
) -> Result<frame_input::Prepared> {
    let r = plan.prepare_batch(
        vec![frame_input::Input {
            index: i,
            pixels: s.read(first + i)?,
        }],
        1,
        256 * 1024 * 1024,
    )?;
    r.into_iter()
        .next()
        .ok_or_else(|| Error("empty prepared batch".into()))?
        .map_err(Into::into)
}
/// Coordinate a complete original-frame workflow. Only its stage products reach observers;
/// no archived intensities or atomic references can be supplied through this interface.
/// # Errors
/// Invalid input/resources, source changes, failed scientific gates, cancellation or observer failure.
pub fn run(
    source: &mut impl FrameSource,
    options: &Options,
    observer: &mut impl Observer,
) -> Result<Output> {
    let start = Instant::now();
    options.check(source.shape())?;
    options.checkpoint()?;
    let shape = [source.shape()[1], source.shape()[2]];
    let angles = bounds(options);
    let frames = angles.len();
    let mut plan = frame_input::Plan {
        shape,
        frames,
        vertical_flip: options.acquisition.vertical_flip,
        reserved_minimum: options.acquisition.reserved_minimum,
        expected_hashes: vec![],
        blocks: vec![],
        dacs: vec![],
    };
    let clock = Instant::now();
    let coarse_shape = frame_validity::Shape {
        height: shape[0],
        width: shape[1],
        block: 16,
    };
    let mut coarse = Vec::new();
    let mut available = Vec::new();
    let mut hashes = Vec::new();
    for i in 0..frames {
        options.checkpoint()?;
        let p = prepared(source, &plan, options.acquisition.first, i)?;
        let data = image(&p, shape);
        let c = frame_validity::coarsen(&data, &p.excluded, coarse_shape)?;
        coarse.extend(c.values);
        available.push(c.available);
        hashes.push(p.digest);
    }
    let masks = frame_validity::estimate(
        &coarse,
        &available,
        coarse_shape,
        frame_validity::SeatOptions::default(),
    )?;
    plan.expected_hashes.push(hashes);
    plan.blocks.push(frame_input::BlockMasks {
        block: 16,
        excluded: masks.excluded,
        fixed: vec![],
    });
    plan.dacs.push(frame_input::DacMasks {
        geometry: options.geometry,
        bounds: angles.clone(),
        block: 16,
        options: frame_validity::DacOptions {
            axis_at_zero: [0., 0., 1.],
            half_opening_degrees: 35.,
        },
        fixed: vec![],
    });
    observer.masks(&plan, &masks.detection_scales)?;
    observer.stage("shadow_masks", clock.elapsed().as_secs_f64())?;
    options.checkpoint()?;
    let clock = Instant::now();
    let positions = positions::detect(source, &plan, options)?;
    observer.positions(&positions)?;
    observer.stage("detection_linking", clock.elapsed().as_secs_f64())?;
    options.checkpoint()?;
    let clock = Instant::now();
    let indexed = grains::index(&positions, options, shape)?;
    observer.indexed(&indexed)?;
    observer.stage("indexing_geometry", clock.elapsed().as_secs_f64())?;
    options.checkpoint()?;
    let clock = Instant::now();
    let measurements = measurement::measure(source, &plan, options, &indexed)?;
    observer.profiles(&measurements)?;
    observer.stage(
        "integration_profile_learning",
        clock.elapsed().as_secs_f64(),
    )?;
    options.checkpoint()?;
    let clock = Instant::now();
    let reduced = scaling::reduce(&measurements, options, &indexed.fit.state.geometry)?;
    observer.reduced(&reduced)?;
    observer.stage("profile_fitting_scaling", clock.elapsed().as_secs_f64())?;
    options.checkpoint()?;
    let clock = Instant::now();
    let (solution, search) = search(&reduced, options)?;
    observer.solution(&solution, &search)?;
    observer.stage("structure_search", clock.elapsed().as_secs_f64())?;
    Ok(Output {
        positions,
        indexed,
        measurements,
        reduced,
        solution,
        search,
        seconds: start.elapsed().as_secs_f64(),
    })
}
fn search(
    r: &Reduced,
    o: &Options,
) -> Result<(
    phasesmith_solution::Solution,
    phasesmith_solution::SharedSearchSummary,
)> {
    use phasesmith_solution::*;
    let observations: Vec<_> = r
        .hkl
        .iter()
        .zip(&r.values)
        .zip(&r.use_mask)
        .filter_map(|((&h, &v), &ok)| {
            ok.then_some(Observation {
                hkl: h,
                intensity: v[0],
                sigma: v[1],
            })
        })
        .collect();
    let memory = estimate_solution_memory(
        &observations,
        o.priors.cell,
        &o.priors.group,
        DenseMemoryBudget::new(o.dense_memory_bytes)?,
    )?;
    let options = SolveOptions {
        seed: o.seed,
        trials: o.search_trials,
        cycles: o.search_cycles,
        phasing: PhasingMethod::DualSpace,
        max_seconds: o.search_seconds,
        r1_target: 0.06,
        completion: false,
        dense_memory_budget: DenseMemoryBudget::new(o.dense_memory_bytes)?,
        cancellation: Some(o.cancellation.clone()),
        delta_sigma: 1.0,
        weak_fraction: 0.2,
        cleanup_cycles: 10,
        polish_iterations: 50,
        assignment_rounds: 4,
        grid_shape: None,
        refinement: Some(AdpMode::Anisotropic),
        symmetry_tolerance_angstrom: 0.15,
        symmetry_iterations: 80,
    };
    let shared = SharedSearchOptions {
        maximum_atoms: memory.maximum_atoms,
        workers: 1,
        joint_assignment_refinement: true,
        cached_repair_evaluation: true,
        beam_width: 4,
        rounds: 8,
        staged_refinement: true,
        screening_iterations: 8,
        refinement_shortlist: 3,
        integrated_peak_density: true,
        peak_density_radius_angstrom: 0.7,
        dual_peak_sigma: 2.0,
        dual_peak_cap: 128,
        dual_checkpoints: true,
        cached_refinement: true,
        symmetry_recycling: true,
        recycling_cycles: 10,
        tangent_refinement: true,
        tangent_in_recycling: true,
        adaptive_counts: true,
        wilson_normalization: true,
        phase_quality_selection: true,
        phase_first_selection: true,
        calculated_shell_normalization: true,
        phase_origin_selection: true,
        adaptive_checkpoints: false,
        score_only_evaluation: true,
        reusable_fourier_workspaces: true,
        work_sliced_phasing: true,
        isotropic_assignment_refinement: true,
        filtered_peak_search: true,
        batched_fft: true,
        real_fft: true,
        fused_phase_workspace: true,
        independent_origin_sampling: false,
        chemical_assignment: true,
        iterative_refinement: true,
        cached_normal_refinement: true,
        multiplicity_weighted_refinement: true,
        ambiguous_assignments: true,
        cached_assignment_refinement: true,
        specialized_isotropic_refinement: true,
        streamed_peak_search: true,
        reuse_normal_matrices: true,
        short_phase_starts: true,
        blocked_normal_matrices: true,
        phase_plateau_stopping: true,
        assignment_convergence_stopping: true,
        late_phase_plateau_stopping: true,
        model_consensus_stopping: true,
        missing_reflections: false,
        resolution_extension: 1.0,
    };
    let solution = solve_elements(
        &observations,
        o.priors.cell,
        &o.priors.group,
        &o.priors.elements,
        &options,
        &shared,
    )?;
    let search = solution
        .shared_search
        .clone()
        .ok_or_else(|| Error("missing native search diagnostics".into()))?;
    Ok((solution, search))
}
