//! Typed interpreter-free coordination of original detector counts through structure search.
//! Kernels retain their own finite supports and analytical derivatives. Source/observer
//! traits carry no Python objects; file formats and optional grading stay in adapters.
// Native v1 admits <=1e8 pixels, <=1e5 indexing points and bounded integer HKLs.
// Coordinate-to-index casts follow explicit footprint/range checks; integer casts
// preserve the public kernels' u32/i32 storage after admission.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names
)]
mod grains;
mod math;
mod measurement;
mod positions;
mod scaling;
mod unknown;
pub use grains::{Competitor, Grain, Indexed};
pub use math::{M3, V3};
pub use measurement::{Measurements, ProfileData};
use phasesmith_crystallography::{SpaceGroup, UnitCell};
use phasesmith_spots::{frame_input, frame_validity, reciprocal::Geometry};
pub use positions::Positions;
pub use scaling::Reduced;
use std::{fmt, time::Instant};
pub use unknown::{DiscoveryOptions, DiscoveryReport, GroupAttempt, PhaseScreenOptions, SearchRun};
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
    /// # Errors
    /// Return an I/O or changed-source error without substituting another frame.
    fn read(&mut self, index: usize) -> Result<Vec<u32>>;
}
/// Declared acquisition and independent source identity.
#[derive(Clone)]
pub struct Acquisition {
    /// Inclusive first original source frame index.
    pub first: usize,
    /// Inclusive last original source frame index.
    pub last: usize,
    /// Exposure start at original frame zero, in degrees.
    pub omega_zero: f64,
    /// Positive original-frame spindle increment, in degrees.
    pub omega_step: f64,
    /// Positive full exposure width, in degrees.
    pub exposure: f64,
    /// Apply the declared detector vertical reversal exactly once.
    pub vertical_flip: bool,
    /// Inclusive u32 detector sentinel threshold; excluded counts remain unchanged.
    pub reserved_minimum: u32,
}
/// Supplied scientific priors; no source HKLs, atomic counts or atomic model.
#[derive(Clone)]
pub struct Priors {
    /// Explicit supplied unit cell, in angstroms and degrees.
    pub cell: UnitCell,
    /// Exact supplied symmetry, or None for measured space-group discovery.
    pub group: Option<SpaceGroup>,
    /// Declared neutral element symbols; no atom counts.
    pub elements: Vec<String>,
    /// Nonempty description of the source and status of supplied priors.
    pub provenance: String,
}
/// Explicit original-image mask parameters; they are scientific inputs, not fitted controls.
#[derive(Clone, Copy)]
pub struct ShadowOptions {
    /// Illumination block side in detector pixels.
    pub illumination_block: usize,
    /// Original-count illumination-ratio estimator controls.
    pub illumination: frame_validity::SeatOptions,
    /// DAC block side in detector pixels; complete footprint/exposure is tested.
    pub aperture_block: usize,
    /// Explicit assumed/sample-known symmetric DAC cone in the sample frame.
    pub aperture: frame_validity::DacOptions,
}
/// Explicit resource limits and algorithm controls for the supported native workflow.
#[derive(Clone)]
pub struct Options {
    /// Required bounded catalogue/search policy when exact symmetry is unknown.
    pub discovery: Option<DiscoveryOptions>,
    /// Declared source selection, orientation and angular intervals.
    pub acquisition: Acquisition,
    /// Initial affine detector calibration in a single laboratory frame.
    pub geometry: Geometry,
    /// Supplied cell, symmetry and element priors with provenance.
    pub priors: Priors,
    /// Exactly 64 predeclared rotation matrices for held-out null checks.
    pub null_rotations: Vec<M3>,
    /// Explicit illumination and whole-exposure aperture assumptions.
    pub shadows: ShadowOptions,
    /// Original-image coarse-frame worker budget, 1 through 32.
    pub mask_workers: usize,
    /// Native original-count preparation workers, 1 through 32.
    pub preparation_workers: usize,
    /// Bounded source snapshots per preparation call, 1 through 32.
    pub preparation_batch: usize,
    /// Maximum owned counts/mask/f64 payload across each frame stage, in bytes.
    pub max_in_flight_frame_bytes: usize,
    /// Cooperative stop token checked between bounded stages and numerical work.
    pub cancellation: phasesmith_solution::Cancellation,
    /// Maximum source-ordered detection tasks in flight, 1 through 32.
    pub detection_workers: usize,
    /// Private native profile-fitting worker budget, 1 through 32.
    pub profile_workers: usize,
    /// Private profile-learning worker budget, 1 through 32.
    pub learning_workers: usize,
    /// Maximum source-ordered integration-frame tasks in flight, 1 through 32.
    pub frame_workers: usize,
    /// Additional profile geometry-cache cap in bytes; zero disables caching.
    pub geometry_cache_bytes: usize,
    /// Original-frame preparation payload cap in bytes.
    pub frame_payload_bytes: usize,
    /// Maximum retained profile samples; at most twenty million.
    pub max_samples: usize,
    /// Positive complete native structure-search time budget, in seconds.
    pub search_seconds: f64,
    /// Per controlled search buffer byte limit; independent of profile caches/RSS.
    pub dense_memory_bytes: usize,
    /// Maximum independent structure phase starts.
    pub search_trials: usize,
    /// Maximum phasing cycles per start.
    pub search_cycles: usize,
    /// Deterministic structure-search seed.
    pub seed: u64,
}
/// Ordered stage observer; errors/cancellation stop before the next stage. Owned products
/// may be archived here without allowing an observer to alter numerical decisions.
pub trait Observer {
    /// Observe measured symmetry evidence before scaling or before candidate searches.
    /// # Errors
    /// Return an output failure to stop the workflow.
    fn symmetry(
        &mut self,
        _name: &str,
        _assessment: &crate::space_group_search::Assessment,
        _hypotheses: &[crate::space_group_search::Hypothesis],
    ) -> Result<()> {
        Ok(())
    }
    /// Observe all completed/failed/unattempted space-group searches.
    /// # Errors
    /// Return an output failure to stop the workflow.
    fn group_search(&mut self, _report: &DiscoveryReport) -> Result<()> {
        Ok(())
    }

    /// Observe a completed stage and its wall seconds, including synchronous output.
    /// # Errors
    /// Return an observer failure to stop before the next stage.
    fn stage(&mut self, _name: &str, _seconds: f64) -> Result<()> {
        Ok(())
    }
    /// Archive the frozen source-hash and composed block-mask plan.
    /// # Errors
    /// Return an observer failure to stop before the next stage.
    fn masks(&mut self, _plan: &frame_input::Plan, _scales: &[f64]) -> Result<()> {
        Ok(())
    }
    /// Observe original linked tracks after shared-mask support checks.
    /// # Errors
    /// Return an observer failure to stop before the next stage.
    fn positions(&mut self, _positions: &Positions) -> Result<()> {
        Ok(())
    }
    /// Observe frozen discovery, calibration and withheld-position evidence.
    /// # Errors
    /// Return an observer failure to stop before the next stage.
    fn indexed(&mut self, _indexed: &Indexed) -> Result<()> {
        Ok(())
    }
    /// Archive original samples and the independently learned profile bank.
    /// # Errors
    /// Return an observer failure to stop before the next stage.
    fn profiles(&mut self, _data: &Measurements) -> Result<()> {
        Ok(())
    }
    /// Observe signed corrected intensities before any atomic-model access.
    /// # Errors
    /// Return an observer failure to stop before the next stage.
    fn reduced(&mut self, _data: &Reduced) -> Result<()> {
        Ok(())
    }
    /// Save completed search candidates and their actual method audits.
    /// # Errors
    /// Return an observer failure to stop before the next stage.
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
    /// Explicit unknown-symmetry evidence and alternatives; absent for a supplied group.
    pub discovery: Option<DiscoveryReport>,
    /// Original linked source positions with full quality flags.
    pub positions: Positions,
    /// Frozen lattice, geometry and withheld-data evidence.
    pub indexed: Indexed,
    /// Original finite pixel observations and independent learned profile bank.
    pub measurements: Measurements,
    /// Signed corrected intensities, conditional uncertainties and selection flags.
    pub reduced: Reduced,
    /// Complete native bounded structure-search result.
    pub solution: phasesmith_solution::Solution,
    /// Actual count-free search audit.
    pub search: phasesmith_solution::SharedSearchSummary,
    /// Measured wall seconds for the complete coordinator, including synchronous observers.
    pub seconds: f64,
}
impl Options {
    pub(super) fn partition_group(&self) -> SpaceGroup {
        self.priors
            .group
            .as_ref()
            .filter(|_| self.discovery.is_none())
            .cloned()
            .unwrap_or_else(|| {
                let rotations: Vec<_> = [-1, 1]
                    .into_iter()
                    .flat_map(|a| {
                        [-1, 1].into_iter().flat_map(move |b| {
                            [-1, 1]
                                .into_iter()
                                .map(move |c| [[a, 0, 0], [0, b, 0], [0, 0, c]])
                        })
                    })
                    .collect();
                crate::space_group_search::rotational_group(&rotations)
                    .expect("orthorhombic metric orbit")
            })
    }

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
        self.check_resources(shape)?;
        self.check_priors()?;
        if self.priors.group.is_none() != self.discovery.is_some() {
            return Err(Error(
                "supply either exact symmetry or a discovery catalogue, never both".into(),
            ));
        }
        if let Some(d) = &self.discovery {
            d.check()?;
            crate::space_group_search::validate_catalogue(
                self.priors.cell,
                &d.hypotheses,
                &d.assessment,
            )?;
        }
        if self.priors.elements.is_empty()
            || self.priors.elements.len() > 8
            || !self.search_seconds.is_finite()
            || self.search_seconds <= 0.
            || self.search_seconds > 3600.
            || !(1..=128).contains(&self.search_trials)
            || !(1..=20_000).contains(&self.search_cycles)
        {
            return Err(Error("invalid element/search controls".into()));
        }
        let _ = phasesmith_solution::DenseMemoryBudget::new(self.dense_memory_bytes)?;
        self.check_shadows(shape)?;
        self.geometry.convert(&[[0., 0., 0.]])?;
        Ok(())
    }
    fn check_priors(&self) -> Result<()> {
        use phasesmith_crystallography::xray_species_metadata;
        let cell = self.priors.cell;
        let geometry = cell.geometry().map_err(|e| Error(e.to_string()))?;
        if [cell.alpha_deg, cell.beta_deg, cell.gamma_deg]
            .iter()
            .any(|a| (a - 90.).abs() > 1e-8)
        {
            return Err(Error(
                "native v1 requires an explicit orthorhombic metric prior".into(),
            ));
        }
        let m = geometry.direct_metric;
        let components = [m[0][0], m[1][1], m[2][2], m[1][2], m[0][2], m[0][1]];
        let scale = components.iter().fold(1f64, |s, x| s.max(x.abs()));
        for equation in &self.partition_group().metric_constraints().equations {
            let residual = equation
                .iter()
                .zip(components)
                .map(|(&a, b)| a as f64 * b)
                .sum::<f64>();
            let coefficients = equation
                .iter()
                .map(|a| a.unsigned_abs() as f64)
                .sum::<f64>();
            if residual.abs() > 1e-10 * scale * coefficients.max(1.) {
                return Err(Error(
                    "cell metric incompatible with supplied exact symmetry".into(),
                ));
            }
        }
        if self
            .priors
            .elements
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != self.priors.elements.len()
        {
            return Err(Error("duplicate supplied element".into()));
        }
        for e in &self.priors.elements {
            let m =
                xray_species_metadata(e).ok_or_else(|| Error("unknown supplied element".into()))?;
            if m.atomic_number <= 1
                || !(1..=2).contains(&e.len())
                || !e.as_bytes()[0].is_ascii_uppercase()
                || e.as_bytes().get(1).is_some_and(|c| !c.is_ascii_lowercase())
            {
                return Err(Error(
                    "search needs distinct neutral non-H element symbols".into(),
                ));
            }
        }
        for &rotation in &self.null_rotations {
            let gram = math::mul(
                rotation,
                std::array::from_fn(|i| std::array::from_fn(|j| rotation[j][i])),
            );
            if gram.iter().enumerate().any(|(i, r)| {
                r.iter()
                    .enumerate()
                    .any(|(j, &v)| (v - f64::from(u8::from(i == j))).abs() > 1e-10)
            }) || (math::det(rotation) - 1.).abs() > 1e-10
            {
                return Err(Error(
                    "null controls must be proper rotation matrices".into(),
                ));
            }
        }
        Ok(())
    }
    fn check_resources(&self, shape: [usize; 3]) -> Result<()> {
        if [
            self.detection_workers,
            self.profile_workers,
            self.learning_workers,
            self.frame_workers,
            self.mask_workers,
            self.preparation_workers,
        ]
        .iter()
        .any(|w| !(1..=32).contains(w))
            || self.max_samples == 0
            || self.max_samples > 20_000_000
            || self.frame_payload_bytes > 1024 * 1024 * 1024
            || self.geometry_cache_bytes > 1024 * 1024 * 1024
            || self.max_in_flight_frame_bytes == 0
            || self.max_in_flight_frame_bytes > 1024 * 1024 * 1024
        {
            return Err(Error("invalid workflow resource limits".into()));
        }
        if !(1..=32).contains(&self.preparation_batch) {
            return Err(Error("invalid preparation batch".into()));
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
        let pixels = shape[1]
            .checked_mul(shape[2])
            .filter(|&n| n <= 100_000_000)
            .ok_or_else(|| Error("detector pixel budget exceeded".into()))?;
        for workers in [
            self.detection_workers,
            self.frame_workers,
            self.mask_workers,
            self.preparation_workers,
        ] {
            if pixels
                .checked_mul(13)
                .and_then(|n| n.checked_mul(workers))
                .is_none_or(|n| n > self.max_in_flight_frame_bytes)
            {
                return Err(Error("owned frame-stage payload budget exceeded".into()));
            }
        }
        if self.acquisition.last - self.acquisition.first + 1 > 225 || shape[1] < 2 || shape[2] < 2
        {
            return Err(Error("native v1 requires 2D detector and at most 225 exposures for its explicit frame design".into()));
        }
        Ok(())
    }
    fn check_shadows(&self, shape: [usize; 3]) -> Result<()> {
        let a = &self.acquisition;
        let nframes = a.last - a.first + 1;
        if nframes < 5 {
            return Err(Error(
                "illumination masking needs at least five exposures".into(),
            ));
        }
        for block in [self.shadows.illumination_block, self.shadows.aperture_block] {
            let sh = frame_validity::Shape {
                height: shape[1],
                width: shape[2],
                block,
            };
            let [h, w] = sh.coarse()?;
            if nframes
                .checked_mul(h * w)
                .is_none_or(|n| n > frame_validity::MAX_CELLS)
            {
                return Err(Error("coarse mask cell budget exceeded".into()));
            }
        }
        if !self.shadows.illumination.threshold.is_finite()
            || !(0.0..1.0).contains(&self.shadows.illumination.threshold)
            || self.shadows.illumination.threshold == 0.
            || self.shadows.illumination.guard_pixels > 4096
        {
            return Err(Error("invalid illumination controls".into()));
        }
        if [
            self.shadows.illumination.reference_first,
            self.shadows.illumination.reference_last,
        ]
        .iter()
        .flatten()
        .any(|&i| i >= nframes)
        {
            return Err(Error("illumination reference outside selected scan".into()));
        }
        frame_validity::dac_mask(
            self.geometry,
            bounds(self)[0],
            frame_validity::Shape {
                height: shape[1],
                width: shape[2],
                block: self.shadows.aperture_block,
            },
            self.shadows.aperture,
        )?;
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
    let clock = Instant::now();
    let (plan, scales) = prepare_validity(source, options, shape)?;
    observer.masks(&plan, &scales)?;
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
    let mut processing = options.clone();
    if let Some(discovery) = &options.discovery {
        let initial = unknown::aperture_assessment(
            &measurements,
            &indexed.fit.state.geometry,
            None,
            discovery,
            options.priors.cell,
        )?;
        observer.symmetry("preliminary-symmetry", &initial, &discovery.hypotheses)?;
        processing.priors.group = Some(crate::space_group_search::rotational_group(
            &initial.laue[initial.processing_laue].rotations,
        )?);
    }
    let reduced = scaling::reduce(&measurements, &processing, &indexed.fit.state.geometry)?;
    observer.reduced(&reduced)?;
    observer.stage("profile_fitting_scaling", clock.elapsed().as_secs_f64())?;
    options.checkpoint()?;
    let clock = Instant::now();
    let (solution, search, discovery) = if let Some(discovery) = &options.discovery {
        let assessment = unknown::aperture_assessment(
            &measurements,
            &indexed.fit.state.geometry,
            Some(&reduced),
            discovery,
            options.priors.cell,
        )?;
        observer.symmetry("space-group-evidence", &assessment, &discovery.hypotheses)?;
        let (best, report) = unknown::search(
            &observations_from_reduction(&reduced),
            options,
            &processing,
            assessment,
        )?;
        observer.group_search(&report)?;
        let (solution,search)=best.ok_or_else(||Error("no space-group candidate found within the declared search budgets; retain evidence and untested alternatives".into()))?;
        (solution, search, Some(report))
    } else {
        let (s, a) = search(&reduced, options)?;
        (s, a, None)
    };
    observer.solution(&solution, &search)?;
    observer.stage("structure_search", clock.elapsed().as_secs_f64())?;
    Ok(Output {
        discovery,
        positions,
        indexed,
        measurements,
        reduced,
        solution,
        search,
        seconds: start.elapsed().as_secs_f64(),
    })
}
// At most `workers` source payloads are retained. Callers/observers consume in
// source order; loading/preparation overlaps earlier numerical jobs. A loader
// failure is queued after preceding frames, and later frames are not read.
fn map_frames<T: Send>(
    source: &mut impl FrameSource,
    plan: &frame_input::Plan,
    o: &Options,
    workers: usize,
    map: impl Fn(frame_input::Prepared) -> Result<T> + Sync,
    mut consume: impl FnMut(usize, T) -> Result<()>,
) -> Result<()> {
    use std::collections::VecDeque;
    std::thread::scope(|scope| {
        let mut pending = VecDeque::new();
        let batch = o.preparation_batch.min(workers);
        plan.admit(batch, o.preparation_workers, o.frame_payload_bytes)?;
        let mut next = 0;
        let map = &map;
        while next < plan.frames {
            o.checkpoint()?;
            let count = batch.min(plan.frames - next);
            while pending.len() > workers - count {
                let (index, task): (usize, std::thread::ScopedJoinHandle<'_, Result<T>>) =
                    pending.pop_front().expect("admitted pending task");
                let row = task
                    .join()
                    .map_err(|_| Error("frame worker panicked".into()))??;
                consume(index, row)?;
            }
            let mut inputs = Vec::new();
            let mut failed = None;
            for index in next..next + count {
                match source.read(o.acquisition.first + index) {
                    Ok(pixels) => inputs.push(frame_input::Input { index, pixels }),
                    Err(error) => {
                        failed = Some((index, error));
                        break;
                    }
                }
            }
            let prepared = if inputs.is_empty() {
                vec![]
            } else {
                plan.prepare_batch(inputs, o.preparation_workers, o.frame_payload_bytes)?
            };
            for (offset, row) in prepared.into_iter().enumerate() {
                let index = next + offset;
                pending.push_back((index, scope.spawn(move || map(row?))));
            }
            if let Some((index, error)) = failed {
                pending.push_back((index, scope.spawn(move || Err(error))));
                break;
            }
            next += count;
        }
        while let Some((index, task)) = pending.pop_front() {
            o.checkpoint()?;
            let row = task
                .join()
                .map_err(|_| Error("frame worker panicked".into()))??;
            consume(index, row)?;
        }
        Ok(())
    })
}
fn prepare_validity(
    source: &mut impl FrameSource,
    options: &Options,
    shape: [usize; 2],
) -> Result<(frame_input::Plan, Vec<f64>)> {
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
    let coarse_shape = frame_validity::Shape {
        height: shape[0],
        width: shape[1],
        block: options.shadows.illumination_block,
    };
    let mut coarse = Vec::new();
    let mut available = Vec::new();
    let mut hashes = Vec::new();
    map_frames(
        source,
        &plan,
        options,
        options.mask_workers,
        |p| {
            let data = image(&p, shape);
            let coarse = frame_validity::coarsen(&data, &p.excluded, coarse_shape)?;
            Ok((p.digest, coarse))
        },
        |_, (digest, c)| {
            coarse.extend(c.values);
            available.push(c.available);
            hashes.push(digest);
            Ok(())
        },
    )?;
    let masks = frame_validity::estimate(
        &coarse,
        &available,
        coarse_shape,
        options.shadows.illumination,
    )?;
    plan.expected_hashes.push(hashes);
    plan.blocks.push(frame_input::BlockMasks {
        block: options.shadows.illumination_block,
        excluded: masks.excluded,
        fixed: vec![],
    });
    let dac_shape = frame_validity::Shape {
        height: shape[0],
        width: shape[1],
        block: options.shadows.aperture_block,
    };
    let [ch, cw] = dac_shape.coarse()?;
    if frames
        .checked_mul(ch * cw)
        .is_none_or(|n| n > frame_validity::MAX_CELLS)
    {
        return Err(Error("cached aperture mask cell budget exceeded".into()));
    }
    let mut dac_excluded = Vec::with_capacity(frames * ch * cw);
    for angle in &angles {
        options.checkpoint()?;
        dac_excluded.extend(
            frame_validity::dac_mask(
                options.geometry,
                *angle,
                dac_shape,
                options.shadows.aperture,
            )?
            .excluded,
        );
    }
    plan.blocks.push(frame_input::BlockMasks {
        block: options.shadows.aperture_block,
        excluded: dac_excluded,
        fixed: vec![],
    });
    Ok((plan, masks.detection_scales))
}
fn search(
    r: &Reduced,
    o: &Options,
) -> Result<(
    phasesmith_solution::Solution,
    phasesmith_solution::SharedSearchSummary,
)> {
    let observations = observations_from_reduction(r);
    search_observations(&observations, o)
}
/// Search supplied signed observations using the same literal validated-v1 strategy.
/// This stage entry point is useful for independent input/optimizer diagnostics.
/// # Errors
/// Invalid observations, priors, resource limits, or a failed native search.
pub fn search_observations(
    observations: &[phasesmith_solution::Observation],
    o: &Options,
) -> Result<(
    phasesmith_solution::Solution,
    phasesmith_solution::SharedSearchSummary,
)> {
    use phasesmith_solution::{
        AdpMode, DenseMemoryBudget, PhasingMethod, SolveOptions, estimate_solution_memory,
        solve_elements,
    };
    let group = o
        .priors
        .group
        .as_ref()
        .ok_or_else(|| Error("exact group missing; use unknown-symmetry coordinator".into()))?;
    let memory = estimate_solution_memory(
        observations,
        o.priors.cell,
        group,
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
        refinement: Some(if o.discovery.is_some() {
            AdpMode::Isotropic
        } else {
            AdpMode::Anisotropic
        }),
        symmetry_tolerance_angstrom: 0.15,
        symmetry_iterations: 80,
    };
    let shared = shared_search_options(memory.maximum_atoms);
    let solution = solve_elements(
        observations,
        o.priors.cell,
        group,
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

#[cfg(test)]
mod tests;

fn shared_search_options(maximum_atoms: usize) -> phasesmith_solution::SharedSearchOptions {
    phasesmith_solution::SharedSearchOptions {
        maximum_atoms,
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
    }
}

fn observations_from_reduction(r: &Reduced) -> Vec<phasesmith_solution::Observation> {
    use phasesmith_solution::Observation;
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
    observations
}

/// Search corrected, unmerged signed measurements without an exact space-group prior.
/// Input must retain genuinely measured weak/negative/forbidden rows; absent coverage
/// is represented by omission, never a synthetic zero. The caller owns correction provenance.
/// The supplied catalogue/index basis and all bounded omissions remain in the report.
/// # Errors
/// Exact group supplied, invalid catalogue/data/resources, or a failed common input preparation.
pub fn search_unknown_observations(
    observations: &[phasesmith_solution::Observation],
    o: &Options,
) -> Result<SearchRun> {
    if o.priors.group.is_some() {
        return Err(Error(
            "unknown-symmetry search must withhold the exact group".into(),
        ));
    }
    let d = o
        .discovery
        .as_ref()
        .ok_or_else(|| Error("discovery policy missing".into()))?;
    d.check()?;
    let assessment = crate::space_group_search::assess(
        observations,
        o.priors.cell,
        &d.hypotheses,
        &d.assessment,
    )?;
    let mut processing = o.clone();
    processing.priors.group = Some(crate::space_group_search::rotational_group(
        &assessment.laue[assessment.processing_laue].rotations,
    )?);
    unknown::search(observations, o, &processing, assessment)
}
