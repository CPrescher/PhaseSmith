//! Shoebox summation over the frames of each reflection's rocking window,
//! streamed over the frames with bounded memory.
use super::{
    HIDDEN, NO_DATA, Reflection, Result, Scan, V3,
    geometry::{Corrections, Polarization, lorentz_polarization, ray_geometry},
    invalid,
    profile::{self, ProfileIntensity, ProfileOptions, ProfileStatus},
    rotate, unit,
};
use crate::spots::{
    Bounds, Image, NO_CENTROID, Spot, integration, local_background, profiles, reciprocal::Geometry,
};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// One frame for measurement: counts and the pixels to leave out.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct Frame {
    /// Row-major counts, `rows × columns`. Masked pixels may be nonfinite.
    pub values: Vec<f64>,
    /// Row-major, `true` excludes a pixel (gaps, dead pixels, shadows, any
    /// mask of this frame). Saturated pixels should stay unmasked so they
    /// can be flagged ([`Noise::saturation`]).
    pub mask: Vec<bool>,
}
impl Frame {
    /// A frame from counts and mask.
    #[must_use]
    pub fn new(values: Vec<f64>, mask: Vec<bool>) -> Self {
        Self { values, mask }
    }
}

/// Random access to the frames of a scan. [`measure`] reads each frame it
/// needs once, from several threads; at most [`MeasureOptions::workers`]
/// frames are held at a time.
pub trait FrameReader: Sync {
    /// Read frame `index` (0-based within the scan).
    /// # Errors
    /// An I/O or format error; the measurement stops with it.
    fn read(&self, index: usize) -> Result<Frame>;
}
impl<F: Fn(usize) -> Result<Frame> + Sync> FrameReader for F {
    fn read(&self, index: usize) -> Result<Frame> {
        self(index)
    }
}

/// Stage reported to a [`Monitor`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Stage {
    /// Reading frames and summing shoeboxes; units are frames.
    Measuring,
    /// Learning the profile bank (one unit).
    LearningProfiles,
    /// Fitting profiles (one unit).
    FittingProfiles,
}

/// Progress and cancellation. Called from worker threads.
pub trait Monitor: Sync {
    /// `done` of `total` units of `stage` are complete; return `false` to
    /// cancel (the measurement then fails with a cancellation error).
    fn progress(&self, stage: Stage, done: usize, total: usize) -> bool;
}
impl<F: Fn(Stage, usize, usize) -> bool + Sync> Monitor for F {
    fn progress(&self, stage: Stage, done: usize, total: usize) -> bool {
        self(stage, done, total)
    }
}
/// A monitor that never cancels.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoMonitor;
impl Monitor for NoMonitor {
    fn progress(&self, _: Stage, _: usize, _: usize) -> bool {
        true
    }
}

/// Local background: a plane fitted, with iterative outlier clipping, to a
/// ring of `margin` pixels outside a guard of `gap` pixels around the peak
/// box. The guards of all peak boxes of the frame (every lattice) are left
/// out of every ring.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Background {
    /// Guard between peak box and ring, pixels. Default 2.
    pub gap: usize,
    /// Ring width, pixels (1–128). Default 5.
    pub margin: usize,
    /// Fewest ring pixels for a plane (≥ 3). Default 20.
    pub minimum: usize,
    /// Outlier rejection in robust standard deviations (2–10). Default 4.
    pub clip: f64,
}
impl Default for Background {
    fn default() -> Self {
        Self {
            gap: 2,
            margin: 5,
            minimum: 20,
            clip: 4.,
        }
    }
}

/// Detector noise model: variance `max(counts, 0) / gain + read_noise²`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Noise {
    /// Photons per count. Default 1.
    pub gain: f64,
    /// Read-noise standard deviation, counts. Default 0.
    pub read_noise: f64,
    /// Counts at and above which a pixel is saturated. Default none.
    pub saturation: Option<f64>,
}
impl Default for Noise {
    fn default() -> Self {
        Self {
            gain: 1.,
            read_noise: 0.,
            saturation: None,
        }
    }
}

/// Controls of [`measure`]. Non-exhaustive: start from `Default` and assign
/// fields.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct MeasureOptions {
    /// Side of the square peak box, pixels (odd, 3–101). Default 9.
    pub box_size: usize,
    /// Local background.
    pub background: Background,
    /// Noise model.
    pub noise: Noise,
    /// Polarisation for the corrections attached to each reflection.
    pub polarization: Polarization,
    /// Profile fitting on the same shoeboxes; `None` (default) sums only.
    pub profile: Option<ProfileOptions>,
    /// Worker threads (1–64), also the most frames held at a time. Default:
    /// available parallelism, at most 8.
    pub workers: usize,
    /// Most shoebox pixels kept for profile fitting (one sample is 25 bytes).
    /// Default twenty million.
    pub max_samples: usize,
}
impl Default for MeasureOptions {
    fn default() -> Self {
        Self {
            box_size: 9,
            background: Background::default(),
            noise: Noise::default(),
            polarization: Polarization::default(),
            profile: None,
            workers: std::thread::available_parallelism().map_or(1, |n| n.get().min(8)),
            max_samples: 20_000_000,
        }
    }
}
impl MeasureOptions {
    fn check(&self) -> Result<()> {
        let b = &self.background;
        if self.box_size.is_multiple_of(2)
            || !(3..=101).contains(&self.box_size)
            || !(1..=64).contains(&self.workers)
            || b.gap > 128
            || !(1..=128).contains(&b.margin)
            || b.minimum < 3
            || !(2.0..=10.0).contains(&b.clip)
            || !self.noise.gain.is_finite()
            || self.noise.gain <= 0.
            || !self.noise.read_noise.is_finite()
            || self.noise.read_noise < 0.
            || self.noise.saturation.is_some_and(|s| !s.is_finite())
            || self.max_samples > 100_000_000
        {
            return Err(invalid("invalid measurement options"));
        }
        if let Some(p) = &self.profile {
            p.check()?;
        }
        Ok(())
    }
}

/// An intensity and its standard uncertainty.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Intensity {
    /// Intensity, counts (NaN when not measured).
    pub value: f64,
    /// Standard uncertainty, counts.
    pub sigma: f64,
}
impl Intensity {
    /// Both multiplied by `factor`.
    #[must_use]
    pub fn scaled(self, factor: f64) -> Self {
        Self {
            value: self.value * factor,
            sigma: self.sigma * factor,
        }
    }
    /// Finite with a positive σ.
    #[must_use]
    pub fn is_measured(&self) -> bool {
        self.value.is_finite() && self.sigma.is_finite() && self.sigma > 0.
    }
}

/// Summation on one frame of a reflection's window.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Partial {
    /// Frame index.
    pub frame: usize,
    /// Background-subtracted sum, counts (NaN without background).
    pub intensity: f64,
    /// Counting-statistics σ, counts.
    pub sigma: f64,
    /// Fitted background per pixel, counts.
    pub background: f64,
    /// Centroid x, y of the positive excess, pixels.
    pub centroid: [f64; 2],
    /// Second-moment widths x, y of the positive excess, pixels.
    pub spread: [f64; 2],
    /// Flags of this frame's box.
    pub flags: u32,
    /// Inside the rocking window (summed); `false` for the extra frames read
    /// for profile fitting.
    pub core: bool,
}

/// One measured reflection: prediction, summation, corrections, optional
/// profile fit and what a reflection-file writer needs.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct MeasuredReflection {
    /// The prediction (lattice, indices, position, d, ζ, window).
    pub reflection: Reflection,
    /// Peak box, pixels.
    pub bounds: Bounds,
    /// Summed intensity over the window, background subtracted, without
    /// corrections; σ from counting statistics of peak and background.
    pub summation: Intensity,
    /// Mean fitted background per peak pixel and frame, counts.
    pub background: f64,
    /// Peak pixels summed over the measured frames.
    pub peak_pixels: usize,
    /// Background pixels used, summed over the measured frames.
    pub background_pixels: usize,
    /// Frames of the window that were measured.
    pub frames_measured: usize,
    /// Observed x, y (pixels) and ω (degrees) of the positive excess; NaN
    /// without one.
    pub centroid: V3,
    /// Summation flags (see [`super::FLAG_NAMES`]), including those of the
    /// prediction; zero means usable.
    pub flags: u32,
    /// Lorentz and polarisation at the predicted position; `None` where
    /// singular.
    pub corrections: Option<Corrections>,
    /// Unit incident-beam direction in the crystal frame (the laboratory
    /// frame at ω = 0, where the lattice basis is given).
    pub incident: V3,
    /// Unit diffracted-beam direction in the crystal frame.
    pub diffracted: V3,
    /// Per-frame sums, in frame order.
    pub partials: Vec<Partial>,
    /// Profile-fitted intensity, when profile fitting ran and this reflection
    /// was fitted.
    pub profile: Option<ProfileIntensity>,
}
impl MeasuredReflection {
    /// Summation usable: no flag and a measured intensity.
    #[must_use]
    pub fn usable(&self) -> bool {
        self.flags == 0 && self.summation.is_measured()
    }
    /// Profile intensity usable: fitted without profile flags and no
    /// prediction flag ([`super::AXIAL`], [`super::HIDDEN`]) nor
    /// [`NO_DATA`].
    #[must_use]
    pub fn profile_usable(&self) -> bool {
        self.flags & (super::AXIAL | HIDDEN | NO_DATA) == 0
            && self
                .profile
                .is_some_and(|p| p.flags == 0 && p.intensity.is_measured())
    }
    /// Lorentz-polarisation corrected summation (NaN without corrections).
    #[must_use]
    pub fn corrected_summation(&self) -> Intensity {
        self.summation
            .scaled(self.corrections.map_or(f64::NAN, |c| c.factor))
    }
    /// Lorentz-polarisation corrected profile intensity.
    #[must_use]
    pub fn corrected_profile(&self) -> Option<Intensity> {
        let f = self.corrections.map_or(f64::NAN, |c| c.factor);
        self.profile.map(|p| p.intensity.scaled(f))
    }
}

/// The result of [`measure`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Measurement {
    /// One row per input reflection, in input order.
    pub reflections: Vec<MeasuredReflection>,
    /// Profile fitting outcome.
    pub profile: ProfileStatus,
    /// Frames read.
    pub frames_read: usize,
}

/// Shoeboxes of all (reflection, frame) pairs, in frame then reflection
/// order: the profile-fit input.
pub(super) struct Groups {
    pub ids: Vec<usize>,
    pub frames: Vec<usize>,
    pub boxes: Vec<Bounds>,
    pub flags: Vec<u32>,
    pub covariance: Vec<[f64; 9]>,
    pub offsets: Vec<usize>,
    pub samples: Vec<[f64; 3]>,
    pub valid: Vec<bool>,
}

struct FrameRows {
    frame: usize,
    core_ids: Vec<usize>,
    core_spots: Vec<Spot>,
    ids: Vec<usize>,
    spots: Vec<Spot>,
    covariance: Vec<[f64; 9]>,
    samples: Vec<[f64; 3]>,
    valid: Vec<bool>,
    hidden: Vec<usize>,
}

/// Peak box of odd side `size` around `(x, y)`, clipped to the detector, and
/// the centre pixel.
fn peak_box(x: f64, y: f64, size: usize, shape: [usize; 2]) -> (Bounds, usize) {
    let cx = ((x + 0.5).floor().max(0.) as usize).min(shape[1] - 1);
    let cy = ((y + 0.5).floor().max(0.) as usize).min(shape[0] - 1);
    let r = size / 2;
    (
        Bounds {
            x0: cx.saturating_sub(r),
            y0: cy.saturating_sub(r),
            x1: (cx + r + 1).min(shape[1]),
            y1: (cy + r + 1).min(shape[0]),
        },
        cy * shape[1] + cx,
    )
}

#[cfg(test)]
pub(super) fn tests_peak_box(x: f64, y: f64, size: usize, shape: [usize; 2]) -> Bounds {
    peak_box(x, y, size, shape).0
}

/// Frames of a reflection's extended window (core ± margin), clipped, and
/// whether the extension was cut by the scan or an excluded frame.
fn extended(r: &Reflection, margin: usize, scan: &Scan) -> ([usize; 2], bool) {
    let first = r.frames[0] as i64 - margin as i64;
    let last = (r.frames[1] + margin) as i64;
    let n = scan.frames as i64;
    let w = [first.max(0) as usize, last.min(n - 1) as usize];
    let cut = first < 0 || last > n - 1 || scan.excluded.iter().any(|&f| f >= w[0] && f <= w[1]);
    (w, cut)
}

/// Measure every reflection on the frames of its rocking window in one pass
/// over the frames (`shape = [rows, columns]`).
///
/// On each frame, the peak boxes of all reflections whose window covers it
/// (every lattice at once) are summed and a background plane is subtracted
/// ([`crate::spots::local_background::extract`]); the frames are added with
/// [`crate::spots::integration::accumulate`]. Frames run in parallel on
/// [`MeasureOptions::workers`] threads, each frame is read once and dropped
/// after use; nothing per pixel of the full image is kept or allocated
/// besides the frame itself. With [`MeasureOptions::profile`], the shoebox
/// pixels are kept (bounded by [`MeasureOptions::max_samples`]) and
/// profiles are learned and fitted afterwards; a failed profile stage is
/// reported in [`Measurement::profile`] and leaves the summation intact.
/// # Errors
/// Invalid inputs or options, a frame-read error, a frame of the wrong
/// shape, a budget exceeded (one million reflections and one million
/// reflection-frame pairs), or cancellation by the monitor.
#[allow(clippy::too_many_lines)]
pub fn measure(
    frames: &impl FrameReader,
    geometry: &Geometry,
    scan: &Scan,
    shape: [usize; 2],
    reflections: &[Reflection],
    options: &MeasureOptions,
    monitor: &impl Monitor,
) -> Result<Measurement> {
    scan.check()?;
    options.check()?;
    geometry.convert(&[])?;
    options.polarization.axes(unit(geometry.beam))?;
    let pixels = shape[0]
        .checked_mul(shape[1])
        .filter(|&n| n > 0 && u32::try_from(n).is_ok())
        .ok_or_else(|| invalid("invalid detector shape"))?;
    let n = reflections.len();
    if n > 1_000_000
        || reflections
            .iter()
            .any(|r| r.frames[0] > r.frames[1] || r.frames[1] >= scan.frames)
    {
        return Err(invalid("invalid reflections or reflection budget exceeded"));
    }
    let margin = options.profile.as_ref().map_or(0, |p| p.margin_frames);
    let mut boxes = Vec::with_capacity(n);
    let mut windows = Vec::with_capacity(n);
    let mut active: Vec<Vec<usize>> = vec![Vec::new(); scan.frames];
    let mut pairs = 0usize;
    let mut sample_count = 0usize;
    for (i, r) in reflections.iter().enumerate() {
        let (b, pixel) = peak_box(r.xyz[0], r.xyz[1], options.box_size, shape);
        boxes.push((b, pixel));
        let (w, cut) = extended(r, margin, scan);
        windows.push((w, cut));
        for (f, list) in active.iter_mut().enumerate().take(w[1] + 1).skip(w[0]) {
            if !scan.is_excluded(f) {
                list.push(i);
                pairs += 1;
                sample_count += (b.x1 - b.x0) * (b.y1 - b.y0);
            }
        }
    }
    if pairs > 1_000_000 {
        return Err(invalid("reflection-frame pair budget exceeded"));
    }
    let keep_samples = options.profile.is_some();
    if keep_samples && sample_count > options.max_samples {
        return Err(invalid("profile sample budget exceeded"));
    }
    let wanted: Vec<usize> = (0..scan.frames)
        .filter(|&f| !active[f].is_empty())
        .collect();
    let total = wanted.len();
    let done = AtomicUsize::new(0);
    let cancelled = AtomicBool::new(false);
    let bg = options.background;
    let one = |f: usize| -> Result<FrameRows> {
        if cancelled.load(Ordering::Relaxed) {
            return Err(invalid("measurement cancelled"));
        }
        let frame = frames.read(f)?;
        if frame.values.len() != pixels || frame.mask.len() != pixels {
            return Err(invalid(format!("frame {f} has the wrong shape")));
        }
        let image = Image {
            values: &frame.values,
            height: shape[0],
            width: shape[1],
            mask: &frame.mask,
            variance: None,
            gain: options.noise.gain,
            read_noise: options.noise.read_noise,
            saturation: options.noise.saturation,
        };
        let ids = active[f].clone();
        // Summation sees the boxes of the rocking windows only, so its
        // results do not depend on profile fitting being requested.
        let core_ids: Vec<usize> = ids
            .iter()
            .copied()
            .filter(|&i| f >= reflections[i].frames[0] && f <= reflections[i].frames[1])
            .collect();
        let core_boxes: Vec<Bounds> = core_ids.iter().map(|&i| boxes[i].0).collect();
        let core_spots = if core_boxes.is_empty() {
            vec![]
        } else {
            local_background::extract_full(
                &image,
                &core_boxes,
                bg.gap,
                bg.margin,
                bg.minimum,
                bg.clip,
            )?
            .0
        };
        let (spots, covariance, samples, valid) = if keep_samples {
            let frame_boxes: Vec<Bounds> = ids.iter().map(|&i| boxes[i].0).collect();
            let (spots, _, covariance, samples, valid) = profiles::extract_frame(
                &image,
                &frame_boxes,
                bg.gap,
                bg.margin,
                bg.minimum,
                bg.clip,
            )?;
            (spots, covariance, samples, valid)
        } else {
            (vec![], vec![], vec![], vec![])
        };
        let hidden = ids
            .iter()
            .copied()
            .filter(|&i| reflections[i].centre_frame == f && frame.mask[boxes[i].1])
            .collect();
        drop(frame);
        let current = done.fetch_add(1, Ordering::Relaxed) + 1;
        if !monitor.progress(Stage::Measuring, current, total) {
            cancelled.store(true, Ordering::Relaxed);
            return Err(invalid("measurement cancelled"));
        }
        Ok(FrameRows {
            frame: f,
            core_ids,
            core_spots,
            ids,
            spots,
            covariance,
            samples,
            valid,
            hidden,
        })
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.workers)
        .build()
        .map_err(|e| invalid(format!("measurement thread pool: {e}")))?;
    let rows = pool.install(|| {
        wanted
            .par_iter()
            .map(|&f| one(f))
            .collect::<Result<Vec<_>>>()
    })?;
    if cancelled.load(Ordering::Relaxed) {
        return Err(invalid("measurement cancelled"));
    }

    // Summation over the core window.
    let mut hidden = vec![false; n];
    let mut groups = Groups {
        ids: vec![],
        frames: vec![],
        boxes: vec![],
        flags: vec![],
        covariance: vec![],
        offsets: vec![0],
        samples: vec![],
        valid: vec![],
    };
    let (mut ids, mut values, mut flags) = (vec![], vec![], vec![]);
    let mut partials: Vec<Vec<Partial>> = vec![Vec::new(); n];
    for row in rows {
        for &i in &row.hidden {
            hidden[i] = true;
        }
        let partial = |spot: &Spot, core: bool| {
            let v = spot.values;
            Partial {
                frame: row.frame,
                intensity: v[2],
                sigma: v[3],
                background: v[4],
                centroid: [v[0], v[1]],
                spread: [v[7], v[8]],
                flags: spot.flags & !NO_CENTROID,
                core,
            }
        };
        for (&i, spot) in row.core_ids.iter().zip(&row.core_spots) {
            partials[i].push(partial(spot, true));
            ids.push(i);
            values.push(spot.values);
            flags.push(spot.flags);
        }
        let mut offset = 0;
        for (k, (&i, spot)) in row.ids.iter().zip(&row.spots).enumerate() {
            let r = &reflections[i];
            if row.frame < r.frames[0] || row.frame > r.frames[1] {
                partials[i].push(partial(spot, false));
            }
            let b = spot.bounds;
            let size = (b.x1 - b.x0) * (b.y1 - b.y0);
            groups.ids.push(i);
            groups.frames.push(row.frame);
            groups.boxes.push(b);
            groups
                .flags
                .push((spot.flags & !NO_CENTROID) | if windows[i].1 { super::PARTIAL } else { 0 });
            groups.covariance.push(row.covariance[k]);
            groups
                .samples
                .extend_from_slice(&row.samples[offset..offset + size]);
            groups
                .valid
                .extend_from_slice(&row.valid[offset..offset + size]);
            groups.offsets.push(groups.samples.len());
            offset += size;
        }
    }
    for p in &mut partials {
        p.sort_by_key(|p| p.frame);
    }
    let (sums, quality) = integration::accumulate(n, &ids, &values, &flags)?;
    drop((ids, values, flags));

    let axis = unit(geometry.axis);
    let beam = unit(geometry.beam);
    let mut out = Vec::with_capacity(n);
    for (i, r) in reflections.iter().enumerate() {
        let s = sums[i];
        let mut flags = quality[i] | r.flags;
        if hidden[i] {
            flags |= HIDDEN;
        }
        let (mut w, mut wx, mut wy, mut wz) = (0., 0., 0., 0.);
        for p in partials[i].iter().filter(|p| p.core) {
            let weight = p.intensity.max(0.);
            if weight > 0. && p.centroid.iter().all(|v| v.is_finite()) {
                let [lo, hi] = scan.bounds(p.frame);
                w += weight;
                wx += weight * p.centroid[0];
                wy += weight * p.centroid[1];
                wz += weight * f64::midpoint(lo, hi);
            }
        }
        let angle = -r.xyz[2].to_radians();
        let ray = ray_geometry([r.xyz[0], r.xyz[1]], geometry).ray;
        out.push(MeasuredReflection {
            reflection: *r,
            bounds: boxes[i].0,
            summation: Intensity {
                value: s[0],
                sigma: s[1],
            },
            background: if s[3] > 0. { s[2] / s[3] } else { f64::NAN },
            peak_pixels: s[3] as usize,
            background_pixels: s[5] as usize,
            frames_measured: s[4] as usize,
            centroid: if w > 0. {
                [wx / w, wy / w, wz / w]
            } else {
                [f64::NAN; 3]
            },
            flags,
            corrections: lorentz_polarization(
                [r.xyz[0], r.xyz[1]],
                geometry,
                &options.polarization,
            )
            .ok(),
            incident: rotate(beam, axis, angle),
            diffracted: rotate(ray, axis, angle),
            partials: std::mem::take(&mut partials[i]),
            profile: None,
        });
    }

    let status = match &options.profile {
        None => ProfileStatus::NotRequested,
        Some(p) => profile::fit(
            &groups, &mut out, geometry, scan, shape, options, p, monitor,
        )?,
    };
    Ok(Measurement {
        reflections: out,
        profile: status,
        frames_read: total,
    })
}
