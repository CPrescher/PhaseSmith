//! Reflection prediction and intensity measurement for refined lattices.
//!
//! The caller supplies frames (through [`FrameReader`](crate::measurement::FrameReader)), the detector
//! [`Geometry`](crate::spots::reciprocal::Geometry), per-frame masks (with
//! each frame), the rotation [`Scan`](crate::measurement::Scan) and one or more refined [`Lattice`](crate::measurement::Lattice)s
//! (row basis in ångström and a Bravais type). Nothing else is assumed: no
//! cell prior, no space group, no recipe. Several lattices (grains) share one
//! detector geometry and are measured in one pass over the frames.
//!
//! 1. [`predict`](crate::measurement::predict) lists every reflection the centring allows up to a
//!    resolution limit that crosses the Ewald sphere inside the scan and
//!    lands on the detector, with the frame window its rocking curve spans.
//! 2. [`estimate_shape`](crate::measurement::estimate_shape) (or [`estimate_shape_from_partials`](crate::measurement::estimate_shape_from_partials),
//!    [`estimate_rocking_from_tracks`](crate::measurement::estimate_rocking_from_tracks)) derives the rocking width and the
//!    spot size from the data.
//! 3. [`measure`](crate::measurement::measure) streams the frames once: a shoebox per reflection over the
//!    frames of its window, a local background plane per frame, summation
//!    with counting-statistics σ and quality flags; optionally profile
//!    fitting on the same shoeboxes ([`ProfileOptions`](crate::measurement::ProfileOptions)).
//! 4. [`lorentz_polarization`](crate::measurement::lorentz_polarization) gives the corrections (also attached to each
//!    measured reflection), and [`merging_statistics`](crate::measurement::merging_statistics) the `R_int`, redundancy
//!    and completeness for the holohedry operations of
//!    [`crate::lattice_symmetry::LatticeSystem::holohedry`].
//!
//! Units: pixels (centre of pixel 0 at 0), degrees for ω, ångström for d,
//! reciprocal positions in 1/Å without 2π. See `docs/measurement.md` for the
//! method and its limits.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::similar_names
)]

mod geometry;
mod integrate;
mod merging;
mod predict;
mod profile;
mod shape;

pub use geometry::{
    Corrections, Polarization, RayGeometry, detector_resolution, lorentz_polarization,
    polarization_factor, ray_geometry,
};
pub use integrate::{
    Background, Frame, FrameReader, Intensity, MeasureOptions, MeasuredReflection, Measurement,
    Monitor, NoMonitor, Noise, Partial, PixelVariance, Stage, measure,
};
pub use merging::{
    Merged, MergingBin, MergingOptions, MergingStatistics, Observation, merge, merging_statistics,
};
pub use predict::{Crossing, PredictOptions, Reflection, crossings, predict};
pub use profile::{ProfileIntensity, ProfileModel, ProfileOptions, ProfileStatus};
pub use shape::{
    ShapeEstimate, estimate_rocking_from_tracks, estimate_shape, estimate_shape_from_partials,
};

use crate::grains::{M3, V3};
use crate::lattice_symmetry::Bravais;
use crate::spots::{SpotError, integration, local_background};

type Result<T> = std::result::Result<T, SpotError>;
fn invalid(message: impl Into<String>) -> SpotError {
    SpotError(message.into())
}

/// Peak box truncated by the detector edge.
pub const EDGE: u32 = crate::spots::EDGE;
/// Masked pixel in a peak box on a measured frame.
pub const MASKED: u32 = crate::spots::MASKED;
/// Saturated pixel in a peak box.
pub const SATURATED: u32 = crate::spots::SATURATED;
/// Peak box shares pixels with another predicted reflection's box on a common
/// frame (same or another lattice); intensities are not deblended.
pub const OVERLAP: u32 = crate::spots::OVERLAP;
/// Too few background pixels for a plane on some frame.
pub const BACKGROUND: u32 = crate::spots::BACKGROUND;
/// Close to the rotation axis: `|ζ|` below [`PredictOptions::axial_zeta`].
pub const AXIAL: u32 = 128;
/// Rocking window cut by the scan start or end or by an excluded frame.
pub const PARTIAL: u32 = 256;
/// Centre pixel masked on the centre frame, or the centre frame excluded.
pub const HIDDEN: u32 = 512;
/// No frame of the window was measured.
pub const NO_DATA: u32 = integration::NO_DATA;
/// Significant signal in the outermost pixels of the peak box.
pub const BOUNDARY: u32 = local_background::BOUNDARY;
/// The background plane describes its ring poorly (many rejected pixels or a
/// large residual).
pub const MODEL: u32 = local_background::MODEL;
/// Names of the summation flags, in bit order.
pub const FLAG_NAMES: [(u32, &str); 11] = [
    (EDGE, "edge"),
    (MASKED, "masked"),
    (SATURATED, "saturated"),
    (OVERLAP, "overlap"),
    (BACKGROUND, "background"),
    (AXIAL, "axial"),
    (PARTIAL, "partial"),
    (HIDDEN, "hidden"),
    (NO_DATA, "no data"),
    (BOUNDARY, "boundary"),
    (MODEL, "model"),
];

/// A rotation scan: frame `i` is exposed from `start + i·step` to that plus
/// `exposure` (degrees). Frames listed in `excluded` are never read; a
/// rocking window that touches one is [`PARTIAL`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Scan {
    /// Exposure start of frame 0, degrees.
    pub start: f64,
    /// Rotation between frame starts, degrees (positive).
    pub step: f64,
    /// Rotation during one exposure, degrees (`0 < exposure ≤ step`).
    pub exposure: f64,
    /// Number of frames.
    pub frames: usize,
    /// Frames not to read (sorted, unique, below `frames`).
    pub excluded: Vec<usize>,
}
impl Scan {
    /// A scan without excluded frames.
    /// # Errors
    /// Nonpositive or nonfinite step or exposure, exposure above the step, or
    /// no frames.
    pub fn new(start: f64, step: f64, exposure: f64, frames: usize) -> Result<Self> {
        let scan = Self {
            start,
            step,
            exposure,
            frames,
            excluded: vec![],
        };
        scan.check()?;
        Ok(scan)
    }
    /// The same scan with these frames excluded (sorted and deduplicated).
    /// # Errors
    /// A frame index outside the scan.
    pub fn with_excluded(mut self, mut excluded: Vec<usize>) -> Result<Self> {
        excluded.sort_unstable();
        excluded.dedup();
        self.excluded = excluded;
        self.check()?;
        Ok(self)
    }
    /// Validate the fields.
    /// # Errors
    /// See [`Scan::new`] and [`Scan::with_excluded`].
    pub fn check(&self) -> Result<()> {
        if !self.start.is_finite()
            || !self.step.is_finite()
            || self.step <= 0.
            || !self.exposure.is_finite()
            || self.exposure <= 0.
            || self.exposure > self.step * (1. + 1e-9)
            || self.frames == 0
            || self.frames > 1_000_000
            || self.excluded.windows(2).any(|w| w[0] >= w[1])
            || self.excluded.last().is_some_and(|&f| f >= self.frames)
        {
            return Err(invalid("invalid rotation scan"));
        }
        Ok(())
    }
    /// Exposure interval of a frame, degrees.
    #[must_use]
    pub fn bounds(&self, frame: usize) -> [f64; 2] {
        let low = self.start + frame as f64 * self.step;
        [low, low + self.exposure]
    }
    /// Rotation range covered by the scan: start of the first exposure to the
    /// end of the last.
    #[must_use]
    pub fn range(&self) -> [f64; 2] {
        [
            self.start,
            self.start + (self.frames - 1) as f64 * self.step + self.exposure,
        ]
    }
    /// Whether a frame is excluded.
    #[must_use]
    pub fn is_excluded(&self, frame: usize) -> bool {
        self.excluded.binary_search(&frame).is_ok()
    }
    /// Frame whose step interval holds `omega`, clamped to the scan.
    #[must_use]
    pub fn frame_of(&self, omega: f64) -> usize {
        let f = ((omega - self.start) / self.step).floor();
        f.clamp(0., (self.frames - 1) as f64) as usize
    }
    /// First and last frame (unclipped) whose exposure overlaps the open
    /// interval `omega ± half`; at least one frame.
    fn window(&self, omega: f64, half: f64) -> [i64; 2] {
        let first = ((omega - half - self.start - self.exposure) / self.step).floor() as i64 + 1;
        let last = ((omega + half - self.start) / self.step).ceil() as i64 - 1;
        [first, last.max(first)]
    }
}

/// A refined lattice: direct basis as rows in ångström, its reciprocal basis
/// (rows, 1/Å, no 2π) and Bravais type. The basis must be the conventional
/// cell of the type (as from [`crate::lattice_symmetry::search`]), so that
/// the centring absences and holohedry operations apply to its indices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lattice {
    direct: M3,
    reciprocal: M3,
    bravais: Bravais,
}
fn inverse_transpose(a: M3) -> Option<M3> {
    let c = |i: usize, j: usize| {
        let (r0, r1) = ((i + 1) % 3, (i + 2) % 3);
        let (c0, c1) = ((j + 1) % 3, (j + 2) % 3);
        a[r0][c0] * a[r1][c1] - a[r0][c1] * a[r1][c0]
    };
    let det = a[0][0] * c(0, 0) + a[0][1] * c(0, 1) + a[0][2] * c(0, 2);
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }
    // (A⁻¹)ᵀ = cofactor matrix / det.
    Some(std::array::from_fn(|i| {
        std::array::from_fn(|j| c(i, j) / det)
    }))
}
impl Lattice {
    /// From the direct basis (rows, Å).
    /// # Errors
    /// A singular or nonfinite basis.
    pub fn new(direct: M3, bravais: Bravais) -> Result<Self> {
        let reciprocal = inverse_transpose(direct)
            .ok_or_else(|| invalid("singular or nonfinite direct basis"))?;
        Ok(Self {
            direct,
            reciprocal,
            bravais,
        })
    }
    /// From the reciprocal basis (rows, 1/Å, no 2π); predictions then use
    /// this basis exactly.
    /// # Errors
    /// A singular or nonfinite basis.
    pub fn from_reciprocal(reciprocal: M3, bravais: Bravais) -> Result<Self> {
        let direct = inverse_transpose(reciprocal)
            .ok_or_else(|| invalid("singular or nonfinite reciprocal basis"))?;
        Ok(Self {
            direct,
            reciprocal,
            bravais,
        })
    }
    /// Direct basis, rows in ångström.
    #[must_use]
    pub fn direct(&self) -> M3 {
        self.direct
    }
    /// Reciprocal basis, rows in 1/Å without 2π: `q = h · B`.
    #[must_use]
    pub fn reciprocal(&self) -> M3 {
        self.reciprocal
    }
    /// Bravais type (centring and holohedry).
    #[must_use]
    pub fn bravais(&self) -> Bravais {
        self.bravais
    }
    /// Reciprocal position `h · B` of indices.
    #[must_use]
    pub fn q(&self, hkl: [i64; 3]) -> V3 {
        row(hkl.map(|x| x as f64), self.reciprocal)
    }
}

// Same arithmetic as the native workflow's helpers, so shared results stay
// bit-identical.
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn unit(a: V3) -> V3 {
    let n = norm(a);
    a.map(|x| x / n)
}
fn row(v: V3, a: M3) -> V3 {
    std::array::from_fn(|j| (0..3).map(|k| v[k] * a[k][j]).sum())
}
fn rotate(v: V3, axis: V3, angle: f64) -> V3 {
    let (s, c) = angle.sin_cos();
    let k = cross(axis, v);
    let d = dot(axis, v);
    std::array::from_fn(|i| v[i] * c + k[i] * s + axis[i] * d * (1. - c))
}
fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    Some(if n.is_multiple_of(2) {
        f64::midpoint(v[n / 2 - 1], v[n / 2])
    } else {
        v[n / 2]
    })
}

#[cfg(test)]
#[path = "../measurement_tests.rs"]
mod tests;
