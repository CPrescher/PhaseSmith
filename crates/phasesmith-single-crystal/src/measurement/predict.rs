//! All reflections of refined lattices over a rotation scan.
use super::{
    AXIAL, HIDDEN, Lattice, PARTIAL, Result, Scan, V3, geometry::ray_geometry, invalid, norm, row,
};
use crate::spots::{prediction, reciprocal::Geometry};

/// Most indices enumerated for one lattice.
const MAX_INDICES: usize = 4_000_000;

/// One Ewald crossing of a reflection on the detector.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Crossing {
    /// Indices in the lattice's basis.
    pub hkl: [i64; 3],
    /// Predicted x, y (pixels) and ω (degrees, the turn nearest the hint).
    pub xyz: V3,
    /// Interplanar spacing, Å.
    pub d: f64,
}

/// Both Ewald crossings of every reflection with `d ≥ d_min` that the
/// lattice's centring allows and that land on a `shape = [rows, columns]`
/// detector (`-0.5 ≤ x < columns − 0.5`, same for y). ω is unwrapped to the
/// turn nearest `hint`; no rotation range is applied. Order: all first roots
/// (the nearer to the hint), then all second roots, each in `h`, `k`, `l`
/// enumeration order.
/// # Errors
/// Invalid geometry, resolution or shape, or more than four million indices.
pub fn crossings(
    lattice: &Lattice,
    geometry: &Geometry,
    shape: [usize; 2],
    d_min: f64,
    hint: f64,
) -> Result<Vec<Crossing>> {
    if !d_min.is_finite() || d_min <= 0. || shape.contains(&0) || !hint.is_finite() {
        return Err(invalid("invalid resolution limit, detector shape or hint"));
    }
    let b = lattice.reciprocal();
    let q_max = 1. / d_min;
    // One more than the bound |h_i| ≤ |a_i| / d_min, against rounding.
    let limits = lattice.direct().map(|a| (norm(a) * q_max).ceil() + 1.);
    let enumerated: f64 = limits.iter().map(|n| 2. * n + 1.).product();
    if enumerated.is_nan() || enumerated > 8. * MAX_INDICES as f64 {
        return Err(invalid("resolution limit gives too many indices"));
    }
    let limits = limits.map(|n| n as i64);
    let centring = lattice.bravais().centring();
    let mut hkl = Vec::new();
    let mut d = Vec::new();
    for x in -limits[0]..=limits[0] {
        for y in -limits[1]..=limits[1] {
            for z in -limits[2]..=limits[2] {
                let h = [x, y, z];
                if !centring.allows(h) {
                    continue;
                }
                let n = norm(row(h.map(|x| x as f64), b));
                if n > 0. && n <= q_max {
                    hkl.push(h);
                    d.push(1. / n);
                }
            }
        }
    }
    if hkl.len() > MAX_INDICES {
        return Err(invalid("resolution limit gives too many indices"));
    }
    let hints = vec![hint; hkl.len()];
    let mut out = Vec::new();
    for other in [false, true] {
        for (i, p) in prediction::predict(*geometry, b, &hkl, &hints, other)?
            .into_iter()
            .enumerate()
        {
            let v = p.values;
            if p.flags != 0
                || v[0] < -0.5
                || v[0] >= shape[1] as f64 - 0.5
                || v[1] < -0.5
                || v[1] >= shape[0] as f64 - 0.5
            {
                continue;
            }
            out.push(Crossing {
                hkl: hkl[i],
                xyz: v,
                d: d[i],
            });
        }
    }
    Ok(out)
}

/// Controls of [`predict`]. Non-exhaustive: start from
/// [`PredictOptions::new`] or `Default` and assign fields.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct PredictOptions {
    /// Resolution limit, Å ([`super::detector_resolution`] gives the
    /// detector corner).
    pub d_min: f64,
    /// Full rocking width for `|ζ| = 1`, degrees: a reflection is measured on
    /// the frames whose exposure overlaps `ω ± rocking_width / (2|ζ|)`
    /// ([`super::estimate_shape`] gives a data-derived value).
    pub rocking_width: f64,
    /// Below this `|ζ|` a reflection is [`AXIAL`]; its window uses `|ζ|`
    /// no smaller than half this value. Default 0.1.
    pub axial_zeta: f64,
    /// The half window never exceeds this many frames nor five rocking
    /// widths, whichever is larger. Default 20.
    pub max_window_frames: usize,
    /// Most reflections over all lattices. Default one million.
    pub max_reflections: usize,
}
impl Default for PredictOptions {
    fn default() -> Self {
        Self {
            d_min: 0.8,
            rocking_width: 1.,
            axial_zeta: 0.1,
            max_window_frames: 20,
            max_reflections: 1_000_000,
        }
    }
}
impl PredictOptions {
    /// Defaults with this resolution limit and rocking width.
    #[must_use]
    pub fn new(d_min: f64, rocking_width: f64) -> Self {
        Self {
            d_min,
            rocking_width,
            ..Self::default()
        }
    }
    fn check(&self) -> Result<()> {
        if !self.d_min.is_finite()
            || self.d_min <= 0.
            || !self.rocking_width.is_finite()
            || self.rocking_width <= 0.
            || !self.axial_zeta.is_finite()
            || !(0.0..1.0).contains(&self.axial_zeta)
            || self.max_window_frames == 0
            || self.max_reflections == 0
        {
            return Err(invalid("invalid prediction options"));
        }
        Ok(())
    }
}

/// One predicted reflection crossing inside the scan.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Reflection {
    /// Index of the lattice in the slice given to [`predict`].
    pub lattice: usize,
    /// Indices in that lattice's basis.
    pub hkl: [i64; 3],
    /// Predicted x, y (pixels) and ω (degrees, inside the scan range).
    pub xyz: V3,
    /// Interplanar spacing, Å.
    pub d: f64,
    /// Rocking factor ζ at the predicted position ([`super::RayGeometry`]).
    pub zeta: f64,
    /// First and last frame of the rocking window, inclusive, clipped to
    /// the scan.
    pub frames: [usize; 2],
    /// Frame whose exposure holds ω (the nearest exposure when ω falls
    /// between exposures), within `frames`.
    pub centre_frame: usize,
    /// [`AXIAL`], [`PARTIAL`] and [`HIDDEN`] as known before measuring.
    pub flags: u32,
}

/// Every reflection of the lattices that the centring allows to
/// `options.d_min`, crossing the Ewald sphere inside the scan's rotation
/// range on the detector (`shape = [rows, columns]`), with its rocking
/// window, sorted by ω, then lattice, then indices. Scans longer than one
/// turn list each crossing once per turn.
///
/// `visible`, when given, is asked whether the predicted pixel `(x, y)`
/// (rounded) is usable on the centre frame; when it is not, the reflection
/// is [`HIDDEN`]. A centre frame in [`Scan::excluded`] also hides it.
/// [`super::measure`] additionally hides reflections whose centre pixel is
/// masked on the centre frame it reads.
/// # Errors
/// Invalid scan, geometry or options, or the reflection budget exceeded.
pub fn predict(
    lattices: &[Lattice],
    geometry: &Geometry,
    scan: &Scan,
    shape: [usize; 2],
    options: &PredictOptions,
    visible: Option<&dyn Fn(usize, usize, usize) -> bool>,
) -> Result<Vec<Reflection>> {
    scan.check()?;
    options.check()?;
    geometry.convert(&[])?;
    let [lo, hi] = scan.range();
    let hint = f64::midpoint(lo, hi);
    let turns = ((hi - lo) / 360.).ceil() as i64 + 1;
    let mut out = Vec::new();
    for (index, lattice) in lattices.iter().enumerate() {
        for c in crossings(lattice, geometry, shape, options.d_min, hint)? {
            for k in -turns..=turns {
                let omega = c.xyz[2] + 360. * k as f64;
                if omega < lo || omega >= hi {
                    continue;
                }
                if out.len() >= options.max_reflections {
                    return Err(invalid("reflection budget exceeded"));
                }
                out.push(reflection(
                    index,
                    c.hkl,
                    [c.xyz[0], c.xyz[1], omega],
                    c.d,
                    geometry,
                    scan,
                    shape,
                    options,
                    visible,
                ));
            }
        }
    }
    out.sort_by(|a, b| {
        a.xyz[2]
            .total_cmp(&b.xyz[2])
            .then(a.lattice.cmp(&b.lattice))
            .then(a.hkl.cmp(&b.hkl))
    });
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn reflection(
    lattice: usize,
    hkl: [i64; 3],
    xyz: V3,
    d: f64,
    geometry: &Geometry,
    scan: &Scan,
    shape: [usize; 2],
    options: &PredictOptions,
    visible: Option<&dyn Fn(usize, usize, usize) -> bool>,
) -> Reflection {
    let zeta = ray_geometry([xyz[0], xyz[1]], geometry).zeta;
    let z = zeta.abs().max(options.axial_zeta / 2.);
    let half = (options.rocking_width / (2. * z))
        .min((options.max_window_frames as f64 * scan.step).max(5. * options.rocking_width));
    let [first, last] = scan.window(xyz[2], half);
    let n = scan.frames as i64;
    let frames = [
        first.clamp(0, n - 1) as usize,
        last.clamp(0, n - 1) as usize,
    ];
    let mut flags = 0;
    if first < 0
        || last > n - 1
        || scan
            .excluded
            .iter()
            .any(|&f| f >= frames[0] && f <= frames[1])
    {
        flags |= PARTIAL;
    }
    if zeta.abs() < options.axial_zeta || zeta.is_nan() {
        flags |= AXIAL;
    }
    // The exposure holding ω, or the nearest one when ω falls between
    // exposures, kept inside the window.
    let mut centre_frame = scan.frame_of(xyz[2]);
    let end = scan.bounds(centre_frame)[1];
    if xyz[2] > end && centre_frame + 1 < scan.frames {
        let next = scan.bounds(centre_frame + 1)[0];
        if next - xyz[2] < xyz[2] - end {
            centre_frame += 1;
        }
    }
    let centre_frame = centre_frame.clamp(frames[0], frames[1]);
    let x = ((xyz[0] + 0.5).floor().max(0.) as usize).min(shape[1] - 1);
    let y = ((xyz[1] + 0.5).floor().max(0.) as usize).min(shape[0] - 1);
    if scan.is_excluded(centre_frame) || visible.is_some_and(|v| !v(centre_frame, x, y)) {
        flags |= HIDDEN;
    }
    Reflection {
        lattice,
        hkl,
        xyz,
        d,
        zeta,
        frames,
        centre_frame,
        flags,
    }
}
