//! Ray geometry at a detector position and the Lorentz and polarisation
//! corrections.
use super::{Result, V3, cross, dot, invalid, norm, unit};
use crate::spots::reciprocal::Geometry;

/// Diffracted-ray geometry at a detector position.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct RayGeometry {
    /// Unit diffracted-ray direction in the laboratory frame.
    pub ray: V3,
    /// Row-major 2 × 2 Jacobian from pixel x/y to the two local angular
    /// coordinates (degrees) along `e1 = ray × beam / |ray × beam|` and
    /// `e2 = ray × e1`.
    pub planar: [f64; 4],
    /// Signed rocking factor `ζ = (ray × beam)·axis / |ray × beam|`; an
    /// angular width `w` perpendicular to the Ewald sphere spans `w / |ζ|`
    /// of rotation.
    pub zeta: f64,
    /// Sample-to-pixel distance, metres.
    pub distance: f64,
}

/// Ray, local angular pixel Jacobian and rocking factor at pixel `xy`.
/// Beam and axis are normalised; the ray is undefined (NaN) along the beam.
#[must_use]
pub fn ray_geometry(xy: [f64; 2], g: &Geometry) -> RayGeometry {
    let beam = unit(g.beam);
    let axis = unit(g.axis);
    let p = std::array::from_fn(|i| g.origin[i] + xy[0] * g.fast[i] + xy[1] * g.slow[i]);
    let length = norm(p);
    let ray = p.map(|v| v / length);
    let e = cross(ray, beam);
    let n = norm(e);
    let zeta = dot(e, axis) / n;
    let e1 = e.map(|v| v / n);
    let e2 = cross(ray, e1);
    let scale = 180. / std::f64::consts::PI / length;
    RayGeometry {
        ray,
        planar: [
            dot(e1, g.fast) * scale,
            dot(e1, g.slow) * scale,
            dot(e2, g.fast) * scale,
            dot(e2, g.slow) * scale,
        ],
        zeta,
        distance: length,
    }
}

/// Linear polarisation of the incident beam: the fraction `fraction` of the
/// intensity has its electric vector along `electric_vector` (perpendicular
/// to the beam), the rest along `beam × electric_vector`. `fraction = 0.5`
/// is unpolarised.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Polarization {
    /// Fraction of the intensity polarised along `electric_vector`, 0–1.
    pub fraction: f64,
    /// Electric-vector direction in the laboratory frame (normalised on use).
    pub electric_vector: V3,
}
impl Default for Polarization {
    /// 0.98 along laboratory y, a typical horizontally polarised synchrotron
    /// beam when y is horizontal (and the value of the native workflow).
    fn default() -> Self {
        Self {
            fraction: 0.98,
            electric_vector: [0., 1., 0.],
        }
    }
}
impl Polarization {
    /// Polarisation with this fraction along this electric vector.
    #[must_use]
    pub fn new(fraction: f64, electric_vector: V3) -> Self {
        Self {
            fraction,
            electric_vector,
        }
    }
    /// An unpolarised beam.
    #[must_use]
    pub fn unpolarized() -> Self {
        Self::new(0.5, [0., 1., 0.])
    }
    pub(super) fn axes(&self, beam: V3) -> Result<(V3, V3)> {
        let e = unit(self.electric_vector);
        if !(0.0..=1.0).contains(&self.fraction)
            || beam.iter().any(|v| !v.is_finite())
            || e.iter().any(|v| !v.is_finite())
            || dot(e, beam).abs() > 1e-8
        {
            return Err(invalid(
                "polarization fraction must be in 0..1 and its vector perpendicular to the beam",
            ));
        }
        Ok((e, cross(beam, e)))
    }
}

/// Polarisation factor `P = f (1 − (s·e)²) + (1 − f)(1 − (s·e⊥)²)` of a
/// unit diffracted ray `s`; the measured intensity is proportional to `P`.
/// # Errors
/// An invalid polarisation (see [`Polarization`]) or beam.
pub fn polarization_factor(ray: V3, beam: V3, polarization: &Polarization) -> Result<f64> {
    if ray.iter().any(|v| !v.is_finite()) {
        return Err(invalid("diffracted ray must be finite"));
    }
    let beam = unit(beam);
    let (e, orth) = polarization.axes(beam)?;
    let f = polarization.fraction;
    Ok(f * (1. - dot(ray, e).powi(2)) + (1. - f) * (1. - dot(ray, orth).powi(2)))
}

/// Lorentz and polarisation terms at one detector position.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Corrections {
    /// Rotation Lorentz factor `L = 1 / |(s × s₀)·axis|` (`= 1 / (sin 2θ |ζ|)`).
    pub lorentz: f64,
    /// Polarisation factor `P` ([`polarization_factor`]).
    pub polarization: f64,
    /// Multiplier that corrects a measured intensity: `1 / (L P)`.
    pub factor: f64,
}

/// Lorentz and polarisation corrections at pixel `xy`; multiply a measured
/// intensity and its σ by [`Corrections::factor`].
/// # Errors
/// An invalid polarisation, or a singular correction (on the rotation axis
/// or along the beam).
pub fn lorentz_polarization(
    xy: [f64; 2],
    g: &Geometry,
    polarization: &Polarization,
) -> Result<Corrections> {
    let beam = unit(g.beam);
    let axis = unit(g.axis);
    let (e, orth) = polarization.axes(beam)?;
    let ray = ray_geometry(xy, g).ray;
    let f = polarization.fraction;
    let p = f * (1. - dot(ray, e).powi(2)) + (1. - f) * (1. - dot(ray, orth).powi(2));
    let inverse_lorentz = dot(cross(ray, beam), axis).abs();
    let factor = inverse_lorentz / p;
    if !factor.is_finite() || factor <= 0. {
        return Err(invalid("singular Lorentz-polarization correction"));
    }
    Ok(Corrections {
        lorentz: 1. / inverse_lorentz,
        polarization: p,
        factor,
    })
}

/// Resolution (Å) at the detector corner farthest from the beam: the
/// smallest d any pixel of a `shape = [rows, columns]` detector records.
#[must_use]
pub fn detector_resolution(g: &Geometry, shape: [usize; 2]) -> f64 {
    let beam = unit(g.beam);
    let (w, h) = (
        shape[1].saturating_sub(1) as f64,
        shape[0].saturating_sub(1) as f64,
    );
    let mut two_theta: f64 = 0.;
    for (x, y) in [(0., 0.), (w, 0.), (0., h), (w, h)] {
        let p: V3 = std::array::from_fn(|i| g.origin[i] + x * g.fast[i] + y * g.slow[i]);
        two_theta = two_theta.max(dot(unit(p), beam).clamp(-1., 1.).acos());
    }
    g.wavelength / (2. * (two_theta / 2.).sin())
}
