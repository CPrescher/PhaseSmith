//! Explicit planar detector geometry and direct-lattice Fourier moments.
use crate::{SpotError, invalid};
use std::f64::consts::TAU;

/// All vectors use one right-handed laboratory basis. Lengths are metres,
/// except wavelength in angstroms. Pixel indices refer to pixel centres.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    /// Laboratory position of pixel (0,0) centre, metres.
    pub origin: [f64; 3],
    /// Laboratory displacement per increasing x pixel, metres.
    pub fast: [f64; 3],
    /// Laboratory displacement per increasing y pixel, metres.
    pub slow: [f64; 3],
    /// Incident-beam direction, normalized on entry.
    pub beam: [f64; 3],
    /// Right-handed sample rotation direction, normalized on entry.
    pub axis: [f64; 3],
    /// Wavelength in angstroms.
    pub wavelength: f64,
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn unit(v: [f64; 3]) -> Result<[f64; 3], SpotError> {
    let norm = dot(v, v).sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return Err(invalid("geometry vectors must have finite nonzero norms"));
    }
    Ok(v.map(|x| x / norm))
}
impl Geometry {
    /// Convert x/y/omega-degree samples to sample-frame reciprocal vectors,
    /// q = R(axis,-omega) (unit(ray)-unit(beam))/wavelength, without 2*pi.
    ///
    /// # Errors
    /// Rejects nonfinite/degenerate geometry or samples.
    pub fn convert(self, samples: &[[f64; 3]]) -> Result<Vec<[f64; 3]>, SpotError> {
        if !self.wavelength.is_finite()
            || self.wavelength <= 0.0
            || self
                .origin
                .iter()
                .chain(&self.fast)
                .chain(&self.slow)
                .any(|x| !x.is_finite())
        {
            return Err(invalid("invalid detector geometry or wavelength"));
        }
        let fast = unit(self.fast)?;
        let slow = unit(self.slow)?;
        if dot(fast, slow).abs() > 1.0 - 1e-10 {
            return Err(invalid("detector pixel directions must be independent"));
        }
        let beam = unit(self.beam)?;
        let axis = unit(self.axis)?;
        samples
            .iter()
            .map(|s| {
                if s.iter().any(|x| !x.is_finite()) {
                    return Err(invalid("samples must be finite x/y/omega"));
                }
                let mut ray = [0.0; 3];
                for (j, v) in ray.iter_mut().enumerate() {
                    *v = self.origin[j] + s[0] * self.fast[j] + s[1] * self.slow[j];
                }
                let ray = unit(ray)?;
                let q = std::array::from_fn(|j| (ray[j] - beam[j]) / self.wavelength);
                let (sn, cs) = (-s[2].to_radians()).sin_cos();
                let cross = [
                    axis[1] * q[2] - axis[2] * q[1],
                    axis[2] * q[0] - axis[0] * q[2],
                    axis[0] * q[1] - axis[1] * q[0],
                ];
                let projection = dot(axis, q);
                let result = std::array::from_fn(|j| {
                    q[j] * cs + cross[j] * sn + axis[j] * projection * (1.0 - cs)
                });
                if result.iter().any(|x| !x.is_finite()) {
                    return Err(invalid("reciprocal conversion overflow"));
                }
                Ok(result)
            })
            .collect()
    }
}

/// Mean cos(2*pi*q.r), its gradient (3) and Hessian (row-major 9), evaluated
/// together in one ordered pass per trial real-space vector. O(N*K), O(K).
///
/// # Errors
/// Rejects empty/nonfinite points, excessive work or numerical overflow.
pub fn fourier(points: &[[f64; 3]], vectors: &[[f64; 3]]) -> Result<Vec<[f64; 13]>, SpotError> {
    if points.is_empty()
        || points
            .iter()
            .chain(vectors)
            .flatten()
            .any(|v| !v.is_finite())
        || points.len().saturating_mul(vectors.len()) > 200_000_000
    {
        return Err(invalid("invalid Fourier arrays or work budget exceeded"));
    }
    let scale = 1.0 / points.len() as f64;
    let mut result = Vec::with_capacity(vectors.len());
    for &v in vectors {
        let mut out = [0.0; 13];
        for &q in points {
            let phase = TAU * dot(q, v);
            let (s, c) = phase.sin_cos();
            out[0] += c;
            for j in 0..3 {
                out[1 + j] -= TAU * q[j] * s * scale;
                for k in 0..3 {
                    out[4 + 3 * j + k] -= TAU * TAU * q[j] * q[k] * c * scale;
                }
            }
        }
        out[0] *= scale;
        if out.iter().any(|v| !v.is_finite()) {
            return Err(invalid("Fourier overflow"));
        }
        result.push(out);
    }
    Ok(result)
}
