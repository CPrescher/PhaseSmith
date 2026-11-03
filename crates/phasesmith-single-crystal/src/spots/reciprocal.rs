//! Explicit planar detector geometry and direct-lattice Fourier moments.
use crate::spots::{SpotError, invalid};
use std::f64::consts::TAU;

/// Reciprocal vector and its observation-coordinate Jacobian.
pub type ReciprocalSample = ([f64; 3], [[f64; 3]; 3]);

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
    /// Reciprocal values and analytical observation Jacobians in one sample pass.
    /// Jacobian rows are q components; columns are x/y pixels and omega degrees.
    /// # Errors
    /// Rejects nonfinite/degenerate geometry, rays or samples.
    pub fn convert_with_jacobian(
        self,
        samples: &[[f64; 3]],
    ) -> Result<Vec<ReciprocalSample>, SpotError> {
        if !self.wavelength.is_finite()
            || self.wavelength <= 0.
            || self
                .origin
                .iter()
                .chain(&self.fast)
                .chain(&self.slow)
                .any(|x| !x.is_finite())
        {
            return Err(invalid("invalid detector geometry or wavelength"));
        }
        if dot(unit(self.fast)?, unit(self.slow)?).abs() > 1. - 1e-10 {
            return Err(invalid("detector pixel directions must be independent"));
        }
        let beam = unit(self.beam)?;
        let axis = unit(self.axis)?;
        let cross = |v: [f64; 3]| {
            [
                axis[1] * v[2] - axis[2] * v[1],
                axis[2] * v[0] - axis[0] * v[2],
                axis[0] * v[1] - axis[1] * v[0],
            ]
        };
        samples
            .iter()
            .map(|s| {
                if s.iter().any(|x| !x.is_finite()) {
                    return Err(invalid("samples must be finite x/y/omega"));
                }
                let ray = std::array::from_fn(|j| {
                    self.origin[j] + s[0] * self.fast[j] + s[1] * self.slow[j]
                });
                let length = dot(ray, ray).sqrt();
                let ray = unit(ray)?;
                let (sn, cs) = (-s[2].to_radians()).sin_cos();
                let rotate = |v: [f64; 3]| {
                    let c = cross(v);
                    let projection = dot(axis, v);
                    std::array::from_fn(|j| {
                        v[j] * cs + c[j] * sn + axis[j] * projection * (1. - cs)
                    })
                };
                let q = rotate(std::array::from_fn(|j| {
                    (ray[j] - beam[j]) / self.wavelength
                }));
                let dx = rotate(std::array::from_fn(|j| {
                    (self.fast[j] - ray[j] * dot(ray, self.fast)) / (self.wavelength * length)
                }));
                let dy = rotate(std::array::from_fn(|j| {
                    (self.slow[j] - ray[j] * dot(ray, self.slow)) / (self.wavelength * length)
                }));
                let dw = cross(q).map(|v| -v * std::f64::consts::PI / 180.);
                let jacobian = std::array::from_fn(|j| [dx[j], dy[j], dw[j]]);
                if q.iter()
                    .chain(jacobian.iter().flatten())
                    .any(|x| !x.is_finite())
                {
                    return Err(invalid("reciprocal conversion overflow"));
                }
                Ok((q, jacobian))
            })
            .collect()
    }

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
    check_fourier(points, vectors)?;
    vectors.iter().map(|&v| fourier_one(points, v)).collect()
}

pub(crate) fn check_fourier(points: &[[f64; 3]], vectors: &[[f64; 3]]) -> Result<(), SpotError> {
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
    Ok(())
}

/// Immutable factors for a complete bounded ascent on one ordered point cloud.
pub(crate) struct PreparedFourier<'a> {
    points: &'a [[f64; 3]],
    coefficients: Vec<[f64; 12]>,
    scale: f64,
}
impl<'a> PreparedFourier<'a> {
    pub(crate) fn new(points: &'a [[f64; 3]]) -> Self {
        Self {
            points,
            scale: 1. / points.len() as f64,
            coefficients: points
                .iter()
                .map(|q| {
                    std::array::from_fn(|j| {
                        if j < 3 {
                            TAU * q[j]
                        } else {
                            let k = j - 3;
                            TAU * TAU * q[k / 3] * q[k % 3]
                        }
                    })
                })
                .collect(),
        }
    }
    pub(crate) fn evaluate(&self, v: [f64; 3]) -> Result<[f64; 13], SpotError> {
        let mut out = [0.; 13];
        for (&q, coefficients) in self.points.iter().zip(&self.coefficients) {
            let (s, c) = (TAU * dot(q, v)).sin_cos();
            out[0] += c;
            for j in 0..3 {
                out[1 + j] -= coefficients[j] * s * self.scale;
            }
            for j in 0..9 {
                out[4 + j] -= coefficients[3 + j] * c * self.scale;
            }
        }
        out[0] *= self.scale;
        if out.iter().any(|x| !x.is_finite()) {
            return Err(invalid("Fourier overflow"));
        }
        Ok(out)
    }
}

pub(crate) fn fourier_one(points: &[[f64; 3]], v: [f64; 3]) -> Result<[f64; 13], SpotError> {
    let scale = 1.0 / points.len() as f64;
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
    Ok(out)
}

/// Two independent vectors share point loads; each lane keeps its ordered sums.
pub(crate) fn fourier_two(
    points: &[[f64; 3]],
    vectors: [[f64; 3]; 2],
) -> Result<[[f64; 13]; 2], SpotError> {
    let scale = 1. / points.len() as f64;
    let mut out = [[0.; 2]; 13];
    for &q in points {
        let mut sine = [0.; 2];
        let mut cosine = [0.; 2];
        for lane in 0..2 {
            let phase = TAU * dot(q, vectors[lane]);
            (sine[lane], cosine[lane]) = phase.sin_cos();
            out[0][lane] += cosine[lane];
        }
        for j in 0..3 {
            for lane in 0..2 {
                out[1 + j][lane] -= TAU * q[j] * sine[lane] * scale;
            }
            for k in 0..3 {
                for lane in 0..2 {
                    out[4 + 3 * j + k][lane] -= TAU * TAU * q[j] * q[k] * cosine[lane] * scale;
                }
            }
        }
    }
    let result = std::array::from_fn(|lane| {
        std::array::from_fn(|j| {
            if j == 0 {
                out[j][lane] * scale
            } else {
                out[j][lane]
            }
        })
    });
    if result.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid("Fourier overflow"));
    }
    Ok(result)
}

#[cfg(test)]
mod prepared_fourier_tests {
    use super::*;
    #[test]
    fn prepared_preserves_every_scalar_moment_bit() {
        let points: Vec<_> = (0..2001)
            .map(|i| {
                [
                    (f64::from(i) * 1.731).sin(),
                    (f64::from(i) * 2.919).cos(),
                    (f64::from(i) * 0.73).sin() * 0.83,
                ]
            })
            .collect();
        let cloud = PreparedFourier::new(&points);
        for vectors in [
            [[1.2, -3.4, 5.6], [0., -0., 0.]],
            [[-7.9, 2.3, 1.1], [7.9, -2.3, -1.1]],
        ] {
            for v in vectors {
                let original = fourier_one(&points, v).unwrap().map(f64::to_bits);
                assert_eq!(original, cloud.evaluate(v).unwrap().map(f64::to_bits));
            }
        }
    }
}
