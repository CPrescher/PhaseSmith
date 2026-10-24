//! Rotation Ewald crossings and planar detector centroids with fused derivatives.
use crate::{SpotError, invalid, reciprocal::Geometry};
use std::f64::consts::PI;

/// No isolated Ewald crossing, including the zero reflection.
pub const NO_CROSSING: u32 = 1;
/// Tangent crossing: angular derivatives are singular.
pub const TANGENT: u32 = 2;
/// Ray is parallel to the detector or intersects behind the sample.
pub const NO_PANEL: u32 = 4;

/// Predicted x/y/omega in pixels/degrees and derivatives wrt row-major B (1/A).
#[derive(Clone, Debug)]
pub struct Prediction {
    /// x, y, omega degrees, unwrapped nearest to the hint.
    pub values: [f64; 3],
    /// Row-major 3 by 9 analytical Jacobian, same pass as values.
    pub jacobian: [f64; 27],
    /// Zero means valid; invalid values and Jacobians are NaN.
    pub flags: u32,
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn normalized(a: [f64; 3]) -> [f64; 3] {
    let length = dot(a, a).sqrt();
    a.map(|v| v / length)
}
fn rotate(a: [f64; 3], v: [f64; 3], angle: f64) -> [f64; 3] {
    let (s, c) = angle.sin_cos();
    let av = dot(a, v);
    let axv = cross(a, v);
    std::array::from_fn(|i| v[i] * c + axv[i] * s + a[i] * av * (1.0 - c))
}
fn failed(flags: u32) -> Prediction {
    Prediction {
        values: [f64::NAN; 3],
        jacobian: [f64::NAN; 27],
        flags,
    }
}

/// Predict the nearest (or other) Ewald root for each indexed reflection.
/// B has reciprocal vectors as rows, so g = h @ B. No 2*pi.
///
/// # Errors
/// Rejects nonfinite/degenerate geometry/B, invalid hints and unbounded work.
#[allow(clippy::too_many_lines)] // One audited centroid/Jacobian pass.
pub fn predict(
    geometry: Geometry,
    basis: [[f64; 3]; 3],
    hkl: &[[i64; 3]],
    hints: &[f64],
    other: bool,
) -> Result<Vec<Prediction>, SpotError> {
    geometry.convert(&[])?;
    let volume = dot(basis[0], cross(basis[1], basis[2]));
    if hkl.len() != hints.len()
        || hkl.len() > 1_000_000
        || hints.iter().any(|v| !v.is_finite() || v.abs() > 1e8)
        || hkl.iter().flatten().any(|v| v.unsigned_abs() > 1_000_000)
        || basis.iter().flatten().any(|v| !v.is_finite())
        || !volume.is_finite()
        || volume.abs() < 1e-15
    {
        return Err(invalid(
            "invalid prediction arrays, reciprocal basis or budgets",
        ));
    }
    let a = normalized(geometry.axis);
    let b = normalized(geometry.beam);
    let n = normalized(cross(geometry.fast, geometry.slow));
    let panel_distance = dot(n, geometry.origin);
    if panel_distance.abs() < 1e-12 {
        return Err(invalid("detector plane intersects sample"));
    }
    let ff = dot(geometry.fast, geometry.fast);
    let ss = dot(geometry.slow, geometry.slow);
    let fs = dot(geometry.fast, geometry.slow);
    let determinant = ff * ss - fs * fs;
    if !determinant.is_finite() || determinant <= 0.0 {
        return Err(invalid("degenerate detector pixel metric"));
    }
    let dual = [
        std::array::from_fn::<_, 3, _>(|i| {
            (ss * geometry.fast[i] - fs * geometry.slow[i]) / determinant
        }),
        std::array::from_fn::<_, 3, _>(|i| {
            (ff * geometry.slow[i] - fs * geometry.fast[i]) / determinant
        }),
    ];
    let mut out = Vec::with_capacity(hkl.len());
    for (h, &hint) in hkl.iter().zip(hints) {
        let g = std::array::from_fn(|j| (0..3).map(|i| h[i] as f64 * basis[i][j]).sum());
        let along = dot(a, g);
        let constant = dot(a, b) * along;
        let u = dot(b, g) - constant;
        let v = dot(b, cross(a, g));
        let radius = u.hypot(v);
        let target = -geometry.wavelength * dot(g, g) / 2.0 - constant;
        if !radius.is_finite() || !target.is_finite() {
            return Err(invalid("prediction overflow"));
        }
        if radius <= 1e-14 || target.abs() > radius {
            out.push(failed(NO_CROSSING));
            continue;
        }
        let phase = v.atan2(u);
        let delta = (target / radius).acos();
        let unwrap = |theta: f64| {
            let deg = theta.to_degrees();
            deg + 360.0 * ((hint - deg) / 360.0 + 0.5).floor()
        };
        let roots = [unwrap(phase + delta), unwrap(phase - delta)];
        let nearest = usize::from((roots[1] - hint).abs() < (roots[0] - hint).abs());
        let omega = roots[if other { 1 - nearest } else { nearest }];
        let angle = omega.to_radians();
        let lab = rotate(a, g, angle);
        let angular_direction = cross(a, lab);
        let slope = dot(b, angular_direction);
        if slope.abs() <= 1e-10 * radius.max(1.0) {
            out.push(failed(TANGENT));
            continue;
        }
        let inverse_beam = rotate(a, b, -angle);
        let theta_gradient: [f64; 3] =
            std::array::from_fn(|j| -(inverse_beam[j] + geometry.wavelength * g[j]) / slope);
        let ray = std::array::from_fn(|i| b[i] + geometry.wavelength * lab[i]);
        let denominator = dot(n, ray);
        let t = panel_distance / denominator;
        if denominator.abs() <= 1e-12 || t <= 0.0 || !t.is_finite() {
            out.push(failed(NO_PANEL));
            continue;
        }
        let displacement = std::array::from_fn(|j| t * ray[j] - geometry.origin[j]);
        let values = [
            dot(dual[0], displacement),
            dot(dual[1], displacement),
            omega,
        ];
        let mut jacobian = [0.0; 27];
        for j in 0..3 {
            let unit = std::array::from_fn(|k| if k == j { 1.0 } else { 0.0 });
            let rotated_unit = rotate(a, unit, angle);
            let dr = std::array::from_fn(|k| {
                geometry.wavelength * (rotated_unit[k] + angular_direction[k] * theta_gradient[j])
            });
            let projection = dot(n, dr) / denominator;
            let dp = std::array::from_fn(|k| t * (dr[k] - ray[k] * projection));
            for (i, &index) in h.iter().enumerate() {
                jacobian[3 * i + j] = dot(dual[0], dp) * index as f64;
                jacobian[9 + 3 * i + j] = dot(dual[1], dp) * index as f64;
                jacobian[18 + 3 * i + j] = theta_gradient[j] * 180.0 / PI * index as f64;
            }
        }
        if values.iter().chain(&jacobian).any(|v| !v.is_finite()) {
            return Err(invalid("prediction/Jacobian overflow"));
        }
        out.push(Prediction {
            values,
            jacobian,
            flags: 0,
        });
    }
    Ok(out)
}
