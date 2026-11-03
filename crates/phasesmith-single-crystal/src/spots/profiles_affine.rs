//! First-principles parallelogram/voxel intersections and boundary-flux partials.
use super::profiles::Grid;

pub(crate) fn valid(a: [f64; 4]) -> bool {
    let norm = a[0].hypot(a[1]) * a[2].hypot(a[3]);
    a.iter().all(|v| v.is_finite())
        && norm.is_finite()
        && norm > 0.
        && (a[0] * a[3] - a[1] * a[2]).abs() > 1e-8 * norm
}

fn area(p: [[f64; 2]; 4], lo: [f64; 2], hi: [f64; 2]) -> f64 {
    let mut points = [[0.; 2]; 12];
    points[..4].copy_from_slice(&p);
    let mut n = 4;
    for (axis, edge, lower) in [
        (0, lo[0], true),
        (0, hi[0], false),
        (1, lo[1], true),
        (1, hi[1], false),
    ] {
        if n == 0 {
            return 0.;
        }
        let mut next = [[0.; 2]; 12];
        let mut count = 0;
        for j in 0..n {
            let a = points[j];
            let b = points[(j + 1) % n];
            let inside_a = if lower {
                a[axis] >= edge
            } else {
                a[axis] <= edge
            };
            let inside_b = if lower {
                b[axis] >= edge
            } else {
                b[axis] <= edge
            };
            if inside_a {
                next[count] = a;
                count += 1;
            }
            if inside_a != inside_b {
                let t = (edge - a[axis]) / (b[axis] - a[axis]);
                next[count] = std::array::from_fn(|k| a[k] + t * (b[k] - a[k]));
                count += 1;
            }
        }
        points = next;
        n = count;
    }
    (0..n)
        .map(|j| {
            let a = points[j];
            let b = points[(j + 1) % n];
            a[0] * b[1] - a[1] * b[0]
        })
        .sum::<f64>()
        .abs()
        / 2.
}

// Chord through the original parallelogram, restricted to the voxel side.
#[allow(clippy::float_cmp)] // Exact parallel edges have a zero denominator; no tolerance changes support.
fn chord(p: [[f64; 2]; 4], axis: usize, v: f64, lo: f64, hi: f64) -> f64 {
    let min = p.iter().map(|a| a[axis]).fold(f64::INFINITY, f64::min);
    let max = p.iter().map(|a| a[axis]).fold(f64::NEG_INFINITY, f64::max);
    if v <= min || v >= max {
        return 0.;
    }
    let mut left = f64::INFINITY;
    let mut right = f64::NEG_INFINITY;
    for j in 0..4 {
        let a = p[j];
        let b = p[(j + 1) % 4];
        if a[axis] == b[axis] {
            continue;
        }
        let t = (v - a[axis]) / (b[axis] - a[axis]);
        if (0.0..=1.0).contains(&t) {
            let w = a[1 - axis] + t * (b[1 - axis] - a[1 - axis]);
            left = left.min(w);
            right = right.max(w);
        }
    }
    (right.min(hi) - left.max(lo)).max(0.)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]
pub(crate) fn weights(
    grid: Grid,
    b: [f64; 6],
    centre: [f64; 3],
    a: [f64; 4],
    zscale: f64,
    mut visit: impl FnMut(usize, f64, [f64; 3]),
) {
    weights_impl::<false>(grid, b, centre, a, zscale, |i, w, d| {
        visit(i, w, [d[0], d[1], d[2]]);
    });
}

pub(crate) fn weights_with_widths(
    grid: Grid,
    b: [f64; 6],
    centre: [f64; 3],
    a: [f64; 4],
    zscale: f64,
    visit: impl FnMut(usize, f64, [f64; 5]),
) {
    weights_impl::<true>(grid, b, centre, a, zscale, visit);
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]
fn weights_impl<const WIDTHS: bool>(
    grid: Grid,
    b: [f64; 6],
    centre: [f64; 3],
    a: [f64; 4],
    zscale: f64,
    mut visit: impl FnMut(usize, f64, [f64; 5]),
) {
    let transform = |x: f64, y: f64| {
        [
            a[0] * (x - centre[0]) + a[1] * (y - centre[1]),
            a[2] * (x - centre[0]) + a[3] * (y - centre[1]),
        ]
    };
    let p = [
        transform(b[0], b[2]),
        transform(b[1], b[2]),
        transform(b[1], b[3]),
        transform(b[0], b[3]),
    ];
    let z0 = (b[4] - centre[2]) * zscale;
    let z1 = (b[5] - centre[2]) * zscale;
    let lower = [
        p.iter().map(|v| v[0]).fold(f64::INFINITY, f64::min),
        p.iter().map(|v| v[1]).fold(f64::INFINITY, f64::min),
        z0.min(z1),
    ];
    let upper = [
        p.iter().map(|v| v[0]).fold(f64::NEG_INFINITY, f64::max),
        p.iter().map(|v| v[1]).fold(f64::NEG_INFINITY, f64::max),
        z0.max(z1),
    ];
    let origin = std::array::from_fn::<_, 3, _>(|j| -(grid.shape[j] as f64) * grid.step[j] / 2.);
    let mut ranges = [(0, 0); 3];
    for j in 0..3 {
        if upper[j] <= origin[j] || lower[j] >= -origin[j] {
            return;
        }
        let start = ((lower[j] - origin[j]) / grid.step[j]).floor().max(0.) as usize;
        let end = (((upper[j] - origin[j]) / grid.step[j]).floor().max(0.) as usize + 1)
            .min(grid.shape[j]);
        ranges[j] = (start.min(grid.shape[j]), end);
    }
    let voxel_area = grid.step[0] * grid.step[1];
    for y in ranges[1].0..ranges[1].1 {
        for x in ranges[0].0..ranges[0].1 {
            let lo = [
                origin[0] + x as f64 * grid.step[0],
                origin[1] + y as f64 * grid.step[1],
            ];
            let hi = [lo[0] + grid.step[0], lo[1] + grid.step[1]];
            let wxy = area(p, lo, hi) / voxel_area;
            if wxy <= 0. {
                continue;
            }
            let right = chord(p, 0, hi[0], lo[1], hi[1]);
            let left = chord(p, 0, lo[0], lo[1], hi[1]);
            let top = chord(p, 1, hi[1], lo[0], hi[0]);
            let bottom = chord(p, 1, lo[1], lo[0], hi[0]);
            let dx = (right - left) / voxel_area;
            let dy = (top - bottom) / voxel_area;
            let dwidth = if WIDTHS {
                (hi[0] * right - lo[0] * left + hi[1] * top - lo[1] * bottom) / voxel_area
                    - 2. * wxy
            } else {
                0.
            };
            for z in ranges[2].0..ranges[2].1 {
                let l = origin[2] + z as f64 * grid.step[2];
                let h = l + grid.step[2];
                let wz = (h.min(upper[2]) - l.max(lower[2])).max(0.) / grid.step[2];
                if wz <= 0. {
                    continue;
                }
                let dz =
                    (f64::from(h < upper[2]) - f64::from(l > lower[2])) / grid.step[2] * zscale;
                visit(
                    (z * grid.shape[1] + y) * grid.shape[0] + x,
                    wxy * wz,
                    [
                        (dx * a[0] + dy * a[2]) * wz,
                        (dx * a[1] + dy * a[3]) * wz,
                        wxy * dz,
                        dwidth * wz,
                        if WIDTHS {
                            wxy * ((h * f64::from(h < upper[2]) - l * f64::from(l > lower[2]))
                                / grid.step[2]
                                - wz)
                        } else {
                            0.
                        },
                    ],
                );
            }
        }
    }
}
