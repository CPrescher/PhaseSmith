#![allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::unnecessary_wraps,
    clippy::needless_range_loop
)]
use super::*;
use crate::lattice_symmetry::{Bravais, LatticeSystem, equivalent_representative, transform_index};
use crate::spots::{Bounds, reciprocal::Geometry};
use std::sync::atomic::{AtomicUsize, Ordering};

// ---------------------------------------------------------------- fixtures

fn geometry() -> Geometry {
    // 300 × 300 pixels of 100 µm at 80 mm, beam centre near (150, 150),
    // vertical rotation axis.
    Geometry {
        origin: [-0.0150, -0.0149, 0.08],
        fast: [0.0001, 0., 0.],
        slow: [0., 0.0001, 0.],
        beam: [0., 0., 1.],
        axis: [0., 1., 0.],
        wavelength: 0.4,
    }
}
const SHAPE: [usize; 2] = [300, 300];

fn rotation(a: f64, b: f64, c: f64) -> M3 {
    let (sa, ca) = a.sin_cos();
    let (sb, cb) = b.sin_cos();
    let (sc, cc) = c.sin_cos();
    let rz = [[ca, -sa, 0.], [sa, ca, 0.], [0., 0., 1.]];
    let ry = [[cb, 0., sb], [0., 1., 0.], [-sb, 0., cb]];
    let rx = [[1., 0., 0.], [0., cc, -sc], [0., sc, cc]];
    let mul = |p: M3, q: M3| -> M3 {
        std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| p[i][k] * q[k][j]).sum()))
    };
    mul(rz, mul(ry, rx))
}
/// Rows of `diag(a, b, c)` rotated by `r`.
fn basis(cell: [f64; 3], r: M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| r[j][i] * cell[i]))
}
fn lattice(bravais: Bravais) -> Lattice {
    Lattice::new(basis([8.1, 9.3, 10.7], rotation(0.3, 0.7, 1.1)), bravais).unwrap()
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
fn uniform(seed: u64) -> f64 {
    ((splitmix(seed) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}
fn gaussian(seed: u64) -> f64 {
    let (u, v) = (uniform(seed), uniform(seed ^ 0x5555_aaaa_5555_aaaa));
    (-2. * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}
/// Deterministic Poisson sample (Knuth below 30, rounded normal above).
fn poisson(lambda: f64, seed: u64) -> f64 {
    if lambda >= 30. {
        return (lambda + lambda.sqrt() * gaussian(seed)).round().max(0.);
    }
    let limit = (-lambda).exp();
    let (mut k, mut p) = (0u64, 1.);
    loop {
        p *= uniform(seed.wrapping_mul(31).wrapping_add(k));
        if p <= limit {
            return k as f64;
        }
        k += 1;
    }
}
fn phi(x: f64) -> f64 {
    super::shape::phi(x)
}

/// A spot: total intensity at (x, y, ω) with spatial σ (pixels) and rocking
/// σ in ω (degrees).
#[derive(Clone, Copy)]
struct Spot {
    x: f64,
    y: f64,
    omega: f64,
    intensity: f64,
    sigma_xy: f64,
    sigma_omega: f64,
}
impl Spot {
    fn pixel(&self, x: usize, y: usize) -> f64 {
        let s = self.sigma_xy;
        let fx = phi((x as f64 + 0.5 - self.x) / s) - phi((x as f64 - 0.5 - self.x) / s);
        let fy = phi((y as f64 + 0.5 - self.y) / s) - phi((y as f64 - 0.5 - self.y) / s);
        fx * fy
    }
    fn frame(&self, b: [f64; 2]) -> f64 {
        phi((b[1] - self.omega) / self.sigma_omega) - phi((b[0] - self.omega) / self.sigma_omega)
    }
    /// Expected counts in a box over frames.
    fn expected(&self, bounds: Bounds, frames: &[usize], scan: &Scan) -> f64 {
        let mut space = 0.;
        for y in bounds.y0..bounds.y1 {
            for x in bounds.x0..bounds.x1 {
                space += self.pixel(x, y);
            }
        }
        let time: f64 = frames.iter().map(|&f| self.frame(scan.bounds(f))).sum();
        self.intensity * space * time
    }
}

type MaskFn = Box<dyn Fn(usize, usize, usize) -> bool + Sync>;
type ExtraFn = Box<dyn Fn(usize, usize) -> f64 + Sync>;
struct Synthetic {
    shape: [usize; 2],
    scan: Scan,
    spots: Vec<Spot>,
    background: [f64; 3],
    noise: bool,
    poisson: bool,
    mask: Option<MaskFn>,
    extra: Option<ExtraFn>,
    reads: AtomicUsize,
}
impl Synthetic {
    fn new(shape: [usize; 2], scan: Scan, spots: Vec<Spot>) -> Self {
        Self {
            shape,
            scan,
            spots,
            background: [20., 0.02, 0.03],
            noise: true,
            poisson: false,
            mask: None,
            extra: None,
            reads: AtomicUsize::new(0),
        }
    }
    fn expectation(&self, f: usize) -> Vec<f64> {
        let [h, w] = self.shape;
        let b = self.background;
        let mut v: Vec<f64> = (0..h * w)
            .map(|i| b[0] + b[1] * (i % w) as f64 + b[2] * (i / w) as f64)
            .collect();
        if let Some(extra) = &self.extra {
            for (i, value) in v.iter_mut().enumerate() {
                *value += extra(i % w, i / w);
            }
        }
        let bounds = self.scan.bounds(f);
        for s in &self.spots {
            let t = s.frame(bounds) * s.intensity;
            if t < 1e-6 {
                continue;
            }
            let r = (6. * s.sigma_xy).ceil() as i64;
            for y in (s.y.round() as i64 - r).max(0)..=(s.y.round() as i64 + r).min(h as i64 - 1) {
                for x in
                    (s.x.round() as i64 - r).max(0)..=(s.x.round() as i64 + r).min(w as i64 - 1)
                {
                    v[y as usize * w + x as usize] += t * s.pixel(x as usize, y as usize);
                }
            }
        }
        v
    }
    fn frame(&self, f: usize) -> Result<Frame> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        let [h, w] = self.shape;
        let mut values = self.expectation(f);
        if self.poisson {
            for (i, v) in values.iter_mut().enumerate() {
                *v = poisson(*v, (f * h * w + i) as u64);
            }
        } else if self.noise {
            for (i, v) in values.iter_mut().enumerate() {
                *v += v.sqrt() * gaussian((f * h * w + i) as u64);
            }
        }
        let mask = (0..h * w)
            .map(|i| self.mask.as_ref().is_some_and(|m| m(f, i % w, i / w)))
            .collect();
        Ok(Frame::new(values, mask))
    }
}

fn scan() -> Scan {
    Scan::new(0., 0.5, 0.5, 80).unwrap()
}

/// Spots of every predicted reflection, intensities spread over three decades
/// with some weak ones and one exactly zero.
fn spots_for(reflections: &[Reflection], sigma: f64) -> Vec<Spot> {
    reflections
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let u = uniform(i as u64 * 7919 + 13);
            Spot {
                x: r.xyz[0],
                y: r.xyz[1],
                omega: r.xyz[2],
                intensity: if i == 3 { 0. } else { 10f64.powf(1.7 + 3. * u) },
                sigma_xy: 1.,
                sigma_omega: sigma / r.zeta.abs().max(0.05),
            }
        })
        .collect()
}

fn window_frames(r: &Reflection, scan: &Scan) -> Vec<usize> {
    (r.frames[0]..=r.frames[1])
        .filter(|&f| !scan.is_excluded(f))
        .collect()
}

// --------------------------------------------------------- basic geometry

#[test]
fn lattice_bases_are_mutually_inverse_transposes() {
    let l = lattice(Bravais::OrthorhombicP);
    let back = Lattice::from_reciprocal(l.reciprocal(), Bravais::OrthorhombicP).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let d: f64 = (0..3)
                .map(|k| l.direct()[i][k] * l.reciprocal()[j][k])
                .sum();
            assert!((d - f64::from(u8::from(i == j))).abs() < 1e-14);
            assert!((back.direct()[i][j] - l.direct()[i][j]).abs() < 1e-12);
        }
    }
    assert!(Lattice::new([[1., 0., 0.], [2., 0., 0.], [0., 0., 1.]], Bravais::CubicP).is_err());
}

#[test]
fn scan_checks_and_windows() {
    assert!(Scan::new(0., 0., 0.5, 3).is_err());
    assert!(Scan::new(0., 0.5, 0.6, 3).is_err());
    assert!(
        Scan::new(0., 0.5, 0.5, 3)
            .unwrap()
            .with_excluded(vec![3])
            .is_err()
    );
    let s = Scan::new(10., 1., 0.8, 5).unwrap();
    assert_eq!(s.range(), [10., 14.8]);
    assert_eq!(s.bounds(2), [12., 12.8]);
    assert_eq!(s.frame_of(12.9), 2);
    // Exposures overlapping 12.5 ± 0.6: frames 1 (11–11.8)? no; 2 and 3? 3 starts at 13.
    assert_eq!(s.window(12.5, 0.6), [2, 3]);
    assert_eq!(s.window(12.5, 0.4), [2, 2]);
}

#[test]
fn ray_jacobian_matches_finite_differences_of_ray_angles() {
    let g = geometry();
    let xy = [40.3, 230.7];
    let r = ray_geometry(xy, &g);
    let e1 = unit(cross(r.ray, unit(g.beam)));
    let e2 = cross(r.ray, e1);
    let h = 1e-3;
    for axis in 0..2 {
        let (mut p, mut m) = (xy, xy);
        p[axis] += h;
        m[axis] -= h;
        let (rp, rm) = (ray_geometry(p, &g).ray, ray_geometry(m, &g).ray);
        for (i, e) in [e1, e2].iter().enumerate() {
            let a = dot(rp, *e).atan2(dot(rp, r.ray)).to_degrees();
            let b = dot(rm, *e).atan2(dot(rm, r.ray)).to_degrees();
            assert!(((a - b) / (2. * h) - r.planar[2 * i + axis]).abs() < 1e-9);
        }
    }
}

#[test]
fn lorentz_and_polarization_follow_their_definitions() {
    let g = geometry();
    for xy in [[20., 30.], [150., 290.], [280., 140.]] {
        let r = ray_geometry(xy, &g);
        let cos2t = dot(r.ray, g.beam);
        let sin2t = (1. - cos2t * cos2t).sqrt();
        let unpolarized = lorentz_polarization(xy, &g, &Polarization::unpolarized()).unwrap();
        assert!((unpolarized.polarization - f64::midpoint(1., cos2t * cos2t)).abs() < 1e-12);
        assert!(
            (unpolarized.lorentz - 1. / (sin2t * r.zeta.abs())).abs() < 1e-9 * unpolarized.lorentz
        );
        assert!(
            (unpolarized.factor - 1. / (unpolarized.lorentz * unpolarized.polarization)).abs()
                < 1e-12 * unpolarized.factor
        );
        // Fully polarised along y: P = 1 − s_y².
        let full = lorentz_polarization(xy, &g, &Polarization::new(1., [0., 1., 0.])).unwrap();
        assert!((full.polarization - (1. - r.ray[1].powi(2))).abs() < 1e-12);
        assert_eq!(
            polarization_factor(r.ray, g.beam, &Polarization::new(1., [0., 1., 0.])).unwrap(),
            full.polarization
        );
    }
    assert!(lorentz_polarization([1., 1.], &g, &Polarization::new(0.9, [0., 0., 1.])).is_err());
    assert!(lorentz_polarization([1., 1.], &g, &Polarization::new(1.2, [0., 1., 0.])).is_err());
}

#[test]
fn detector_resolution_is_the_far_corner() {
    let g = geometry();
    let d = detector_resolution(&g, SHAPE);
    // Farthest corner (0, 299): offsets −0.0150, −0.0149 + 0.0299 = 0.0150.
    let r = 0.0150f64.hypot(0.0150);
    let tt = (r / 0.08).atan();
    assert!((d - 0.4 / (2. * (tt / 2.).sin())).abs() < 1e-9);
}

// ------------------------------------------------------------- prediction

#[test]
fn crossings_agree_with_the_spot_predictor_and_respect_centring_and_resolution() {
    let g = geometry();
    for bravais in [
        Bravais::OrthorhombicP,
        Bravais::OrthorhombicI,
        Bravais::OrthorhombicF,
    ] {
        let l = lattice(bravais);
        let c = crossings(&l, &g, SHAPE, 1.6, 20.).unwrap();
        assert!(c.len() > 10, "{}", c.len());
        for x in &c {
            assert!(bravais.centring().allows(x.hkl));
            assert!(x.d >= 1.6 - 1e-12);
            assert!((x.d - 1. / norm(l.q(x.hkl))).abs() < 1e-12);
            assert!(x.xyz[0] >= -0.5 && x.xyz[0] < 299.5 && x.xyz[1] >= -0.5 && x.xyz[1] < 299.5);
        }
    }
    // Brute force: every reflection with d ≥ d_min appears in both roots
    // when on the detector.
    let l = lattice(Bravais::OrthorhombicP);
    let c = crossings(&l, &g, SHAPE, 1.6, 20.).unwrap();
    let mut expected = 0;
    for h in -8..=8i64 {
        for k in -8..=8i64 {
            for m in -8..=8i64 {
                let hkl = [h, k, m];
                if hkl == [0; 3] || norm(l.q(hkl)) > 1. / 1.6 {
                    continue;
                }
                for other in [false, true] {
                    let p =
                        crate::spots::prediction::predict(g, l.reciprocal(), &[hkl], &[20.], other)
                            .unwrap()
                            .remove(0);
                    let v = p.values;
                    if p.flags == 0 && v[0] >= -0.5 && v[0] < 299.5 && v[1] >= -0.5 && v[1] < 299.5
                    {
                        expected += 1;
                        assert!(c.iter().any(|x| x.hkl == hkl && x.xyz == v));
                    }
                }
            }
        }
    }
    assert_eq!(c.len(), expected);
    let p = crossings(&l, &g, SHAPE, 1.6, 20.).unwrap().len();
    let i = crossings(&lattice(Bravais::OrthorhombicI), &g, SHAPE, 1.6, 20.)
        .unwrap()
        .len();
    assert!(i * 10 > p * 4 && i * 10 < p * 6, "{i} of {p}");
}

#[test]
fn prediction_windows_flags_and_order() {
    let g = geometry();
    let s = scan().with_excluded(vec![40]).unwrap();
    let options = PredictOptions::new(1.6, 1.5);
    let r = predict(
        &[lattice(Bravais::OrthorhombicP)],
        &g,
        &s,
        SHAPE,
        &options,
        None,
    )
    .unwrap();
    assert!(r.len() > 100, "{}", r.len());
    assert!(r.windows(2).all(|w| w[0].xyz[2] <= w[1].xyz[2]));
    let [lo, hi] = s.range();
    for x in &r {
        assert!(x.xyz[2] >= lo && x.xyz[2] < hi);
        assert!(x.frames[0] <= x.centre_frame && x.centre_frame <= x.frames[1]);
        let half = 1.5 / (2. * x.zeta.abs().max(0.05));
        let half = half.min((20. * 0.5f64).max(7.5));
        // Every frame overlapping ω ± half is inside the window.
        for f in 0..s.frames {
            let b = s.bounds(f);
            let overlaps = b[0] < x.xyz[2] + half && b[1] > x.xyz[2] - half;
            assert_eq!(overlaps, f >= x.frames[0] && f <= x.frames[1], "{x:?} {f}");
        }
        let cut = x.xyz[2] - half < lo
            || x.xyz[2] + half > hi
            || (x.frames[0]..=x.frames[1]).contains(&40);
        assert_eq!(x.flags & PARTIAL != 0, cut);
        assert_eq!(x.flags & AXIAL != 0, x.zeta.abs() < 0.1);
        assert_eq!(x.flags & HIDDEN != 0, x.centre_frame == 40);
    }
    assert!(r.iter().any(|x| x.flags & PARTIAL != 0));
    assert!(r.iter().any(|x| x.flags == 0));
    // A visibility test hides reflections on the left half.
    let left = |_: usize, x: usize, _: usize| x >= 150;
    let v = predict(
        &[lattice(Bravais::OrthorhombicP)],
        &g,
        &s,
        SHAPE,
        &options,
        Some(&left),
    )
    .unwrap();
    for x in &v {
        let pixel = (x.xyz[0] + 0.5).floor().max(0.) as usize;
        assert_eq!(x.flags & HIDDEN != 0, pixel < 150 || x.centre_frame == 40);
    }
    // Two lattices are merged and tagged.
    let two = predict(
        &[
            lattice(Bravais::OrthorhombicP),
            lattice(Bravais::OrthorhombicI),
        ],
        &g,
        &s,
        SHAPE,
        &options,
        None,
    )
    .unwrap();
    assert!(two.iter().any(|x| x.lattice == 1));
    assert_eq!(two.iter().filter(|x| x.lattice == 0).count(), r.len());
}

#[test]
fn a_scan_longer_than_one_turn_lists_each_crossing_per_turn() {
    let g = geometry();
    let one = Scan::new(0., 2., 2., 180).unwrap();
    let two = Scan::new(0., 2., 2., 360).unwrap();
    let options = PredictOptions::new(2.5, 2.);
    let l = [lattice(Bravais::OrthorhombicP)];
    let a = predict(&l, &g, &one, SHAPE, &options, None).unwrap();
    let b = predict(&l, &g, &two, SHAPE, &options, None).unwrap();
    assert_eq!(b.len(), 2 * a.len());
    for r in &a {
        assert!(
            b.iter()
                .any(|x| x.hkl == r.hkl && (x.xyz[2] - r.xyz[2] - 360.).abs() < 1e-9)
        );
    }
}

// ------------------------------------------------------------- summation

fn synthetic_lattice(sigma: f64) -> (Synthetic, Vec<Reflection>, PredictOptions) {
    let s = scan();
    let options = PredictOptions::new(1.6, 6. * sigma);
    let reflections = predict(
        &[lattice(Bravais::OrthorhombicP)],
        &geometry(),
        &s,
        SHAPE,
        &options,
        None,
    )
    .unwrap();
    let spots = spots_for(&reflections, sigma);
    (Synthetic::new(SHAPE, s, spots), reflections, options)
}

fn pulls(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let rms = (values.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
    (mean, rms)
}

#[test]
fn summation_recovers_known_intensities_on_a_sloped_background() {
    let (data, reflections, _) = synthetic_lattice(0.25);
    let options = MeasureOptions::default();
    let m = measure(
        &|f| data.frame(f),
        &geometry(),
        &data.scan,
        SHAPE,
        &reflections,
        &options,
        &NoMonitor,
    )
    .unwrap();
    assert_eq!(m.reflections.len(), reflections.len());
    assert_eq!(m.profile, ProfileStatus::NotRequested);
    // Every frame read once.
    assert_eq!(data.reads.load(Ordering::Relaxed), m.frames_read);
    let mut z = Vec::new();
    for (r, spot) in m.reflections.iter().zip(&data.spots) {
        if !r.usable() {
            continue;
        }
        let expected = spot.expected(
            r.bounds,
            &window_frames(&r.reflection, &data.scan),
            &data.scan,
        );
        assert!(expected > 0.99 * spot.intensity || spot.intensity == 0.);
        z.push((r.summation.value - expected) / r.summation.sigma);
        // σ is counting statistics: at least the peak's Poisson part.
        assert!(r.summation.sigma >= (expected.max(0.)).sqrt() * 0.9);
        assert_eq!(
            r.frames_measured,
            r.reflection.frames[1] - r.reflection.frames[0] + 1
        );
        let plane = 20. + 0.02 * spot.x + 0.03 * spot.y;
        assert!(
            (r.background - plane).abs() < 2.,
            "{} {plane}",
            r.background
        );
        if spot.intensity > 2000. {
            assert!((r.centroid[0] - spot.x).abs() < 0.2 && (r.centroid[1] - spot.y).abs() < 0.2);
            assert!((r.centroid[2] - spot.omega).abs() < 0.3);
        }
        let c = r.corrections.unwrap();
        assert!((r.corrected_summation().value - r.summation.value * c.factor).abs() < 1e-9);
    }
    let names: Vec<(String, usize)> = FLAG_NAMES
        .iter()
        .map(|(bit, name)| {
            (
                (*name).to_string(),
                m.reflections.iter().filter(|r| r.flags & bit != 0).count(),
            )
        })
        .collect();
    assert!(
        z.len() > 80,
        "{} of {}: {names:?}",
        z.len(),
        m.reflections.len()
    );
    let (mean, rms) = pulls(&z);
    assert!(mean.abs() < 0.25, "mean pull {mean}");
    assert!((0.75..1.3).contains(&rms), "rms pull {rms}");
    assert!(z.iter().all(|v| v.abs() < 5.));
    // The zero-intensity reflection is kept, weak and possibly negative.
    let zero = &m.reflections[3];
    assert!(zero.summation.value.is_finite());
    assert!(zero.summation.value.abs() < 5. * zero.summation.sigma);
    // Crystal-frame beams: q = (s − s₀)/λ rotated back equals h·B.
    let l = lattice(Bravais::OrthorhombicP);
    for r in m.reflections.iter().take(20) {
        let q = l.q(r.reflection.hkl);
        for j in 0..3 {
            let v = (r.diffracted[j] - r.incident[j]) / 0.4;
            assert!((v - q[j]).abs() < 1e-9, "{v} {}", q[j]);
        }
    }
}

#[test]
fn profile_fitting_recovers_known_intensities_on_the_same_shoeboxes() {
    let (data, reflections, _) = synthetic_lattice(0.25);
    let options = MeasureOptions {
        profile: Some(ProfileOptions::default()),
        ..MeasureOptions::default()
    };
    let stages = std::sync::Mutex::new(Vec::new());
    let monitor = |stage: Stage, _: usize, _: usize| {
        stages.lock().unwrap().push(stage);
        true
    };
    let m = measure(
        &|f| data.frame(f),
        &geometry(),
        &data.scan,
        SHAPE,
        &reflections,
        &options,
        &monitor,
    )
    .unwrap();
    let ProfileStatus::Fitted(model) = &m.profile else {
        panic!("{:?}", m.profile);
    };
    assert!(model.teachers >= 20);
    assert!(stages.lock().unwrap().contains(&Stage::FittingProfiles));
    let mut zp = Vec::new();
    let mut zs = Vec::new();
    for (r, spot) in m.reflections.iter().zip(&data.spots) {
        // Margin frames are read but not summed.
        assert_eq!(
            r.partials.iter().filter(|p| p.core).count(),
            r.frames_measured
        );
        if !r.profile_usable() || !r.usable() {
            continue;
        }
        let p = r.profile.unwrap();
        zp.push((p.intensity.value - spot.intensity) / p.intensity.sigma);
        let expected = spot.expected(
            r.bounds,
            &window_frames(&r.reflection, &data.scan),
            &data.scan,
        );
        zs.push((r.summation.value - expected) / r.summation.sigma);
        assert!(p.mass > 0.9);
        // The profile fit is at least as precise as summation for weak spots.
        if spot.intensity < 300. {
            assert!(p.intensity.sigma < 1.05 * r.summation.sigma);
        }
    }
    assert!(zp.len() > 80, "{}", zp.len());
    let (mean, rms) = pulls(&zp);
    assert!(mean.abs() < 0.5, "mean profile pull {mean}");
    assert!((0.6..1.6).contains(&rms), "rms profile pull {rms}");
    let (mean, _) = pulls(&zs);
    assert!(mean.abs() < 0.3);
}

#[test]
fn a_failed_profile_stage_leaves_the_summation_intact() {
    let (data, reflections, _) = synthetic_lattice(0.25);
    let options = MeasureOptions {
        profile: Some(ProfileOptions {
            minimum_total_teachers: 100_000,
            ..ProfileOptions::default()
        }),
        ..MeasureOptions::default()
    };
    let m = measure(
        &|f| data.frame(f),
        &geometry(),
        &data.scan,
        SHAPE,
        &reflections,
        &options,
        &NoMonitor,
    )
    .unwrap();
    assert!(matches!(m.profile, ProfileStatus::Failed(_)));
    // Summation is identical with and without profile fitting.
    let plain = measure(
        &|f| data.frame(f),
        &geometry(),
        &data.scan,
        SHAPE,
        &reflections,
        &MeasureOptions::default(),
        &NoMonitor,
    )
    .unwrap();
    for (a, b) in m.reflections.iter().zip(&plain.reflections) {
        assert_eq!(a.flags, b.flags);
        assert_eq!(a.summation.value.to_bits(), b.summation.value.to_bits());
        assert_eq!(a.summation.sigma.to_bits(), b.summation.sigma.to_bits());
        assert_eq!(
            a.partials.iter().filter(|p| p.core).count(),
            b.partials.len()
        );
    }
    assert!(m.reflections.iter().filter(|r| r.usable()).count() > 80);
    assert!(m.reflections.iter().all(|r| r.profile.is_none()));
}

// ------------------------------------------------------------------ flags

fn manual(x: f64, y: f64, omega: f64, frames: [usize; 2]) -> Reflection {
    Reflection {
        lattice: 0,
        hkl: [1, 2, 3],
        xyz: [x, y, omega],
        d: 2.,
        zeta: 1.,
        frames,
        centre_frame: omega.floor() as usize,
        flags: 0,
    }
}

#[test]
fn every_summation_flag_is_raised_where_it_applies() {
    let shape = [64, 128];
    let s = Scan::new(0., 1., 1., 10)
        .unwrap()
        .with_excluded(vec![8, 9])
        .unwrap();
    let spot = |x: f64, y: f64, intensity: f64| Spot {
        x,
        y,
        omega: 4.5,
        intensity,
        sigma_xy: 0.8,
        sigma_omega: 0.3,
    };
    // 0 clean, 1 edge, 2 masked, 3 saturated, 4/5 overlap, 6 background,
    // 7 boundary (signal 3 px off), 8 model, 9 hidden, 10 no data.
    let reflections = vec![
        manual(20., 20., 4.5, [3, 5]),
        manual(1., 40., 4.5, [3, 5]),
        manual(40., 20., 4.5, [3, 5]),
        manual(60., 20., 4.5, [3, 5]),
        manual(80., 20., 4.5, [3, 5]),
        manual(84., 21., 4.5, [3, 5]),
        manual(20., 45., 4.5, [3, 5]),
        manual(40., 45., 4.5, [3, 5]),
        manual(100., 45., 4.5, [3, 5]),
        manual(60., 45., 4.5, [3, 5]),
        manual(110., 15., 8.5, [8, 9]),
    ];
    let mut spots: Vec<Spot> = reflections
        .iter()
        .map(|r| spot(r.xyz[0], r.xyz[1], 3000.))
        .collect();
    spots[3].intensity = 40000.;
    spots[7].x += 3.;
    let mut data = Synthetic::new(shape, s.clone(), spots);
    data.background = [50., 0., 0.];
    data.noise = false;
    data.mask = Some(Box::new(|f, x, y| {
        (f == 4 && x == 42 && y == 21)
            // A ring of masked pixels around reflection 6 (box ±3, guard 2, ring 5).
            || (f == 4 && (x as i64 - 20).abs().max((y as i64 - 45).abs()) >= 6 && (x as i64 - 20).abs().max((y as i64 - 45).abs()) <= 10)
            || (f == 4 && x == 60 && y == 45)
    }));
    data.extra = Some(Box::new(|x, y| {
        let near = (x as i64 - 100).abs().max((y as i64 - 45).abs());
        if (6..=10).contains(&near) {
            if (x + y) % 2 == 0 { 45. } else { -45. }
        } else {
            0.
        }
    }));
    let options = MeasureOptions {
        box_size: 7,
        noise: Noise {
            saturation: Some(5000.),
            ..Noise::default()
        },
        ..MeasureOptions::default()
    };
    let m = measure(
        &|f| data.frame(f),
        &geometry(),
        &s,
        shape,
        &reflections,
        &options,
        &NoMonitor,
    )
    .unwrap();
    let f: Vec<u32> = m.reflections.iter().map(|r| r.flags).collect();
    assert_eq!(f[0], 0, "{:?}", m.reflections[0]);
    assert!(m.reflections[0].usable());
    // The truncated box also has signal in its outermost column.
    assert_eq!(f[1] & !BOUNDARY, EDGE);
    assert_eq!(f[2], MASKED);
    assert_eq!(f[3], SATURATED);
    // The neighbour's signal also reaches the box border.
    assert_eq!(f[4] & !BOUNDARY, OVERLAP);
    assert_eq!(f[5] & !BOUNDARY, OVERLAP);
    assert_eq!(f[6] & BACKGROUND, BACKGROUND);
    assert!(m.reflections[6].summation.value.is_nan());
    assert_eq!(f[7] & BOUNDARY, BOUNDARY);
    assert_eq!(f[8] & MODEL, MODEL);
    assert_eq!(f[9] & HIDDEN, HIDDEN);
    // Its whole window is excluded in the measuring scan.
    assert_eq!(f[10], NO_DATA | PARTIAL | HIDDEN);
    assert_eq!(m.reflections[10].frames_measured, 0);
    // The clean reflection: expected counts within σ (no noise added).
    let r = &m.reflections[0];
    let e = data.spots[0].expected(r.bounds, &[3, 4, 5], &s);
    assert!((r.summation.value - e).abs() < 1e-6 * e);
    assert_eq!(FLAG_NAMES.len(), 11);
}

#[test]
fn reflections_of_two_lattices_are_measured_together_and_their_overlaps_flagged() {
    let s = scan();
    let g = geometry();
    let a = lattice(Bravais::OrthorhombicP);
    // A second grain 0.15° away: most reflections land on the first one's.
    let b = Lattice::new(
        basis(
            [8.1, 9.3, 10.7],
            rotation(0.3 + 0.15f64.to_radians(), 0.7, 1.1),
        ),
        Bravais::OrthorhombicP,
    )
    .unwrap();
    // A third grain far away in orientation overlaps only by chance.
    let c = Lattice::new(
        basis([8.1, 9.3, 10.7], rotation(1.3, 0.2, 0.4)),
        Bravais::OrthorhombicP,
    )
    .unwrap();
    let options = PredictOptions::new(1.6, 1.5);
    let reflections = predict(&[a, b, c], &g, &s, SHAPE, &options, None).unwrap();
    let spots = spots_for(&reflections, 0.25);
    let data = Synthetic::new(SHAPE, s.clone(), spots);
    let m = measure(
        &|f| data.frame(f),
        &g,
        &s,
        SHAPE,
        &reflections,
        &MeasureOptions::default(),
        &NoMonitor,
    )
    .unwrap();
    // One pass: each needed frame read once for all three lattices.
    assert_eq!(data.reads.load(Ordering::Relaxed), m.frames_read);
    let overlapped = |l: usize| {
        m.reflections
            .iter()
            .filter(|r| r.reflection.lattice == l && r.flags & OVERLAP != 0)
            .count() as f64
            / m.reflections
                .iter()
                .filter(|r| r.reflection.lattice == l)
                .count() as f64
    };
    assert!(overlapped(1) > 0.5, "{}", overlapped(1));
    assert!(overlapped(2) < 0.3, "{}", overlapped(2));
    // Every flagged pair really shares pixels on a common frame.
    for (i, r) in m.reflections.iter().enumerate() {
        let shares = m.reflections.iter().enumerate().any(|(j, o)| {
            j != i
                && o.bounds.x0 < r.bounds.x1
                && r.bounds.x0 < o.bounds.x1
                && o.bounds.y0 < r.bounds.y1
                && r.bounds.y0 < o.bounds.y1
                && o.reflection.frames[0] <= r.reflection.frames[1]
                && r.reflection.frames[0] <= o.reflection.frames[1]
        });
        assert_eq!(r.flags & OVERLAP != 0, shares, "{i}");
    }
    // Isolated reflections of the third lattice are recovered.
    let z: Vec<f64> = m
        .reflections
        .iter()
        .zip(&data.spots)
        .filter(|(r, _)| r.reflection.lattice == 2 && r.usable())
        .map(|(r, s)| {
            let e = s.expected(
                r.bounds,
                &window_frames(&r.reflection, &data.scan),
                &data.scan,
            );
            (r.summation.value - e) / r.summation.sigma
        })
        .collect();
    assert!(z.len() > 30);
    assert!(pulls(&z).0.abs() < 0.4);
}

// ------------------------------------------------------------ cancellation

#[test]
fn a_monitor_cancels_between_frames() {
    let (data, reflections, _) = synthetic_lattice(0.25);
    let options = MeasureOptions {
        workers: 2,
        ..MeasureOptions::default()
    };
    let monitor = |_: Stage, done: usize, _: usize| done < 3;
    let e = measure(
        &|f| data.frame(f),
        &geometry(),
        &data.scan,
        SHAPE,
        &reflections,
        &options,
        &monitor,
    )
    .unwrap_err();
    assert_eq!(error_kind(&e), ErrorKind::Cancelled);
    let read = data.reads.load(Ordering::Relaxed);
    assert!(read < 20, "{read}");
    // A failing reader stops the pass with its error.
    let failing = |f: usize| {
        if f == 30 {
            Err(invalid("disk"))
        } else {
            data.frame(f)
        }
    };
    let e = measure(
        &failing,
        &geometry(),
        &data.scan,
        SHAPE,
        &reflections,
        &options,
        &NoMonitor,
    )
    .unwrap_err();
    assert_eq!(e.0, "frame read failed at frame 30: disk");
    assert_eq!(error_kind(&e), ErrorKind::FrameRead);
}

// --------------------------------------------------------- bounded memory

#[test]
fn the_frame_pass_keeps_only_box_local_work_arrays() {
    // A large detector with few boxes: the background coverage index of a
    // frame is a tiny fraction of one byte per pixel (the former version
    // allocated two u32 arrays of the full image per frame).
    let shape = [1024, 1024];
    let s = Scan::new(0., 1., 1., 6).unwrap();
    let reflections: Vec<Reflection> = (0..40)
        .map(|i| manual(30. + 24. * f64::from(i), 500., 2.5, [1, 4]))
        .collect();
    let boxes: Vec<Bounds> = reflections
        .iter()
        .map(|r| super::integrate::tests_peak_box(r.xyz[0], r.xyz[1], 9, shape))
        .collect();
    let coverage =
        crate::spots::local_background::SparseCoverage::new(&boxes, 2, shape[1], shape[0]);
    assert!(
        coverage.heap_bytes() * 100 < shape[0] * shape[1],
        "{}",
        coverage.heap_bytes()
    );
    let spots = reflections
        .iter()
        .map(|r| Spot {
            x: r.xyz[0],
            y: r.xyz[1],
            omega: 2.5,
            intensity: 1000.,
            sigma_xy: 1.,
            sigma_omega: 0.4,
        })
        .collect();
    let mut data = Synthetic::new(shape, s.clone(), spots);
    data.noise = false;
    // At most `workers` frames are held: count frames read but not finished.
    let live = AtomicUsize::new(0);
    let peak = AtomicUsize::new(0);
    let reader = |f: usize| {
        let now = live.fetch_add(1, Ordering::SeqCst) + 1;
        peak.fetch_max(now, Ordering::SeqCst);
        data.frame(f)
    };
    let monitor = |_: Stage, _: usize, _: usize| {
        live.fetch_sub(1, Ordering::SeqCst);
        true
    };
    let m = measure(
        &reader,
        &geometry(),
        &s,
        shape,
        &reflections,
        &MeasureOptions {
            workers: 3,
            ..MeasureOptions::default()
        },
        &monitor,
    )
    .unwrap();
    assert!(peak.load(Ordering::SeqCst) <= 3);
    assert!(m.reflections.iter().all(MeasuredReflection::usable));
}

// ------------------------------------------------------------ rocking width

#[test]
fn rocking_width_from_synthetic_tracks() {
    let g = geometry();
    let s = Scan::new(0., 0.1, 0.1, 400).unwrap();
    let sigma = 0.3;
    let r = predict(
        &[lattice(Bravais::OrthorhombicP)],
        &g,
        &s,
        SHAPE,
        &PredictOptions::new(1.6, 1.),
        None,
    )
    .unwrap();
    let tracks: Vec<Vec<[f64; 4]>> = r
        .iter()
        .filter(|x| x.zeta.abs() > 0.2)
        .map(|x| {
            let spot = Spot {
                x: x.xyz[0],
                y: x.xyz[1],
                omega: x.xyz[2],
                intensity: 1e4,
                sigma_xy: 1.,
                sigma_omega: sigma / x.zeta.abs(),
            };
            (0..s.frames)
                .filter_map(|f| {
                    let w = spot.intensity * spot.frame(s.bounds(f));
                    let b = s.bounds(f);
                    (w > 20.).then_some([x.xyz[0], x.xyz[1], f64::midpoint(b[0], b[1]), w])
                })
                .collect()
        })
        .filter(|t: &Vec<[f64; 4]>| !t.is_empty())
        .collect();
    let e = estimate_rocking_from_tracks(&tracks, &g, &s).unwrap();
    assert!((e.rocking_sigma - sigma).abs() < 0.1 * sigma, "{e:?}");
    assert!((e.rocking_width - 6. * e.rocking_sigma).abs() < 1e-12);
    assert!(e.spot_sigma.is_none());
    assert!(!e.at_step_floor);
    assert!(estimate_rocking_from_tracks(&tracks[..2], &g, &s).is_err());
    // Single-frame tracks: σ 0, the step floor, and it says so.
    let single: Vec<Vec<[f64; 4]>> = tracks.iter().map(|t| vec![t[0]]).collect();
    let f = estimate_rocking_from_tracks(&single, &g, &s).unwrap();
    assert!(f.at_step_floor && f.rocking_sigma == 0. && f.rocking_width == s.step);
}

#[test]
fn rocking_width_and_spot_size_from_the_frames() {
    // σ = 0.25° with 0.5° frames: resolved through frame-boundary sharing.
    for sigma in [0.25, 0.6] {
        let (data, _, _) = synthetic_lattice(sigma);
        let e = estimate_shape(
            &|f| data.frame(f),
            &geometry(),
            &data.scan,
            SHAPE,
            &[lattice(Bravais::OrthorhombicP)],
            1.6,
            &MeasureOptions::default(),
            &NoMonitor,
        )
        .unwrap();
        assert!(
            (e.rocking_sigma - sigma).abs() < 0.2 * sigma,
            "{sigma}: {e:?}"
        );
        let spot = e.spot_sigma.unwrap();
        assert!(spot.iter().all(|&v| (v - 1.).abs() < 0.35), "{e:?}");
        assert!(matches!(e.box_size, Some(9 | 11)), "{e:?}");
    }
}

// ------------------------------------------------------- holohedry, merging

fn metric(system: LatticeSystem) -> M3 {
    let (a, b, c) = (5.1, 6.3, 7.9);
    let cell = match system {
        LatticeSystem::Triclinic => [a, b, c, 81., 97., 103.],
        LatticeSystem::Monoclinic => [a, b, c, 90., 104., 90.],
        LatticeSystem::Orthorhombic => [a, b, c, 90., 90., 90.],
        LatticeSystem::Tetragonal => [a, a, c, 90., 90., 90.],
        LatticeSystem::Rhombohedral | LatticeSystem::Hexagonal => [a, a, c, 90., 90., 120.],
        LatticeSystem::Cubic => [a, a, a, 90., 90., 90.],
    };
    let r = cell[3..]
        .iter()
        .map(|v: &f64| v.to_radians().cos())
        .collect::<Vec<_>>();
    let l = [cell[0], cell[1], cell[2]];
    [
        [l[0] * l[0], l[0] * l[1] * r[2], l[0] * l[2] * r[1]],
        [l[0] * l[1] * r[2], l[1] * l[1], l[1] * l[2] * r[0]],
        [l[0] * l[2] * r[1], l[1] * l[2] * r[0], l[2] * l[2]],
    ]
}

#[test]
fn holohedries_are_closed_groups_that_keep_the_metric() {
    for bravais in Bravais::ALL {
        let ops = bravais.holohedry();
        assert_eq!(ops.len(), bravais.holohedry_order(), "{bravais:?}");
        assert!(ops.contains(&[[-1, 0, 0], [0, -1, 0], [0, 0, -1]]));
        for a in &ops {
            for b in &ops {
                let p: [[i32; 3]; 3] = std::array::from_fn(|i| {
                    std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum())
                });
                assert!(ops.contains(&p));
            }
            // Wᵀ G W = G for the direct metric.
            let g = metric(bravais.system());
            for i in 0..3 {
                for j in 0..3 {
                    let v: f64 = (0..3)
                        .flat_map(|k| (0..3).map(move |l| (k, l)))
                        .map(|(k, l)| f64::from(a[k][i]) * g[k][l] * f64::from(a[l][j]))
                        .sum();
                    assert!((v - g[i][j]).abs() < 1e-9, "{bravais:?} {a:?}");
                }
            }
        }
    }
    assert_eq!(LatticeSystem::Rhombohedral.laue_symbol(), "-3m");
    // Rhombohedral: the obverse centring condition is kept by every operation.
    let r = Bravais::RhombohedralR;
    for w in r.holohedry() {
        for hkl in [[1, 0, 1], [0, 1, 2], [1, 1, 0]] {
            assert!(r.centring().allows(hkl));
            assert!(r.centring().allows(transform_index(hkl, &w)));
        }
    }
}

#[test]
fn merging_statistics_on_synthetic_equivalents() {
    for system in [
        LatticeSystem::Triclinic,
        LatticeSystem::Monoclinic,
        LatticeSystem::Orthorhombic,
        LatticeSystem::Tetragonal,
        LatticeSystem::Hexagonal,
        LatticeSystem::Cubic,
    ] {
        let bravais = match system {
            LatticeSystem::Triclinic => Bravais::TriclinicP,
            LatticeSystem::Monoclinic => Bravais::MonoclinicP,
            LatticeSystem::Orthorhombic => Bravais::OrthorhombicP,
            LatticeSystem::Tetragonal => Bravais::TetragonalP,
            LatticeSystem::Hexagonal => Bravais::HexagonalP,
            _ => Bravais::CubicP,
        };
        let g = metric(system);
        // A direct basis with that metric (Cholesky rows).
        let a0 = g[0][0].sqrt();
        let a1 = [g[0][1] / a0, (g[1][1] - (g[0][1] / a0).powi(2)).sqrt()];
        let a2x = g[0][2] / a0;
        let a2y = (g[1][2] - a1[0] * a2x) / a1[1];
        let a2z = (g[2][2] - a2x * a2x - a2y * a2y).sqrt();
        let lat =
            Lattice::new([[a0, 0., 0.], [a1[0], a1[1], 0.], [a2x, a2y, a2z]], bravais).unwrap();
        let ops = system.holohedry();
        let d_min = 2.2;
        // Every reflection to d_min, observed in every equivalent setting:
        // complete, redundancy = orbit sizes, R_int zero without noise.
        let mut obs = Vec::new();
        for h in -6..=6i64 {
            for k in -6..=6i64 {
                for l in -6..=6i64 {
                    let hkl = [h, k, l];
                    if hkl == [0; 3] || norm(lat.q(hkl)) > 1. / d_min {
                        continue;
                    }
                    let rep = equivalent_representative(hkl, &ops);
                    let truth = 100.
                        + (splitmix((rep[0] * 1000 + rep[1] * 37 + rep[2] + 50_000) as u64) % 1000)
                            as f64;
                    obs.push(Observation::new(hkl, truth, 10., 1. / norm(lat.q(hkl))));
                }
            }
        }
        let options = MergingOptions {
            d_min: Some(d_min),
            lattice: Some(lat),
            ..MergingOptions::default()
        };
        let s = merging_statistics(&obs, &ops, &options).unwrap();
        assert_eq!(s.overall.r_int, Some(0.), "{system:?}");
        assert_eq!(s.overall.completeness, Some(1.), "{system:?}");
        assert_eq!(s.overall.observations, obs.len());
        assert!((s.overall.redundancy - obs.len() as f64 / s.overall.unique as f64).abs() < 1e-12);
        let shell_unique: usize = s.shells.iter().map(|b| b.unique).sum();
        assert_eq!(shell_unique, s.overall.unique);
        assert!(s.shells.iter().all(|b| b.completeness == Some(1.)));
        // Unique count matches the orbit count of the operations.
        let reps: std::collections::BTreeSet<_> = obs
            .iter()
            .map(|o| equivalent_representative(o.hkl, &ops))
            .collect();
        assert_eq!(reps.len(), s.overall.unique);
        // Alternate +10 / −10 within each pair: R_int = Σ|I−⟨I⟩| / ΣI.
        let mut noisy = obs.clone();
        let mut seen = std::collections::BTreeMap::new();
        for o in &mut noisy {
            let rep = equivalent_representative(o.hkl, &ops);
            let n: &mut usize = seen.entry(rep).or_insert(0);
            o.intensity += if n.is_multiple_of(2) { 10. } else { -10. };
            *n += 1;
        }
        let s2 = merging_statistics(&noisy, &ops, &options).unwrap();
        let merged = merge(&noisy, &ops);
        let (mut num, mut den) = (0., 0.);
        for m in merged.iter().filter(|m| m.observations >= 2) {
            for o in noisy
                .iter()
                .filter(|o| equivalent_representative(o.hkl, &ops) == m.hkl)
            {
                num += (o.intensity - m.intensity).abs();
                den += o.intensity;
            }
        }
        assert!(
            (s2.overall.r_int.unwrap() - num / den).abs() < 1e-12,
            "{system:?}"
        );
        assert!(s2.overall.r_meas.unwrap() > s2.overall.r_int.unwrap());
    }
    // Two observations of one reflection: R_int = 20 / 200.
    let ops = LatticeSystem::Orthorhombic.holohedry();
    let obs = [
        Observation::new([1, 2, 3], 110., 5., 2.),
        Observation::new([-1, 2, -3], 90., 5., 2.),
        Observation::new([2, 0, 0], 200., 5., 3.),
    ];
    let s = merging_statistics(&obs, &ops, &MergingOptions::default()).unwrap();
    assert!((s.overall.r_int.unwrap() - 0.1).abs() < 1e-15);
    assert!((s.overall.r_meas.unwrap() - 0.1 * 2f64.sqrt()).abs() < 1e-15);
    assert_eq!(s.overall.unique, 2);
    assert_eq!(s.overall.multiples, 2);
    assert_eq!(s.signal.len(), 4);
    assert_eq!(s.signal[3].unique, 1); // mean I/σ 40
    assert_eq!(s.signal[2].unique, 1); // 20
    assert!(merging_statistics(&[], &ops, &MergingOptions::default()).is_err());
}

// ------------------------------------------------- review regressions (PR 12)

#[test]
fn merging_leaves_out_observations_beyond_d_min_and_shells_cover_all_possible() {
    let lat = Lattice::new(
        [[5., 0., 0.], [0., 6., 0.], [0., 0., 7.]],
        Bravais::OrthorhombicP,
    )
    .unwrap();
    let ops = LatticeSystem::Orthorhombic.holohedry();
    let mut obs = Vec::new();
    for h in -7..=7i64 {
        for k in -8..=8i64 {
            for l in -9..=9i64 {
                let hkl = [h, k, l];
                let d = 1. / norm(lat.q(hkl));
                if hkl != [0; 3] && d >= 0.8 && hkl == equivalent_representative(hkl, &ops) {
                    obs.push(Observation::new(hkl, 100., 10., d));
                }
            }
        }
    }
    // d_min coarser than the data: completeness may not exceed one.
    let options = MergingOptions {
        d_min: Some(1.0),
        lattice: Some(lat),
        ..MergingOptions::default()
    };
    let s = merging_statistics(&obs, &ops, &options).unwrap();
    assert_eq!(s.overall.completeness, Some(1.));
    assert!(s.beyond_d_min > 0);
    assert_eq!(s.overall.observations + s.beyond_d_min, obs.len());
    for b in &s.shells {
        assert_eq!(b.completeness, Some(1.), "{b:?}");
    }
    // Low-angle reflections missing (beam stop): shells see them.
    let cut: Vec<Observation> = obs.iter().copied().filter(|o| o.d <= 3.).collect();
    let s = merging_statistics(&cut, &ops, &options).unwrap();
    let possible: usize = s.shells.iter().map(|b| b.possible.unwrap()).sum();
    let unique: usize = s.shells.iter().map(|b| b.unique).sum();
    assert_eq!(possible, s.overall.possible.unwrap());
    assert_eq!(unique, s.overall.unique);
    assert!(s.shells[0].completeness.unwrap() < 1.);
    assert!(s.shells[0].limits[0] > 3.);
    // Nothing within the limit is an error.
    let options = MergingOptions {
        d_min: Some(10.),
        ..MergingOptions::default()
    };
    assert!(merging_statistics(&cut, &ops, &options).is_err());
}

#[test]
fn the_measuring_scan_decides_partial_and_hidden() {
    let g = geometry();
    let s = scan();
    let reflections = predict(
        &[lattice(Bravais::OrthorhombicP)],
        &g,
        &s,
        SHAPE,
        &PredictOptions::new(1.6, 2.),
        None,
    )
    .unwrap();
    let data = Synthetic::new(SHAPE, s.clone(), spots_for(&reflections, 0.3));
    let measuring = s.clone().with_excluded((30..34).collect()).unwrap();
    let m = measure(
        &|f| data.frame(f),
        &g,
        &measuring,
        SHAPE,
        &reflections,
        &MeasureOptions::default(),
        &NoMonitor,
    )
    .unwrap();
    let mut touched = 0;
    for r in &m.reflections {
        let w = r.reflection.frames;
        let cut = (w[0]..=w[1]).any(|f| (30..34).contains(&f));
        assert_eq!(
            r.flags & PARTIAL != 0,
            cut || r.reflection.flags & PARTIAL != 0
        );
        assert_eq!(
            r.flags & HIDDEN != 0,
            (30..34).contains(&r.reflection.centre_frame) || r.reflection.flags & HIDDEN != 0
        );
        if cut {
            touched += 1;
            assert!(!r.usable());
        }
    }
    assert!(touched > 5, "{touched}");
}

#[test]
fn centre_frames_lie_in_the_window_when_exposures_leave_gaps() {
    let g = geometry();
    let s = Scan::new(0., 1., 0.3, 60).unwrap();
    let r = predict(
        &[lattice(Bravais::OrthorhombicP)],
        &g,
        &s,
        SHAPE,
        &PredictOptions::new(1.6, 0.05),
        None,
    )
    .unwrap();
    assert!(r.len() > 50);
    for x in &r {
        assert!(
            x.frames[0] <= x.centre_frame && x.centre_frame <= x.frames[1],
            "{x:?}"
        );
        let b = s.bounds(x.centre_frame);
        // The centre exposure holds ω or is the nearest one.
        let distance = (b[0] - x.xyz[2]).max(x.xyz[2] - b[1]).max(0.);
        for f in x.frames[0]..=x.frames[1] {
            let o = s.bounds(f);
            assert!(distance <= (o[0] - x.xyz[2]).max(x.xyz[2] - o[1]).max(0.) + 1e-12);
        }
    }
    // The centre pixel masked on every frame hides every reflection.
    let pixels: Vec<(usize, usize)> = r
        .iter()
        .map(|x| {
            (
                (x.xyz[0] + 0.5).floor() as usize,
                (x.xyz[1] + 0.5).floor() as usize,
            )
        })
        .collect();
    let mut data = Synthetic::new(SHAPE, s.clone(), vec![]);
    data.noise = false;
    data.mask = Some(Box::new(move |_, x, y| pixels.contains(&(x, y))));
    let m = measure(
        &|f| data.frame(f),
        &g,
        &s,
        SHAPE,
        &r,
        &MeasureOptions::default(),
        &NoMonitor,
    )
    .unwrap();
    assert!(m.reflections.iter().all(|x| x.flags & HIDDEN != 0));
}

#[test]
fn summation_sigma_matches_the_scatter_at_low_photon_counting_background() {
    // Weak reflections (30–230 counts) on Poisson backgrounds of 0.02–20
    // counts per pixel; pull SD of (I − I_expected)/σ for both variance rules.
    let g = geometry();
    let s = scan();
    let reflections = predict(
        &[lattice(Bravais::OrthorhombicP)],
        &g,
        &s,
        SHAPE,
        &PredictOptions::new(1.6, 1.5),
        None,
    )
    .unwrap();
    let mut table = Vec::new();
    for b in [0.02, 0.2, 2., 20.] {
        let spots: Vec<Spot> = spots_for(&reflections, 0.25)
            .into_iter()
            .enumerate()
            .map(|(i, mut sp)| {
                sp.intensity = 30. + 200. * uniform(i as u64 * 104_729 + 7);
                sp
            })
            .collect();
        let mut data = Synthetic::new(SHAPE, s.clone(), spots);
        data.background = [b, 0., 0.];
        data.poisson = true;
        let mut row = vec![b];
        for variance in [
            PixelVariance::FlooredAtOne,
            PixelVariance::CountsOrBackground,
        ] {
            let mut options = MeasureOptions::default();
            options.noise.variance = variance;
            let m = measure(
                &|f| data.frame(f),
                &g,
                &s,
                SHAPE,
                &reflections,
                &options,
                &NoMonitor,
            )
            .unwrap();
            let z: Vec<f64> = m
                .reflections
                .iter()
                .zip(&data.spots)
                .filter(|(r, _)| r.usable())
                .map(|(r, sp)| {
                    let e = sp.expected(r.bounds, &window_frames(&r.reflection, &s), &s);
                    (r.summation.value - e) / r.summation.sigma
                })
                .collect();
            let n = z.len() as f64;
            let mean = z.iter().sum::<f64>() / n;
            let sd = (z.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.)).sqrt();
            assert!(z.len() > 60);
            assert!(mean.abs() < 0.3, "{b} {variance:?} {mean}");
            row.push(sd);
        }
        table.push(row);
    }
    eprintln!("background  pull SD floored-at-one  pull SD counts-or-background");
    for r in &table {
        eprintln!("{:>10}  {:>21.3}  {:>28.3}", r[0], r[1], r[2]);
    }
    for r in &table {
        assert!((0.85..1.2).contains(&r[2]), "{table:?}");
    }
    assert!(table[0][1] < 0.75, "{table:?}");
}

#[test]
fn block_accumulation_equals_one_pass_and_the_pair_budget_names_its_option() {
    let mut ids = Vec::new();
    let mut values = Vec::new();
    let mut flags = Vec::new();
    for k in 0..200usize {
        let i = (k * 7) % 23;
        let mut v = [0.; 11];
        v[2] = (k as f64).sin() * 100.;
        v[3] = 1. + (k % 5) as f64;
        v[4] = 3.;
        v[9] = 81.;
        v[10] = 200.;
        ids.push(i);
        values.push(v);
        flags.push(if k % 31 == 0 { 2 } else { 0 });
    }
    let one = crate::spots::integration::accumulate(25, &ids, &values, &flags).unwrap();
    let blocks = super::integrate::accumulate_blocks(13, 25, &ids, &values, &flags).unwrap();
    let bits = |r: &(Vec<[f64; 6]>, Vec<u32>)| {
        r.0.iter()
            .flatten()
            .map(|v| v.to_bits())
            .chain(r.1.iter().map(|&f| u64::from(f)))
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&one), bits(&blocks));
    let (data, reflections, _) = synthetic_lattice(0.25);
    let options = MeasureOptions {
        max_pairs: 10,
        ..MeasureOptions::default()
    };
    let e = measure(
        &|f| data.frame(f),
        &geometry(),
        &data.scan,
        SHAPE,
        &reflections,
        &options,
        &NoMonitor,
    )
    .unwrap_err();
    assert!(e.0.contains("MeasureOptions::max_pairs"), "{}", e.0);
    assert_eq!(error_kind(&e), ErrorKind::Other);
    assert_eq!(data.reads.load(Ordering::Relaxed), 0);
}

#[test]
fn caller_facing_details_from_the_first_dioptas_integration() {
    let (data, reflections, _) = synthetic_lattice(0.25);
    // Default workers = 0: the call runs in the caller's rayon pool, so the
    // pool bounds the frames held.
    assert_eq!(MeasureOptions::default().workers, 0);
    let live = AtomicUsize::new(0);
    let peak = AtomicUsize::new(0);
    let reader = |f: usize| {
        let now = live.fetch_add(1, Ordering::SeqCst) + 1;
        peak.fetch_max(now, Ordering::SeqCst);
        data.frame(f)
    };
    let monitor = |_: Stage, _: usize, _: usize| {
        live.fetch_sub(1, Ordering::SeqCst);
        true
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let mut m = pool
        .install(|| {
            measure(
                &reader,
                &geometry(),
                &data.scan,
                SHAPE,
                &reflections,
                &MeasureOptions::default(),
                &monitor,
            )
        })
        .unwrap();
    assert!(peak.load(Ordering::SeqCst) <= 2);
    // usable() needs a finite correction.
    let i = m
        .reflections
        .iter()
        .position(MeasuredReflection::usable)
        .unwrap();
    m.reflections[i].corrections = None;
    assert!(!m.reflections[i].usable());
    // Errors name the field.
    let e = measure(
        &reader,
        &geometry(),
        &data.scan,
        SHAPE,
        &reflections,
        &MeasureOptions {
            box_size: 8,
            ..MeasureOptions::default()
        },
        &NoMonitor,
    )
    .unwrap_err();
    assert!(e.0.contains("MeasureOptions::box_size"), "{}", e.0);
    // Profile flag names cover every bit a fit can set.
    let named: u32 = PROFILE_FLAG_NAMES.iter().map(|f| f.0).fold(0, |a, b| a | b);
    let bits = crate::spots::profiles::PARTIAL
        | crate::spots::profiles::FAILED
        | crate::spots::profiles::NO_PROFILE
        | crate::spots::profiles::MISMATCH
        | EDGE
        | MASKED
        | SATURATED
        | OVERLAP
        | BACKGROUND
        | PARTIAL
        | MODEL;
    assert_eq!(named, bits);
}

#[test]
fn measurement_boundary_validations_reject_nonfinite_and_oversized_inputs() {
    let pol = Polarization::unpolarized();
    assert!(polarization_factor([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], &pol).is_err());
    assert!(polarization_factor([0.0, 0.0, 1.0], [f64::NAN, 0.0, 1.0], &pol).is_err());
    assert!(polarization_factor([f64::NAN, 0.0, 1.0], [0.0, 0.0, 1.0], &pol).is_err());

    let ops = crate::lattice_symmetry::LatticeSystem::Triclinic.holohedry();
    let bad_d = [Observation {
        hkl: [1, 0, 0],
        d: f64::INFINITY,
        intensity: 10.0,
        sigma: 1.0,
    }];
    assert!(merging_statistics(&bad_d, &ops, &MergingOptions::default()).is_err());

    let ok_obs = [Observation {
        hkl: [1, 0, 0],
        d: 2.0 * (1.0 - 5e-10),
        intensity: 10.0,
        sigma: 1.0,
    }];
    let bad_bins = MergingOptions {
        signal_bins: vec![f64::NAN, 3.0],
        ..MergingOptions::default()
    };
    assert!(merging_statistics(&ok_obs, &ops, &bad_bins).is_err());

    // Observation inside the d_min tolerance band must not invert shell bounds.
    let tol_opts = MergingOptions {
        d_min: Some(2.0),
        shells: 4,
        ..MergingOptions::default()
    };
    let stats = merging_statistics(&ok_obs, &ops, &tol_opts).expect("tolerance band stats");
    for shell in &stats.shells {
        assert!(shell.limits[0].is_finite() && shell.limits[1].is_finite());
        assert!(shell.limits[0] >= shell.limits[1] - 1e-12);
    }

    // Excessive scan span must be rejected by Scan::check.
    assert!(Scan::new(0.0, 1.0e9, 0.1, 10).is_err());

    // Suggested box_size must never exceed MeasureOptions::check's upper bound (101, odd).
    let (data, reflections, _) = synthetic_lattice(0.25);
    let mut m = measure(
        &|f| data.frame(f),
        &geometry(),
        &data.scan,
        SHAPE,
        &reflections,
        &MeasureOptions::default(),
        &NoMonitor,
    )
    .unwrap();
    for reflection in &mut m.reflections {
        for partial in &mut reflection.partials {
            partial.spread = [25.0, 25.0];
        }
    }
    let estimate = estimate_shape_from_partials(&m, &data.scan).unwrap();
    assert_eq!(estimate.box_size, Some(101));
    let accepted = MeasureOptions {
        box_size: estimate.box_size.unwrap(),
        ..MeasureOptions::default()
    };
    assert!(
        measure(
            &|f| data.frame(f),
            &geometry(),
            &data.scan,
            SHAPE,
            &reflections[..1],
            &accepted,
            &NoMonitor,
        )
        .is_ok()
    );
}
