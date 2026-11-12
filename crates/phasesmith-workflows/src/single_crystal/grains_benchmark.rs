//! Ignored release benchmark of the complete native indexing coordinator.
use super::super::{Acquisition, Geometry, Priors, ShadowOptions, UnitCell};
use super::*;
use phasesmith_crystallography::SymmetryOperation;
use phasesmith_single_crystal::spots::{frame_validity, prediction::Prediction};
use std::{fs, path::Path, time::Instant};
fn options() -> Options {
    let identity = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    Options {
        discovery: None,
        acquisition: Acquisition {
            first: 0,
            last: 7,
            omega_zero: -40.,
            omega_step: 10.,
            exposure: 5.,
            vertical_flip: true,
            reserved_minimum: u32::MAX - 1,
        },
        geometry: Geometry {
            origin: [-0.0006, -0.0006, 0.15],
            fast: [0., 0.000_075, 0.],
            slow: [0.000_075, 0., 0.],
            beam: [0., 0., 1.],
            axis: [-1., 0., 0.],
            wavelength: 0.2846,
        },
        priors: Priors {
            cell: UnitCell {
                a_angstrom: 8.,
                b_angstrom: 6.,
                c_angstrom: 10.,
                alpha_deg: 90.,
                beta_deg: 90.,
                gamma_deg: 90.,
            },
            group: Some(SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap()),
            elements: vec!["Si".into()],
            provenance: "synthetic independent coordinator test".into(),
        },
        null_rotations: vec![identity; 64],
        shadows: ShadowOptions {
            illumination_block: 4,
            illumination: frame_validity::SeatOptions::default(),
            aperture_block: 4,
            aperture: frame_validity::DacOptions {
                axis_at_zero: [0., 0., 1.],
                half_opening_degrees: 35.,
            },
        },
        cancellation: phasesmith_single_crystal::solution::Cancellation::default(),
        mask_workers: 2,
        preparation_workers: 2,
        preparation_batch: 2,
        max_in_flight_frame_bytes: 1024 * 1024,
        detection_workers: 4,
        profile_workers: 2,
        learning_workers: 2,
        frame_workers: 2,
        geometry_cache_bytes: 0,
        frame_payload_bytes: 1024 * 1024,
        max_samples: 20_000_000,
        search_seconds: 10.,
        dense_memory_bytes: 16_000_000,
        search_trials: 1,
        search_cycles: 32,
        seed: 0,
    }
}

trait Bits {
    fn bits(&self, out: &mut Vec<u8>);
}
macro_rules! scalar { ($($t:ty),*) => { $(impl Bits for $t { fn bits(&self, out: &mut Vec<u8>) { out.extend(self.to_le_bytes()); } })* }; }
scalar!(u32, u64, i64);
impl Bits for usize {
    fn bits(&self, out: &mut Vec<u8>) {
        (*self as u64).bits(out);
    }
}
impl Bits for bool {
    fn bits(&self, out: &mut Vec<u8>) {
        out.push(u8::from(*self));
    }
}
impl Bits for f64 {
    fn bits(&self, out: &mut Vec<u8>) {
        self.to_bits().bits(out);
    }
}
impl<T: Bits, const N: usize> Bits for [T; N] {
    fn bits(&self, out: &mut Vec<u8>) {
        for x in self {
            x.bits(out);
        }
    }
}
impl<T: Bits> Bits for Vec<T> {
    fn bits(&self, out: &mut Vec<u8>) {
        self.len().bits(out);
        for x in self {
            x.bits(out);
        }
    }
}
macro_rules! fields { ($s:expr,$o:expr,$($field:ident),*) => { $($s.$field.bits($o);)* }; }
impl Bits for Geometry {
    fn bits(&self, o: &mut Vec<u8>) {
        fields!(self, o, origin, fast, slow, beam, axis, wavelength);
    }
}
impl Bits for Prediction {
    fn bits(&self, o: &mut Vec<u8>) {
        fields!(self, o, values, jacobian, flags);
    }
}
impl Bits for geometry::State {
    fn bits(&self, o: &mut Vec<u8>) {
        fields!(self, o, basis, geometry, basis_jacobian, detector_jacobian);
    }
}
impl Bits for geometry::Fit {
    fn bits(&self, o: &mut Vec<u8>) {
        fields!(
            self,
            o,
            parameters,
            state,
            initial,
            predictions,
            eligible,
            converged,
            at_bounds,
            scaled_condition,
            maximum_parameter_correlation,
            covariance,
            evaluations,
            history,
            rms,
            counts
        );
    }
}
impl Bits for Grain {
    fn bits(&self, o: &mut Vec<u8>) {
        fields!(
            self,
            o,
            direct,
            hkl,
            indexed,
            training,
            withheld,
            accepted,
            null_counts
        );
    }
}
impl Bits for Competitor {
    fn bits(&self, o: &mut Vec<u8>) {
        fields!(
            self, o, candidate, basis, hkl, crossings, training, withheld, rms
        );
    }
}
impl Bits for Indexed {
    fn bits(&self, o: &mut Vec<u8>) {
        fields!(
            self,
            o,
            grains,
            selected,
            track_ids,
            hkl,
            observed,
            training,
            eligible,
            fit,
            competitors
        );
    }
}
fn fixture(path: &Path) -> (Options, Vec<V3>) {
    let bytes = fs::read(path).unwrap();
    let (words, remainder) = bytes.as_chunks::<8>();
    assert_eq!(remainder, []);
    let mut chunks = words.iter();
    let mut number = || f64::from_le_bytes(*chunks.next().unwrap());
    let vectors: [[f64; 3]; 5] = std::array::from_fn(|_| std::array::from_fn(|_| number()));
    let mut o = options();
    o.geometry = Geometry {
        origin: vectors[0],
        fast: vectors[1],
        slow: vectors[2],
        beam: vectors[3],
        axis: vectors[4],
        wavelength: number(),
    };
    let c: [f64; 6] = std::array::from_fn(|_| number());
    o.priors.cell = UnitCell {
        a_angstrom: c[0],
        b_angstrom: c[1],
        c_angstrom: c[2],
        alpha_deg: c[3],
        beta_deg: c[4],
        gamma_deg: c[5],
    };
    o.null_rotations = (0..64)
        .map(|_| std::array::from_fn(|_| std::array::from_fn(|_| number())))
        .collect();
    let n = u64::from_le_bytes(*chunks.next().unwrap()) as usize;
    let points = (0..n)
        .map(|_| std::array::from_fn(|_| f64::from_le_bytes(*chunks.next().unwrap())))
        .collect();
    assert!(chunks.next().is_none());
    (o, points)
}
fn positions(observed: &[V3], o: &mut Options) -> Positions {
    let start = observed
        .iter()
        .map(|v| v[2])
        .fold(f64::INFINITY, f64::min)
        .floor()
        - 1.;
    let end = observed
        .iter()
        .map(|v| v[2])
        .fold(f64::NEG_INFINITY, f64::max)
        .ceil()
        + 1.;
    let n = ((end - start) / 0.5).ceil() as usize;
    o.acquisition.first = 0;
    o.acquisition.last = n - 1;
    o.acquisition.omega_zero = start;
    o.acquisition.omega_step = 0.5;
    o.acquisition.exposure = 0.5;
    let frames: Vec<_> = observed
        .iter()
        .map(|v| ((v[2] - start) / 0.5).floor() as usize)
        .collect();
    Positions {
        boxes: vec![],
        ranges: frames.iter().map(|&i| [i, i + 1]).collect(),
        values: observed
            .iter()
            .map(|v| [v[0], v[1], v[2], 0., 1.])
            .collect(),
        flags: vec![0; observed.len()],
        original_flags: vec![0; observed.len()],
        observations: observed
            .iter()
            .zip(&frames)
            .enumerate()
            .map(|(i, (v, &f))| [i as f64, f as f64, v[0], v[1], v[2]])
            .collect(),
        angles: (0..n)
            .map(|i| [start + 0.5 * i as f64, start + 0.5 * (i + 1) as f64])
            .collect(),
        hashes: vec![],
    }
}
#[test]
#[ignore = "release benchmark; requires PHASESMITH_NATIVE_GRAIN_INPUT and OUTPUT"]
fn complete_native_grain_benchmark() {
    if std::env::var_os("PHASESMITH_NATIVE_GRAIN_INPUT").is_none()
        && std::env::var_os("PHASESMITH_NATIVE_GRAIN_OUTPUT").is_none()
    {
        eprintln!("native grain benchmark skipped: configure its input and output paths");
        return;
    }
    let repeats =
        std::env::var("PHASESMITH_NATIVE_GRAIN_REPEATS").map_or(1, |n| n.parse().unwrap());
    let mut expected = None;
    for _ in 0..repeats {
        let bytes = run_once();
        if let Some(previous) = &expected {
            assert_eq!(&bytes, previous, "repeated complete results differ");
        } else {
            expected = Some(bytes);
        }
    }
}
fn run_once() -> Vec<u8> {
    let input = std::env::var("PHASESMITH_NATIVE_GRAIN_INPUT").unwrap();
    let output = std::env::var("PHASESMITH_NATIVE_GRAIN_OUTPUT").unwrap();
    let (mut o, points) = fixture(Path::new(&input));
    let mut bytes = Vec::new();
    let full = Path::new(&input)
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .starts_with("index-");
    let p = full.then(|| positions(&points, &mut o));
    let training: Vec<_> = (0..points.len()).map(|i| i % 5 != 0).collect();
    let train: Vec<_> = points
        .iter()
        .zip(&training)
        .filter_map(|(&q, &t)| t.then_some(q))
        .collect();
    let start = Instant::now();
    let elapsed;
    let message = if let Some(p) = p {
        let result = index(&p, &o, [4096, 4096]);
        elapsed = start.elapsed().as_secs_f64();
        match result {
            Ok(result) => {
                true.bits(&mut bytes);
                result.bits(&mut bytes);
                format!(
                    "grains={} selected={} fit_counts={:?} competitors={}",
                    result.grains.len(),
                    result.selected,
                    result.fit.counts,
                    result.competitors.len()
                )
            }
            Err(e) => {
                false.bits(&mut bytes);
                bytes.extend(e.0.as_bytes());
                format!("error={e}")
            }
        }
    } else {
        let result = discover_and_validate(&points, &training, &train, &o);
        elapsed = start.elapsed().as_secs_f64();
        match result {
            Ok(result) => {
                true.bits(&mut bytes);
                result.bits(&mut bytes);
                format!(
                    "grains={} accepted={}",
                    result.len(),
                    result.iter().filter(|g| g.accepted).count()
                )
            }
            Err(e) => {
                false.bits(&mut bytes);
                bytes.extend(e.0.as_bytes());
                format!("error={e}")
            }
        }
    };
    fs::write(output, &bytes).unwrap();
    println!("NATIVE_BENCH seconds={elapsed:.9} {message}");
    bytes
}
