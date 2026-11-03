use super::*;
use phasesmith_crystallography::SymmetryOperation;

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
struct Source {
    reads: usize,
    change: bool,
}
impl FrameSource for Source {
    fn shape(&self) -> [usize; 3] {
        [8, 16, 16]
    }
    fn read(&mut self, index: usize) -> Result<Vec<u32>> {
        self.reads += 1;
        let mut counts = vec![100; 256];
        if self.change && self.reads > 8 && index == 0 {
            counts[0] = 101;
        }
        Ok(counts)
    }
}
struct StopAfterMask {
    seen: bool,
    geometry: Geometry,
}
impl Observer for StopAfterMask {
    fn masks(&mut self, p: &frame_input::Plan, _: &[f64]) -> Result<()> {
        assert_eq!(p.expected_hashes[0].len(), 8);
        assert_eq!(p.blocks.len(), 2);
        assert!(p.dacs.is_empty());
        for (i, angle) in bounds(&options()).iter().enumerate() {
            let sh = frame_validity::Shape {
                height: 16,
                width: 16,
                block: 4,
            };
            let expected =
                frame_validity::dac_mask(self.geometry, *angle, sh, options().shadows.aperture)?
                    .excluded;
            assert_eq!(&p.blocks[1].excluded[i * 16..(i + 1) * 16], &expected);
        }
        self.seen = true;
        Err(Error("intentional observer stop after native masks".into()))
    }
}
#[test]
fn shared_masks_are_frozen_before_any_position_or_search_work() {
    let o = options();
    let mut s = Source {
        reads: 0,
        change: false,
    };
    let mut observer = StopAfterMask {
        seen: false,
        geometry: o.geometry,
    };
    let err = run(&mut s, &o, &mut observer).err().expect("observer stop");
    assert!(err.0.contains("intentional observer stop"));
    assert!(observer.seen);
    assert_eq!(s.reads, 8);
}
#[test]
fn cancellation_and_payload_admission_do_not_read_source_frames() {
    for kind in 0..4 {
        let mut o = options();
        match kind {
            0 => {
                let _ = o.cancellation.cancel();
            }
            1 => o.frame_payload_bytes = 1,
            2 => o.max_in_flight_frame_bytes = 1,
            _ => o.acquisition.last = 8,
        }
        let mut source = Source {
            reads: 0,
            change: false,
        };
        assert!(run(&mut source, &o, &mut NoObserver).is_err());
        assert_eq!(source.reads, 0);
    }
}
#[test]
fn original_count_change_between_masks_and_detection_is_an_error() {
    let mut source = Source {
        reads: 0,
        change: true,
    };
    let err = run(&mut source, &options(), &mut NoObserver)
        .err()
        .expect("source changed");
    assert!(err.0.contains("source frame hash differs"), "{err}");
    assert!(source.reads <= 12);
}
#[test]
fn source_order_is_preserved_across_preparation_and_worker_completion() {
    let o = options();
    let p = frame_input::Plan {
        shape: [16, 16],
        frames: 8,
        vertical_flip: false,
        reserved_minimum: u32::MAX,
        expected_hashes: vec![],
        blocks: vec![],
        dacs: vec![],
    };
    let mut source = Source {
        reads: 0,
        change: false,
    };
    let mut result = vec![];
    map_frames(
        &mut source,
        &p,
        &o,
        4,
        |p| {
            if p.index == 0 {
                std::thread::yield_now();
            }
            Ok(p.index)
        },
        |i, row| {
            assert_eq!(i, row);
            result.push(i);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(result, (0..8).collect::<Vec<_>>());
}
#[test]
fn callback_failure_prevents_later_source_batches_and_joins_workers() {
    let o = options();
    let p = frame_input::Plan {
        shape: [16, 16],
        frames: 8,
        vertical_flip: false,
        reserved_minimum: u32::MAX,
        expected_hashes: vec![],
        blocks: vec![],
        dacs: vec![],
    };
    let mut source = Source {
        reads: 0,
        change: false,
    };
    let err = map_frames(
        &mut source,
        &p,
        &o,
        4,
        |p| Ok(p.index),
        |i, _| {
            if i == 0 {
                Err(Error("stop first source observer".into()))
            } else {
                panic!("later rows must not reach observer")
            }
        },
    )
    .unwrap_err();
    assert!(err.0.contains("stop first"));
    assert_eq!(source.reads, 4);
}
