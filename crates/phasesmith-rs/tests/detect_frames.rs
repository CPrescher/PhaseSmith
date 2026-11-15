//! Local and full second detection passes on every frame of a real scan.
//!
//! `PHASESMITH_DETECT_FRAMES=<file.h5> PHASESMITH_DETECT_DATASET=<path>
//! cargo test --release -p phasesmith --features hdf5 --test detect_frames -- --ignored --nocapture`
#![cfg(all(feature = "hdf5", feature = "spots"))]
use phasesmith::hdf5::{Limits, Reader};
use phasesmith::spots::{Image, Options, detect, detect_two_pass};
use std::time::{Duration, Instant};

#[test]
#[ignore = "requires PHASESMITH_DETECT_FRAMES and PHASESMITH_DETECT_DATASET"]
fn local_second_pass_matches_the_full_pass_on_recorded_frames() {
    let (Ok(path), Ok(dataset)) = (
        std::env::var("PHASESMITH_DETECT_FRAMES"),
        std::env::var("PHASESMITH_DETECT_DATASET"),
    ) else {
        eprintln!("detection comparison skipped: configure the frame file and dataset");
        return;
    };
    let reader = Reader::open(&path, &dataset, Limits::default()).unwrap();
    let [frames, height, width] = reader.shape();
    let options = Options::default();
    let (mut local, mut full, mut spots) = (Duration::ZERO, Duration::ZERO, 0);
    for index in 0..frames {
        let pixels = reader.read_frame(index).unwrap().pixels;
        // Reserved counts (module gaps, defective pixels) are masked.
        let mask: Vec<bool> = pixels.iter().map(|&p| p >= u32::MAX - 1).collect();
        let values: Vec<f64> = pixels.iter().map(|&p| f64::from(p)).collect();
        let image = Image {
            values: &values,
            height,
            width,
            mask: &mask,
            variance: None,
            gain: 1.0,
            read_noise: 0.0,
            saturation: None,
        };
        let start = Instant::now();
        let a = detect(&image, options).unwrap();
        local += start.elapsed();
        let start = Instant::now();
        let b = detect_two_pass(&image, options).unwrap();
        full += start.elapsed();
        assert_eq!(a.rejected_small, b.rejected_small, "frame {index}");
        assert_eq!(a.spots.len(), b.spots.len(), "frame {index}");
        for (x, y) in a.spots.iter().zip(&b.spots) {
            assert_eq!(x.bounds, y.bounds, "frame {index}");
            assert_eq!(x.flags, y.flags, "frame {index}");
            assert_eq!(
                x.values.map(f64::to_bits),
                y.values.map(f64::to_bits),
                "frame {index}"
            );
        }
        spots += a.spots.len();
    }
    eprintln!(
        "{frames} frames, {spots} spots identical; local {local:?}, full {full:?}, speed-up {:.2}x",
        full.as_secs_f64() / local.as_secs_f64()
    );
}
