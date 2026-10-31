//! Native fixtures exercise bundled filters without a Python writer/reader.
#![allow(clippy::float_cmp)] // Unmodified integer counts and exact scalar round trips.
use phasesmith_hdf5::{Limits, PixelType, Reader};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "phasesmith-hdf5-{}-{}.h5",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )))
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn bundled_bitshuffle_lz4_and_standard_filters_preserve_counts_and_metadata() {
    hdf5::sync::sync(hdf5_bitshuffle::register_bitshuffle_plugin);
    for filter in [
        None,
        Some(hdf5::filters::Filter::deflate(4)),
        Some(hdf5::filters::Filter::lzf()),
        Some(hdf5::filters::Filter::user(32008, &[0, 2])),
    ] {
        let fixture = Fixture::new();
        let counts = [
            0_u32,
            70_000,
            u32::MAX - 1,
            u32::MAX,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
        ];
        {
            let file = hdf5::File::create(&fixture.0).unwrap();
            let mut builder = file.new_dataset::<u32>().shape((2, 3, 4)).chunk((1, 3, 4));
            if let Some(filter) = &filter {
                builder = builder.set_filters(std::slice::from_ref(filter));
            }
            builder
                .create("counts")
                .unwrap()
                .write_raw(&counts)
                .unwrap();
            file.new_dataset::<f64>()
                .shape(())
                .create("pixel_size")
                .unwrap()
                .write_scalar(&75e-6)
                .unwrap();
        }
        let reader = Reader::open(&fixture.0, "counts", Limits::default()).unwrap();
        assert_eq!(reader.shape(), [2, 3, 4]);
        assert_eq!(reader.pixel_type(), PixelType::U32);
        assert_eq!(reader.scalar("pixel_size").unwrap(), 75e-6);
        assert!(reader.scalar("counts").is_err());
        assert!(reader.scalar("missing").is_err());
        assert_eq!(reader.read_frame(0).unwrap().pixels, counts[..12]);
        assert_eq!(reader.read_frame(1).unwrap().pixels, counts[12..]);
        assert!(reader.read_frame(2).is_err());
        assert!(reader.filters().iter().all(|f| f.decode_enabled));
        if let Some(filter) = filter {
            assert_eq!(reader.filters()[0].id, filter.id());
        }
    }
}

#[test]
fn unsigned16_is_promoted_exactly_and_no_display_flip_occurs() {
    let fixture = Fixture::new();
    {
        let file = hdf5::File::create(&fixture.0).unwrap();
        file.new_dataset::<u16>()
            .shape((1, 2, 3))
            .create("counts")
            .unwrap()
            .write_raw(&[0, 65535, 2, 3, 4, 5])
            .unwrap();
    }
    let reader = Reader::open(&fixture.0, "counts", Limits::default()).unwrap();
    assert_eq!(reader.pixel_type(), PixelType::U16);
    assert_eq!(reader.read_frame(0).unwrap().pixels, [0, 65535, 2, 3, 4, 5]);
}

#[test]
fn rank_type_shape_frame_and_chunk_limits_are_checked_before_pixel_reads() {
    let fixture = Fixture::new();
    {
        let file = hdf5::File::create(&fixture.0).unwrap();
        file.new_dataset::<u32>()
            .shape((2, 3, 4))
            .chunk((2, 3, 4))
            .create("counts")
            .unwrap();
        file.new_dataset::<f64>()
            .shape((2, 3, 4))
            .create("float")
            .unwrap();
        file.new_dataset::<u32>()
            .shape((3, 4))
            .create("rank")
            .unwrap();
        file.new_dataset::<f64>()
            .shape(())
            .create("nan")
            .unwrap()
            .write_scalar(&f64::NAN)
            .unwrap();
    }
    for dataset in ["missing", "float", "rank"] {
        assert!(Reader::open(&fixture.0, dataset, Limits::default()).is_err());
    }
    for limits in [
        Limits {
            frames: 1,
            ..Limits::default()
        },
        Limits {
            frame_pixels: 11,
            ..Limits::default()
        },
        Limits {
            chunk_bytes: 95,
            ..Limits::default()
        },
        Limits {
            frame_pixels: 0,
            ..Limits::default()
        },
    ] {
        assert!(Reader::open(&fixture.0, "counts", limits).is_err());
    }
    assert!(
        Reader::open(&fixture.0, "counts", Limits::default())
            .unwrap()
            .scalar("nan")
            .is_err()
    );
}

#[test]
fn independent_h5py_bitshuffle_fixture_decodes_without_external_plugins() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/counts-bitshuffle-lz4.h5");
    let reader = Reader::open(path, "detector/data", Limits::default()).unwrap();
    assert_eq!(reader.filters()[0].id, 32008);
    assert_eq!(
        reader.read_frame(0).unwrap().pixels[..4],
        [0, 70_000, u32::MAX - 1, u32::MAX]
    );
    assert_eq!(
        reader.read_frame(1).unwrap().pixels,
        (12_u32..24).collect::<Vec<_>>()
    );
}

#[test]
fn virtual_sources_cannot_bypass_chunk_admission_or_silently_supply_missing_frames() {
    let fixture = Fixture::new();
    {
        let file = hdf5::File::create(&fixture.0).unwrap();
        file.new_dataset::<u32>()
            .shape((2, 3, 4))
            .create("counts")
            .unwrap();
        file.new_dataset::<u32>()
            .shape((2, 3, 4))
            .virtual_map(
                ".",
                "/counts",
                (2, 3, 4),
                (.., .., ..),
                (2, 3, 4),
                (.., .., ..),
            )
            .create("virtual")
            .unwrap();
    }
    let error = Reader::open(&fixture.0, "virtual", Limits::default()).unwrap_err();
    assert!(error.to_string().contains("virtual detector"));
}
