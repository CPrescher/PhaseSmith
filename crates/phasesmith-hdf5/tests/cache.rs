//! Immutable decoded-count cache identity, mutation isolation and payload admission.
use phasesmith_hdf5::{Limits, cache::CachedReader};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/counts-bitshuffle-lz4.h5")
}

#[test]
fn decoded_cache_is_bounded_source_scoped_and_caller_mutation_isolated() {
    let reader = CachedReader::open(fixture(), "detector/data", Limits::default(), 48).unwrap();
    let mut first = reader.read_frame(0).unwrap();
    let original = first.pixels.clone();
    first.pixels.fill(42);
    assert_eq!(reader.read_frame(0).unwrap().pixels, original);
    reader.read_frame(1).unwrap();
    assert_eq!(reader.read_frame(0).unwrap().pixels, original);
    let stats = reader.stats().unwrap();
    assert_eq!(stats.requests, 4);
    assert_eq!(stats.hits, 2);
    assert_eq!(stats.decodes, 2);
    assert_eq!(stats.frames, 1);
    assert_eq!(stats.payload_bytes, 48);
    let tiny = CachedReader::open(fixture(), "detector/data", Limits::default(), 1).unwrap();
    for _ in 0..2 {
        assert_eq!(tiny.read_frame(0).unwrap().pixels, original);
    }
    assert_eq!(tiny.stats().unwrap().hits, 0);
    assert_eq!(tiny.stats().unwrap().payload_bytes, 0);
    assert!(reader.read_frame(2).is_err());
}

#[cfg(unix)]
#[test]
fn changed_source_clears_cache_and_fails_closed() {
    use std::io::{Read, Seek, SeekFrom, Write};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "phasesmith-source-cache-{}-{}.h5",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::copy(fixture(), &path).unwrap();
    let reader = CachedReader::open(&path, "detector/data", Limits::default(), 1024).unwrap();
    reader.read_frame(0).unwrap();
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    let mut first = [0_u8];
    file.read_exact(&mut first).unwrap();
    file.seek(SeekFrom::Start(0)).unwrap();
    file.write_all(&first).unwrap();
    file.sync_all().unwrap();
    drop(file);
    assert!(
        reader
            .read_frame(0)
            .err()
            .unwrap()
            .to_string()
            .contains("source changed")
    );
    let stats = reader.stats().unwrap();
    assert_eq!(stats.frames, 0);
    assert_eq!(stats.payload_bytes, 0);
    assert_eq!(stats.invalidations, 1);
    drop(reader);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn cached_external_links_require_explicit_source_validation() {
    let path = std::env::temp_dir().join(format!(
        "phasesmith-linked-source-{}.h5",
        std::process::id()
    ));
    let file = hdf5::File::create(&path).unwrap();
    file.link_external(fixture().to_str().unwrap(), "/detector/data", "foreign")
        .unwrap();
    file.close().unwrap();
    assert!(
        CachedReader::open(&path, "foreign", Limits::default(), 1024)
            .err()
            .unwrap()
            .to_string()
            .contains("self-contained")
    );
    let ordinary = CachedReader::open(&path, "foreign", Limits::default(), 0).unwrap();
    assert_eq!(ordinary.read_frame(0).unwrap().pixels.len(), 12);
    drop(ordinary);
    std::fs::remove_file(path).unwrap();
}
