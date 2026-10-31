//! Inspect original frame hashes with a standalone bundled native reader.
use phasesmith_hdf5::{Limits, Reader};
use sha2::{Digest, Sha256};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: inspect_frames FILE DATASET [FRAME ...]")?;
    let dataset = args.next().ok_or("dataset path required")?;
    let reader = Reader::open(path, &dataset, Limits::default())?;
    println!(
        "HDF5 {:?}, shape {:?}, source {:?}, filters {:?}",
        phasesmith_hdf5::library_version(),
        reader.shape(),
        reader.pixel_type(),
        reader.filters()
    );
    for argument in args {
        let index: usize = argument.parse()?;
        let frame = reader.read_frame(index)?;
        let mut digest = Sha256::new();
        for count in frame.pixels {
            digest.update(count.to_le_bytes());
        }
        println!("frame {index} sha256 {:x}", digest.finalize());
    }
    Ok(())
}
