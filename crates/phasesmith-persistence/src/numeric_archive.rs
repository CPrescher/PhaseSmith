//! Lossless numeric NPY/NPZ output, independent of Python and project schemas.
use std::{collections::BTreeSet, fs::OpenOptions, io::Write, path::Path};
use zip::{CompressionMethod, ZipWriter, write::SimpleFileOptions};

use crate::PersistenceError;

/// One contiguous C-order numeric array, retaining its exact stored bytes.
pub struct NumericArray<'a> {
    /// Flat archive key (ASCII letters, digits and underscore).
    pub name: &'a str,
    /// NPY dtype descriptor, including byte order.
    pub dtype: &'a str,
    /// Array dimensions; empty denotes a scalar.
    pub shape: &'a [usize],
    /// Exact numeric payload, including signed zeros and NaN bit patterns.
    pub bytes: &'a [u8],
}

fn invalid(message: &str) -> PersistenceError {
    PersistenceError::InvalidArray {
        message: message.into(),
    }
}

fn header(array: &NumericArray<'_>) -> Result<Vec<u8>, PersistenceError> {
    if array.name.is_empty()
        || !array
            .name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return Err(invalid(
            "numeric archive keys must be flat ASCII identifiers",
        ));
    }
    let width: usize = match array.dtype {
        "|b1" | "|u1" | "|i1" => 1,
        "<i2" | ">i2" | "<u2" | ">u2" | "<f2" | ">f2" => 2,
        "<i4" | ">i4" | "<u4" | ">u4" | "<f4" | ">f4" => 4,
        "<i8" | ">i8" | "<u8" | ">u8" | "<f8" | ">f8" => 8,
        _ => return Err(invalid("unsupported numeric archive dtype")),
    };
    if array.shape.len() > 32 {
        return Err(invalid("numeric archive rank exceeds 32"));
    }
    let bytes = array.shape.iter().try_fold(width, |n, &d| n.checked_mul(d));
    if bytes != Some(array.bytes.len()) {
        return Err(invalid("numeric archive shape/byte count mismatch"));
    }
    let shape = array
        .shape
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let comma = if array.shape.len() == 1 { "," } else { "" };
    let mut text = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': ({shape}{comma}), }}",
        array.dtype
    );
    let padding = (64 - ((10 + text.len() + 1) % 64)) % 64;
    text.extend(std::iter::repeat_n(' ', padding));
    text.push('\n');
    let length = u16::try_from(text.len()).map_err(|_| invalid("numeric NPY header too long"))?;
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(text.as_bytes());
    Ok(bytes)
}

/// Write a new deterministic NPZ file using native Rust lossless compression.
///
/// Level zero stores uncompressed members; 1–9 select DEFLATE effort. At most
/// 4096 arrays and 1 GiB total raw payload are accepted. Validate before creating
/// the destination; existing files are never overwritten. Numeric bytes are
/// not interpreted or changed. Ordinary streaming compressor workspace is extra.
///
/// # Errors
/// Invalid keys, dtypes, shapes, limits, existing paths or I/O failures.
pub fn write_numeric_npz(
    path: &Path,
    arrays: &[NumericArray<'_>],
    level: u8,
) -> Result<(), PersistenceError> {
    if level > 9 || arrays.len() > 4096 {
        return Err(invalid("numeric archive compression/array limit exceeded"));
    }
    let mut names = BTreeSet::new();
    let mut total = 0_usize;
    let mut headers = Vec::with_capacity(arrays.len());
    for array in arrays {
        if !names.insert(array.name) {
            return Err(invalid("duplicate numeric archive key"));
        }
        total = total
            .checked_add(array.bytes.len())
            .ok_or_else(|| invalid("archive size overflow"))?;
        if total > 1024 * 1024 * 1024 {
            return Err(invalid("numeric archive raw payload exceeds 1 GiB"));
        }
        headers.push(header(array)?);
    }
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let result = (|| {
        let mut archive = ZipWriter::new(file);
        let options = SimpleFileOptions::default()
            .last_modified_time(zip::DateTime::default())
            .compression_method(if level == 0 {
                CompressionMethod::Stored
            } else {
                CompressionMethod::Deflated
            })
            .compression_level(if level == 0 {
                None
            } else {
                Some(i64::from(level))
            });
        for (array, header) in arrays.iter().zip(headers) {
            archive
                .start_file(format!("{}.npy", array.name), options)
                .map_err(|e| invalid(&e.to_string()))?;
            archive.write_all(&header)?;
            archive.write_all(array.bytes)?;
        }
        archive.finish().map_err(|e| invalid(&e.to_string()))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(path);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::Read,
        sync::atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn numeric_archive_is_lossless_deterministic_and_exclusive_without_python() {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "phasesmith-numeric-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let data = [0_u64, (-0_f64).to_bits(), 0x7ff8_0000_0000_0123]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect::<Vec<_>>();
        let array = NumericArray {
            name: "counts",
            dtype: "<f8",
            shape: &[3],
            bytes: &data,
        };
        for level in [0, 1, 6] {
            let path = root.join(format!("level-{level}.npz"));
            write_numeric_npz(&path, std::slice::from_ref(&array), level).unwrap();
            let first = std::fs::read(&path).unwrap();
            assert!(write_numeric_npz(&path, std::slice::from_ref(&array), level).is_err());
            assert_eq!(first, std::fs::read(&path).unwrap());
            let repeat = root.join(format!("repeat-{level}.npz"));
            write_numeric_npz(&repeat, std::slice::from_ref(&array), level).unwrap();
            assert_eq!(first, std::fs::read(repeat).unwrap());
            let mut zip = zip::ZipArchive::new(std::io::Cursor::new(first)).unwrap();
            let mut member = Vec::new();
            zip.by_name("counts.npy")
                .unwrap()
                .read_to_end(&mut member)
                .unwrap();
            assert_eq!(&member[..8], b"\x93NUMPY\x01\x00");
            let length = usize::from(u16::from_le_bytes([member[8], member[9]]));
            assert_eq!(&member[10 + length..], &data);
        }
        let path = root.join("invalid.npz");
        let bad = NumericArray {
            shape: &[4],
            ..array
        };
        assert!(write_numeric_npz(&path, &[bad], 1).is_err());
        assert!(!path.exists());
        std::fs::remove_dir_all(root).unwrap();
    }
}
