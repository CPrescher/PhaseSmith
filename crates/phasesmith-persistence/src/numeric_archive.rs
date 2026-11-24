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

static TEMP_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

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
    if path.exists() {
        return Err(PersistenceError::Io(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "destination numeric archive already exists",
        )));
    }
    let parent = path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("archive.npz");
    let temp_path = loop {
        let id = TEMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let candidate = parent.join(format!(".{file_name}.tmp-{}-{id}", std::process::id()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => break (candidate, file),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(PersistenceError::Io(error)),
        }
    };
    let (temp_path, file) = temp_path;
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
        let file = archive.finish().map_err(|e| invalid(&e.to_string()))?;
        file.sync_all()?;
        drop(file);
        publish_exclusive(&temp_path, path, |src, dst| std::fs::hard_link(src, dst))
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result
}

fn publish_exclusive(
    temp_path: &Path,
    destination: &Path,
    link: impl FnOnce(&Path, &Path) -> std::io::Result<()>,
) -> Result<(), PersistenceError> {
    match link(temp_path, destination) {
        Ok(()) => {
            let _ = std::fs::remove_file(temp_path);
            sync_parent_dir_best_effort(destination);
            Ok(())
        }
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::Unsupported | std::io::ErrorKind::PermissionDenied
            ) =>
        {
            // Filesystems without hard links (e.g. FAT/exFAT/CIFS returning EPERM or ENOTSUP):
            // create the destination exclusively so a concurrent creator can never be overwritten,
            // then stream and sync the staged archive bytes.
            let mut target = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(destination)?;
            let copy_result = (|| -> std::io::Result<()> {
                let mut staged = std::fs::File::open(temp_path)?;
                std::io::copy(&mut staged, &mut target)?;
                target.sync_all()?;
                Ok(())
            })();
            drop(target);
            if let Err(copy_error) = copy_result {
                let _ = std::fs::remove_file(destination);
                return Err(PersistenceError::Io(copy_error));
            }
            let _ = std::fs::remove_file(temp_path);
            sync_parent_dir_best_effort(destination);
            Ok(())
        }
        Err(error) => Err(PersistenceError::Io(error)),
    }
}

#[cfg(not(windows))]
fn sync_parent_dir_best_effort(path: &Path) {
    let parent = path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if let Ok(dir) = std::fs::File::open(parent) {
        let _ = dir.sync_all();
    }
}

#[cfg(windows)]
fn sync_parent_dir_best_effort(_path: &Path) {}

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

    #[test]
    fn publish_exclusive_fallback_handles_unsupported_and_permission_denied_without_overwrite() {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "phasesmith-numeric-fallback-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();

        for (name, kind) in [
            ("unsupported.npz", std::io::ErrorKind::Unsupported),
            ("eperm.npz", std::io::ErrorKind::PermissionDenied),
        ] {
            let staged = root.join(format!(".{name}.tmp"));
            let dest = root.join(name);
            std::fs::write(&staged, b"staged-payload").unwrap();
            publish_exclusive(&staged, &dest, |_, _| Err(std::io::Error::from(kind))).unwrap();
            assert_eq!(std::fs::read(&dest).unwrap(), b"staged-payload");
            assert!(!staged.exists());

            // Must refuse to overwrite an existing destination even on fallback path.
            let staged_second = root.join(format!(".{name}.tmp2"));
            std::fs::write(&staged_second, b"new-payload").unwrap();
            let err = publish_exclusive(&staged_second, &dest, |_, _| {
                Err(std::io::Error::from(kind))
            })
            .unwrap_err();
            assert!(matches!(
                err,
                PersistenceError::Io(ref io_error)
                    if io_error.kind() == std::io::ErrorKind::AlreadyExists
            ));
            assert_eq!(std::fs::read(&dest).unwrap(), b"staged-payload");
            std::fs::remove_file(&staged_second).unwrap();
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}
