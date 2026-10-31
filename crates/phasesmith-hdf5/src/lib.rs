//! Read-only, bounded HDF5 detector frames with bundled native compression.
//!
//! HDF5, zlib and Bitshuffle/LZ4 build from pinned Cargo sources; no installed
//! HDF5, Python interpreter or external filter path is required. LZF is also
//! bundled. Other filters must be available to HDF5 and are reported explicitly.
//! This adapter never flips, normalizes, masks or otherwise changes counts.

pub mod cache;

use hdf5::{Dataset, File};
use std::fmt::{Display, Formatter};
use std::path::Path;

/// I/O, unsupported-format or resource-limit failure.
#[derive(Debug)]
pub struct Error(String);
impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
impl From<hdf5::Error> for Error {
    fn from(value: hdf5::Error) -> Self {
        Self(value.to_string())
    }
}

/// Allocation/shape admission before reading detector pixels.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Maximum number of frames in a dataset.
    pub frames: usize,
    /// Maximum pixels per frame (u32 output uses four bytes per pixel).
    pub frame_pixels: usize,
    /// Maximum decoded bytes in any source chunk.
    pub chunk_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            frames: 1_000_000,
            frame_pixels: 100_000_000,
            chunk_bytes: 268_435_456,
        }
    }
}

/// Source pixel storage; unsigned values are promoted losslessly to u32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelType {
    /// Unsigned 16-bit counts.
    U16,
    /// Unsigned 32-bit counts, including detector-reserved sentinel values.
    U32,
}

/// Required filter and decoder availability at reader creation.
#[derive(Clone, Debug)]
pub struct Filter {
    /// Standard HDF5 filter identifier.
    pub id: i32,
    /// Whether the linked native library can decode this filter.
    pub decode_enabled: bool,
    /// Original user-filter configuration parameters; empty for built-in filters.
    pub parameters: Vec<u32>,
}

/// Original detector pixels in native row-major array orientation.
pub struct Frame {
    /// Detector height and width.
    pub shape: [usize; 2],
    /// All counts, including zero, high finite counts and reserved u32 values.
    pub pixels: Vec<u32>,
}

/// A read-only dataset handle. All HDF5 calls use the upstream library mutex.
/// Dataset rows are interpreted explicitly as `(frame, detector_y, detector_x)`.
#[derive(Debug)]
pub struct Reader {
    file: File,
    dataset: Dataset,
    shape: [usize; 3],
    pixel_type: PixelType,
    filters: Vec<Filter>,
}
impl Reader {
    /// Open exactly the caller-specified 3D unsigned-count dataset read-only.
    /// # Errors
    /// Rejects unsupported types/ranks, unavailable filters, malformed dimensions
    /// or frame/chunk allocations above the declared limits.
    pub fn open(path: impl AsRef<Path>, dataset: &str, limits: Limits) -> Result<Self, Error> {
        if limits.frames == 0 || limits.frame_pixels == 0 || limits.chunk_bytes == 0 {
            return Err(Error("HDF5 read limits must be positive".into()));
        }
        hdf5::sync::sync(|| std::panic::catch_unwind(hdf5_bitshuffle::register_bitshuffle_plugin))
            .map_err(|_| Error("could not register bundled Bitshuffle/LZ4 decoder".into()))?;
        let file = File::open(path)?;
        let dataset = file.dataset(dataset)?;
        if dataset.layout() == hdf5::plist::dataset_create::Layout::Virtual {
            return Err(Error("virtual detector datasets require separate source/chunk validation; open an explicit source dataset".into()));
        }
        let dimensions = dataset.shape();
        let shape: [usize; 3] = dimensions
            .try_into()
            .map_err(|_| Error("detector dataset must have shape (frames,height,width)".into()))?;
        if shape.contains(&0)
            || shape[0] > limits.frames
            || shape[1]
                .checked_mul(shape[2])
                .is_none_or(|n| n > limits.frame_pixels)
        {
            return Err(Error("detector dataset exceeds frame/pixel limits".into()));
        }
        let dtype = dataset.dtype()?;
        let pixel_type = if dtype.is::<u16>() {
            PixelType::U16
        } else if dtype.is::<u32>() {
            PixelType::U32
        } else {
            return Err(Error(
                "detector counts must be unsigned uint16 or uint32".into(),
            ));
        };
        if let Some(chunks) = dataset.chunk() {
            let bytes = chunks
                .into_iter()
                .try_fold(dtype.size(), usize::checked_mul);
            if bytes.is_none_or(|n| n > limits.chunk_bytes) {
                return Err(Error(
                    "detector dataset exceeds decoded chunk-byte limit".into(),
                ));
            }
        }
        let filters: Vec<_> = dataset
            .filters()
            .into_iter()
            .map(|filter| {
                let (id, parameters) = match &filter {
                    hdf5::filters::Filter::User(id, params) => (*id, params.clone()),
                    _ => (filter.id(), Vec::new()),
                };
                Filter {
                    id,
                    decode_enabled: filter.decode_enabled(),
                    parameters,
                }
            })
            .collect();
        if let Some(filter) = filters.iter().find(|f| !f.decode_enabled) {
            return Err(Error(format!(
                "HDF5 filter {} has no decoder; bundled filters include Bitshuffle/LZ4 (32008), gzip and LZF",
                filter.id
            )));
        }
        Ok(Self {
            file,
            dataset,
            shape,
            pixel_type,
            filters,
        })
    }

    /// Original frame count, height and width.
    #[must_use]
    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// Source storage type before exact promotion to u32.
    #[must_use]
    pub fn pixel_type(&self) -> PixelType {
        self.pixel_type
    }

    /// Actual required dataset filters and availability.
    #[must_use]
    pub fn filters(&self) -> &[Filter] {
        &self.filters
    }

    /// Read one original frame without flipping, normalization or sentinel changes.
    /// # Errors
    /// Rejects out-of-range indices or a failed native HDF5 decode.
    pub fn read_frame(&self, index: usize) -> Result<Frame, Error> {
        if index >= self.shape[0] {
            return Err(Error("HDF5 frame index outside dataset".into()));
        }
        let values = self.dataset.read_slice_2d::<u32, _>((index, .., ..))?;
        let pixels = values.iter().copied().collect();
        Ok(Frame {
            shape: [self.shape[1], self.shape[2]],
            pixels,
        })
    }

    /// Read one explicitly named finite scalar metadata dataset as f64.
    /// # Errors
    /// Rejects non-scalar, non-numeric, nonfinite or unreadable metadata.
    pub fn scalar(&self, path: &str) -> Result<f64, Error> {
        let dataset = self.file.dataset(path)?;
        if !dataset.shape().is_empty() {
            return Err(Error("metadata dataset must be scalar".into()));
        }
        let value = dataset.read_scalar::<f64>()?;
        if !value.is_finite() {
            return Err(Error("metadata scalar must be finite".into()));
        }
        Ok(value)
    }
}

/// Runtime native-library version, useful for external-data provenance.
#[must_use]
pub fn library_version() -> (u8, u8, u8) {
    hdf5::library_version()
}
