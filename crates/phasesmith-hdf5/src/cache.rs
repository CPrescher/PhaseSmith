//! Source-bound immutable decoded-count cache; no normalized or masked data.
use crate::{Error, Filter, Frame, Limits, PixelType, Reader};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::SystemTime,
};

#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stamp {
    bytes: u64,
    modified: SystemTime,
    created: Option<SystemTime>,
    #[cfg(unix)]
    identity: (u64, u64, i64, i64),
}
fn stamp(path: &Path) -> Result<Stamp, Error> {
    let meta = std::fs::metadata(path).map_err(|e| Error(e.to_string()))?;
    Ok(Stamp {
        bytes: meta.len(),
        modified: meta.modified().map_err(|e| Error(e.to_string()))?,
        created: meta.created().ok(),
        #[cfg(unix)]
        identity: (meta.dev(), meta.ino(), meta.ctime(), meta.ctime_nsec()),
    })
}
fn copy_frame(frame: &Frame) -> Result<Frame, Error> {
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(frame.pixels.len())
        .map_err(|e| Error(e.to_string()))?;
    pixels.extend_from_slice(&frame.pixels);
    Ok(Frame {
        shape: frame.shape,
        pixels,
    })
}
/// Cache payload and source-read counters, separate from process memory.
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheStats {
    /// Declared count-payload cap.
    pub budget_bytes: usize,
    /// Admitted count-vector capacity in bytes.
    pub payload_bytes: usize,
    /// Resident immutable original frames.
    pub frames: usize,
    /// Successful caller requests.
    pub requests: usize,
    /// Requests served by the cache.
    pub hits: usize,
    /// Successful native HDF5 decodes.
    pub decodes: usize,
    /// Changed-source failures, which clear cached payloads.
    pub invalidations: usize,
}
struct State {
    frames: BTreeMap<usize, Arc<Frame>>,
    stats: CacheStats,
}
/// One read-only source handle with deterministic prefix admission and no eviction.
///
/// Cache only original u32 decoder output. A hit returns a new owned copy, so
/// callers cannot mutate cached counts. Source path/metadata identity is checked
/// before every cached read; changes clear the cache and fail until reopening.
/// Each instance is independent and is never persisted or shared across runs.
pub struct CachedReader {
    reader: Reader,
    path: PathBuf,
    stamp: Stamp,
    state: Mutex<State>,
}
impl CachedReader {
    /// Open a source with a 0–4 GiB explicit decoded-payload cap; zero streams.
    /// # Errors
    /// Invalid budget, file identity, shape, filters or original reader limits.
    pub fn open(
        path: impl AsRef<Path>,
        dataset: &str,
        limits: Limits,
        budget: usize,
    ) -> Result<Self, Error> {
        if budget > usize::try_from(4_u64 * 1024 * 1024 * 1024).unwrap_or(usize::MAX) {
            return Err(Error("decoded-frame cache exceeds 4 GiB".into()));
        }
        let path = std::fs::canonicalize(path).map_err(|e| Error(e.to_string()))?;
        let identity = stamp(&path)?;
        let reader = Reader::open(&path, dataset, limits)?;
        if budget > 0 {
            let actual = std::fs::canonicalize(reader.dataset.filename())
                .map_err(|e| Error(e.to_string()))?;
            if actual != path || !reader.dataset.dcpl()?.get_external()?.is_empty() {
                return Err(Error(
                    "decoded cache requires a self-contained source dataset".into(),
                ));
            }
        }
        Ok(Self {
            reader,
            path,
            stamp: identity,
            state: Mutex::new(State {
                frames: BTreeMap::new(),
                stats: CacheStats {
                    budget_bytes: budget,
                    ..Default::default()
                },
            }),
        })
    }
    /// Dataset dimensions.
    #[must_use]
    pub fn shape(&self) -> [usize; 3] {
        self.reader.shape()
    }
    /// Original unsigned source dtype.
    #[must_use]
    pub fn pixel_type(&self) -> PixelType {
        self.reader.pixel_type()
    }
    /// Original compression filters.
    #[must_use]
    pub fn filters(&self) -> &[Filter] {
        self.reader.filters()
    }
    /// Explicit finite scalar metadata.
    /// # Errors
    /// Non-scalar/invalid metadata or I/O failure.
    pub fn scalar(&self, path: &str) -> Result<f64, Error> {
        self.reader.scalar(path)
    }
    /// Current resource/source-read counters.
    /// # Errors
    /// Poisoned state lock.
    pub fn stats(&self) -> Result<CacheStats, Error> {
        self.state
            .lock()
            .map(|s| s.stats)
            .map_err(|e| Error(e.to_string()))
    }
    /// Read exact original counts; each return owns its array independently.
    /// # Errors
    /// Changed source, invalid index, failed decode or poisoned state lock.
    pub fn read_frame(&self, index: usize) -> Result<Frame, Error> {
        let budget = self.stats()?.budget_bytes;
        if budget > 0 && stamp(&self.path).map_or(true, |current| current != self.stamp) {
            let mut state = self.state.lock().map_err(|e| Error(e.to_string()))?;
            state.frames.clear();
            state.stats.payload_bytes = 0;
            state.stats.frames = 0;
            state.stats.invalidations += 1;
            return Err(Error(
                "source changed while decoded-frame cache was open".into(),
            ));
        }
        {
            let mut state = self.state.lock().map_err(|e| Error(e.to_string()))?;
            if let Some(frame) = state.frames.get(&index).cloned() {
                let output = copy_frame(&frame)?;
                state.stats.requests += 1;
                state.stats.hits += 1;
                return Ok(output);
            }
        }
        let frame = self.reader.read_frame(index)?;
        let mut state = self.state.lock().map_err(|e| Error(e.to_string()))?;
        state.stats.requests += 1;
        state.stats.decodes += 1;
        let bytes = frame.pixels.capacity().checked_mul(size_of::<u32>());
        if !state.frames.contains_key(&index)
            && bytes.is_some_and(|n| n <= budget.saturating_sub(state.stats.payload_bytes))
        {
            if let Ok(cached) = copy_frame(&frame) {
                let actual = cached.pixels.capacity() * size_of::<u32>();
                if actual <= budget.saturating_sub(state.stats.payload_bytes) {
                    state.stats.payload_bytes += actual;
                    state.stats.frames += 1;
                    state.frames.insert(index, Arc::new(cached));
                }
            }
        }
        Ok(frame)
    }
}
