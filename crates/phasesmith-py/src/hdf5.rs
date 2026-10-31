//! Optional array-boundary adapter for bundled native HDF5 frame I/O.
use npy::{IntoPyArray, PyArray2, ndarray::Array2};
use phasesmith_hdf5::{Limits, PixelType, Reader};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::path::PathBuf;
use std::sync::Mutex;

fn error(value: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(value.to_string())
}

#[pyclass(name = "NativeHdf5Frames", module = "phasesmith._core")]
struct NativeHdf5Frames {
    reader: Mutex<Option<Reader>>,
}

impl NativeHdf5Frames {
    fn with<T>(&self, f: impl FnOnce(&Reader) -> PyResult<T>) -> PyResult<T> {
        let reader = self.reader.lock().map_err(error)?;
        f(reader
            .as_ref()
            .ok_or_else(|| error("HDF5 reader is closed"))?)
    }
}

#[pymethods]
impl NativeHdf5Frames {
    #[new]
    #[pyo3(signature=(path, dataset, max_frames=1_000_000, max_frame_pixels=100_000_000, max_chunk_bytes=268_435_456))]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        dataset: String,
        max_frames: usize,
        max_frame_pixels: usize,
        max_chunk_bytes: usize,
    ) -> PyResult<Self> {
        let reader = py
            .detach(move || {
                Reader::open(
                    path,
                    &dataset,
                    Limits {
                        frames: max_frames,
                        frame_pixels: max_frame_pixels,
                        chunk_bytes: max_chunk_bytes,
                    },
                )
            })
            .map_err(error)?;
        Ok(Self {
            reader: Mutex::new(Some(reader)),
        })
    }
    #[getter]
    fn shape(&self) -> PyResult<[usize; 3]> {
        self.with(|r| Ok(r.shape()))
    }
    #[getter]
    fn dtype(&self) -> PyResult<&'static str> {
        self.with(|r| {
            Ok(match r.pixel_type() {
                PixelType::U16 => "uint16",
                PixelType::U32 => "uint32",
            })
        })
    }
    #[getter]
    fn filters(&self) -> PyResult<Vec<(i32, bool, Vec<u32>)>> {
        self.with(|r| {
            Ok(r.filters()
                .iter()
                .map(|f| (f.id, f.decode_enabled, f.parameters.clone()))
                .collect())
        })
    }
    fn read_frame<'py>(
        &self,
        py: Python<'py>,
        index: usize,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let frame = py.detach(|| self.with(|r| r.read_frame(index).map_err(error)))?;
        Ok(Array2::from_shape_vec(frame.shape, frame.pixels)
            .expect("validated frame shape")
            .into_pyarray(py))
    }
    fn scalar(&self, py: Python<'_>, path: String) -> PyResult<f64> {
        py.detach(|| self.with(|r| r.scalar(&path).map_err(error)))
    }
    fn close(&self) -> PyResult<()> {
        self.reader.lock().map_err(error)?.take();
        Ok(())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeHdf5Frames>()?;
    module.add("HDF5_VERSION", phasesmith_hdf5::library_version())?;
    Ok(())
}
