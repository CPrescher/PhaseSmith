//! Owned array snapshots for bounded native original-count frame preparation.
use npy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArray3, ndarray::Array2};
use phasesmith_single_crystal::spots::{
    frame_input as kernel, frame_validity::DacOptions, reciprocal::Geometry,
};
use pyo3::{exceptions::PyValueError, prelude::*};

type BlockInput<'a> = (
    usize,
    PyReadonlyArray3<'a, bool>,
    Option<PyReadonlyArray2<'a, bool>>,
);
type DacInput<'a> = (
    PyReadonlyArray2<'a, f64>,
    PyReadonlyArray2<'a, f64>,
    usize,
    [f64; 3],
    f64,
    Option<PyReadonlyArray2<'a, bool>>,
);
type Prepared<'py> = (
    usize,
    Option<Bound<'py, PyArray2<u32>>>,
    Option<Bound<'py, PyArray2<bool>>>,
    Option<String>,
    Option<String>,
);
fn error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}
fn fixed(a: Option<PyReadonlyArray2<'_, bool>>, shape: [usize; 2]) -> PyResult<Vec<bool>> {
    a.map(|a| {
        if a.as_array().dim() != (shape[0], shape[1]) {
            return Err(error("fixed mask shape differs"));
        }
        Ok(a.as_slice()?.to_vec())
    })
    .transpose()
    .map(Option::unwrap_or_default)
}

#[pyclass(name = "NativeFramePreparation", module = "phasesmith._core")]
struct NativeFramePreparation {
    plan: kernel::Plan,
}
#[pymethods]
impl NativeFramePreparation {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        shape: [usize; 2],
        frames: usize,
        vertical_flip: bool,
        reserved_minimum: u32,
        hashes: Vec<Vec<String>>,
        blocks: Vec<BlockInput<'_>>,
        dacs: Vec<DacInput<'_>>,
    ) -> PyResult<Self> {
        let blocks = blocks
            .into_iter()
            .map(|(block, a, f)| {
                if block == 0
                    || a.as_array().dim()
                        != (frames, shape[0].div_ceil(block), shape[1].div_ceil(block))
                {
                    return Err(error("block mask shape differs"));
                }
                Ok(kernel::BlockMasks {
                    block,
                    excluded: a.as_slice()?.to_vec(),
                    fixed: fixed(f, shape)?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let dacs = dacs
            .into_iter()
            .map(|(vectors, bounds, block, axis, opening, f)| {
                if vectors.as_array().dim() != (5, 3) || bounds.as_array().dim() != (frames, 2) {
                    return Err(error("DAC geometry/bounds shape differs"));
                }
                let v: Vec<[f64; 3]> = vectors
                    .as_slice()?
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|r| [r[0], r[1], r[2]])
                    .collect();
                Ok(kernel::DacMasks {
                    geometry: Geometry {
                        origin: v[0],
                        fast: v[1],
                        slow: v[2],
                        beam: v[3],
                        axis: v[4],
                        wavelength: 1.,
                    },
                    bounds: bounds
                        .as_slice()?
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|r| [r[0], r[1]])
                        .collect(),
                    block,
                    options: DacOptions {
                        axis_at_zero: axis,
                        half_opening_degrees: opening,
                    },
                    fixed: fixed(f, shape)?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let plan = kernel::Plan {
            shape,
            frames,
            vertical_flip,
            reserved_minimum,
            expected_hashes: hashes,
            blocks,
            dacs,
        };
        plan.validate().map_err(error)?;
        Ok(Self { plan })
    }
    #[getter]
    fn source_hash_providers(&self) -> usize {
        self.plan.expected_hashes.len()
    }
    fn admit(&self, count: usize, workers: usize, budget: usize) -> PyResult<()> {
        self.plan.admit(count, workers, budget).map_err(error)
    }
    fn prepare<'py>(
        &self,
        py: Python<'py>,
        frames: Vec<(usize, PyReadonlyArray2<'py, u32>)>,
        workers: usize,
        budget: usize,
    ) -> PyResult<Vec<Prepared<'py>>> {
        self.plan
            .admit(frames.len(), workers, budget)
            .map_err(error)?;
        let shape = self.plan.shape;
        let inputs = frames
            .iter()
            .map(|(index, a)| {
                if a.as_array().dim() != (shape[0], shape[1]) {
                    return Err(error("source frame shape differs"));
                }
                Ok(kernel::Input {
                    index: *index,
                    pixels: a.as_slice()?.to_vec(),
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let prepared = py
            .detach(|| self.plan.prepare_batch(inputs, workers, budget))
            .map_err(error)?;
        frames
            .iter()
            .zip(prepared)
            .map(|((index, _), result)| match result {
                Ok(row) => Ok((
                    row.index,
                    Some(
                        Array2::from_shape_vec((shape[0], shape[1]), row.pixels)
                            .map_err(error)?
                            .into_pyarray(py),
                    ),
                    Some(
                        Array2::from_shape_vec((shape[0], shape[1]), row.excluded)
                            .map_err(error)?
                            .into_pyarray(py),
                    ),
                    Some(row.digest),
                    None,
                )),
                Err(e) => Ok((*index, None, None, None, Some(e.to_string()))),
            })
            .collect()
    }
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeFramePreparation>()
}
