"""Optional bundled native HDF5 detector-frame input; no h5py or plugin package.

Build the extension with ``--features hdf5``. Counts are returned as owned u32
arrays in original detector orientation, with no normalization or sentinel mask.
The native reader supports source uint16 and uint32, promoted exactly to u32.
"""

from phasesmith import _core


class Hdf5Frames:
    """Read a declared (frame, y, x) dataset with bounded native allocations.

    This read-only handle owns its file/dataset. Use a context manager or call
    ``close()``. Metadata paths are explicit scalar datasets; there is no search
    for calibration or scan angles. Required compression filters are reported in
    ``filters`` and checked before pixels are read.
    """

    def __init__(
        self,
        path,
        dataset,
        *,
        max_frames=1_000_000,
        max_frame_pixels=100_000_000,
        max_chunk_bytes=268_435_456,
        cache_bytes=0,
    ):
        if not hasattr(_core, "NativeHdf5Frames"):
            raise RuntimeError(
                "rebuild PhaseSmith with --features hdf5 for bundled native HDF5 input"
            )
        for value in (max_frames, max_frame_pixels, max_chunk_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("HDF5 read limits must be positive integers")
        if not isinstance(dataset, str) or not dataset or "\0" in dataset:
            raise ValueError("dataset must be a nonempty HDF5 path")
        if type(cache_bytes) is not int or not 0 <= cache_bytes <= 4 * 1024**3:
            raise ValueError("decoded-frame cache must be in 0..4 GiB")
        self._reader = _core.NativeHdf5Frames(
            path, dataset, max_frames, max_frame_pixels, max_chunk_bytes, cache_bytes
        )

    @property
    def shape(self):
        """Original frame count, detector height and width."""
        return tuple(self._reader.shape)

    @property
    def dtype(self):
        """Source count storage type, before exact u32 promotion."""
        return self._reader.dtype

    @property
    def filters(self):
        """Required filter IDs, decoder availability and source parameters."""
        return tuple((f[0], f[1], tuple(f[2])) for f in self._reader.filters)

    @property
    def library_version(self):
        """Version of the bundled native HDF5 library."""
        return tuple(_core.HDF5_VERSION)

    @property
    def cache_stats(self):
        """Decoded-payload admission and actual source-read counters."""
        return dict(self._reader.cache_stats)

    def read_frame(self, index):
        """Read one owned u32 frame, preserving all original pixel counts."""
        if type(index) is not int or not 0 <= index < self.shape[0]:
            raise ValueError("frame index outside source dataset")
        return self._reader.read_frame(index)

    def scalar(self, path):
        """Read one explicitly named finite scalar metadata dataset as f64."""
        if not isinstance(path, str) or not path or "\0" in path:
            raise ValueError("metadata path must be nonempty")
        return self._reader.scalar(path)

    def close(self):
        """Close the native file and dataset; repeated calls are harmless."""
        self._reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
