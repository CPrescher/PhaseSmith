"""Independent unnormalized 3D DFT conventions, using NumPy's FFT backend."""

import numpy as np


def transform_3d_reference(values, *, inverse=False):
    """C-order XYZ transform; inverse uses the positive exponent without scaling."""
    values = np.asarray(values, dtype=np.complex128)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("finite three-dimensional complex input required")
    return np.fft.ifftn(values) * values.size if inverse else np.fft.fftn(values)
