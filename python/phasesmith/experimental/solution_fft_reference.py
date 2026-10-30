"""Independent unnormalized 3D DFT conventions, using NumPy's FFT backend."""

import numpy as np


def transform_3d_reference(values, *, inverse=False):
    """C-order XYZ transform; inverse uses the positive exponent without scaling."""
    values = np.asarray(values, dtype=np.complex128)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("finite three-dimensional complex input required")
    return np.fft.ifftn(values) * values.size if inverse else np.fft.fftn(values)


def real_density_transform_reference(values, *, to_density=False):
    """Positive real DFT, or real part of a negative arbitrary-complex DFT."""
    values = np.asarray(values, dtype=np.complex128)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("finite three-dimensional complex input required")
    if not to_density and np.any(values.imag != 0):
        raise ValueError("density input must be real")
    return np.fft.fftn(values).real if to_density else np.fft.ifftn(values.real) * values.size
