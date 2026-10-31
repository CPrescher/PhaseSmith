"""Native lossless, pickle-free numeric NPZ writing with exclusive creation."""

import numpy as np

from phasesmith import _core


def save_npz(path, arrays, *, compression_level=1):
    """Write exact C-order numeric bytes using the Rust NPZ writer.

    Level 0 is uncompressed; levels 1-9 select lossless DEFLATE effort. The
    native boundary accepts at most 1 GiB of raw array data and copies it before
    releasing the interpreter. Existing files are never overwritten. Array
    names must be flat ASCII identifiers; objects/strings/complex are rejected.
    """
    if type(compression_level) is not int or not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be an integer in 0..9")
    rows = []
    total = 0
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind not in "biuf":
            raise ValueError("only numeric real and boolean arrays can be archived")
        total += array.nbytes
        if total > 1024**3:
            raise ValueError("numeric archive raw payload exceeds 1 GiB")
        rows.append(
            (
                name,
                array.dtype.str,
                array.shape,
                np.ascontiguousarray(array).view(np.uint8).reshape(-1),
            )
        )
    _core.native_write_npz(str(path), rows, compression_level)
