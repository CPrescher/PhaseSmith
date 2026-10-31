# SPDX-License-Identifier: MIT
"""Independent NumPy/SciPy mask reference, never called by production kernels.

This retains the readable Dioptas-compatible equations for numerical comparisons.
SciPy is an optional reference-test dependency, not a runtime masking backend.
See THIRD_PARTY_NOTICES.md for the upstream MIT attribution.
"""

import numpy as np
from scipy import ndimage


class SeatShadowUnavailable(ValueError):
    """Independent reference cannot infer a shadow mask."""


def coarse_reference(image, mask, block):
    """Independent finite/masked block median, including partial edge blocks."""
    import warnings

    valid = np.isfinite(image) & ~mask
    a = np.where(valid, image, np.nan)
    a = np.pad(a, ((0, (-a.shape[0]) % block), (0, (-a.shape[1]) % block)), constant_values=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmedian(
            a.reshape(a.shape[0] // block, block, a.shape[1] // block, block), axis=(1, 3)
        )
    return np.nan_to_num(result), bool(valid.any())


def _dot_range(rays, axis, normal, low, high):
    """Exact extrema of ray dot R(axis,omega)normal on a closed interval."""
    parallel = (axis @ normal) * axis
    c = rays @ parallel
    a, b = rays @ (normal - parallel), rays @ np.cross(axis, normal)
    ends = (c + a * np.cos(low) + b * np.sin(low), c + a * np.cos(high) + b * np.sin(high))
    minimum, maximum = np.minimum(*ends), np.maximum(*ends)
    phase, radius = np.arctan2(b, a), np.hypot(a, b)
    period = 2 * np.pi
    for point, value, maximum_point in (
        (phase, c + radius, True),
        (phase + np.pi, c - radius, False),
    ):
        occurs = np.ceil((low - point) / period) <= np.floor((high - point) / period)
        if maximum_point:
            maximum = np.where(occurs, np.maximum(maximum, value), maximum)
        else:
            minimum = np.where(occurs, np.minimum(minimum, value), minimum)
    return minimum, maximum


def estimate_reference(coarse, available, shape, options):
    """Float32 spatial filters, reference ratios and side-component masks."""
    count = len(available)
    first = (
        round(0.37 * (count - 1)) if options.reference_first is None else options.reference_first
    )
    last = round(0.59 * (count - 1)) if options.reference_last is None else options.reference_last
    opts = options
    smooth = ndimage.gaussian_filter(
        ndimage.median_filter(np.asarray(coarse, dtype=np.float32), size=(1, 3, 3)), sigma=(0, 1, 1)
    )
    references = np.linspace(first, last, min(5, last - first + 1)).astype(int)
    reference = np.median(smooth[references], axis=0)
    if not np.any(reference > 40):
        raise SeatShadowUnavailable("illuminated reference background must exceed 40 counts")
    ratio = smooth / np.maximum(reference, 1)
    scales = np.quantile(ratio[:, reference > 40], 0.9, axis=1)
    if np.any((scales <= 0) & available):
        raise SeatShadowUnavailable(
            "wholly dark unmasked frames have unknown visibility; supply reviewed masks or geometry"
        )
    ratio /= np.maximum(scales[:, None, None], 1e-6)
    support = reference > 25
    _, xx = np.indices(reference.shape)
    sides = (xx * opts.block < shape[1] * 350 / 2048) | (xx * opts.block > shape[1] * 1700 / 2048)
    excluded = []
    for index, frame in enumerate(ratio):
        if not available[index]:
            excluded.append(np.ones(reference.shape, bool))
            continue
        candidate = ndimage.binary_fill_holes(
            ndimage.binary_closing((frame < opts.threshold) & support)
        )
        labels, _ = ndimage.label(candidate)
        sizes = np.bincount(labels.ravel())
        connected = np.unique(labels[sides])
        connected = connected[(connected != 0) & (sizes[connected] >= 20)]
        keep = np.isin(labels, connected)
        margin = int(np.ceil(opts.guard_pixels / opts.block))
        excluded.append(ndimage.binary_dilation(keep, iterations=margin) if margin else keep)
    return np.asarray(excluded), scales, references
