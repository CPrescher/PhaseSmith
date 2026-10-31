"""Non-mutating diagnostics for finite-aperture reflection integration.

This reporting layer uses existing measurements, never evaluates a peak profile
or changes an intensity, sigma, mask or scale. See docs/single-crystal-integration-quality.md.
"""

from dataclasses import dataclass
from enum import IntFlag
from typing import ClassVar

import numpy as np

from .integration import ReflectionTable
from .prediction import _readonly

__all__ = [
    "IntegrationQuality",
    "QualityIssue",
    "QualityOptions",
    "SupportAssessment",
    "assess_integration",
    "assess_support",
    "issue_actions",
]


class QualityIssue(IntFlag):
    NO_MEASUREMENTS = 1
    MASKED = 2
    SATURATED = 4
    OVERLAP = 8
    BACKGROUND = 16
    POSITION = 32
    WINDOW_END = 64
    NEGATIVE = 128
    WINDOW_CHANGE = 256
    DETECTOR_EDGE = 512
    ANGULAR = 1024
    SIGNAL_BOUNDARY = 2048
    BACKGROUND_MODEL = 4096


ACTIONS = (
    (
        QualityIssue.SIGNAL_BOUNDARY,
        "Signal reaches the spatial boundary: inspect a larger aperture and spot tails.",
    ),
    (
        QualityIssue.BACKGROUND_MODEL,
        "Background plane needs review: inspect rejected pixels, gradients and contamination.",
    ),
    (
        QualityIssue.ANGULAR,
        "Requested angular support crosses a scan edge or gap; scaling cannot "
        "recover missing counts.",
    ),
    (
        QualityIssue.NO_MEASUREMENTS,
        "No predicted measurements: inspect the selected cell, geometry and exposure coverage.",
    ),
    (
        QualityIssue.MASKED,
        "Signal pixels were excluded: inspect masks and shadows; missing counts "
        "need explicit exclusion or a validated model.",
    ),
    (
        QualityIssue.SATURATED,
        "Saturated signal: inspect detector limits and exclude or remeasure affected reflections.",
    ),
    (
        QualityIssue.OVERLAP,
        "Overlapping prediction apertures: inspect competing grains; separate "
        "profiles or exclude affected observations.",
    ),
    (
        QualityIssue.BACKGROUND,
        "Insufficient background: inspect the local ring and neighboring peaks "
        "before changing background support.",
    ),
    (
        QualityIssue.POSITION,
        "Strong spots are offset from predictions: inspect cell/orientation, "
        "calibration and spot contamination.",
    ),
    (
        QualityIssue.WINDOW_END,
        "Significant signal reaches a measured angular endpoint: compare a wider "
        "window and inspect the rocking curve.",
    ),
    (
        QualityIssue.NEGATIVE,
        "Many significantly negative sums: inspect background subtraction and "
        "the uncertainty assumptions.",
    ),
    (
        QualityIssue.WINDOW_CHANGE,
        "Strong intensities change with the alternate window: inspect signal "
        "tails, background and neighboring peaks before accepting them.",
    ),
    (
        QualityIssue.DETECTOR_EDGE,
        "Signal apertures touch the detector edge: inspect truncation and "
        "exclude or model incomplete measurements.",
    ),
)


def issue_actions(flags):
    """Stable, advisory-only actions; no diagnostic is an automatic correction."""
    return tuple(message for bit, message in ACTIONS if int(flags) & int(bit))


@dataclass(frozen=True)
class QualityOptions:
    strong_sigma: float = 3.0
    endpoint_fraction: float = 0.10
    window_change_fraction: float = 0.10
    position_pixels: float = 2.0
    min_position_spots: int = 5
    negative_fraction: float = 0.10
    min_negative_measurements: int = 10

    def __post_init__(self):
        for name in (
            "strong_sigma",
            "endpoint_fraction",
            "window_change_fraction",
            "position_pixels",
            "negative_fraction",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, complex))
                or not np.isscalar(value)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
            if name.endswith("fraction") and value >= 1:
                raise ValueError(f"{name} must lie in (0,1)")
            object.__setattr__(self, name, float(value))
        for name in ("min_position_spots", "min_negative_measurements"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not 1 <= value <= 1_000_000
            ):
                raise ValueError(f"{name} must be an integer in [1,1000000]")
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True)
class IntegrationQuality:
    frame_values: np.ndarray
    frame_issues: np.ndarray
    reflection_values: np.ndarray
    reflection_issues: np.ndarray
    options: QualityOptions
    comparison_available: bool
    columns: ClassVar = (
        "omega",
        "n_measurements",
        "n_finite",
        "n_strong",
        "n_flagged",
        "masked_pixel_fraction",
        "median_background",
        "median_i_over_sigma",
        "n_negative_strong",
        "median_position_offset",
        "n_window_end",
        "n_window_changed",
        "n_masked",
        "n_saturated",
        "n_overlap",
        "n_background",
        "n_edge",
    )
    reflection_columns: ClassVar = ("endpoint_fraction", "relative_window_change", "n_observations")

    def __post_init__(self):
        for name in ("frame_values", "frame_issues", "reflection_values", "reflection_issues"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    def column(self, name):
        return self.frame_values[:, self.columns.index(name)]

    def actions(self, frame):
        if type(frame) is not int or not 0 <= frame < len(self.frame_values):
            raise IndexError("frame outside quality report")
        return issue_actions(self.frame_issues[frame])


def _validate(table):
    if not isinstance(table, ReflectionTable):
        raise TypeError("table must be ReflectionTable")
    n, m, nf = len(table), len(table.frame_indices), len(table.omega_bounds)
    shapes = dict(
        hkl=(n, 3),
        predicted=(n, 3),
        grain_ids=(n,),
        values=(n, 6),
        flags=(n,),
        observation_ids=(m,),
        frame_indices=(m,),
        boxes=(m, 4),
        frame_values=(m, 11),
        frame_flags=(m,),
        omega_bounds=(nf, 2),
    )
    if n > 1_000_000 or m > 1_000_000 or nf < 1:
        raise ValueError("quality work budget exceeded or no exposure bounds")
    if any(getattr(table, name).shape != shape for name, shape in shapes.items()):
        raise ValueError("inconsistent reflection table shapes")
    for name, high in (("observation_ids", n), ("frame_indices", nf)):
        a = getattr(table, name)
        if a.dtype.kind not in "iu" or (a < 0).any() or (a >= high).any():
            raise ValueError(f"invalid {name}")
    if (np.diff(table.frame_indices.astype(np.int64)) < 0).any():
        raise ValueError("measurement rows must be in source-frame order")
    if (
        table.boxes.dtype.kind not in "iu"
        or (table.boxes < 0).any()
        or (table.boxes[:, 2:] <= table.boxes[:, :2]).any()
    ):
        raise ValueError("invalid measurement boxes")
    if not np.isfinite(table.omega_bounds).all() or not np.isfinite(table.predicted).all():
        raise ValueError("nonfinite exposure bounds or predictions")
    if table.frame_flags.dtype.kind not in "iu" or table.flags.dtype.kind not in "iu":
        raise ValueError("quality flags must be integer arrays")
    if (table.boxes > np.iinfo(np.uint32).max).any():
        raise ValueError("measurement boxes exceed detector range")
    pixels = table.frame_values[:, 9:11]
    if not np.isfinite(pixels).all() or (pixels < 0).any():
        raise ValueError("invalid measurement pixel counts")
    area = (table.boxes[:, 2] - table.boxes[:, 0]).astype(float) * (
        table.boxes[:, 3] - table.boxes[:, 1]
    )
    if (pixels[:, 0] > area).any():
        raise ValueError("usable signal pixels exceed aperture area")
    return n, m, nf, area


def assess_integration(table, *, comparison=None, options=None):
    """Summarize each image and link problems to individual predicted crossings.

    An optional comparison must use identical predictions, raw input fingerprints
    and noise assumptions. It may use different finite measurement windows.
    No fitted scale, symmetry assumption, outlier rejection or correction is applied.
    """
    options = QualityOptions() if options is None else options
    if not isinstance(options, QualityOptions):
        raise TypeError("options must be QualityOptions")
    n, m, nf, area = _validate(table)
    ids = table.observation_ids.astype(np.int64, copy=False)
    frames = table.frame_indices.astype(np.int64, copy=False)
    v, raw = table.frame_values, table.frame_flags
    counts = np.bincount(ids, minlength=n)
    valid = np.isfinite(v[:, 2]) & np.isfinite(v[:, 3]) & (v[:, 3] > 0)
    strong = valid & (v[:, 2] > options.strong_sigma * v[:, 3])
    negative = valid & (v[:, 2] < -options.strong_sigma * v[:, 3])
    finite_counts = np.bincount(ids[valid], minlength=n)
    positive_sum = np.bincount(ids[valid], weights=np.maximum(v[valid, 2], 0), minlength=n)
    first, last = np.full(n, m, np.int64), np.full(n, -1, np.int64)
    np.minimum.at(first, ids, np.arange(m))
    np.maximum.at(last, ids, np.arange(m))
    endpoint = np.full(n, np.nan)
    can_check = (counts >= 3) & (finite_counts == counts) & (positive_sum > 0)
    cids = np.flatnonzero(can_check)
    endpoint[cids] = np.maximum(v[first[cids], 2], v[last[cids], 2]) / positive_sum[cids]
    end_significant = np.zeros(n, bool)
    end_significant[cids] = strong[first[cids]] | strong[last[cids]]
    reaches_end = can_check & end_significant & (endpoint >= options.endpoint_fraction)
    reflection_issues = np.zeros(n, np.uint32)
    reflection_issues[reaches_end] |= int(QualityIssue.WINDOW_END)
    reflection_issues[(table.flags & 256) != 0] |= int(QualityIssue.ANGULAR)
    reflection_issues[counts == 0] |= int(QualityIssue.NO_MEASUREMENTS)
    relative = np.full(n, np.nan)
    changed = np.zeros(n, bool)
    if comparison is not None:
        _validate(comparison)
        for name in ("hkl", "predicted", "grain_ids", "omega_bounds"):
            if not np.array_equal(getattr(table, name), getattr(comparison, name)):
                raise ValueError(f"comparison has different {name}")
        if (
            len(table.frame_hashes) != nf
            or tuple(table.frame_hashes) != tuple(comparison.frame_hashes)
            or table.noise_model != comparison.noise_model
            or table.noise_parameters is None
            or table.noise_parameters != comparison.noise_parameters
        ):
            raise ValueError("comparison requires identical input fingerprints and noise model")
        a, b = table.values, comparison.values
        eligible = (
            ((table.flags & np.uint32(~2048 & 0xFFFFFFFF)) == 0)
            & ((comparison.flags & np.uint32(~2048 & 0xFFFFFFFF)) == 0)
            & np.isfinite(a[:, :2]).all(axis=1)
            & np.isfinite(b[:, :2]).all(axis=1)
            & (a[:, 1] > 0)
            & (b[:, 1] > 0)
            & (a[:, 0] > options.strong_sigma * a[:, 1])
        )
        relative[eligible] = (b[eligible, 0] - a[eligible, 0]) / np.abs(a[eligible, 0])
        changed = eligible & (np.abs(relative) >= options.window_change_fraction)
        reflection_issues[changed] |= int(QualityIssue.WINDOW_CHANGE)
    frame_values = np.zeros((nf, len(IntegrationQuality.columns)))
    frame_values[:, 0] = table.omega_bounds.mean(axis=1)
    frame_issues = np.zeros(nf, np.uint32)
    offsets = np.r_[0, np.cumsum(np.bincount(frames, minlength=nf))]
    mapped_flags = (
        (1, QualityIssue.DETECTOR_EDGE),
        (2, QualityIssue.MASKED),
        (4, QualityIssue.SATURATED),
        (8, QualityIssue.OVERLAP),
        (16, QualityIssue.BACKGROUND),
    )
    for raw_flag, issue in mapped_flags:
        affected = (raw & raw_flag) != 0
        np.bitwise_or.at(reflection_issues, ids[affected], np.uint32(issue))
    for flag, issue in (
        (2048, QualityIssue.SIGNAL_BOUNDARY),
        (4096, QualityIssue.BACKGROUND_MODEL),
    ):
        affected = (raw & flag) != 0
        np.bitwise_or.at(reflection_issues, ids[affected], np.uint32(issue))
        np.bitwise_or.at(frame_issues, frames[affected], np.uint32(issue))
    # Iterate source frames; observation arithmetic stays in contiguous arrays.
    for frame in range(nf):
        s, e = offsets[frame : frame + 2]
        row = frame_values[frame]
        if s == e:
            frame_issues[frame] = int(QualityIssue.NO_MEASUREMENTS)
            row[[5, 6, 7, 9]] = np.nan
            continue
        fv, ff, rid = v[s:e], raw[s:e], ids[s:e]
        ok, ss, neg = valid[s:e], strong[s:e], negative[s:e]
        row[1:5] = [e - s, ok.sum(), ss.sum(), np.count_nonzero(ff)]
        row[5] = 1 - fv[:, 9].sum() / area[s:e].sum()
        bg_ok = np.isfinite(fv[:, 4]) & ((ff & 16) == 0)
        row[6] = np.median(fv[bg_ok, 4]) if bg_ok.any() else np.nan
        row[7] = np.median(fv[ok, 2] / fv[ok, 3]) if ok.any() else np.nan
        row[8] = neg.sum()
        pos_ok = ss & (ff == 0) & np.isfinite(fv[:, :2]).all(axis=1)
        offset = np.linalg.norm(fv[pos_ok, :2] - table.predicted[rid[pos_ok], :2], axis=1)
        row[9] = np.median(offset) if len(offset) else np.nan
        if len(offset) >= options.min_position_spots and row[9] >= options.position_pixels:
            frame_issues[frame] |= int(QualityIssue.POSITION)
            reflection_issues[rid[pos_ok][offset >= options.position_pixels]] |= int(
                QualityIssue.POSITION
            )
        if np.any((table.flags[rid] & 256) != 0):
            frame_issues[frame] |= int(QualityIssue.ANGULAR)
        row[10:12] = [np.count_nonzero(reaches_end[rid]), np.count_nonzero(changed[rid])]
        if row[10]:
            frame_issues[frame] |= int(QualityIssue.WINDOW_END)
        if row[11]:
            frame_issues[frame] |= int(QualityIssue.WINDOW_CHANGE)
        if (
            ok.sum() >= options.min_negative_measurements
            and neg.sum() / ok.sum() >= options.negative_fraction
        ):
            frame_issues[frame] |= int(QualityIssue.NEGATIVE)
            reflection_issues[rid[neg]] |= int(QualityIssue.NEGATIVE)
        for column, (raw_flag, issue) in zip((16, 12, 13, 14, 15), mapped_flags, strict=True):
            row[column] = np.count_nonzero(ff & raw_flag)
            if row[column]:
                frame_issues[frame] |= int(issue)
    reflection_values = np.column_stack([endpoint, relative, counts])
    return IntegrationQuality(
        frame_values,
        frame_issues,
        reflection_values,
        reflection_issues,
        options,
        comparison is not None,
    )


@dataclass(frozen=True)
class SupportAssessment:
    """Descriptive support checks; passes is not absolute intensity validation.

    Reasons: 1 baseline measurement flag, 2 weak/nonpositive baseline,
    4 baseline angular endpoint, 8 alternate measurement flag,
    16 alternate angular endpoint, 32 excessive/nonfinite relative change.
    """

    relative_changes: np.ndarray
    reasons: np.ndarray
    passes: np.ndarray

    def __post_init__(self):
        for name in ("relative_changes", "reasons", "passes"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


def assess_support(table, comparisons, *, max_relative_change=0.05, strong_sigma=5.0):
    """Check fixed predictions against >=2 explicitly chosen alternate measurements.

    Input/noise fingerprints must match. Correlated window differences are
    sensitivity checks, not independent significance tests. Choose spatial,
    angular and background controls before examining the results.
    Original measurements are never filtered, corrected or rescaled.
    """
    for value, name in (
        (max_relative_change, "max_relative_change"),
        (strong_sigma, "strong_sigma"),
    ):
        if (
            isinstance(value, (bool, complex))
            or not np.isscalar(value)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    comparisons = tuple(comparisons)
    if not 2 <= len(comparisons) <= 32:
        raise ValueError("supply between two and 32 alternate measurements")
    original = assess_integration(table)
    reasons = np.zeros((len(table), len(comparisons)), np.uint32)
    reasons[table.flags != 0] |= 1
    strong = (
        np.isfinite(table.values[:, :2]).all(axis=1)
        & (table.values[:, 1] > 0)
        & (table.values[:, 0] > strong_sigma * table.values[:, 1])
    )
    reasons[~strong] |= 2
    reasons[(original.reflection_issues & int(QualityIssue.WINDOW_END)) != 0] |= 4
    changes = np.full(reasons.shape, np.nan)
    for column, other in enumerate(comparisons):
        assess_integration(table, comparison=other)  # checks provenance and identity
        alternate = assess_integration(other)
        reasons[other.flags != 0, column] |= 8
        reasons[(alternate.reflection_issues & int(QualityIssue.WINDOW_END)) != 0, column] |= 16
        np.divide(
            other.values[:, 0] - table.values[:, 0],
            abs(table.values[:, 0]),
            out=changes[:, column],
            where=strong,
        )
        bad = ~np.isfinite(changes[:, column]) | (abs(changes[:, column]) > max_relative_change)
        reasons[bad, column] |= 32
    return SupportAssessment(changes, reasons, np.all(reasons == 0, axis=1))
