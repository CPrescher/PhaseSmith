"""Local empirical 3D profiles, with native learning, box integration and GLS.

See docs/experimental-3d-profiles.md. No image reader, GUI or DIALS dependency.
"""

import hashlib
from dataclasses import dataclass, field
from enum import IntFlag
from typing import ClassVar

import numpy as np

from phasesmith import _core

from .integration import IntegrationOptions, ReflectionTable, integrate_series
from .prediction import _readonly


class ProfileFlag(IntFlag):
    PARTIAL = 8192
    FAILED = 16384
    NO_PROFILE = 32768
    MISMATCH = 131072
    SHAPE_EXTRAPOLATED = 524288


@dataclass(frozen=True)
class ProfileExecution:
    """Native resource controls; zero cache bytes selects the streaming evaluator."""

    workers: int = 1
    geometry_cache_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        if type(self.workers) is not int or not 1 <= self.workers <= 32:
            raise ValueError("profile workers must be in 1..32")
        if (
            type(self.geometry_cache_bytes) is not int
            or not 0 <= self.geometry_cache_bytes <= 1024**3
        ):
            raise ValueError("profile geometry cache must be in 0..1 GiB")


def _execution(value):
    value = ProfileExecution() if value is None else value
    if not isinstance(value, ProfileExecution):
        raise TypeError("execution must be ProfileExecution")
    return value


@dataclass(frozen=True)
class ProfileOptions:
    shape: tuple = (19, 19, 9)  # x, y, omega; array storage is bank,omega,y,x.
    step: tuple = (1.0, 1.0, 1.0)
    regions: tuple = (2, 2, 2)
    min_references: int = 8
    reference_snr: float = 20.0
    cutoff: float = 0.005
    iterations: int = 16
    minimum_mass: float = 0.90
    reference_weighting: str = "equal"

    def __post_init__(self):
        if self.reference_weighting not in ("equal", "inverse_variance"):
            raise ValueError("reference_weighting must be equal or inverse_variance")
        for name, lo, hi in (("shape", 3, 65), ("regions", 1, 4)):
            v = tuple(getattr(self, name))
            if len(v) != 3 or any(
                isinstance(x, (bool, np.bool_))
                or not isinstance(x, (int, np.integer))
                or not lo <= x <= hi
                for x in v
            ):
                raise ValueError(f"invalid {name}")
            if name == "shape" and any(x % 2 == 0 for x in v):
                raise ValueError("profile grid dimensions must be odd")
            object.__setattr__(self, name, tuple(map(int, v)))
        step = np.asarray(self.step)
        if (
            step.shape != (3,)
            or step.dtype.kind not in "fiu"
            or not np.isfinite(step).all()
            or (step <= 0).any()
        ):
            raise ValueError("step must contain three finite positive values")
        object.__setattr__(self, "step", tuple(map(float, step)))
        for name, lo, hi in (("min_references", 2, 1_000_000), ("iterations", 1, 100)):
            v = getattr(self, name)
            if (
                isinstance(v, (bool, np.bool_))
                or not isinstance(v, (int, np.integer))
                or not lo <= v <= hi
            ):
                raise ValueError(f"invalid {name}")
        for name, lo, hi in (
            ("reference_snr", 3.0, 1e6),
            ("cutoff", 0.0, 0.2),
            ("minimum_mass", 0.0, 1.0),
        ):
            v = getattr(self, name)
            if (
                isinstance(v, (bool, complex))
                or not np.isscalar(v)
                or not np.isfinite(v)
                or not lo <= v <= hi
            ):
                raise ValueError(f"invalid {name}")
        if self.cutoff >= 0.2:
            raise ValueError("cutoff must be below 0.2")


@dataclass(frozen=True)
class ProfileObservations:
    summation: ReflectionTable
    samples: np.ndarray  # background-subtracted signal, independent variance, background
    valid: np.ndarray
    offsets: np.ndarray  # one sample-offset per reflection/exposure group plus sentinel
    covariance: np.ndarray  # per-exposure fitted-background covariance flattened (g,9)
    centres: np.ndarray
    scales: np.ndarray | None = None
    planar_transforms: np.ndarray | None = None
    _digest: str = field(init=False, repr=False)

    def __post_init__(self):
        t = self.summation
        ng = len(t.observation_ids)
        s = np.asarray(self.samples)
        if s.dtype.kind not in "fiu" or s.ndim != 2 or s.shape[1] != 3 or len(s) > 20_000_000:
            raise ValueError("samples must be real (m,3), within the 20-million sample budget")
        if np.shape(self.valid) != (len(s),) or np.asarray(self.valid).dtype != bool:
            raise ValueError("valid must be a boolean vector of sample length")
        offsets = np.asarray(self.offsets)
        if (
            offsets.shape != (ng + 1,)
            or offsets.dtype.kind not in "iu"
            or offsets[0] != 0
            or offsets[-1] != len(s)
            or (np.diff(offsets.astype(np.int64)) < 0).any()
        ):
            raise ValueError("invalid profile group offsets")
        areas = np.prod(t.boxes[:, 2:] - t.boxes[:, :2], axis=1)
        if not np.array_equal(np.diff(offsets), areas):
            raise ValueError("sample groups differ from signal box sizes")
        if (
            np.shape(self.covariance) != (ng, 9)
            or np.asarray(self.covariance).dtype.kind not in "fiu"
        ):
            raise ValueError("covariance must be a real (groups,9) array")
        cov = np.asarray(self.covariance).reshape(-1, 3, 3)
        finite = np.isfinite(cov).all(axis=(1, 2))
        if np.any(~finite & ((t.frame_flags & 16) == 0)):
            raise ValueError("nonfinite covariance needs an explicit failed background flag")
        c = cov[finite]
        if len(c):
            scale = np.maximum(np.max(abs(c), axis=(1, 2)), 1.0)
            if np.any(np.max(abs(c - c.transpose(0, 2, 1)), axis=(1, 2)) > 1e-10 * scale):
                raise ValueError("background covariance must be symmetric")
            nonzero = np.any(c != 0, axis=(1, 2))
            if np.any(np.linalg.eigvalsh(c[nonzero])[:, 0] <= 0):
                raise ValueError("fitted-plane covariance must be positive definite or all zero")
        centres = np.asarray(self.centres)
        if (
            centres.shape != (len(t), 3)
            or centres.dtype.kind not in "fiu"
            or not np.isfinite(centres).all()
        ):
            raise ValueError("centres must be finite real (reflections,3)")
        scales = (
            np.ones_like(centres, dtype=float) if self.scales is None else np.asarray(self.scales)
        )
        if (
            scales.shape != centres.shape
            or scales.dtype.kind not in "fiu"
            or not np.isfinite(scales).all()
            or (abs(scales) < 1e-8).any()
        ):
            raise ValueError("scales must be finite nonzero real (reflections,3)")
        object.__setattr__(self, "scales", scales)
        planar = (
            np.broadcast_to(np.eye(2), (len(t), 2, 2))
            if self.planar_transforms is None
            else self.planar_transforms
        )
        object.__setattr__(self, "planar_transforms", _planar(planar, len(t)))
        ok = np.asarray(self.valid)
        if not np.isfinite(s[ok]).all() or (s[ok, 1] <= 0).any():
            raise ValueError("valid samples require finite values and positive variances")
        for name, dtype in (
            ("samples", float),
            ("valid", bool),
            ("offsets", np.int64),
            ("covariance", float),
            ("centres", float),
            ("scales", float),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))
        h = hashlib.sha256()
        h.update(str((t.frame_hashes, t.noise_parameters, t.options)).encode())
        for a in (
            self.samples,
            self.valid,
            self.offsets,
            self.covariance,
            self.centres,
            self.scales,
            self.planar_transforms,
            t.hkl,
            t.omega_bounds,
            t.boxes,
            t.observation_ids,
            t.frame_indices,
            t.frame_flags,
            t.flags,
        ):
            h.update(memoryview(a).cast("B"))
        object.__setattr__(self, "_digest", h.hexdigest())

    def _native(self):
        t = self.summation
        angles = np.sort(t.omega_bounds[t.frame_indices], axis=1)
        return (
            self.samples,
            self.valid,
            self.offsets,
            t.boxes,
            t.observation_ids,
            angles,
            self.covariance,
            self.centres,
            np.ascontiguousarray(t.frame_flags | (t.flags[t.observation_ids] & 256)),
            self.scales,
            self.planar_transforms.reshape(-1, 4),
        )

    def fingerprint(self):
        return self._digest


def prepare_profiles(
    frame_loader,
    omega_bounds,
    predictions,
    *,
    mask_loader=None,
    variance_loader=None,
    grain_ids=None,
    options=None,
    gain=1.0,
    read_noise=0.0,
    saturation=None,
    centres=None,
    scales=None,
    planar_transforms=None,
    angular_half_widths=None,
    progress=None,
    max_samples=20_000_000,
):
    """Stream raw frames once, preserving summed counts and fitted-plane covariance.

    centres may supply independently measured (x,y,omega) positions. Otherwise
    frozen predictions are used. The profile learner never sees external intensities.
    """
    options = options or IntegrationOptions(
        pixel_radius=11,
        omega_half_width=4.5,
        background_model="plane",
        background_gap=3,
        background_margin=6,
    )
    if options.background_model != "plane":
        raise ValueError("profile preparation requires the guarded plane background")
    if (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, int)
        or not 1 <= max_samples <= 20_000_000
    ):
        raise ValueError("invalid profile sample budget")
    samples = []
    valid = []
    cov = []
    count = 0

    def sink(ids, frame, boxes, covariance, data, usable):
        nonlocal count
        count += len(data)
        if count > max_samples:
            raise ValueError("profile sample budget exceeded")
        samples.append(data)
        valid.append(usable)
        cov.append(covariance)

    summed = integrate_series(
        frame_loader,
        omega_bounds,
        predictions,
        mask_loader=mask_loader,
        variance_loader=variance_loader,
        grain_ids=grain_ids,
        options=options,
        gain=gain,
        read_noise=read_noise,
        saturation=saturation,
        progress=progress,
        _profile_sink=sink,
        _profile_half_widths=angular_half_widths,
    )
    areas = np.prod(summed.boxes[:, 2:] - summed.boxes[:, :2], axis=1)
    return ProfileObservations(
        summed,
        np.concatenate(samples) if samples else np.empty((0, 3)),
        np.concatenate(valid) if valid else np.empty(0, bool),
        np.r_[0, np.cumsum(areas)],
        np.concatenate(cov) if cov else np.empty((0, 9)),
        summed.predicted if centres is None else centres,
        scales,
        planar_transforms,
    )


@dataclass(frozen=True)
class ProfileModel:
    profiles: np.ndarray  # bank, flattened z/y/x
    counts: np.ndarray
    shifts: np.ndarray
    reference_ids: np.ndarray
    training_fingerprint: str
    detector_shape: tuple
    angular_range: tuple
    options: ProfileOptions
    reference_centres: np.ndarray | None = None
    execution: dict | None = None

    def __post_init__(self):
        n = 1 + int(np.prod(self.options.regions))
        p = np.asarray(self.profiles)
        if (
            p.shape != (n, int(np.prod(self.options.shape)))
            or p.dtype.kind not in "fiu"
            or not np.isfinite(p).all()
            or (p < 0).any()
        ):
            raise ValueError("invalid nonnegative profile bank")
        masses = p.sum(axis=1)
        if abs(masses[0] - 1) > 1e-10 or np.any((masses > 1e-12) & (abs(masses - 1) > 1e-10)):
            raise ValueError("nonempty profiles must have unit mass")
        if (
            np.shape(self.counts) != (n,)
            or np.asarray(self.counts).dtype.kind not in "iu"
            or (np.asarray(self.counts) < 0).any()
        ):
            raise ValueError("invalid reference counts")
        if np.shape(self.shifts) != (n, 3) or not np.isfinite(self.shifts).all():
            raise ValueError("invalid model centre shifts")
        refs = np.asarray(self.reference_ids)
        if (
            refs.ndim != 1
            or refs.dtype.kind not in "iu"
            or (refs < 0).any()
            or (np.diff(refs.astype(np.int64)) <= 0).any()
        ):
            raise ValueError("reference IDs must be sorted distinct nonnegative integers")
        _regions(np.empty((0, 3)), self.detector_shape, self.angular_range, self.options.regions)
        if self.reference_centres is not None:
            c = np.asarray(self.reference_centres)
            if c.shape != (len(refs), 3) or c.dtype.kind not in "fiu" or not np.isfinite(c).all():
                raise ValueError("invalid measured reference centres")
            object.__setattr__(self, "reference_centres", _readonly(c, float))
        for name, dtype in (
            ("profiles", float),
            ("counts", np.int64),
            ("shifts", float),
            ("reference_ids", np.int64),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))

    @property
    def volumes(self):
        nx, ny, nz = self.options.shape
        return self.profiles.reshape(-1, nz, ny, nx)


def _regions(centres, detector_shape, angular_range, regions):
    if len(detector_shape) != 2 or any(type(v) is not int or v <= 0 for v in detector_shape):
        raise ValueError("detector_shape must contain positive integer height,width")
    lo, hi = angular_range
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("invalid angular model range")
    limits = np.array([detector_shape[1], detector_shape[0], hi - lo])
    unit = (centres - [0, 0, lo]) / limits
    index = np.minimum(np.maximum(np.floor(unit * regions), 0), np.array(regions) - 1).astype(
        np.int64
    )
    return np.ascontiguousarray((index[:, 2] * regions[1] + index[:, 1]) * regions[0] + index[:, 0])


def learn_profiles(observations, reference_mask, *, detector_shape, options=None, execution=None):
    """Learn only from the supplied training population, then native quality gates.

    Detector/angle cells with too few references explicitly fall back to bank 0.
    Learning reference IDs are returned for auditing and out-of-sample checks.
    """
    options = options or ProfileOptions()
    if not isinstance(options, ProfileOptions) or not isinstance(observations, ProfileObservations):
        raise TypeError("expected ProfileOptions and ProfileObservations")
    mask = np.asarray(reference_mask)
    if mask.dtype != bool or mask.shape != (len(observations.centres),):
        raise ValueError("reference_mask must be a boolean vector per reflection")
    mask = mask & (observations.summation.flags == 0)
    angular_range = (
        float(observations.summation.omega_bounds.min()),
        float(observations.summation.omega_bounds.max()),
    )
    regions = _regions(observations.centres, detector_shape, angular_range, options.regions)
    execution = _execution(execution)
    p, counts, shifts, used, reference_centres, stats = _core.experimental_profile_learn(
        observations._native(),
        regions,
        np.ascontiguousarray(mask),
        options.shape,
        options.step,
        1 + int(np.prod(options.regions)),
        options.min_references,
        options.reference_snr,
        options.cutoff,
        options.iterations,
        options.reference_weighting == "inverse_variance",
        execution.workers,
        execution.geometry_cache_bytes,
    )
    return ProfileModel(
        p,
        counts,
        shifts,
        np.flatnonzero(used),
        observations.fingerprint(),
        tuple(detector_shape),
        angular_range,
        options,
        reference_centres[used],
        stats,
    )


@dataclass(frozen=True)
class ProfileIntegration:
    summation: ReflectionTable
    values: np.ndarray
    flags: np.ndarray
    model_ids: np.ndarray
    reference_membership: np.ndarray
    coefficients: np.ndarray | None = None
    coefficient_covariance: np.ndarray | None = None
    variation_factors: np.ndarray | None = None
    execution: dict | None = None
    columns: ClassVar = (
        "intensity",
        "sigma",
        "observed_mass",
        "reduced_chi_square",
        "n_pixels",
        "correlation",
        "dchi_dI",
        "dchi_dx",
        "dchi_dy",
        "dchi_domega",
    )

    def __post_init__(self):
        for name in ("values", "flags", "model_ids", "reference_membership"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

        for name in ("coefficients", "coefficient_covariance", "variation_factors"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _readonly(getattr(self, name)))

    def column(self, name):
        return self.values[:, self.columns.index(name)]


def fit_profiles(
    observations,
    model,
    *,
    expected_noise=True,
    variation_factors=None,
    fixed_fractions=None,
    allow_shape_refinement=False,
    execution=None,
):
    """Signed GLS amplitudes with explicit mass, residual and source quality flags.

    Frozen predicted/independent centres plus learned local median offsets are
    used; weak observations do not undergo an intensity-driven position search.
    """
    if not isinstance(observations, ProfileObservations) or not isinstance(model, ProfileModel):
        raise TypeError("expected ProfileObservations and ProfileModel")
    if type(expected_noise) is not bool:
        raise ValueError("expected_noise must be boolean")
    execution = _execution(execution)
    o = model.options
    regions = (
        _regions(observations.centres, model.detector_shape, model.angular_range, o.regions) + 1
    )
    fallback = (model.counts[regions] < o.min_references) | (
        model.profiles[regions].sum(axis=1) == 0
    )
    regions[fallback] = 0
    noise = observations.summation.noise_parameters
    if noise is None:
        raise ValueError("profile fitting requires explicit noise parameters")
    if observations.summation.noise_model == "supplied_independent_pixel_variance":
        expected_noise = False
    extra = {}
    if variation_factors is None:
        values, flags, stats = _core.experimental_profile_fit(
            observations._native(),
            model.profiles,
            o.shape,
            o.step,
            regions,
            model.shifts,
            o.minimum_mass,
            expected_noise,
            float(noise[0]),
            float(noise[1]),
            execution.workers,
            execution.geometry_cache_bytes,
        )
        extra["execution"] = stats
    else:
        factors = np.asarray(variation_factors)
        if (
            factors.ndim != 2
            or factors.shape[1] != 3
            or not 2 <= len(factors) <= 4
            or factors.dtype.kind not in "fiu"
            or not np.isfinite(factors).all()
            or (factors < 0.5).any()
            or (factors > 2.5).any()
            or not np.array_equal(factors[0], np.ones(3))
        ):
            raise ValueError(
                "variation_factors require 2..4 finite triples in [.5,2.5], starting with ones"
            )
        if model.options.cutoff != 0:
            raise ValueError("profile variations require retained tails: learn with cutoff=0")
        if fixed_fractions is None and not allow_shape_refinement:
            raise ValueError(
                "free shape refinement is a training diagnostic; use "
                "learn_profile_variations and fit_learned_profiles"
            )
        fixed = np.empty((0, 4))
        if fixed_fractions is not None:
            f = np.asarray(fixed_fractions)
            if (
                f.shape != (len(observations.centres), len(factors))
                or f.dtype.kind not in "fiu"
                or not np.isfinite(f).all()
                or (f < 0).any()
                or not np.allclose(f.sum(axis=1), 1.0, atol=1e-10, rtol=0)
            ):
                raise ValueError("fixed_fractions must be nonnegative unit-sum rows")
            fixed = np.zeros((len(f), 4))
            fixed[:, : len(factors)] = f
        values, flags, coeff, cov = _core.experimental_profile_fit_variations(
            observations._native(),
            model.profiles,
            o.shape,
            o.step,
            regions,
            model.shifts,
            np.ascontiguousarray(factors, float),
            fixed,
            o.minimum_mass,
            expected_noise,
            float(noise[0]),
            float(noise[1]),
        )
        k = len(factors)
        extra = dict(
            coefficients=coeff[:, :k],
            coefficient_covariance=cov.reshape(-1, 4, 4)[:, :k, :k],
            variation_factors=factors,
        )
    # Profile flags refer to the learned foreground; rectangular sum flags
    # remain untouched in result.summation for inspection and comparison.
    flags |= observations.summation.flags & 1024
    membership = np.zeros(len(regions), bool)
    if observations.fingerprint() == model.training_fingerprint:
        if (model.reference_ids >= len(membership)).any():
            raise ValueError("model reference IDs exceed the training dataset")
        membership[model.reference_ids] = True
    return ProfileIntegration(observations.summation, values, flags, regions, membership, **extra)


def evaluate_profile(
    profile,
    bounds,
    *,
    step=(1.0, 1.0, 1.0),
    centre=(0.0, 0.0, 0.0),
    scales=(1.0, 1.0, 1.0),
    planar_transform=None,
):
    """Exact voxel/pixel-exposure mass and three position partials for a z/y/x grid."""
    p, b, c, s, scale = _evaluation_inputs(profile, bounds, centre, step, scales)
    if planar_transform is not None:
        a = _planar(np.asarray(planar_transform)[None], 1)[0]
        return _readonly(
            _core.experimental_profile_evaluate_affine(
                np.ascontiguousarray(p, float).ravel(),
                p.shape[::-1],
                tuple(map(float, s)),
                np.ascontiguousarray(b, float),
                tuple(map(float, c)),
                tuple(a.ravel()),
                tuple(map(float, scale)),
            )
        )
    endpoints = (np.asarray(b, float).reshape(-1, 3, 2) - c[None, :, None]) * scale[None, :, None]
    transformed = np.sort(endpoints, axis=2).reshape(-1, 6)
    values = _core.experimental_profile_evaluate(
        np.ascontiguousarray(p, float).ravel(),
        p.shape[::-1],
        tuple(map(float, s)),
        np.ascontiguousarray(transformed),
        (0.0, 0.0, 0.0),
    )
    values[:, 1:] *= scale
    return _readonly(values)


def _evaluation_inputs(profile, bounds, centre, step, scales):
    p = np.asarray(profile)
    b = np.asarray(bounds)
    c = np.asarray(centre)
    s = np.asarray(step)
    if p.ndim != 3 or p.dtype.kind not in "fiu" or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("profile must be a finite nonnegative real z/y/x array")
    if (
        b.ndim != 2
        or b.shape[1] != 6
        or b.dtype.kind not in "fiu"
        or not np.isfinite(b).all()
        or np.any(b[:, ::2] >= b[:, 1::2])
    ):
        raise ValueError("bounds must contain finite positive-volume sample boxes")
    if (
        c.shape != (3,)
        or c.dtype.kind not in "fiu"
        or not np.isfinite(c).all()
        or s.shape != (3,)
        or s.dtype.kind not in "fiu"
        or not np.isfinite(s).all()
        or (s <= 0).any()
    ):
        raise ValueError("invalid centre or grid steps")
    scale = np.asarray(scales)
    if (
        scale.shape != (3,)
        or scale.dtype.kind not in "fiu"
        or not np.isfinite(scale).all()
        or (abs(scale) < 1e-8).any()
    ):
        raise ValueError("scales must be three finite nonzero real numbers")
    return p, b, c, s, scale


def evaluate_profile_shape(
    profile,
    bounds,
    *,
    step=(1.0, 1.0, 1.0),
    centre=(0.0, 0.0, 0.0),
    scales=(1.0, 1.0, 1.0),
    planar_transform=None,
):
    """Mass and fused partials for x/y/omega and log spatial/rocking width.

    Width derivatives dilate the finite support together with its voxels.
    The grid has the supplied current ``step``; derivatives are local to it.
    At exactly coincident voxel/sample boundaries the strict interior edge
    convention applies; centered finite differences must avoid those points.
    """
    p, b, c, s, scale = _evaluation_inputs(profile, bounds, centre, step, scales)
    a = np.eye(2) if planar_transform is None else _planar(np.asarray(planar_transform)[None], 1)[0]
    return _readonly(
        _core.experimental_profile_evaluate_shape(
            np.ascontiguousarray(p, float).ravel(),
            p.shape[::-1],
            tuple(map(float, s)),
            np.ascontiguousarray(b, float),
            tuple(map(float, c)),
            tuple(a.ravel()),
            tuple(map(float, scale)),
        )
    )


def rotation_factors(xy, geometry):
    """Signed zeta = spindle dot (outgoing cross incident)/|cross|.

    Kabsch (2010), section 2.3, doi:10.1107/S0907444909047374.
    Local rocking coordinate is zeta*(omega-omega_B), in degrees. Values
    near zero identify crossings poorly conditioned for rotation integration.
    """
    from .reciprocal import PanelGeometry

    a = np.asarray(xy)
    if not isinstance(geometry, PanelGeometry):
        raise TypeError("geometry must be PanelGeometry")
    if a.ndim != 2 or a.shape[1] != 2 or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError("xy must be finite real (n,2)")
    origin, fast, slow, beam, axis = geometry.vectors
    outgoing = origin + a[:, 0, None] * fast + a[:, 1, None] * slow
    cross = np.cross(outgoing, beam / np.linalg.norm(beam))
    norms = np.linalg.norm(cross, axis=1)
    return _readonly(
        np.divide(
            cross @ (axis / np.linalg.norm(axis)), norms, out=np.zeros(len(a)), where=norms > 0
        )
    )


def _planar(values, n):
    a = np.asarray(values)
    if a.shape != (n, 2, 2) or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError("planar_transforms must be finite real (reflections,2,2)")
    norm = np.linalg.norm(a[:, 0], axis=1) * np.linalg.norm(a[:, 1], axis=1)
    det = np.linalg.det(a)
    if not np.isfinite(norm).all() or np.any(norm == 0) or np.any(abs(det) <= 1e-8 * norm):
        raise ValueError("planar_transforms must be nonsingular")
    return _readonly(a, float)


def scattering_transforms(xy, geometry):
    """Local e1/e2 beam-direction Jacobians, degrees per detector pixel.

    First-order Kabsch (2010) eq. 2.3: e1=S cross S0; e2=S cross e1.
    Pixel footprints are integrated as parallelograms, not enclosing boxes.
    """
    zeta = rotation_factors(xy, geometry)
    a = np.asarray(xy, float)
    origin, fast, slow, beam, _ = geometry.vectors
    ray = origin + a[:, 0, None] * fast + a[:, 1, None] * slow
    length = np.linalg.norm(ray, axis=1)
    unit = ray / length[:, None]
    e1 = np.cross(unit, beam / np.linalg.norm(beam))
    norm = np.linalg.norm(e1, axis=1)
    if np.any(norm <= 1e-12):
        raise ValueError("scattering plane undefined on the direct beam")
    e1 /= norm[:, None]
    e2 = np.cross(unit, e1)
    jac = np.stack(
        [np.column_stack([e1 @ fast, e1 @ slow]), np.column_stack([e2 @ fast, e2 @ slow])], axis=1
    )
    jac *= (180 / np.pi / length)[:, None, None]
    return _planar(jac, len(a)), zeta


SCATTERING_VARIATIONS = ((1.0, 1.0, 1.0), (1.6, 1.6, 1.0), (1.0, 1.0, 1.6), (1.6, 1.6, 1.6))


@dataclass(frozen=True)
class LearnedProfileVariations:
    """Strong-reference shape fractions; targets never fit their own fractions."""

    factors: np.ndarray
    centres: np.ndarray
    fractions: np.ndarray
    reference_ids: np.ndarray
    model_ids: np.ndarray
    source_fingerprint: str
    neighbours: int = 16

    def __post_init__(self):
        f = np.asarray(self.factors)
        c = np.asarray(self.centres)
        p = np.asarray(self.fractions)
        if (
            f.ndim != 2
            or f.shape[1] != 3
            or not 2 <= len(f) <= 4
            or f.dtype.kind not in "fiu"
            or not np.isfinite(f).all()
            or (f < 0.5).any()
            or (f > 2.5).any()
            or not np.array_equal(f[0], np.ones(3))
        ):
            raise ValueError("invalid variation factors")
        if (
            c.ndim != 2
            or c.shape[1] != 3
            or len(c) < 2
            or c.dtype.kind not in "fiu"
            or not np.isfinite(c).all()
        ):
            raise ValueError("invalid reference centres")
        if (
            p.shape != (len(c), len(f))
            or p.dtype.kind not in "fiu"
            or not np.isfinite(p).all()
            or (p < 0).any()
            or not np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-10)
        ):
            raise ValueError("invalid reference shape fractions")
        for name in ("reference_ids", "model_ids"):
            a = np.asarray(getattr(self, name))
            if a.shape != (len(c),) or a.dtype.kind not in "iu" or (a < 0).any():
                raise ValueError("invalid reference indices")
        if (np.diff(self.reference_ids) <= 0).any() or (np.asarray(self.model_ids) >= 65).any():
            raise ValueError("invalid reference index order")
        if not isinstance(self.source_fingerprint, str) or not self.source_fingerprint:
            raise ValueError("source fingerprint required")
        if (
            isinstance(self.neighbours, bool)
            or not isinstance(self.neighbours, int)
            or not 2 <= self.neighbours <= 128
        ):
            raise ValueError("invalid neighbour count")
        for name in ("factors", "centres", "fractions", "reference_ids", "model_ids"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    def predict(self, observations, model):
        o = model.options
        regions = (
            _regions(observations.centres, model.detector_shape, model.angular_range, o.regions) + 1
        )
        regions[
            (model.counts[regions] < o.min_references) | (model.profiles[regions].sum(axis=1) == 0)
        ] = 0
        scale = np.array(
            [
                model.detector_shape[1],
                model.detector_shape[0],
                model.angular_range[1] - model.angular_range[0],
            ]
        )
        output = np.empty((len(regions), len(self.factors)))
        same = observations.fingerprint() == self.source_fingerprint
        for bank in np.unique(regions):
            targets = np.flatnonzero(regions == bank)
            refs = np.flatnonzero(self.model_ids == bank)
            if len(refs) < 2:
                refs = np.arange(len(self.centres))
            distance = np.sum(
                ((observations.centres[targets, None, :] - self.centres[refs][None, :, :]) / scale)
                ** 2,
                axis=2,
            )
            if same:
                distance = np.where(
                    targets[:, None] == self.reference_ids[refs][None, :], np.inf, distance
                )
            count = min(self.neighbours, len(refs) - (1 if same else 0))
            if count < 1:
                raise ValueError("no independent variation references")
            nearest = np.argsort(distance, axis=1, kind="stable")[:, :count]
            selected = self.fractions[refs[nearest]]
            fraction = np.median(selected, axis=1)
            empty = fraction.sum(axis=1) <= 0
            fraction[empty] = selected[empty].mean(axis=1)
            output[targets] = fraction / fraction.sum(axis=1)[:, None]
        return _readonly(output)


def select_observations(observations, ids):
    """Select ordered unique reflection rows, preserving all-domain background fits.

    Selection happens after joint collection/exclusion. It never rereads pixels
    or refits backgrounds; signed observations and fitted covariance are retained.
    """
    if not isinstance(observations, ProfileObservations):
        raise TypeError("observations must be ProfileObservations")
    indices = np.asarray(ids)
    if (
        indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or not len(indices)
        or np.any(indices >= len(observations.centres))
        or np.any(indices < 0)
        or np.any(indices[1:] <= indices[:-1])
    ):
        raise ValueError("ids must be sorted unique in-range integer reflection rows")
    return _reference_observations(observations, indices.astype(np.int64))


def _reference_observations(observations, ids):
    """Copy reference groups only, retaining the original all-spot background fit."""
    from dataclasses import replace

    o = observations
    t = o.summation
    keep = np.isin(t.observation_ids, ids)
    groups = np.flatnonzero(keep)
    sizes = np.diff(o.offsets)[groups]
    prefix = np.r_[0, np.cumsum(sizes)]
    pixels = np.repeat(o.offsets[groups] - prefix[:-1], sizes) + np.arange(prefix[-1])
    mapping = np.full(len(o.centres), -1, np.int64)
    mapping[ids] = np.arange(len(ids))
    table = replace(
        t,
        hkl=t.hkl[ids],
        predicted=t.predicted[ids],
        grain_ids=t.grain_ids[ids],
        values=t.values[ids],
        flags=t.flags[ids],
        observation_ids=mapping[t.observation_ids[groups]],
        frame_indices=t.frame_indices[groups],
        boxes=t.boxes[groups],
        frame_values=t.frame_values[groups],
        frame_flags=t.frame_flags[groups],
        background_models=None if t.background_models is None else t.background_models[groups],
        angular_half_widths=None if t.angular_half_widths is None else t.angular_half_widths[ids],
    )
    return ProfileObservations(
        table,
        o.samples[pixels],
        o.valid[pixels],
        prefix,
        o.covariance[groups],
        o.centres[ids],
        o.scales[ids],
        o.planar_transforms[ids],
    )


def learn_profile_variations(
    observations, model, *, factors=SCATTERING_VARIATIONS, minimum_snr=20.0, neighbours=16
):
    """Learn width-mixture fractions only from the model's strong references.

    The free-fraction fit is a training diagnostic. Weak intensities must use
    predicted fixed fractions, avoiding selection bias from fitting their noise.
    """
    if (
        isinstance(neighbours, bool)
        or not isinstance(neighbours, int)
        or not 2 <= neighbours <= 128
    ):
        raise ValueError("neighbours must be an integer in [2,128]")
    if (
        isinstance(minimum_snr, bool)
        or np.asarray(minimum_snr).dtype.kind not in "fiu"
        or not np.isscalar(minimum_snr)
        or not np.isfinite(minimum_snr)
        or minimum_snr < 10
    ):
        raise ValueError("variation references require minimum_snr>=10")
    if observations.fingerprint() != model.training_fingerprint:
        raise ValueError("variation learning requires the original profile-training observations")
    from dataclasses import replace

    source = observations
    if model.reference_centres is not None:
        ids = model.reference_ids
        centres = observations.centres.copy()
        bank = (
            _regions(centres, model.detector_shape, model.angular_range, model.options.regions) + 1
        )
        bank[
            (model.counts[bank] < model.options.min_references)
            | (model.profiles[bank].sum(axis=1) == 0)
        ] = 0
        local = model.shifts[bank[ids]] / observations.scales[ids]
        centres[ids, :2] = (
            model.reference_centres[:, :2]
            - np.linalg.solve(observations.planar_transforms[ids], local[:, :2, None])[:, :, 0]
        )
        source = replace(observations, centres=centres)
    ids = model.reference_ids
    source = _reference_observations(source, ids)
    fit = fit_profiles(source, model, variation_factors=factors, allow_shape_refinement=True)
    good = (
        (
            (fit.flags & (1 | 2 | 4 | 8 | 16 | 256 | 1024 | 4096 | 8192 | 16384 | 32768 | 131072))
            == 0
        )
        & np.isfinite(fit.values[:, :2]).all(axis=1)
        & (fit.values[:, 0] > minimum_snr * fit.values[:, 1])
    )
    ids = ids[good]
    if len(ids) < 2:
        raise ValueError("too few strong shape-variation references")
    fractions = fit.coefficients[good] / fit.values[good, 0, None]
    return LearnedProfileVariations(
        np.asarray(factors, float),
        observations.centres[ids],
        fractions,
        ids,
        fit.model_ids[good],
        observations.fingerprint(),
        neighbours,
    )


def fit_learned_profiles(observations, model, variations, *, expected_noise=True):
    """Fit signed amplitudes with independently learned, fixed shape fractions."""
    if not isinstance(variations, LearnedProfileVariations):
        raise TypeError("expected LearnedProfileVariations")
    fractions = variations.predict(observations, model)
    result = fit_profiles(
        observations,
        model,
        expected_noise=expected_noise,
        variation_factors=variations.factors,
        fixed_fractions=fractions,
    )
    outside = (observations.centres[:, 2] < variations.centres[:, 2].min()) | (
        observations.centres[:, 2] > variations.centres[:, 2].max()
    )
    flags = result.flags.copy()
    flags[outside] |= int(ProfileFlag.SHAPE_EXTRAPOLATED)
    from dataclasses import replace

    return replace(result, flags=flags)
