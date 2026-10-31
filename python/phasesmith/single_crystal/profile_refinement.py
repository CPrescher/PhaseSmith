"""Strong-only centre/width refinement and frozen weak-profile extraction.

The caller declares which strong observations may fit shape and supplies fixed
sensor centroid offsets. No structure-factor or displacement-parameter target
enters model choice. Baseline aperture and fixed-profile values are retained.
"""

from dataclasses import dataclass, replace

import numpy as np

from phasesmith import _core

from .profiles import (
    ProfileIntegration,
    fit_profiles,
)

__all__ = ["RefinedProfiles", "ShapeFitOptions", "calibrate_sensor_offsets", "refine_profiles"]


@dataclass(frozen=True)
class ShapeFitOptions:
    workers: int = 1
    iterations: int = 3
    maximum_iterations: int = 80
    centre_pixels: float = 1.0
    centre_degrees: float = 0.4
    spatial_range: tuple = (0.65, 1.5)
    rocking_range: tuple = (0.5, 1.5)
    regularizer: tuple = (0.01, 0.01, 0.16, 0.04, 0.04)
    weak_design_counts: float = 1000.0
    weak_spatial_width: float = 0.95

    def __post_init__(self):
        if type(self.workers) is not int or not 1 <= self.workers <= 32:
            raise ValueError("profile workers must be in 1..32")
        if (
            type(self.iterations) is not int
            or not 1 <= self.iterations <= 10
            or type(self.maximum_iterations) is not int
            or not 1 <= self.maximum_iterations <= 200
        ):
            raise ValueError("invalid profile fit iteration budget")
        scalars = np.asarray(
            [
                self.centre_pixels,
                self.centre_degrees,
                self.weak_design_counts,
                self.weak_spatial_width,
            ]
        )
        if not np.isfinite(scalars).all() or np.any(scalars <= 0):
            raise ValueError("profile fit scales must be positive")
        for interval in (self.spatial_range, self.rocking_range):
            if (
                len(interval) != 2
                or not np.isfinite(interval).all()
                or not 0.25 <= interval[0] < 1 < interval[1] <= 3
            ):
                raise ValueError("width bounds must bracket one within [.25,3]")
        r = np.asarray(self.regularizer)
        if r.shape != (5,) or not np.isfinite(r).all() or np.any(r < 0):
            raise ValueError("invalid shape regularizer")


def _native_options(o):
    if not isinstance(o, ShapeFitOptions):
        raise TypeError("options must be ShapeFitOptions")
    return (
        o.workers,
        o.iterations,
        o.maximum_iterations,
        o.centre_pixels,
        o.centre_degrees,
        o.spatial_range,
        o.rocking_range,
        o.regularizer,
        o.weak_design_counts,
        o.weak_spatial_width,
    )


def _noise(observations):
    t = observations.summation
    gain, read = t.noise_parameters[:2]
    return t.noise_model != "supplied_independent_pixel_variance", float(gain), float(read)


class _Spot:
    """Diagnostic view of the native objective, also used by independent tests."""

    def __init__(self, observations, model, bank):
        self._native = _core.NativeProfileSpot(
            observations._native(),
            model.profiles,
            model.options.shape,
            model.options.step,
            int(bank),
            model.shifts,
            model.options.minimum_mass,
            *_noise(observations),
        )
        self.__dict__.update(self._native.snapshot())

    def mass(self, parameters):
        return self._native.mass(list(parameters))

    def set_noise(self, mass, intensity, foreground=False):
        self._native.set_noise(np.ascontiguousarray(mass), float(intensity), foreground)
        self.weights = self._native.snapshot()["weights"]

    def normal(self, design):
        return self._native.normal(np.ascontiguousarray(design))

    def objective(self, parameters, regularizer):
        return self._native.objective(list(parameters), list(regularizer))


def profile_centroid_offsets(observations, model, model_ids, log_spatial_width=0.0):
    """Physical template centroid minus predicted position, in detector pixels.

    Includes the bank shift and finite template first moment. Sensor centroid
    corrections subtract this quantity to obtain the profile origin.
    """
    n = len(observations.centres)
    ids = np.asarray(model_ids)
    widths = np.asarray(log_spatial_width)
    if ids.shape != (n,) or ids.dtype.kind not in "iu" or np.any(ids < 0):
        raise ValueError("invalid profile assignments")
    if widths.dtype.kind not in "fiu" or not np.isfinite(widths).all():
        raise ValueError("invalid log spatial widths")
    widths = np.ascontiguousarray(np.broadcast_to(widths, (n,)), dtype=float)
    return _core.experimental_profile_centroids(
        model.profiles,
        model.options.shape,
        model.options.step,
        model.shifts,
        np.ascontiguousarray(ids, dtype=np.int64),
        observations.scales,
        observations.planar_transforms.reshape(n, 4),
        widths,
    )


@dataclass(frozen=True)
class RefinedProfiles:
    baseline: ProfileIntegration
    integration: ProfileIntegration
    parameters: np.ndarray
    choices: np.ndarray
    weak_mask: np.ndarray
    converged: np.ndarray


def refine_profiles(
    observations,
    model,
    choices,
    *,
    weak_mask=None,
    sensor_offsets=None,
    options=None,
    frozen_parameters=None,
    progress=None,
    execution=None,
):
    """0=fixed, 1=strong x/y/omega+width, 2=strong x/y only.

    Choices are explicit frozen inputs. This function does not select them by
    comparing with an external integration or an atomic refinement. Sensor
    offsets are physical centroids, not profile-origin shifts. Rows are never
    removed; a failed conditional fit retains baseline values with FAILED bit.
    """
    o = ShapeFitOptions() if options is None else options
    n = len(observations.centres)
    c = np.asarray(choices)
    weak = np.zeros(n, bool) if weak_mask is None else np.asarray(weak_mask)
    offsets = np.zeros((n, 2)) if sensor_offsets is None else np.asarray(sensor_offsets)
    if (
        c.shape != (n,)
        or c.dtype.kind not in "iu"
        or np.any(c > 2)
        or np.any(c < 0)
        or weak.shape != (n,)
        or weak.dtype != bool
        or np.any(weak & (c != 0))
        or offsets.shape != (n, 2)
        or offsets.dtype.kind not in "fiu"
        or not np.isfinite(offsets).all()
    ):
        raise ValueError("invalid frozen profile choices, weak mask or sensor centroid offsets")
    if weak.any() and sensor_offsets is None:
        raise ValueError("weak updates require explicitly declared sensor centroid offsets")
    frozen = None if frozen_parameters is None else np.asarray(frozen_parameters)
    if frozen is not None and (
        frozen.shape != (n, 5) or frozen.dtype.kind not in "fiu" or not np.isfinite(frozen).all()
    ):
        raise ValueError("frozen parameters must be finite real (reflections,5)")
    baseline = fit_profiles(observations, model, execution=execution)
    result = _core.experimental_profile_refine(
        observations._native(),
        model.profiles,
        model.options.shape,
        model.options.step,
        baseline.model_ids,
        model.shifts,
        model.options.minimum_mass,
        *_noise(observations),
        np.ascontiguousarray(c, dtype=np.uint8),
        np.ascontiguousarray(weak),
        np.ascontiguousarray(offsets, dtype=float),
        _native_options(o),
        baseline.values,
        baseline.flags,
        None if frozen is None else np.ascontiguousarray(frozen, dtype=float),
        progress,
    )
    integrated = replace(baseline, values=result["values"], flags=result["flags"])
    return RefinedProfiles(
        baseline, integrated, result["parameters"], c.copy(), weak.copy(), result["converged"]
    )


def calibrate_sensor_offsets(
    observations, model, strong_result, layout, training, *, provenance, **options
):
    """Learn the declared sensor map from independent strong-fit centroids.

    The fixed mask must exclude validation/test teachers. Includes the learned
    template first moment: profile origin is not the physical spot centroid.
    """
    from .sensor_offsets import fit_sensor_offsets

    mask = np.asarray(training)
    n = len(observations.centres)
    if mask.shape != (n,) or mask.dtype != bool:
        raise ValueError("sensor training must be an explicit boolean mask")
    ids = np.flatnonzero(
        (strong_result.choices != 0)
        & strong_result.converged
        & (strong_result.integration.flags == 0)
    )
    if not len(ids):
        raise ValueError("no valid strong centroid teachers")
    centroids = profile_centroid_offsets(
        observations, model, strong_result.integration.model_ids, strong_result.parameters[:, 3]
    )
    target = strong_result.parameters[:, :2] + centroids
    return fit_sensor_offsets(
        layout,
        observations.centres[ids, :2],
        target[ids],
        mask[ids],
        provenance=provenance,
        **options,
    )
