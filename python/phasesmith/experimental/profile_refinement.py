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
    _reference_observations,
    evaluate_profile_shape,
    fit_profiles,
)


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
        if type(self.workers) is not int or not 1 <= self.workers <= 4:
            raise ValueError("profile workers must be in 1..4")
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


class _Spot:
    def __init__(self, observations, model, bank):
        self.o, self.model, self.bank = observations, model, int(bank)
        t = observations.summation
        self.profile = model.volumes[self.bank]
        shift = model.shifts[self.bank]
        self.actual = observations.centres[0].copy()
        self.actual[:2] += np.linalg.solve(
            observations.planar_transforms[0], shift[:2] / observations.scales[0, :2]
        )
        self.actual[2] += shift[2] / observations.scales[0, 2]
        bounds, basis = [], []
        for g, box in enumerate(t.boxes):
            x0, y0, x1, y1 = box
            yy, xx = np.mgrid[y0:y1, x0:x1]
            xx, yy = xx.ravel(), yy.ravel()
            a, b = sorted(t.omega_bounds[t.frame_indices[g]])
            bounds.append(
                np.c_[
                    xx - 0.5, xx + 0.5, yy - 0.5, yy + 0.5, np.full(len(xx), a), np.full(len(xx), b)
                ]
            )
            basis.append(
                np.c_[
                    np.ones(len(xx)),
                    (xx - (x0 + x1 - 1) / 2) / max((x1 - x0) / 2, 1),
                    (yy - (y0 + y1 - 1) / 2) / max((y1 - y0) / 2, 1),
                ]
            )
        self.bounds = np.concatenate(bounds) if bounds else np.empty((0, 6))
        self.basis = np.ascontiguousarray(np.concatenate(basis) if basis else np.empty((0, 3)))
        self.signal = np.ascontiguousarray(
            np.where(observations.valid, observations.samples[:, 0], 0.0)
        )
        self.cov = np.ascontiguousarray(
            np.where(np.isfinite(observations.covariance), observations.covariance, 0.0)
        )
        self.offsets = np.ascontiguousarray(observations.offsets, np.int64)
        self.weights = None

    def mass(self, parameters):
        steps = np.array(self.model.options.step) * np.exp(
            [parameters[3], parameters[3], parameters[4]]
        )
        return evaluate_profile_shape(
            self.profile,
            self.bounds,
            step=steps,
            centre=self.actual + parameters[:3],
            scales=self.o.scales[0],
            planar_transform=self.o.planar_transforms[0],
        )

    def set_noise(self, mass, intensity, foreground=False):
        t = self.o.summation
        if t.noise_model == "supplied_independent_pixel_variance":
            variance = self.o.samples[:, 1]
        else:
            gain, read = t.noise_parameters[:2]
            variance = np.maximum(
                1.0,
                np.maximum(0.0, self.o.samples[:, 2] + max(intensity, 0) * mass) / gain + read**2,
            )
        valid = self.o.valid & ((mass > 0) if foreground else True)
        self.weights = np.ascontiguousarray(
            np.divide(1.0, variance, out=np.zeros_like(variance), where=valid)
        )

    def normal(self, design):
        return _core.experimental_profile_normal(
            self.signal,
            self.weights,
            self.basis,
            self.offsets,
            self.cov,
            np.ascontiguousarray(design),
        )

    def objective(self, parameters, regularizer):
        gram = self.normal(self.mass(parameters))
        den, num = gram[0, 0], gram[0, -1]
        if den <= 0:
            return 1e20, np.zeros(5)
        amplitude = num / den
        value = gram[-1, -1] - num * amplitude + np.sum(regularizer * parameters**2)
        derivative = (
            -2 * amplitude * (gram[1:6, -1] - amplitude * gram[1:6, 0])
            + 2 * regularizer * parameters
        )
        return float(value), derivative

    def native_fit(self, parameters):
        native = list(self.o._native())
        native[7] = np.ascontiguousarray(self.o.centres + parameters[:3])
        steps = tuple(
            np.array(self.model.options.step)
            * np.exp([parameters[3], parameters[3], parameters[4]])
        )
        gain, read = self.o.summation.noise_parameters[:2]
        values, flags = _core.experimental_profile_fit(
            tuple(native),
            self.model.profiles,
            self.model.options.shape,
            steps,
            np.array([self.bank], np.int64),
            self.model.shifts,
            self.model.options.minimum_mass,
            self.o.summation.noise_model != "supplied_independent_pixel_variance",
            float(gain),
            float(read),
        )
        return values[0], flags[0]

    def refine(self, initial, choice, options):
        from scipy.optimize import minimize

        parameters = np.zeros(5)
        bounds = [(-options.centre_pixels, options.centre_pixels)] * 2 + [
            (-options.centre_degrees, options.centre_degrees),
            tuple(np.log(options.spatial_range)),
            tuple(np.log(options.rocking_range)),
        ]
        if choice == 2:
            bounds[2:] = [(0.0, 0.0)] * 3
        reg = np.array(options.regularizer)
        intensity = initial
        converged = True
        for _ in range(options.iterations):
            self.set_noise(self.mass(parameters)[:, 0], intensity)
            before = self.objective(parameters, reg)[0]
            result = minimize(
                self.objective,
                parameters,
                args=(reg,),
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options=dict(maxiter=options.maximum_iterations, ftol=1e-9, gtol=1e-4, maxls=30),
            )
            converged &= bool(result.success)
            if np.isfinite(result.fun) and result.fun <= before + 1e-5:
                parameters = result.x
            gram = self.normal(self.mass(parameters))
            if gram[0, 0] <= 0:
                raise ValueError("nonidentifiable strong profile")
            intensity = gram[0, -1] / gram[0, 0]
        values, flags = self.finish(parameters, choice)
        return values, flags, parameters, converged

    def finish(self, parameters, choice):
        """Apply frozen strong shape parameters with conditional joint sigma."""
        values, flags = self.native_fit(parameters)
        mass = self.mass(parameters)
        self.set_noise(mass[:, 0], float(values[0]), foreground=True)
        active = [0, 1, 2] if choice == 2 else list(range(6))
        design = mass[:, active].copy()
        design[:, 1:] *= values[0]
        normal = self.normal(design)[:-1, :-1]
        diag = np.sqrt(np.maximum(normal.diagonal(), 1e-30))
        scaled = normal / diag[:, None] / diag[None]
        eig, vectors = np.linalg.eigh((scaled + scaled.T) / 2)
        if eig.min() < -1e-7:
            raise ValueError("nonpositive profile normal matrix")
        covariance = (vectors * np.where(eig > 1e-9, 1 / np.maximum(eig, 1e-300), 0.0)) @ vectors.T
        values[1] = max(values[1], np.sqrt(max(0.0, covariance[0, 0])) / diag[0])
        return values, flags

    def fixed(self, parameters, design_counts):
        mass = self.mass(parameters)[:, 0]
        self.set_noise(mass, design_counts, foreground=True)
        observed = np.ascontiguousarray(np.where(self.o.valid, self.o.samples[:, 1], 0.0))
        intensity, sigma = _core.experimental_fixed_profile(
            self.signal,
            observed,
            self.weights,
            self.basis,
            self.offsets,
            self.cov,
            np.ascontiguousarray(mass),
        )
        return intensity, sigma, float(mass[self.o.valid].sum())


def profile_centroid_offsets(observations, model, model_ids, log_spatial_width=0.0):
    """Physical template centroid minus predicted position, in detector pixels.

    This includes both bank shift and template first moment. A learned sensor
    centroid correction must subtract this quantity to obtain profile origin.
    """
    nx, ny, nz = model.options.shape
    z, y, x = np.meshgrid(
        *[
            (np.arange(n) - (n - 1) / 2) * s
            for n, s in zip((nz, ny, nx), model.options.step[::-1], strict=True)
        ],
        indexing="ij",
    )
    moments = model.profiles @ np.c_[x.ravel(), y.ravel(), z.ravel()]
    ids = np.asarray(model_ids)
    local = (
        model.shifts[ids, :2] + moments[ids, :2] * np.exp(np.asarray(log_spatial_width)[..., None])
    ) / observations.scales[:, :2]
    return np.linalg.solve(observations.planar_transforms, local[..., None])[..., 0]


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
    baseline = fit_profiles(observations, model)
    values, flags = baseline.values.copy(), baseline.flags.copy()
    parameters = np.zeros((n, 5))
    converged = np.ones(n, bool)
    centroid = profile_centroid_offsets(
        observations, model, baseline.model_ids, np.log(o.weak_spatial_width)
    )
    selected = np.flatnonzero((c != 0) | weak)

    def integrate_one(i):
        one = _reference_observations(observations, np.array([i], np.int64))
        spot = _Spot(one, model, baseline.model_ids[i])
        value, flag = baseline.values[i].copy(), baseline.flags[i]
        parameters = np.zeros(5)
        success = True
        try:
            if c[i]:
                if frozen is None:
                    value, flag, parameters, success = spot.refine(value[0], c[i], o)
                else:
                    parameters = frozen[i].copy()
                    value, flag = spot.finish(parameters, c[i])
            else:
                parameters[3] = np.log(o.weak_spatial_width)
                parameters[:2] = offsets[i] - centroid[i]
                value[:3] = spot.fixed(parameters, o.weak_design_counts)
                # Remaining diagnostics belong to the baseline, not this fit.
                value[3:] = np.nan
        except (ValueError, np.linalg.LinAlgError):
            flag |= 16384
            success = False
        return i, value, flag, parameters, success

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=o.workers) as pool:
        # Source-order consumption and independent arrays preserve determinism.
        for done, (i, value, flag, parameter, success) in enumerate(
            pool.map(integrate_one, selected)
        ):
            values[i], flags[i], parameters[i], converged[i] = value, flag, parameter, success
            if progress is not None:
                progress(done + 1, len(selected))
    integrated = replace(baseline, values=values, flags=flags)
    return RefinedProfiles(baseline, integrated, parameters, c.copy(), weak.copy(), converged)


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
