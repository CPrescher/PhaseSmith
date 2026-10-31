"""Independent NumPy/SciPy strong-profile control for small axis-aligned fixtures.

This module never imports the native extension or calls the production adapter.
It integrates finite piecewise-constant voxels directly, including analytical
translation and log-width partials, and uses dense covariance matrices.
"""

from types import SimpleNamespace

import numpy as np

from .intensity_reference import profile_precision
from .profiles_reference import sample_bounds


def shape(profile, bounds, step, centre, scales, parameters):
    """Finite box/voxel intersections; contact without overlap has zero mass/partial."""
    p = np.asarray(parameters)
    step = np.asarray(step) * np.exp([p[3], p[3], p[4]])
    scale = np.asarray(scales)
    endpoints = (np.asarray(bounds).reshape(-1, 3, 2) - (centre + p[:3])[None, :, None]) * scale[
        None, :, None
    ]
    endpoints.sort(axis=2)
    z, y, x = np.indices(profile.shape)
    index = np.c_[x.ravel(), y.ravel(), z.ravel()]
    lo = (index - np.array(profile.shape[::-1]) / 2) * step
    hi = lo + step
    overlap = np.maximum(
        np.minimum(hi, endpoints[:, None, :, 1]) - np.maximum(lo, endpoints[:, None, :, 0]), 0
    )
    fraction = overlap / step
    dhi = hi < endpoints[:, None, :, 1]
    dlo = lo > endpoints[:, None, :, 0]
    translation = (dhi.astype(float) - dlo.astype(float)) / step
    width = (dhi * hi - dlo * lo) / step - fraction
    translation[overlap <= 0] = 0
    width[overlap <= 0] = 0
    deriv = []
    dilation = []
    for j in range(3):
        others = np.prod(fraction[:, :, [k for k in range(3) if k != j]], axis=2)
        deriv.append((translation[:, :, j] * others) @ profile.ravel() * scale[j])
        dilation.append((width[:, :, j] * others) @ profile.ravel())
    return np.c_[
        np.prod(fraction, axis=2) @ profile.ravel(), *deriv, dilation[0] + dilation[1], dilation[2]
    ]


class ReferenceSpot:
    """Dense single-reflection reference, deliberately separate from production."""

    def __init__(self, observations, model, bank):
        if len(observations.centres) != 1 or not np.array_equal(
            observations.planar_transforms[0], np.eye(2)
        ):
            raise ValueError("small reference requires one axis-aligned reflection")
        self.o, self.model, self.bank = observations, model, bank
        t = observations.summation
        self.bounds, self.basis = sample_bounds(
            observations.offsets, t.boxes, np.sort(t.omega_bounds[t.frame_indices], axis=1)
        )
        self.centre = observations.centres[0] + model.shifts[bank] / observations.scales[0]
        self.signal = np.where(observations.valid, observations.samples[:, 0], 0)
        self.covariance = np.where(np.isfinite(observations.covariance), observations.covariance, 0)

    def mass(self, parameters):
        return shape(
            self.model.volumes[self.bank],
            self.bounds,
            self.model.options.step,
            self.centre,
            self.o.scales[0],
            parameters,
        )

    def set_noise(self, mass, intensity, foreground=False):
        o = self.o
        if o.summation.noise_model == "supplied_independent_pixel_variance":
            variance = o.samples[:, 1]
        else:
            gain, read = o.summation.noise_parameters[:2]
            variance = np.maximum(
                1, np.maximum(0, o.samples[:, 2] + max(0, intensity) * mass) / gain + read**2
            )
        self.weights = np.where(o.valid & ((mass > 0) if foreground else True), 1 / variance, 0)
        self.precision = profile_precision(self.weights, self.basis, o.offsets, self.covariance)

    def objective(self, p, regularizer):
        mass = self.mass(p)
        wp = self.precision @ mass[:, 0]
        amplitude = self.signal @ wp / (mass[:, 0] @ wp)
        residual = self.signal - amplitude * mass[:, 0]
        value = residual @ self.precision @ residual + np.sum(regularizer * p**2)
        gradient = -2 * amplitude * mass[:, 1:].T @ self.precision @ residual + 2 * regularizer * p
        return value, gradient

    def fit(self, initial, choice, options):
        from scipy.optimize import minimize

        p = np.zeros(5)
        intensity = initial
        bounds = [(-options.centre_pixels, options.centre_pixels)] * 2 + [
            (-options.centre_degrees, options.centre_degrees),
            tuple(np.log(options.spatial_range)),
            tuple(np.log(options.rocking_range)),
        ]
        if choice == 2:
            bounds[2:] = [(0, 0)] * 3
        for _ in range(options.iterations):
            self.set_noise(self.mass(p)[:, 0], intensity)
            result = minimize(
                self.objective,
                p,
                args=(np.array(options.regularizer),),
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options=dict(maxiter=options.maximum_iterations, ftol=1e-9, gtol=1e-4, maxls=30),
            )
            p = result.x
            mass = self.mass(p)[:, 0]
            intensity = mass @ self.precision @ self.signal / (mass @ self.precision @ mass)
        # Independent fixed-shape final integration: supplied variance first,
        # then the same seven expected-count updates when explicitly requested.
        mass = self.mass(p)
        expected = self.o.summation.noise_model != "supplied_independent_pixel_variance"
        for iteration in range(8 if expected else 1):
            if iteration == 0:
                weights = np.where(self.o.valid & (mass[:, 0] > 0), 1 / self.o.samples[:, 1], 0)
                self.precision = profile_precision(
                    weights, self.basis, self.o.offsets, self.covariance
                )
            else:
                self.set_noise(mass[:, 0], intensity, foreground=True)
            den = mass[:, 0] @ self.precision @ mass[:, 0]
            intensity = mass[:, 0] @ self.precision @ self.signal / den
            sigma = 1 / np.sqrt(den)
        self.set_noise(mass[:, 0], intensity, foreground=True)
        design = mass[:, : 3 if choice == 2 else 6].copy()
        design[:, 1:] *= intensity
        normal = design.T @ self.precision @ design
        diag = np.sqrt(np.maximum(normal.diagonal(), 1e-30))
        scaled = normal / diag[:, None] / diag[None]
        eig, vectors = np.linalg.eigh((scaled + scaled.T) / 2)
        joint = np.sum(vectors[0, eig > 1e-9] ** 2 / eig[eig > 1e-9])
        return SimpleNamespace(
            parameters=p, intensity=intensity, sigma=max(sigma, np.sqrt(max(0, joint)) / diag[0])
        )
