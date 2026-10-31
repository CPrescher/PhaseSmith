"""Portable experimental profile/intensity reduction, with an optional safe CLI.

Image reading, geometry refinement and structural refinement belong to callers.
No data-specific script imports, source coefficients or atomic targets are used.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path

import numpy as np

from .intensity_response import SlabResponse
from .profile_refinement import ShapeFitOptions, refine_profiles
from .profiles import ProfileOptions, learn_profiles
from .reduction_cache import ReductionCache, cache_key


@dataclass(frozen=True)
class ReductionConfig:
    profile: ProfileOptions = field(default_factory=lambda: ProfileOptions(cutoff=0.0))
    shape_fit: ShapeFitOptions = field(default_factory=ShapeFitOptions)
    profile_coordinates: str = "detector"
    strong_policy: str = "fixed"
    strong_snr: float = 50.0
    response: SlabResponse | None = None
    polarization_axis: tuple | None = None
    polarization_fraction: float = 0.98
    sensor: dict | None = None
    empirical: dict | None = None
    equivalent_error_floor: bool = False

    def __post_init__(self):
        if not isinstance(self.profile, ProfileOptions) or not isinstance(
            self.shape_fit, ShapeFitOptions
        ):
            raise TypeError("profile and shape_fit require typed options")
        if self.profile_coordinates not in ("detector", "scattering"):
            raise ValueError("unknown profile coordinates")
        if self.strong_policy not in ("fixed", "strong_xy", "strong_full"):
            raise ValueError("unknown strong-profile policy")
        if (
            not np.isscalar(self.strong_snr)
            or not np.isfinite(self.strong_snr)
            or self.strong_snr < 20
        ):
            raise ValueError("strong shape fits require declared SNR >=20")
        if (
            not np.isscalar(self.polarization_fraction)
            or not np.isfinite(self.polarization_fraction)
            or not 0 <= self.polarization_fraction <= 1
        ):
            raise ValueError("polarization fraction must lie in [0,1]")
        if self.polarization_axis is not None:
            v = np.asarray(self.polarization_axis)
            if (
                v.shape != (3,)
                or v.dtype.kind not in "fiu"
                or not np.isfinite(v).all()
                or np.linalg.norm(v) <= 0
            ):
                raise ValueError("polarization axis must be an explicit nonzero lab vector")
        if type(self.equivalent_error_floor) is not bool:
            raise ValueError("equivalent_error_floor must be boolean")
        if self.equivalent_error_floor and self.empirical is None:
            raise ValueError("equivalent errors require explicit scaling/symmetry families")
        if self.sensor is not None:
            from .sensor_offsets import SensorLayout, SensorOffsets

            if not isinstance(self.sensor, dict) or set(self.sensor) != {
                "layout",
                "coefficients",
                "provenance",
            }:
                raise ValueError("sensor recipe requires layout, coefficients and provenance")
            SensorOffsets(
                SensorLayout(**self.sensor["layout"]),
                self.sensor["coefficients"],
                self.sensor["provenance"],
            )
        if self.empirical is not None:
            from .empirical_scaling import ScalingOptions

            required = {"equivalence_operations", "symmetry_provenance", "detector_shape"}
            allowed = required | {
                "frame_gauge",
                "node_gauge",
                "grid",
                "options",
                "reference_snr",
                "relative_variance_floor",
            }
            if (
                not isinstance(self.empirical, dict)
                or not required <= self.empirical.keys()
                or self.empirical.keys() - allowed
            ):
                raise ValueError("empirical recipe has missing or unknown fields")
            ScalingOptions(**self.empirical.get("options", {}))
        # Strict round-trip catches objects/nonfinite arrays in JSON recipes.
        json.dumps(asdict(self), allow_nan=False)

    @classmethod
    def from_dict(cls, value):
        record = dict(value)
        if "profile" in record:
            record["profile"] = ProfileOptions(**record["profile"])
        if "shape_fit" in record:
            record["shape_fit"] = ShapeFitOptions(**record["shape_fit"])
        if record.get("response") is not None:
            record["response"] = SlabResponse(**record["response"])
        return cls(**record)


def implementation_identity():
    names = (
        "phasesmith._core",
        *(
            "phasesmith.experimental." + n
            for n in (
                "reduction",
                "profiles",
                "profile_refinement",
                "sensor_offsets",
                "intensity_response",
                "empirical_scaling",
                "intensity_uncertainty",
                "reduction_cache",
                "integration",
                "reciprocal",
            )
        ),
    )
    hashes = {
        name: hashlib.sha256(Path(import_module(name).__file__).read_bytes()).hexdigest()
        for name in names
    }
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def profile_geometry(xy, geometry):
    """Ray, exact local pixel Jacobian and rocking zeta, including a fixed map."""
    from .reciprocal import mapped_xy

    a = np.asarray(xy, float)
    if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
        raise ValueError("xy must be finite (n,2)")
    plane = a if geometry.pixel_map is None else mapped_xy(geometry.pixel_map, a)
    origin, fast, slow, beam, axis = geometry.vectors
    beam, axis = beam / np.linalg.norm(beam), axis / np.linalg.norm(axis)
    ray = origin + plane[:, :1] * fast + plane[:, 1:2] * slow
    length = np.linalg.norm(ray, axis=1)
    ray /= length[:, None]
    e1 = np.cross(ray, beam)
    norm = np.linalg.norm(e1, axis=1)
    if np.any(norm < 1e-10):
        raise ValueError("scattering profile undefined on direct beam")
    zeta = e1 @ axis / norm
    e1 /= norm[:, None]
    e2 = np.cross(ray, e1)
    jac = np.stack([np.c_[e1 @ fast, e1 @ slow], np.c_[e2 @ fast, e2 @ slow]], axis=1)
    jac *= (180 / np.pi / length)[:, None, None]
    if geometry.pixel_map is not None:
        inv = np.asarray(geometry.pixel_map.inverse_jacobian(plane))
        if inv.shape != (len(a), 2, 2) or not np.isfinite(inv).all():
            raise ValueError("invalid pixel-map Jacobian")
        jac = jac @ np.linalg.inv(inv)
    return ray, jac, zeta


def physical_factors(centres, geometry, config):
    """Rotation Lorentz/polarization and explicit absolute slab efficiency.

    Correction L/P with L=|axis dot (s1 cross s0)|, as for a rotation scan.
    No powder solid-angle factor is applied to discrete integrated spots.
    Physical corrections require a declared polarization axis. Slab response
    gives absorption efficiency only; threshold/charge sharing are not modeled.
    """
    ray, _, _ = profile_geometry(np.asarray(centres)[:, :2], geometry)
    beam, axis = geometry.vectors[3:]
    beam, axis = beam / np.linalg.norm(beam), axis / np.linalg.norm(axis)
    factor = np.ones(len(ray))
    applied = []
    if config.polarization_axis is not None:
        e = np.asarray(config.polarization_axis, float)
        e /= np.linalg.norm(e)
        if abs(e @ beam) > 1e-8:
            raise ValueError("polarization vector must be perpendicular to incident beam")
        orth = np.cross(beam, e)
        p = config.polarization_fraction * (1 - (ray @ e) ** 2) + (
            1 - config.polarization_fraction
        ) * (1 - (ray @ orth) ** 2)
        lorentz = abs(np.cross(ray, beam) @ axis)
        if np.any(p <= 1e-12) or np.any(lorentz <= 1e-10):
            raise ValueError("Lorentz/polarization correction singular for this reflection set")
        factor *= lorentz / p
        applied.append("rotation Lorentz / polarization")
    if config.response is not None:
        if not np.isclose(
            config.response.wavelength_angstrom, geometry.wavelength, rtol=1e-8, atol=0
        ):
            raise ValueError("slab attenuation wavelength differs from calibration")
        normal = np.cross(*geometry.vectors[1:3])
        normal /= np.linalg.norm(normal)
        cosine = np.clip(abs(ray @ normal), 0, 1)
        factor /= config.response.evaluate(cosine)[:, 0]
        applied.append("declared homogeneous-slab absorption efficiency")
    if not np.isfinite(factor).all() or np.any(factor <= 0):
        raise ValueError("physical intensity correction is nonfinite or nonpositive")
    return factor, applied


@dataclass(frozen=True)
class ReductionResult:
    aperture: object
    profile: object
    intensity_sigma: np.ndarray
    statistical_sigma: np.ndarray
    parameters: np.ndarray
    choices: np.ndarray
    weak_mask: np.ndarray
    physical_factor: np.ndarray
    empirical_correction: np.ndarray
    metadata: dict


def reduce_observations(
    observations,
    detector_shape,
    config=None,
    *,
    geometry=None,
    model=None,
    reference_mask=None,
    choices=None,
    weak_mask=None,
    sensor_centroids=None,
    frozen_parameters=None,
    cache=None,
    progress=None,
):
    """Complete signed profile path, retaining the exact aperture table and rows.

    Sensor maps in configuration are frozen maps, or the caller may supply
    physical centroid offsets from ``fit_sensor_offsets``. A weak row never
    fits its own position/width. Scaling requires explicit equivalence matrices
    plus provenance; it is never inferred from metric lattice symmetry.
    """
    from .empirical_scaling import ScalingOptions, fit_empirical_scale
    from .sensor_offsets import SensorLayout, SensorOffsets

    c = ReductionConfig() if config is None else config
    if not isinstance(c, ReductionConfig):
        raise TypeError("config must be ReductionConfig")
    t, n = observations.summation, len(observations.centres)
    refs = np.ones(n, bool) if reference_mask is None else np.asarray(reference_mask)
    if refs.shape != (n,) or refs.dtype != bool:
        raise ValueError("reference mask must be boolean per observation")
    store = (
        None
        if cache is None
        else (cache if isinstance(cache, ReductionCache) else ReductionCache(cache))
    )
    identity = implementation_identity()
    if model is None:
        bank_key = cache_key(
            "profile-learning",
            dict(
                observations=observations.fingerprint(),
                detector_shape=list(detector_shape),
                profile=asdict(c.profile),
            ),
            dict(reference_mask=refs),
            implementation=identity,
        )
        bank = None if store is None else store.load(bank_key)
        if bank is None:
            model = learn_profiles(
                observations, refs, detector_shape=tuple(detector_shape), options=c.profile
            )
            if store is not None:
                store.store(
                    bank_key,
                    dict(
                        profiles=model.profiles,
                        counts=model.counts,
                        shifts=model.shifts,
                        reference_ids=model.reference_ids,
                        reference_centres=model.reference_centres,
                        angular_range=np.array(model.angular_range),
                    ),
                )
        else:
            from .profiles import ProfileModel

            model = ProfileModel(
                bank["profiles"],
                bank["counts"],
                bank["shifts"],
                bank["reference_ids"],
                observations.fingerprint(),
                tuple(detector_shape),
                tuple(bank["angular_range"]),
                c.profile,
                bank["reference_centres"],
            )
    elif asdict(model.options) != asdict(c.profile) or tuple(model.detector_shape) != tuple(
        detector_shape
    ):
        raise ValueError("supplied profile bank differs from reduction configuration")
    if choices is None:
        choices = np.zeros(n, np.int64)
        if c.strong_policy != "fixed":
            use = (
                (t.flags == 0)
                & (t.values[:, 0] > c.strong_snr * t.values[:, 1])
                & (t.values[:, 1] > 0)
            )
            choices[use] = 2 if c.strong_policy == "strong_xy" else 1
    if c.sensor is not None:
        if sensor_centroids is not None:
            raise ValueError("supply one sensor-centroid source")
        layout = SensorLayout(**c.sensor["layout"])
        sensor = SensorOffsets(layout, c.sensor["coefficients"], c.sensor["provenance"])
        apply_to = (
            (np.asarray(choices) == 0) & (t.flags == 0)
            if weak_mask is None
            else np.asarray(weak_mask)
        )
        if apply_to.shape != (n,) or apply_to.dtype != bool:
            raise ValueError("invalid sensor application mask")
        sensor_centroids = np.zeros((n, 2))
        sensor_centroids[apply_to] = sensor.evaluate(observations.centres[apply_to, :2])
    if weak_mask is None:
        weak_mask = (
            np.zeros(n, bool)
            if sensor_centroids is None
            else (np.asarray(choices) == 0) & (t.flags == 0)
        )
    arrays = dict(
        reference_mask=refs,
        choices=np.asarray(choices),
        weak_mask=np.asarray(weak_mask),
        profiles=model.profiles,
        shifts=model.shifts,
        counts=model.counts,
        reference_ids=model.reference_ids,
        reference_centres=np.empty((0, 3))
        if model.reference_centres is None
        else model.reference_centres,
        sensor_centroids=np.empty((0, 2)) if sensor_centroids is None else sensor_centroids,
        frozen_parameters=np.empty((0, 5)) if frozen_parameters is None else frozen_parameters,
    )
    key = cache_key(
        "profiles",
        dict(
            config=asdict(c),
            observations=observations.fingerprint(),
            detector_shape=list(detector_shape),
        ),
        arrays,
        implementation=identity,
    )
    cached = None if store is None else store.load(key)
    if cached is None:
        refined = refine_profiles(
            observations,
            model,
            choices,
            weak_mask=weak_mask,
            sensor_offsets=sensor_centroids,
            options=c.shape_fit,
            frozen_parameters=frozen_parameters,
            progress=progress,
        )
        profile, parameters = refined.integration, refined.parameters
        if store is not None:
            store.store(
                key,
                dict(
                    values=profile.values,
                    flags=profile.flags,
                    model_ids=profile.model_ids,
                    reference_membership=profile.reference_membership,
                    parameters=parameters,
                ),
            )
    else:
        from .profiles import ProfileIntegration

        profile = ProfileIntegration(
            t,
            cached["values"],
            cached["flags"],
            cached["model_ids"],
            cached["reference_membership"],
        )
        parameters = cached["parameters"]
    if (
        c.response is not None or c.polarization_axis is not None or c.empirical is not None
    ) and geometry is None:
        raise ValueError("physical/scaling corrections require the actual selected geometry")
    factor, corrections = (
        (np.ones(n), [])
        if geometry is None
        else physical_factors(observations.centres, geometry, c)
    )
    corrected = profile.values[:, :2] * factor[:, None]
    empirical = np.ones(n)
    scaling_record = None
    error_record = None
    statistical_sigma = corrected[:, 1].copy()
    if c.empirical is not None:
        design, groups, training, variance = empirical_design(
            observations, geometry, c.empirical, corrected, profile.flags
        )
        options = ScalingOptions(**c.empirical.get("options", {}))
        fit = fit_empirical_scale(
            design,
            np.where(np.isfinite(corrected[:, 0]), corrected[:, 0], 0.0),
            variance,
            groups,
            training,
            options=options,
        )
        if not fit.converged:
            raise ValueError(
                "empirical scaling did not converge; retain unscaled intensities for review"
            )
        corrected = corrected * fit.correction[:, None]
        empirical = fit.correction
        statistical_sigma = corrected[:, 1].copy()
        if c.equivalent_error_floor:
            from .intensity_uncertainty import estimate_equivalent_error

            usable = np.isfinite(corrected).all(axis=1) & (statistical_sigma > 0)
            _, usable_groups = np.unique(groups[usable], return_inverse=True)
            errors = estimate_equivalent_error(
                corrected[usable, 0],
                statistical_sigma[usable],
                usable_groups,
                fit.reference_mask[usable],
                np.isin(groups[usable], np.unique(groups[training])),
            )
            corrected[usable, 1] = errors.sigma
            error_record = dict(
                relative=errors.relative,
                training_pairs=errors.training_pairs,
                training_families=errors.training_families,
                training_robust_width=errors.training_robust_width,
                interpretation="equivalent-pair observation error; no scale covariance",
            )
        scaling_record = dict(
            coefficients=fit.coefficients.tolist(),
            reference_ids=np.flatnonzero(fit.reference_mask).tolist(),
            converged=fit.converged,
            history=fit.history,
            uncertainty="conditional counting sigma; scale covariance excluded",
        )
        corrections.append("within-equivalent empirical frame/panel/ray scale")
    metadata = dict(
        schema="phasesmith.experimental.intensity-reduction.v1",
        config=asdict(c),
        implementation=identity,
        observations=observations.fingerprint(),
        profile_cache_key=key,
        cache_hit=cached is not None,
        reflections=n,
        profile_fitting_backend="phasesmith-spots native bounded L-BFGS-B",
        empirical_scaling_backend="phasesmith-spots native bounded least squares and IRLS",
        corrections=corrections,
        scaling=scaling_record,
        error_model=error_record,
        intensity="signed profile amplitudes; physical/scaling factors explicit",
        aperture="unchanged finite-window counting baseline; frame curves belong to this baseline",
        uncertainty="conditional counting/background and strong-fit covariance; "
        "shared shape, sensor and scale errors excluded",
    )
    return ReductionResult(
        t,
        profile,
        corrected,
        statistical_sigma,
        parameters,
        np.asarray(choices).copy(),
        np.asarray(weak_mask).copy(),
        factor,
        empirical,
        metadata,
    )


def empirical_design(observations, geometry, recipe, corrected, flags):
    """Construct the shared model with explicit symmetry and fixed gauges."""
    from .empirical_scaling import ScalingDesign, paired_mask, panel_mesh, ray_harmonics

    operations = np.asarray(recipe.get("equivalence_operations"))
    provenance = recipe.get("symmetry_provenance")
    if (
        operations.ndim != 3
        or operations.shape[1:] != (3, 3)
        or operations.dtype.kind not in "iu"
        or not 1 <= len(operations) <= 96
        or not isinstance(provenance, str)
        or not provenance.strip()
        or np.any(abs(abs(np.linalg.det(operations)) - 1.0) > 1e-10)
    ):
        raise ValueError(
            "explicit integer intensity-equivalence operations and provenance required"
        )
    t = observations.summation
    # Label by lexicographically smallest explicitly equivalent Miller index;
    # Friedel/inversion equivalence must be explicitly included if intended.
    equivalent = t.hkl @ operations.transpose(0, 2, 1)
    canonical = np.array([min(map(tuple, equivalent[:, i])) for i in range(len(t))], np.int64)
    _, groups = np.unique(np.c_[t.grain_ids, canonical], axis=0, return_inverse=True)
    nf = len(t.omega_bounds)
    gauge = recipe.get("frame_gauge", nf // 2)
    if type(gauge) is not int or not 0 <= gauge < nf:
        raise ValueError("frame gauge outside scan")
    frame = np.argmin(abs(observations.centres[:, 2, None] - t.omega_bounds.mean(axis=1)), axis=1)
    frames = np.delete(np.eye(nf)[frame], gauge, axis=1)
    mesh = panel_mesh(
        observations.centres[:, :2],
        tuple(recipe["detector_shape"]),
        tuple(recipe.get("grid", [5, 5])),
    )
    node_gauge = recipe.get("node_gauge", mesh.shape[1] // 2)
    if type(node_gauge) is not int or not 0 <= node_gauge < mesh.shape[1]:
        raise ValueError("node gauge outside detector mesh")
    detector = np.delete(mesh, node_gauge, axis=1)
    ray = profile_geometry(observations.centres[:, :2], geometry)[0]
    axis = geometry.vectors[4] / np.linalg.norm(geometry.vectors[4])
    beam = geometry.vectors[3] / np.linalg.norm(geometry.vectors[3])
    theta = -np.deg2rad(observations.centres[:, 2])

    def rotate(v):
        return (
            np.cos(theta)[:, None] * v
            + np.sin(theta)[:, None] * np.cross(axis, v)
            + (1 - np.cos(theta))[:, None] * (v @ axis)[:, None] * axis
        )

    harmonics = ray_harmonics(rotate(ray)) + ray_harmonics(rotate(np.tile(-beam, (len(ray), 1))))
    p = frames.shape[1] + detector.shape[1] + 8
    if p > 256:
        raise ValueError(
            "scan exceeds empirical frame parameter budget; use a declared reduced design"
        )
    a, b = frames.shape[1], detector.shape[1]
    reg = []
    q = np.zeros((a, p))
    q[:, :a] = np.eye(a) / 0.25
    reg.append(q)
    full = np.zeros((nf, p))
    full[np.delete(np.arange(nf), gauge), np.arange(a)] = 1
    reg.append(np.diff(full, n=2, axis=0) / 0.10)
    q = np.zeros((b, p))
    q[:, a : a + b] = np.eye(b) / 0.15
    reg.append(q)
    _, sv, vt = np.linalg.svd(harmonics / np.sqrt(len(ray)), full_matrices=False)
    q = np.zeros((len(sv), p))
    q[:, a + b :] = sv[:, None] * vt / 0.25
    reg.append(q)
    design = ScalingDesign(
        frames,
        detector,
        harmonics,
        np.concatenate(reg),
        np.r_[np.full(a, -np.log(2)), np.full(b, -0.5), np.full(8, -1.0)],
        np.r_[np.full(a, np.log(2)), np.full(b, 1.0), np.full(8, 1.0)],
    )
    intensity, sigma = corrected.T
    snr = recipe.get("reference_snr", 10.0)
    if not np.isscalar(snr) or not np.isfinite(snr) or snr < 3:
        raise ValueError("invalid empirical reference SNR")
    eligible = (flags == 0) & np.isfinite(intensity) & (sigma > 0) & (intensity > snr * sigma)
    # Explicit training families, otherwise deterministic complete-family split.
    # The withheld fifth is preserved for independent fit diagnostics.
    folds = np.array(
        [
            int(hashlib.sha256(np.asarray(h, "<i8").tobytes()).hexdigest()[:8], 16) % 5
            for h in canonical
        ]
    )
    training = paired_mask(groups, eligible & (folds != 0))
    floor = recipe.get("relative_variance_floor", 0.02)
    if not np.isscalar(floor) or not np.isfinite(floor) or not 0 <= floor <= 1:
        raise ValueError("invalid relative variance floor for scaling references")
    variance = np.ones(len(t))
    variance[eligible] = (sigma[eligible] / intensity[eligible]) ** 2 + floor**2
    return design, groups, training, variance


def main():
    import argparse

    from .reduction_io import load_case

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    observations, model = load_case(args.case)
    config = ReductionConfig.from_dict(cfg["reduction"])
    geometry = None
    if "geometry_vectors" in cfg:
        from .reciprocal import PanelGeometry

        geometry = PanelGeometry(np.asarray(cfg["geometry_vectors"]), cfg["wavelength_angstrom"])
    result = reduce_observations(
        observations,
        tuple(cfg["detector_shape"]),
        config,
        geometry=geometry,
        model=model,
        cache=args.cache,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "intensities.npz",
        hkl=result.aperture.hkl,
        intensity_sigma=result.intensity_sigma,
        profile_values=result.profile.values,
        profile_flags=result.profile.flags,
        statistical_sigma=result.statistical_sigma,
        aperture_values=result.aperture.values,
        choices=result.choices,
        parameters=result.parameters,
        weak_mask=result.weak_mask,
        physical_factor=result.physical_factor,
        empirical_correction=result.empirical_correction,
    )
    (args.output / "report.json").write_text(
        json.dumps(result.metadata, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
