"""Native CW Pawley least squares with explicit area and overlap conventions.

Areas are independent powder-family integrated intensities, not structure factors.
See https://phasesmith.readthedocs.io/en/latest/pawley/ for supported boundaries.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from .. import _core
from ..control import CancellationToken
from ..crystallography import UnitCell
from ..instrument import ConstantWavelengthInstrument, FcjGeometry
from ..pattern import PowderPattern
from ..symmetry import SpaceGroup, SymmetryOperation
from .background import (
    ChebyshevBackground,
    CompositeBackground,
    DifferentiableBackground,
    PointBackground,
    PolynomialBackground,
)
from .core import (
    AffineConstraint,
    Bounds,
    Constraint,
    ConstraintTransform,
    FixedConstraint,
    LinearConstraint,
    ParameterKey,
    ParameterSet,
    ParameterSpec,
)
from .lattice import CwLatticeReflectionDomain, LatticeParameterBounds, LatticeParameterization

__all__ = [
    "PawleyCalculation",
    "PawleyInput",
    "PawleyOptions",
    "PawleyPhase",
    "PawleyProject",
    "PawleyResult",
    "build_parameter_set",
    "calculate",
    "parameter_key",
    "refine",
]


def _array(value, name, dtype=np.float64, ndim=1):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise TypeError(f"{name} must be real-valued")
    if dtype == np.int64 and not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    a = np.array(value, dtype=dtype, order="C", copy=True)
    if a.ndim != ndim or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    a.flags.writeable = False
    return a


def parameter_key(family: str, owner: str, name: str) -> ParameterKey:
    """Stable intensity/profile/background/lattice identity."""
    return ParameterKey(f"pawley_{family}", owner, name)


@dataclass(frozen=True, slots=True)
class PawleyPhase:
    """Fixed ordered family list, optionally tied to bounded cell geometry."""

    phase_id: str
    reflection_ids: tuple[str, ...]
    two_theta_deg: NDArray[np.float64]
    intensities: NDArray[np.float64]
    hkl: NDArray[np.int64] | None = None
    lattice: CwLatticeReflectionDomain | None = None

    def __post_init__(self):
        ids = tuple(self.reflection_ids)
        if not self.phase_id or "/" in self.phase_id or len(set(ids)) != len(ids):
            raise ValueError("phase and reflection IDs must be nonempty and unique")
        if any(not v or "/" in v for v in ids):
            raise ValueError("invalid reflection identity")
        positions = _array(self.two_theta_deg, "two_theta_deg")
        areas = _array(self.intensities, "intensities")
        if len(ids) == 0 or positions.shape != areas.shape or len(ids) != len(areas):
            raise ValueError("reflection arrays must have matching nonempty shapes")
        if np.any((positions <= 0) | (positions >= 180)):
            raise ValueError("positions must lie inside (0, 180) degrees")
        object.__setattr__(self, "reflection_ids", ids)
        object.__setattr__(self, "two_theta_deg", positions)
        object.__setattr__(self, "intensities", areas)
        if self.hkl is not None:
            hkl = _array(self.hkl, "hkl", np.int64, 2)
            if hkl.shape != (len(ids), 3):
                raise ValueError("hkl must have shape (reflections, 3)")
            object.__setattr__(self, "hkl", hkl)
        if self.lattice is not None and self.hkl is None:
            raise ValueError("lattice phases require HKLs")

    @classmethod
    def from_cif(
        cls,
        phase_id: str,
        path_or_text: str | Path,
        *,
        wavelength_angstrom: float,
        two_theta_range: tuple[float, float],
        bounds: LatticeParameterBounds | None = None,
        initial_intensity: float = 1.0,
    ) -> PawleyPhase:
        """Read cell/symmetry from CIF; atom coordinates do not determine areas."""
        from ..io import read_cif

        structure = read_cif(path_or_text).structure
        return cls.from_cell(
            phase_id,
            structure.cell,
            structure.space_group,
            wavelength_angstrom=wavelength_angstrom,
            two_theta_range=two_theta_range,
            bounds=bounds,
            initial_intensity=initial_intensity,
        )

    @classmethod
    def from_cell(
        cls,
        phase_id: str,
        cell: UnitCell,
        space_group: SpaceGroup,
        *,
        wavelength_angstrom: float,
        two_theta_range: tuple[float, float],
        bounds: LatticeParameterBounds | None = None,
        initial_intensity: float = 1.0,
    ) -> PawleyPhase:
        """Generate a bounded family superset in Rust; no atom model is needed.

        The range must include the desired profile-tail margin. Tighten lattice
        bounds explicitly when a broad default box would generate too many peaks.
        """
        par = LatticeParameterization(space_group, cell)
        bounds = bounds or LatticeParameterBounds.around(
            par, relative_length=0.01, angle_delta_deg=0.5
        )
        domain = CwLatticeReflectionDomain(
            space_group, par, bounds, wavelength_angstrom, *two_theta_range, initial_intensity
        )
        phase = {
            "id": phase_id,
            "reflection_ids": [],
            "two_theta_deg": [],
            "intensities": [],
            "hkl": [],
            "lattice": _domain(domain),
        }
        i = {
            "x_deg": list(two_theta_range),
            "observed_y": [0.0, 0.0],
            "uncertainty": None,
            "mask": None,
            "background_y": [0.0, 0.0],
            "instrument": [wavelength_angstrom, 0.0, 0.0, 0.001, 0.001, 0.0],
            "axial": None,
            "phases": [phase],
            "background": None,
            "signed_intensities": False,
            "parameters": None,
            "constraints": [],
        }
        record = _record(i, PawleyOptions())
        return _decode_phase(json.loads(_core._pawley_prepare(record))["input"]["phases"][0])


@dataclass(frozen=True, slots=True)
class PawleyOptions:
    """Scientific controls; dense allocation limit counts f64 elements, not bytes."""

    support_fwhm: float = 20.0
    use_uncertainty: bool = True
    max_elements: int = 50_000_000
    rank_tolerance: float = 1e-10
    tolerance: float = 1e-9
    damping: float = 1e-6
    max_active_iterations: int = 2000

    def __post_init__(self):
        controls = (self.support_fwhm, self.rank_tolerance, self.tolerance, self.damping)
        if not np.isfinite(controls).all() or min(controls) <= 0:
            raise ValueError("Pawley controls must be positive and finite")
        if self.rank_tolerance >= 1 or self.tolerance >= 1:
            raise ValueError("Pawley tolerances must be below one")
        for value in (self.max_elements, self.max_active_iterations):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("Pawley allocation/iteration limits must be positive integers")
        if not isinstance(self.use_uncertainty, bool):
            raise TypeError("use_uncertainty must be boolean")


_DEFAULT_OPTIONS = PawleyOptions()


@dataclass(frozen=True, slots=True)
class PawleyInput:
    """One CW histogram and ordered phases, with additive linear background."""

    pattern: PowderPattern
    instrument: ConstantWavelengthInstrument
    phases: tuple[PawleyPhase, ...]
    background: DifferentiableBackground | None = None
    axial_geometry: FcjGeometry | None = None
    signed_intensities: bool = False
    parameters: ParameterSet | None = None
    constraints: tuple[Constraint, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "phases", tuple(self.phases))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        if not self.phases:
            raise ValueError("Pawley requires at least one phase")
        # Native preparation validates the same scientific boundary used by Rust callers.
        prepared = json.loads(_core._pawley_prepare(_record(_input(self), PawleyOptions())))
        if self.parameters is None:
            object.__setattr__(self, "parameters", _parameters(prepared["input"]["parameters"]))


def build_parameter_set(
    request: PawleyInput, *, profile_parameters: tuple[str, ...] = (), lattice: bool = False
) -> ParameterSet:
    """Select CW coefficients and independent cells; areas/background remain selected.

    Customize individual bounds, selections or values using dataclasses.replace
    on returned ParameterSpec records. Constraints use existing refinement types.
    """
    names = {"u_deg2", "v_deg2", "w_deg2", "x_deg", "y_deg"}
    if not set(profile_parameters) <= names:
        raise ValueError("unsupported Pawley profile parameter")
    return ParameterSet(
        [
            replace(s, refine=s.key.name in profile_parameters)
            if s.key.module == "pawley_profile"
            else replace(s, refine=lattice)
            if s.key.module == "pawley_lattice"
            else s
            for s in request.parameters.specs
        ]
    )


@dataclass(frozen=True, slots=True)
class PawleyCalculation:
    """Immutable values and analytical Jacobian over scaled free coordinates."""

    calculated_y: NDArray[np.float64]
    background_y: NDArray[np.float64]
    intensities: NDArray[np.float64]
    positions: NDArray[np.float64]
    jacobian: NDArray[np.float64]
    residual: NDArray[np.float64]
    weighted_residual: NDArray[np.float64]
    included: NDArray[np.bool_]
    rp: float
    rwp: float
    chi_square: float
    reduced_chi_square: float
    inactive_columns: tuple[int, ...]
    unobserved_reflections: tuple[tuple[str, str], ...]
    coincident_groups: tuple[tuple[tuple[tuple[str, str], ...], float], ...]


@dataclass(frozen=True, slots=True)
class PawleyResult:
    """Accepted fit, explicit rank/bound diagnostics, and portable native checkpoint."""

    calculation: PawleyCalculation
    termination_reason: str
    rank: int
    observed_free_parameters: int
    active_bounds: tuple[int, ...]
    active_width_bounds: bool
    covariance: NDArray[np.float64] | None
    covariance_limitation: str | None
    history: NDArray[np.float64]
    free: NDArray[np.float64]
    checkpoint: str
    parameters: ParameterSet
    diagnostics: Mapping[str, float | int | str | None]

    @property
    def intensities(self):
        """Fitted family areas in input order."""
        return self.calculation.intensities


def _calculation(raw):
    for value in raw.values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    fields = PawleyCalculation.__dataclass_fields__
    values = {key: raw[key] for key in fields}
    values["inactive_columns"] = tuple(values["inactive_columns"])
    values["unobserved_reflections"] = tuple(values["unobserved_reflections"])
    values["coincident_groups"] = tuple(
        (tuple(members), area) for members, area in values["coincident_groups"]
    )
    return PawleyCalculation(**values)


def calculate(request: PawleyInput, options: PawleyOptions = _DEFAULT_OPTIONS) -> PawleyCalculation:
    """Evaluate the input through the native fused value/derivative pass."""
    return _calculation(_core._pawley_calculate(_record(_input(request), options)))


def refine(
    request: PawleyInput,
    options: PawleyOptions = _DEFAULT_OPTIONS,
    *,
    checkpoint: str | None = None,
    max_iterations: int = 100,
    max_evaluations: int = 1000,
    max_rejections: int = 20,
    max_seconds: float | None = None,
    cancellation: CancellationToken | None = None,
    progress: Callable[[dict], None] | None = None,
) -> PawleyResult:
    """Run native bounded joint least squares with the GIL released.

    Iteration ceilings are cumulative across accepted-state restarts. Runtime
    evaluation/time budgets apply to this invocation. Progress receives native
    boundary events; cancellation may be requested from another Python thread.
    """
    record = _core._pawley_prepare(_record(_input(request), options))
    if checkpoint is not None:
        saved = json.loads(_core._pawley_prepare(checkpoint))
        current = json.loads(record)
        if saved["input"] != current["input"] or saved["options"] != current["options"]:
            raise ValueError("Pawley checkpoint does not match request/options")
        record = checkpoint
    raw = _core._pawley_refine(
        record,
        max_iterations,
        max_evaluations,
        max_rejections,
        max_seconds,
        None if cancellation is None else cancellation._native,
        progress,
    )
    calculation = _calculation(raw)
    transform = ConstraintTransform(request.parameters, request.constraints)
    fitted = request.parameters.replace_values(transform.unpack(raw["free"]))
    return PawleyResult(
        calculation,
        raw["termination_reason"],
        raw["rank"],
        raw["observed_free_parameters"],
        tuple(raw["active_bounds"]),
        raw["active_width_bounds"],
        raw["covariance"],
        raw["covariance_limitation"],
        raw["history"],
        raw["free"],
        raw["checkpoint"],
        fitted,
        MappingProxyType(raw["diagnostics"]),
    )


@dataclass(slots=True)
class PawleyProject:
    """Standalone versioned native project, separate from structural project bundles."""

    input: PawleyInput
    options: PawleyOptions = _DEFAULT_OPTIONS
    checkpoint: str | None = None
    _cancellation: CancellationToken = field(
        default_factory=CancellationToken, init=False, repr=False
    )

    def calculate(self) -> PawleyCalculation:
        """Evaluate the accepted checkpoint when present, otherwise the input."""
        if self.checkpoint is None:
            return calculate(self.input, self.options)
        saved = json.loads(_core._pawley_prepare(self.checkpoint))
        current = json.loads(_core._pawley_prepare(_record(_input(self.input), self.options)))
        if saved["input"] != current["input"] or saved["options"] != current["options"]:
            raise ValueError("stale Pawley project checkpoint")
        return _calculation(_core._pawley_calculate(self.checkpoint))

    def stop(self) -> bool:
        """Request cooperative cancellation of this project's active native solve."""
        return self._cancellation.request()

    def refine(self, **runtime) -> PawleyResult:
        """Refine or resume, retaining the last returned accepted checkpoint."""
        self._cancellation = runtime.pop("cancellation", None) or CancellationToken()
        result = refine(
            self.input,
            self.options,
            checkpoint=self.checkpoint,
            cancellation=self._cancellation,
            **runtime,
        )
        self.checkpoint = result.checkpoint
        return result

    def save(self, path: str | Path) -> None:
        """Save a new finite JSON project; existing files are never overwritten."""
        record = _core._pawley_prepare(_record(_input(self.input), self.options))
        if self.checkpoint is not None:
            saved = json.loads(_core._pawley_prepare(self.checkpoint))
            current = json.loads(record)
            if saved["input"] != current["input"] or saved["options"] != current["options"]:
                raise ValueError("stale Pawley project checkpoint")
            record = self.checkpoint
        with Path(path).open("x", encoding="utf-8") as stream:
            stream.write(record)

    @classmethod
    def load(cls, path: str | Path, *, max_bytes: int = 256 * 1024 * 1024) -> PawleyProject:
        """Validate file size, version, scientific state and checkpoint digest."""
        if not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        with Path(path).open("rb") as stream:
            data = stream.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("Pawley project byte limit exceeded")
        record = _core._pawley_prepare(data.decode("utf-8"))
        w = json.loads(record)
        return cls(
            _decode_input(w["input"]),
            PawleyOptions(**w["options"]),
            record if w["checkpoint"] is not None else None,
        )


def _key(k):
    return [k.module, k.owner_id, k.name]


def _record(i, options):
    return json.dumps(
        {
            "format": "phasesmith-pawley",
            "version": 1,
            "input": i,
            "options": asdict(options),
            "checkpoint": None,
        },
        allow_nan=False,
    )


def _domain(d):
    if d is None:
        return None
    cell = d.parameterization.to_cell(d.parameterization.reference_values)
    return {
        "cell": list(cell.as_tuple()),
        "operations": [
            {
                "rotation": op.rotation.tolist(),
                "translation": [[v.numerator, v.denominator] for v in op.translation],
            }
            for op in d.space_group.operations
        ],
        "lower": d.bounds.lower.tolist(),
        "upper": d.bounds.upper.tolist(),
        "wavelength_angstrom": d.wavelength_angstrom,
        "visible_two_theta_deg": [d.visible_two_theta_min_deg, d.visible_two_theta_max_deg],
        "initial_intensity": d.initial_intensity,
        "merge_friedel": d.merge_friedel,
        "max_candidates": d.max_candidates,
        "guard_scale": d.guard_scale,
    }


def _background(b):
    if b is None:
        return None
    w = {"id": b.background_id}
    if isinstance(b, CompositeBackground):
        return dict(w, type="composite", components=[_background(v) for v in b.components])
    if isinstance(b, PointBackground):
        return dict(w, type="point", knot_x=list(b.knot_x), values=list(b.values))
    if isinstance(b, ChebyshevBackground):
        return dict(
            w, type="chebyshev", coefficients=list(b.coefficients), domain_deg=list(b.domain_deg)
        )
    if isinstance(b, PolynomialBackground):
        return dict(w, type="polynomial", coefficients=list(b.coefficients))
    raise TypeError("Pawley supports native coefficient-invariant backgrounds")


def _constraint(c):
    w = {"target": _key(c.target)}
    if isinstance(c, FixedConstraint):
        return dict(w, type="fixed", value=c.value)
    if isinstance(c, AffineConstraint):
        return dict(
            w, type="affine", source=_key(c.source), multiplier=c.multiplier, offset=c.offset
        )
    if isinstance(c, LinearConstraint):
        return dict(w, type="linear", terms=[[_key(k), v] for k, v in c.terms], offset=c.offset)
    raise TypeError("unsupported constraint")


def _input(r):
    p, i, a = r.pattern, r.instrument, r.axial_geometry

    def optional(v):
        return None if v is None else v.tolist()

    specs = (
        None
        if r.parameters is None
        else [
            dict(
                key=_key(s.key),
                value=s.value,
                unit=s.unit,
                lower=s.bounds.lower if np.isfinite(s.bounds.lower) else None,
                upper=s.bounds.upper if np.isfinite(s.bounds.upper) else None,
                scale=s.scale,
                refine=s.refine,
            )
            for s in r.parameters.specs
        ]
    )
    return dict(
        x_deg=p.x.tolist(),
        observed_y=optional(p.observed_y),
        uncertainty=optional(p.uncertainty),
        mask=optional(p.mask),
        background_y=p.background.tolist(),
        instrument=[i.wavelength_angstrom, i.u_deg2, i.v_deg2, i.w_deg2, i.x_deg, i.y_deg],
        axial=None if a is None else [a.sample_over_radius, a.detector_over_radius],
        phases=[
            dict(
                id=v.phase_id,
                reflection_ids=list(v.reflection_ids),
                two_theta_deg=v.two_theta_deg.tolist(),
                intensities=v.intensities.tolist(),
                hkl=[] if v.hkl is None else v.hkl.tolist(),
                lattice=_domain(v.lattice),
            )
            for v in r.phases
        ],
        background=_background(r.background),
        signed_intensities=r.signed_intensities,
        parameters=specs,
        constraints=[_constraint(c) for c in r.constraints],
    )


def _parameters(specs):
    return ParameterSet(
        [
            ParameterSpec(
                ParameterKey(*s["key"]),
                s["value"],
                s["unit"],
                Bounds(
                    -np.inf if s["lower"] is None else s["lower"],
                    np.inf if s["upper"] is None else s["upper"],
                ),
                s["scale"],
                s["refine"],
            )
            for s in specs
        ]
    )


def _decode_phase(p):
    w = p["lattice"]
    domain = None
    if w is not None:
        cell = UnitCell(*w["cell"])
        group = SpaceGroup(
            tuple(
                SymmetryOperation(v["rotation"], [Fraction(*t) for t in v["translation"]])
                for v in w["operations"]
            )
        )
        par = LatticeParameterization(group, cell)
        bounds = LatticeParameterBounds(par, w["lower"], w["upper"])
        domain = CwLatticeReflectionDomain(
            group,
            par,
            bounds,
            w["wavelength_angstrom"],
            *w["visible_two_theta_deg"],
            w["initial_intensity"],
            w["merge_friedel"],
            w["max_candidates"],
            w["guard_scale"],
        )
    return PawleyPhase(
        p["id"],
        tuple(p["reflection_ids"]),
        p["two_theta_deg"],
        p["intensities"],
        None if not p["hkl"] else np.asarray(p["hkl"], dtype=np.int64),
        domain,
    )


def _decode_background(b):
    if b is None:
        return None
    if b["type"] == "polynomial":
        return PolynomialBackground(b["id"], tuple(b["coefficients"]))
    if b["type"] == "chebyshev":
        return ChebyshevBackground(b["id"], tuple(b["coefficients"]), tuple(b["domain_deg"]))
    if b["type"] == "point":
        return PointBackground(b["id"], tuple(b["knot_x"]), tuple(b["values"]))
    return CompositeBackground(b["id"], tuple(_decode_background(v) for v in b["components"]))


def _decode_input(i):
    constraints = []
    for c in i["constraints"]:
        target = ParameterKey(*c["target"])
        if c["type"] == "fixed":
            constraints.append(FixedConstraint(target, c["value"]))
        elif c["type"] == "affine":
            constraints.append(
                AffineConstraint(target, ParameterKey(*c["source"]), c["multiplier"], c["offset"])
            )
        else:
            constraints.append(
                LinearConstraint(
                    target, tuple((ParameterKey(*k), v) for k, v in c["terms"]), c["offset"]
                )
            )
    return PawleyInput(
        PowderPattern(
            i["x_deg"],
            observed_y=i["observed_y"],
            uncertainty=i["uncertainty"],
            mask=i["mask"],
            background=i["background_y"],
        ),
        ConstantWavelengthInstrument(*i["instrument"]),
        tuple(_decode_phase(p) for p in i["phases"]),
        _decode_background(i["background"]),
        None if i["axial"] is None else FcjGeometry(*i["axial"]),
        i["signed_intensities"],
        _parameters(i["parameters"]),
        tuple(constraints),
    )
