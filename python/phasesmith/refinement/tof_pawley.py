"""Native single-/multi-bank TOF Pawley least squares in microsecond density units.

Cells are shared by phase identity. Areas, calibration, profiles and backgrounds
are bank-local; no incident-spectrum or structural amplitude is reapplied.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .. import _core
from ..control import CancellationToken
from ..crystallography import UnitCell
from ..instrument import TofInstrument
from ..pattern import TofPowderPattern
from ..symmetry import SpaceGroup, SymmetryOperation
from .core import (
    AffineConstraint,
    Constraint,
    FixedConstraint,
    LinearConstraint,
    ParameterKey,
    ParameterSet,
)
from .lattice import LatticeParameterBounds, LatticeParameterization
from .pawley import (
    PawleyCalculation,
    PawleyOptions,
    PawleyResult,
    _array,
    _calculation,
    _constraint,
    _key,
    _parameters,
    _result,
)
from .tof_multibank import TofSharedLatticePhase

__all__ = [
    "TofPawleyBackground",
    "TofPawleyBank",
    "TofPawleyInput",
    "TofPawleyPhase",
    "TofPawleyProject",
    "build_parameter_set",
    "calculate",
    "refine",
]
_DEFAULT_OPTIONS = PawleyOptions()


@dataclass(frozen=True, slots=True)
class TofPawleyPhase:
    """Ordered fixed families; HKLs use a matching shared cell when supplied."""

    phase_id: str
    reflection_ids: tuple[str, ...]
    d_spacing_angstrom: NDArray[np.float64]
    intensities: NDArray[np.float64]
    hkl: NDArray[np.int64] | None = None

    def __post_init__(self):
        object.__setattr__(self, "reflection_ids", tuple(self.reflection_ids))
        for name in ("d_spacing_angstrom", "intensities"):
            object.__setattr__(self, name, _array(getattr(self, name), name))
        if self.hkl is not None:
            hkl = _array(self.hkl, "hkl", np.int64, 2)
            if hkl.shape != (len(self.intensities), 3):
                raise ValueError("HKLs must have shape (families, 3)")
            if np.any(hkl < np.iinfo(np.int32).min) or np.any(hkl > np.iinfo(np.int32).max):
                raise ValueError("HKLs exceed native int32 range")
            object.__setattr__(self, "hkl", hkl)
        if (
            len(self.reflection_ids) != len(self.intensities)
            or self.d_spacing_angstrom.shape != self.intensities.shape
        ):
            raise ValueError("TOF family shape mismatch")


@dataclass(frozen=True, slots=True)
class TofPawleyBackground:
    """Linear Chebyshev coefficients with an explicit microsecond domain."""

    coefficients: tuple[float, ...]
    domain_us: tuple[float, float]
    background_id: str = "background"

    def __post_init__(self):
        object.__setattr__(self, "coefficients", tuple(self.coefficients))
        object.__setattr__(self, "domain_us", tuple(self.domain_us))


@dataclass(frozen=True, slots=True)
class TofPawleyBank:
    """One density observation bank with explicit normalization provenance.

    Bin-count data must be converted to densities with matching uncertainties
    before construction. The normalization text records that external step.
    """

    bank_id: str
    pattern: TofPowderPattern
    instrument: TofInstrument
    phases: tuple[TofPawleyPhase, ...]
    normalization: str
    background: TofPawleyBackground | None = None

    def __post_init__(self):
        object.__setattr__(self, "phases", tuple(self.phases))
        if not isinstance(self.pattern, TofPowderPattern):
            raise TypeError("pattern must be TofPowderPattern")
        if not isinstance(self.instrument, TofInstrument):
            raise TypeError("instrument must be TofInstrument")


@dataclass(frozen=True, slots=True)
class TofPawleyInput:
    """One atomic joint objective, with cells shared by stable phase identity.

    Supplied HKL lists must include the desired profile-tail and allowed-cell
    margins; topology is fixed throughout all trials and restarts.
    """

    banks: tuple[TofPawleyBank, ...]
    shared_lattice: tuple[TofSharedLatticePhase, ...] = ()
    signed_intensities: bool = False
    tail_log: float = 20.0
    parameters: ParameterSet | None = None
    constraints: tuple[Constraint, ...] = ()

    def __post_init__(self):
        for name in ("banks", "shared_lattice", "constraints"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        record = json.loads(_core._tof_pawley_prepare(_record(self, _DEFAULT_OPTIONS)))
        if self.parameters is None:
            object.__setattr__(self, "parameters", _parameters(record["input"]["parameters"]))

    @property
    def sample_offsets(self) -> tuple[int, ...]:
        """Boundaries for splitting concatenated calculated/residual sample arrays."""
        return tuple(np.cumsum([0] + [len(b.pattern.tof_us) for b in self.banks]).tolist())


def build_parameter_set(
    request: TofPawleyInput,
    *,
    profile_parameters: dict[str, tuple[str, ...]] | None = None,
    lattice: bool = False,
) -> ParameterSet:
    """Select bank-local native coefficient names and optional shared cells.

    Names are zero/difc/difa/difb/alpha/beta0/beta1/betaq/sigma0/sigma1/
    sigma2/sigmaq/x/y/z. At least one bank DIFC must stay fixed when cell
    lengths vary. Set physical bounds through the returned ParameterSet.
    """
    selected = profile_parameters or {}
    known = {
        (s.key.owner_id, s.key.name)
        for s in request.parameters.specs
        if s.key.module == "pawley_profile"
    }
    if any((bank, name) not in known for bank, names in selected.items() for name in names):
        raise ValueError("unknown TOF bank/profile parameter")
    return ParameterSet(
        tuple(
            replace(
                s,
                refine=(
                    s.key.module in ("pawley_intensity", "pawley_background")
                    or (lattice and s.key.module == "pawley_lattice")
                    or (
                        s.key.module == "pawley_profile"
                        and s.key.name in selected.get(s.key.owner_id, ())
                    )
                ),
            )
            for s in request.parameters.specs
        )
    )


def calculate(
    request: TofPawleyInput, options: PawleyOptions = _DEFAULT_OPTIONS
) -> PawleyCalculation:
    """Evaluate concatenated bank values/products; positions are microseconds."""
    return _calculation(_core._tof_pawley_calculate(_record(request, options)))


def refine(
    request: TofPawleyInput,
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
    """Bounded joint native fit; each accepted checkpoint contains every bank."""
    record = _bound_record(request, options, checkpoint)
    raw = _core._tof_pawley_refine(
        record,
        max_iterations,
        max_evaluations,
        max_rejections,
        max_seconds,
        None if cancellation is None else cancellation._native,
        progress,
    )
    return _result(request, raw)


def _bound_record(request, options, checkpoint):
    record = _core._tof_pawley_prepare(_record(request, options))
    if checkpoint is None:
        return record
    saved = json.loads(_core._tof_pawley_prepare(checkpoint))
    current = json.loads(record)
    if saved["input"] != current["input"] or saved["options"] != current["options"]:
        raise ValueError("stale TOF Pawley checkpoint")
    return checkpoint


@dataclass(slots=True)
class TofPawleyProject:
    """Versioned native joint state with atomic exclusive saves and exact resume."""

    input: TofPawleyInput
    options: PawleyOptions = _DEFAULT_OPTIONS
    checkpoint: str | None = None
    _cancellation: CancellationToken = field(
        default_factory=CancellationToken, init=False, repr=False
    )

    def calculate(self) -> PawleyCalculation:
        """Evaluate the last accepted state, rejecting stale project edits."""
        return _calculation(
            _core._tof_pawley_calculate(_bound_record(self.input, self.options, self.checkpoint))
        )

    def stop(self) -> bool:
        """Request cancellation at a native safe boundary."""
        return self._cancellation.request()

    def refine(self, **runtime) -> PawleyResult:
        """Refine/resume and retain the last jointly accepted state."""
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
        """Atomically save a new native JSON record without overwriting files."""
        _core._tof_pawley_save(_bound_record(self.input, self.options, self.checkpoint), str(path))

    @classmethod
    def load(cls, path: str | Path, *, max_bytes: int = 256 * 1024 * 1024) -> TofPawleyProject:
        """Bounded load with schema, scientific-state and checkpoint-digest checks."""
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        with Path(path).open("rb") as stream:
            data = stream.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("TOF Pawley project byte limit exceeded")
        record = _core._tof_pawley_prepare(data.decode("utf-8"))
        w = json.loads(record)
        return cls(
            _decode_input(w["input"]),
            PawleyOptions(**w["options"]),
            record if w["checkpoint"] is not None else None,
        )


def _record(request, options):
    def optional(v):
        return None if v is None else v.tolist()

    specs = (
        None
        if request.parameters is None
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
            for s in request.parameters.specs
        ]
    )
    banks = []
    for b in request.banks:
        p = b.pattern
        bg = b.background
        banks.append(
            dict(
                id=b.bank_id,
                tof_us=p.tof_us.tolist(),
                observed_y=optional(p.observed_y),
                uncertainty=optional(p.uncertainty),
                mask=optional(p.mask),
                background_y=p.background.tolist(),
                instrument=list(b.instrument.as_tuple()),
                phases=[
                    dict(
                        id=v.phase_id,
                        reflection_ids=list(v.reflection_ids),
                        hkl=[] if v.hkl is None else v.hkl.tolist(),
                        d_spacing_angstrom=v.d_spacing_angstrom.tolist(),
                        intensities=v.intensities.tolist(),
                    )
                    for v in b.phases
                ],
                background=None
                if bg is None
                else dict(
                    id=bg.background_id,
                    coefficients=list(bg.coefficients),
                    domain_us=list(bg.domain_us),
                ),
                normalization=b.normalization,
            )
        )
    cells = [
        dict(
            id=c.phase_id,
            cell=list(c.initial_cell.as_tuple()),
            reference_cell=list(
                c.parameterization.to_cell(c.parameterization.reference_values).as_tuple()
            ),
            operations=[
                dict(
                    rotation=op.rotation.tolist(),
                    translation=[[v.numerator, v.denominator] for v in op.translation],
                )
                for op in c.parameterization.space_group.operations
            ],
            lower=c.bounds.lower.tolist(),
            upper=c.bounds.upper.tolist(),
        )
        for c in request.shared_lattice
    ]
    return json.dumps(
        dict(
            format="phasesmith-tof-pawley",
            version=1,
            input=dict(
                banks=banks,
                shared_lattice=cells,
                signed_intensities=request.signed_intensities,
                tail_log=request.tail_log,
                parameters=specs,
                constraints=[_constraint(c) for c in request.constraints],
            ),
            options=asdict(options),
            checkpoint=None,
        ),
        allow_nan=False,
    )


def _decode_input(i):
    banks = []
    for b in i["banks"]:
        bg = b["background"]
        banks.append(
            TofPawleyBank(
                b["id"],
                TofPowderPattern(
                    b["tof_us"],
                    observed_y=b["observed_y"],
                    uncertainty=b["uncertainty"],
                    mask=b["mask"],
                    background=b["background_y"],
                ),
                TofInstrument(*b["instrument"]),
                tuple(
                    TofPawleyPhase(
                        p["id"],
                        tuple(p["reflection_ids"]),
                        p["d_spacing_angstrom"],
                        p["intensities"],
                        None if not p["hkl"] else np.asarray(p["hkl"], dtype=np.int64),
                    )
                    for p in b["phases"]
                ),
                b["normalization"],
                None
                if bg is None
                else TofPawleyBackground(
                    tuple(bg["coefficients"]), tuple(bg["domain_us"]), bg["id"]
                ),
            )
        )
    cells = []
    for c in i["shared_lattice"]:
        cell = UnitCell(*c["cell"])
        group = SpaceGroup(
            tuple(
                SymmetryOperation(op["rotation"], [Fraction(*v) for v in op["translation"]])
                for op in c["operations"]
            )
        )
        par = LatticeParameterization(group, UnitCell(*c["reference_cell"]))
        cells.append(
            TofSharedLatticePhase(
                c["id"], par, LatticeParameterBounds(par, c["lower"], c["upper"]), cell
            )
        )
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
    return TofPawleyInput(
        tuple(banks),
        tuple(cells),
        i["signed_intensities"],
        i["tail_log"],
        _parameters(i["parameters"]),
        tuple(constraints),
    )
