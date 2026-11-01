"""Original HDF5 counts to structure-search candidates through the Rust coordinator.

The frozen v1 recipe requires a supplied primitive orthorhombic metric, EIGER
pixel-size metadata, PONI geometry and explicit spindle/DAC assumptions. Python
translates inputs and notifications only; all numerical processing and file
products are native. No standalone structure refinement is performed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.io.hdf5 import Hdf5Frames

from .solution import SolutionCancellation

__all__ = ["WorkflowConfig", "WorkflowResult", "available", "run_workflow"]


def available() -> bool:
    """Whether this ordinary build includes the complete native file workflow."""
    return hasattr(_core, "_run_single_crystal_workflow")


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    """Explicit physical inputs for the frozen orthorhombic HDF5 workflow v1.

    Angles are degrees, cell lengths angstrom. Frame indices are zero-based and
    inclusive; omega_zero describes original frame zero. Axes use the native
    laboratory frame. Output must be new and outside Git checkouts. PONI supplies
    wavelength; EIGER metadata supplies pixel sizes. Counts assume gain 1/read
    noise 0, with combined illumination/DAC masks and up to ten native workers.
    """

    raw_path: str
    poni_path: str
    output: str
    dataset: str
    cell: tuple[float, float, float, float, float, float]
    elements: tuple[str, ...]
    first: int
    last: int
    omega_zero: float
    omega_step: float
    exposure: float
    rotation_axis: tuple[float, float, float]
    dac_axis: tuple[float, float, float]
    dac_half_opening: float
    vertical_flip: bool
    effort: str = "auto"
    provenance: str = (
        "User-supplied cell/elements and explicitly declared acquisition/DAC geometry."
    )

    def __post_init__(self):
        for name in ("raw_path", "poni_path", "output", "dataset", "provenance"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "\0" in value:
                raise ValueError(f"{name} must be a nonempty string")
        cell = np.asarray(self.cell, dtype=np.float64)
        if cell.shape != (6,) or not np.isfinite(cell).all():
            raise ValueError("cell needs six finite parameters")
        UnitCell(*cell)
        if (cell[:3] <= 0).any() or not np.all(cell[3:] == 90):
            raise ValueError("workflow v1 requires a supplied orthorhombic cell (90 degree angles)")
        object.__setattr__(self, "cell", tuple(float(x) for x in cell))
        elements = tuple(self.elements)
        if (
            not 1 <= len(elements) <= 8
            or len(set(elements)) != len(elements)
            or any(not isinstance(s, str) or not re.fullmatch(r"[A-Z][a-z]?", s) for s in elements)
        ):
            raise ValueError("supply 1-8 distinct element symbols, without atom counts")
        object.__setattr__(self, "elements", elements)
        if any(type(v) is not int or v < 0 for v in (self.first, self.last)):
            raise ValueError("frame indices must be nonnegative integers")
        if not 5 <= self.last - self.first + 1 <= 225:
            raise ValueError("workflow v1 supports 5-225 inclusive exposures")
        for name in ("omega_zero", "omega_step", "exposure", "dac_half_opening"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.omega_step <= 0 or self.exposure <= 0 or not 0 < self.dac_half_opening < 90:
            raise ValueError("positive step/exposure and DAC half-opening in (0,90) required")
        for name in ("rotation_axis", "dac_axis"):
            axis = np.asarray(getattr(self, name), dtype=np.float64)
            if (
                axis.shape != (3,)
                or not np.isfinite(axis).all()
                or not np.isclose(np.linalg.norm(axis), 1.0, rtol=0, atol=1e-8)
            ):
                raise ValueError(f"{name} needs three finite components of a unit vector")
            object.__setattr__(self, name, tuple(float(x) for x in axis))
        if type(self.vertical_flip) is not bool or self.effort not in ("fast", "auto", "thorough"):
            raise ValueError("explicit orientation and fast/auto/thorough effort required")

    def to_record(self) -> dict:
        """Small portable recipe, containing no counts, caches or atomic reference."""
        return {"schema": "phasesmith.single-crystal-workflow-config.v1", **asdict(self)}

    @classmethod
    def from_record(cls, record: dict) -> WorkflowConfig:
        """Read only this versioned configuration; reject unexpected fields."""
        values = dict(record)
        if values.pop("schema", None) != "phasesmith.single-crystal-workflow-config.v1":
            raise ValueError("unsupported workflow configuration schema")
        return cls(**values)


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """Durable native products and candidate assessment; arrays remain on disk."""

    output: Path
    report: Mapping

    @property
    def accepted(self) -> bool:
        """Whether the unchanged search acceptance checks passed."""
        return self.report.get("accepted") is True

    @property
    def candidate_cif(self) -> Path | None:
        """Selected setting CIF, or full-cell P1 candidate when no setting is available."""
        for name in ("candidate/space-group.cif", "candidate/best.cif"):
            path = self.output / name
            if path.is_file():
                return path
        return None


def _packet(config: WorkflowConfig) -> dict:
    protocol = json.loads(_core._single_crystal_workflow_protocol())
    acquisition = protocol["acquisition"]
    acquisition.update(
        first_frame=config.first,
        last_frame_inclusive=config.last,
        omega_at_original_frame_zero_degrees=config.omega_zero,
        step_degrees=config.omega_step,
        exposure_width_degrees=config.exposure,
        axis=config.rotation_axis,
        vertical_flip=config.vertical_flip,
        angle_provenance=config.provenance,
    )
    protocol["shadow_mask"]["geometric"].update(
        axis_at_omega_zero=config.dac_axis, half_opening_degrees=config.dac_half_opening
    )
    protocol["space_group_search"]["effort"]["mode"] = config.effort
    raw_path = str(Path(config.raw_path).expanduser().resolve())
    with Hdf5Frames(raw_path, config.dataset) as frames:
        shape = frames.shape
        if config.last >= shape[0]:
            raise ValueError("selected exposures exceed the HDF5 scan")
    return dict(
        raw_path=raw_path,
        poni_path=str(Path(config.poni_path).expanduser().resolve()),
        raw_array=dict(dataset=config.dataset, shape=shape),
        priors=dict(
            cell_angstrom_degrees=config.cell,
            elements=config.elements,
            provenance=config.provenance,
        ),
        rotations=[],
        protocol=protocol,
        output=str(Path(config.output).expanduser().resolve()),
        through="search",
    )


def run_workflow(
    config: WorkflowConfig,
    *,
    cancellation: SolutionCancellation | None = None,
    progress: Callable[[str, float], None] | None = None,
) -> WorkflowResult:
    """Run the complete native workflow, releasing the GIL during numerical work.

    Completed-stage callbacks run on the caller's worker thread. Marshal GUI
    updates through signals. Cancellation is cooperative, joins bounded work,
    raises InterruptedError and retains already written products for diagnosis.
    A successful result may still be unaccepted; inspect ``accepted`` and the
    saved alternatives. Source files are hashed before and after processing.
    """
    if not available():
        raise ImportError("install a PhaseSmith build with native single-crystal workflow support")
    if not isinstance(config, WorkflowConfig):
        raise TypeError("config must be WorkflowConfig")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")
    token = SolutionCancellation() if cancellation is None else cancellation
    if not isinstance(token, SolutionCancellation):
        raise TypeError("cancellation must be SolutionCancellation")
    if token.is_cancelled:
        raise InterruptedError("single-crystal workflow cancelled")
    packet = _packet(config)
    report = json.loads(
        _core._run_single_crystal_workflow(
            json.dumps(packet, allow_nan=False), packet["output"], token._native, progress
        )
    )
    return WorkflowResult(Path(packet["output"]), MappingProxyType(report))
