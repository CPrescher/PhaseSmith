"""Lossless native mixed-method project bundles with cell-only Pawley analyses."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from . import _core
from .radiation import RadiationProbe
from .refinement.pawley import PawleyOptions, PawleyProject, _decode_input, _input, _record
from .refinement.tof_pawley import (
    TofPawleyProject,
    _bound_record,
)
from .refinement.tof_pawley import (
    _decode_input as _decode_tof_input,
)

__all__ = ["ProjectBundle"]


def _serialized(project: PawleyProject) -> str:
    if not isinstance(project, PawleyProject):
        raise TypeError("project must be PawleyProject")
    record = _core._pawley_prepare(_record(_input(project.input), project.options))
    if project.checkpoint is None:
        return record
    saved = json.loads(_core._pawley_prepare(project.checkpoint))
    current = json.loads(record)
    if saved["input"] != current["input"] or saved["options"] != current["options"]:
        raise ValueError("stale Pawley project checkpoint")
    return project.checkpoint


@dataclass(frozen=True, slots=True)
class ProjectBundle:
    """Immutable native snapshot preserving every supported analysis family.

    Use ``pawley()`` to obtain an independently editable analysis, then
    ``with_pawley()`` to insert its updated accepted state into a new snapshot.
    Existing Rietveld and TOF analyses remain intact, including their checkpoints.
    """

    _native: object

    @classmethod
    def load(cls, path: str | Path) -> ProjectBundle:
        """Load native bundle formats 1-7 through the bounded Rust codec."""
        return cls(_core._ProjectBundle.load(str(path)))

    @classmethod
    def from_pawley(
        cls,
        project: PawleyProject,
        *,
        probe: RadiationProbe,
        project_id: str = "project",
        histogram_id: str = "histogram",
        name: str = "Pawley analysis",
    ) -> ProjectBundle:
        """Create a shared cell-only project without inventing structural atoms.

        Radiation metadata is required explicitly because free family areas do
        not distinguish X-ray and neutron experiments by themselves.
        """
        if not isinstance(probe, RadiationProbe):
            raise TypeError("probe must be RadiationProbe")
        return cls(
            _core._ProjectBundle.from_pawley(
                _serialized(project), str(probe), project_id, histogram_id, name
            )
        )

    @classmethod
    def from_tof_pawley(
        cls,
        project: TofPawleyProject,
        *,
        project_id: str = "project",
        analysis_id: str = "tof-pawley",
        name: str = "TOF Pawley analysis",
    ) -> ProjectBundle:
        """Create a shared bundle with one density histogram for each participating bank."""
        if not isinstance(project, TofPawleyProject):
            raise TypeError("project must be TofPawleyProject")
        record = _bound_record(project.input, project.options, project.checkpoint)
        return cls(_core._ProjectBundle.from_tof_pawley(record, project_id, analysis_id, name))

    @property
    def tof_pawley_analyses(self) -> tuple[str, ...]:
        """Joint TOF analysis IDs in stored order."""
        return tuple(self._native.tof_pawley_analyses)

    def tof_pawley(self, analysis_id: str) -> TofPawleyProject:
        """Obtain an independently editable, atomically resumable multi-bank analysis."""
        record = self._native.tof_pawley(analysis_id)
        wire = json.loads(record)
        return TofPawleyProject(
            _decode_tof_input(wire["input"]),
            PawleyOptions(**wire["options"]),
            record if wire["checkpoint"] is not None else None,
        )

    def with_tof_pawley(self, analysis_id: str, project: TofPawleyProject) -> ProjectBundle:
        """Insert a complete joint analysis after checking all shared histogram owners."""
        if not isinstance(project, TofPawleyProject):
            raise TypeError("project must be TofPawleyProject")
        record = _bound_record(project.input, project.options, project.checkpoint)
        return ProjectBundle(self._native.with_tof_pawley(analysis_id, record))

    @property
    def pawley_histograms(self) -> tuple[str, ...]:
        """Histogram IDs owning Pawley analyses, in stable stored order."""
        return tuple(self._native.pawley_histograms)

    @property
    def analysis_counts(self) -> Mapping[str, int]:
        """Counts of all retained methods, including non-Pawley analyses."""
        return MappingProxyType(self._native.analysis_counts)

    def pawley(self, histogram_id: str) -> PawleyProject:
        """Return an independently editable, resumable Pawley analysis."""
        record = self._native.pawley(histogram_id)
        wire = json.loads(record)
        return PawleyProject(
            _decode_input(wire["input"]),
            PawleyOptions(**wire["options"]),
            record if wire["checkpoint"] is not None else None,
        )

    def with_pawley(self, histogram_id: str, project: PawleyProject) -> ProjectBundle:
        """Insert or replace one analysis after validating shared histogram state."""
        return ProjectBundle(self._native.with_pawley(histogram_id, _serialized(project)))

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Save format 7 atomically; preserve unrelated files in the directory."""
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be boolean")
        self._native.save(str(path), overwrite)
