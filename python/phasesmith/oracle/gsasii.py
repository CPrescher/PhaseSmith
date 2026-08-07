"""Public GSASIIscriptable adapter.

GSAS-II is imported only inside adapter calls, so it is never required by the
normal package. Returned values are copies and no GSAS-II wrapper escapes this
module.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ReflectionTable:
    """One phase's reflection list and stable public metadata."""

    phase: str
    ref_list: NDArray[np.float64]
    histogram_type: str
    superspace: bool


@dataclass(frozen=True, slots=True)
class OracleSnapshot:
    """Plain arrays extracted from one GSAS-II powder histogram."""

    histogram_name: str
    x: NDArray[np.float64]
    ycalc: NDArray[np.float64]
    background: NDArray[np.float64]
    reflections: tuple[ReflectionTable, ...]


def _import_gsasii(gsas_root: str | Path | None) -> ModuleType:
    if gsas_root is not None:
        root = str(Path(gsas_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    try:
        from GSASII import GSASIIscriptable as scripting
    except ImportError:
        try:
            import G2script as scripting
        except ImportError as error:
            raise RuntimeError(
                "GSAS-II scripting is unavailable; pass gsas_root or install its G2script shortcut"
            ) from error
    return scripting


def _plain_array(values: Any) -> NDArray[np.float64]:
    """Copy a normal or masked GSAS-II array into ordinary float64 storage."""

    return np.ascontiguousarray(np.ma.filled(values, np.nan), dtype=np.float64)


def extract_snapshot(histogram: Any) -> OracleSnapshot:
    """Extract the public validation surface from a ``G2PwdrData`` wrapper."""

    reflection_tables = []
    for phase, payload in sorted(histogram.reflections().items()):
        reflection_tables.append(
            ReflectionTable(
                phase=str(phase),
                ref_list=_plain_array(payload["RefList"]),
                histogram_type=str(payload.get("Type", "")),
                superspace=bool(payload.get("Super", False)),
            )
        )
    return OracleSnapshot(
        histogram_name=str(histogram.name),
        x=_plain_array(histogram.getdata("X")),
        ycalc=_plain_array(histogram.getdata("Ycalc")),
        background=_plain_array(histogram.getdata("Background")),
        reflections=tuple(reflection_tables),
    )


def extract_project(
    project_file: str | Path,
    *,
    histogram: int | str = 0,
    gsas_root: str | Path | None = None,
) -> OracleSnapshot:
    """Open a GPX project and extract one powder histogram via public APIs."""

    scripting = _import_gsasii(gsas_root)
    project = scripting.G2Project(str(Path(project_file).resolve()))
    return extract_snapshot(project.histogram(histogram))
