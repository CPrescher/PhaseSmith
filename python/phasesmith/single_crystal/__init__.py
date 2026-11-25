"""Single-crystal diffraction: original frames through structure search.

Submodules load on demand. Native numerical kernels ship in ordinary builds;
HDF5 and shared crystallography types retain their existing adapter ownership.
See docs/single-crystal.md for supported methods, units and limitations.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import (
        cell_comparison,
        cell_metrics,
        empirical_scaling,
        frame_input,
        frame_validity,
        geometry_refinement,
        grains,
        indexing,
        integration,
        integration_quality,
        intensity_response,
        intensity_uncertainty,
        omega_background,
        orthorhombic,
        ownership,
        prediction,
        profile_refinement,
        profiles,
        reciprocal,
        recovery,
        reduction,
        reduction_cache,
        reduction_io,
        retention,
        rotation,
        rows,
        scan_scaling,
        search,
        sensor_offsets,
        solution,
        solution_symmetry,
        spots,
        workflow,
    )

__all__ = [
    "cell_comparison",
    "cell_metrics",
    "empirical_scaling",
    "frame_input",
    "frame_validity",
    "geometry_refinement",
    "grains",
    "indexing",
    "integration",
    "integration_quality",
    "intensity_response",
    "intensity_uncertainty",
    "omega_background",
    "orthorhombic",
    "ownership",
    "prediction",
    "profile_refinement",
    "profiles",
    "reciprocal",
    "recovery",
    "reduction",
    "reduction_cache",
    "reduction_io",
    "retention",
    "rotation",
    "rows",
    "scan_scaling",
    "search",
    "sensor_offsets",
    "solution",
    "solution_symmetry",
    "spots",
    "workflow",
]


def __getattr__(name: str) -> ModuleType:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = _import_module(f"{__name__}.{name}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
