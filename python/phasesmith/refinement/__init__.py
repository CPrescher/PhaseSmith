"""Shared refinement infrastructure and method-specific submodules.

Residual, constraint, solver, covariance, and QPA equations:
https://phasesmith.readthedocs.io/en/latest/mathematics/refinement/

Method-owned records and entry points live in :mod:`phasesmith.refinement.lebail`,
:mod:`phasesmith.refinement.rietveld`, and the named TOF/workflow modules.
"""

from __future__ import annotations

import importlib as _importlib
import warnings as _warnings

from . import lebail, readiness, rietveld, tof_lebail, tof_multibank, tof_structural, workflow
from .background import (
    AmorphousBackground,
    AmorphousPeak,
    ChebyshevBackground,
    CompositeBackground,
    DifferentiableBackground,
    PointBackground,
    PolynomialBackground,
)
from .core import (
    AffineConstraint,
    Bounds,
    ConstraintTransform,
    FixedConstraint,
    LeastSquaresOptimizer,
    LinearConstraint,
    ParameterKey,
    ParameterSet,
    ParameterSpec,
    ResidualEvaluation,
    ResidualOptions,
    TerminationReason,
    evaluate_residuals,
    jacobian_vector_product,
    transpose_jacobian_vector_product,
)
from .lattice import (
    CwLatticeReflectionDomain,
    CwStructuralReflectionDomain,
    GeneratedReflectionDomainResult,
    GeneratedStructuralReflectionDomainResult,
    LatticeParameterBounds,
    LatticeParameterization,
    LatticeReflectionGeometry,
    cw_lattice_geometry,
    tof_lattice_geometry,
)
from .runtime import (
    CheckpointCallback,
    CheckpointCallbackError,
    ConsoleRefinementLogger,
    JsonLinesRefinementLogger,
    RefinementEvent,
    RefinementEventKind,
    RefinementLimits,
    RefinementLogger,
    RefinementRuntime,
    RefinementStopped,
)
from .scipy import ScipyLeastSquaresAdapter

__all__ = [
    "AffineConstraint",
    "AmorphousBackground",
    "AmorphousPeak",
    "Bounds",
    "ChebyshevBackground",
    "CheckpointCallback",
    "CheckpointCallbackError",
    "CompositeBackground",
    "ConsoleRefinementLogger",
    "ConstraintTransform",
    "CwLatticeReflectionDomain",
    "CwStructuralReflectionDomain",
    "DifferentiableBackground",
    "FixedConstraint",
    "GeneratedReflectionDomainResult",
    "GeneratedStructuralReflectionDomainResult",
    "JsonLinesRefinementLogger",
    "LatticeParameterBounds",
    "LatticeParameterization",
    "LatticeReflectionGeometry",
    "LeastSquaresOptimizer",
    "LinearConstraint",
    "ParameterKey",
    "ParameterSet",
    "ParameterSpec",
    "PointBackground",
    "PolynomialBackground",
    "RefinementEvent",
    "RefinementEventKind",
    "RefinementLimits",
    "RefinementLogger",
    "RefinementRuntime",
    "RefinementStopped",
    "ResidualEvaluation",
    "ResidualOptions",
    "ScipyLeastSquaresAdapter",
    "TerminationReason",
    "cw_lattice_geometry",
    "evaluate_residuals",
    "jacobian_vector_product",
    "lebail",
    "readiness",
    "rietveld",
    "tof_lattice_geometry",
    "tof_lebail",
    "tof_multibank",
    "tof_structural",
    "transpose_jacobian_vector_product",
    "workflow",
]

_DEPRECATED_EXPORT_MODULES = {
    "TOF_INSTRUMENT_PARAMETERS": "tof_multibank",
    "CoincidentReflectionGroup": "lebail",
    "IntensityExtractionResult": "lebail",
    "IterationRecord": "lebail",
    "LeBailCheckpoint": "lebail",
    "LeBailInput": "lebail",
    "LeBailOptions": "lebail",
    "LeBailPhase": "lebail",
    "LeBailResult": "lebail",
    "ParameterChange": "lebail",
    "ReadinessSeverity": "readiness",
    "ReflectionIntensity": "lebail",
    "RietveldCalculationResult": "rietveld",
    "RietveldCheckpoint": "rietveld",
    "RietveldInput": "rietveld",
    "RietveldIterationRecord": "rietveld",
    "RietveldOptions": "rietveld",
    "RietveldParameterChange": "rietveld",
    "RietveldParameterCorrelation": "rietveld",
    "RietveldParameterSelection": "rietveld",
    "RietveldReadinessDiagnostic": "readiness",
    "RietveldReadinessReport": "readiness",
    "RietveldRecipe": "workflow",
    "RietveldResult": "rietveld",
    "RietveldStage": "workflow",
    "RietveldStageResult": "workflow",
    "RietveldWorkflowResult": "workflow",
    "SiteCoordinateModel": "rietveld",
    "StructuralTofBank": "tof_structural",
    "StructuralTofBankResult": "tof_structural",
    "StructuralTofCancellation": "tof_structural",
    "StructuralTofIteration": "tof_structural",
    "StructuralTofMultiBankCheckpoint": "tof_structural",
    "StructuralTofMultiBankInput": "tof_structural",
    "StructuralTofMultiBankProvenance": "tof_structural",
    "StructuralTofMultiBankResult": "tof_structural",
    "StructuralTofParameterChange": "tof_structural",
    "StructuralTofRefinementOptions": "tof_structural",
    "StructuralTofRequestProvenance": "tof_structural",
    "StructuralTofSelection": "tof_structural",
    "StructuralTofSourceDigest": "tof_structural",
    "TofBankInstrumentModel": "tof_multibank",
    "TofBankInstrumentState": "tof_multibank",
    "TofChebyshevBackground": "tof_lebail",
    "TofGeometryCorrelation": "tof_multibank",
    "TofGeometryDiagnostics": "tof_multibank",
    "TofGeometryParameterKey": "tof_multibank",
    "TofInstrumentParameter": "tof_multibank",
    "TofInstrumentParameterBound": "tof_multibank",
    "TofInstrumentParameterChange": "tof_multibank",
    "TofLatticeParameterChange": "tof_multibank",
    "TofLeBailBank": "tof_multibank",
    "TofLeBailCancellation": "tof_lebail",
    "TofLeBailCheckpoint": "tof_lebail",
    "TofLeBailInput": "tof_lebail",
    "TofLeBailIteration": "tof_lebail",
    "TofLeBailOptions": "tof_lebail",
    "TofLeBailPhase": "tof_lebail",
    "TofLeBailResult": "tof_lebail",
    "TofMultiBankGeometryBankResult": "tof_multibank",
    "TofMultiBankGeometryCheckpoint": "tof_multibank",
    "TofMultiBankGeometryInput": "tof_multibank",
    "TofMultiBankGeometryIteration": "tof_multibank",
    "TofMultiBankGeometryOptions": "tof_multibank",
    "TofMultiBankGeometryResult": "tof_multibank",
    "TofMultiBankMetrics": "tof_multibank",
    "TofSharedLatticePhase": "tof_multibank",
    "TofSharedLatticeState": "tof_multibank",
    "background_parameter_key": "lebail",
    "build_parameter_set": "lebail",
    "extract_intensities": "lebail",
    "initialize_intensities": "lebail",
    "instrument_parameter_key": "lebail",
    "intelligent_rietveld_recipe": "workflow",
    "iterate_once": "lebail",
    "lattice_parameter_key": "lebail",
    "phase_scale_key": "lebail",
    "refine": "lebail",
    "refine_structural_tof_multibank": "tof_structural",
    "refine_tof_lebail": "tof_lebail",
    "refine_tof_multibank_geometry": "tof_multibank",
    "reflection_position_key": "lebail",
    "review_rietveld_input": "readiness",
    "run_rietveld_recipe": "workflow",
    "validate_rietveld_recipe": "workflow",
}


def __getattr__(name: str) -> object:
    """Resolve one-release-cycle compatibility aliases for method-owned APIs."""

    owner = _DEPRECATED_EXPORT_MODULES.get(name)
    if owner is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name = f"{__name__}.{owner}"
    _warnings.warn(
        f"phasesmith.refinement.{name} moved to {module_name}.{name}; "
        "the aggregate alias is deprecated and will be removed in 1.0",
        DeprecationWarning,
        stacklevel=2,
    )
    value = getattr(_importlib.import_module(module_name), name)
    globals()[name] = value
    return value
