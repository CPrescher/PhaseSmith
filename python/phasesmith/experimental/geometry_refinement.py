"""Compatibility import; use phasesmith.single_crystal.geometry_refinement."""

import importlib as _importlib
import sys as _sys
import warnings as _warnings

_TARGET = "phasesmith.single_crystal.geometry_refinement"
if __name__ == "__main__":
    raise SystemExit(_importlib.import_module(_TARGET).main())

_warnings.warn(
    f"{__name__} moved to {_TARGET}; the compatibility import will be removed in 1.0",
    DeprecationWarning,
    stacklevel=2,
)
_sys.modules[__name__] = _importlib.import_module(_TARGET)
