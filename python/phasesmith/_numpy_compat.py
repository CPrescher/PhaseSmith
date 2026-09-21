"""NumPy names that differ across the supported dependency versions."""

import numpy as np

# NumPy 2 renamed trapz; use its current name without dropping NumPy 1.26.
trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
