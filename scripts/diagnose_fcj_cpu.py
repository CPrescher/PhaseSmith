"""Temporary CPU-dispatch probe; does not modify the installed package."""
import math
import sys
from pathlib import Path
import numpy as np
import pytest
from phasesmith import reference

class ScalarTrig:
    def __getattr__(self, name):
        mapped = {'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                  'arccos': math.acos, 'arcsin': math.asin, 'arctan': math.atan}
        if name in mapped:
            def evaluate(value):
                values = np.asarray(value)
                if values.ndim == 0:
                    return mapped[name](float(values))
                return np.fromiter((mapped[name](float(v)) for v in values.flat), dtype=float).reshape(values.shape)
            return evaluate
        return getattr(np, name)

np.show_runtime()
print('MODE', sys.argv[1], flush=True)
if sys.argv[1] == 'scalar':
    reference.np = ScalarTrig()
root = Path(__file__).resolve().parents[1]
status = pytest.main(['-q', str(root/'tests/test_profile_accuracy.py'), str(root/'tests/test_wavelength_components.py'), '-k', 'fast_fcj_against_independent_and_high_order_integrals or fused_fcj_doublet_matches_independent_reference'])
print('PYTEST_EXIT', status, flush=True)
