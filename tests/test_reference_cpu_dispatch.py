"""Reference angular coordinates must not depend on NumPy's SIMD trig path."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from phasesmith import reference


def _write_probe(path: Path) -> None:
    outputs = {}
    components = reference.wavelength_component_positions(
        np.array([49.85, 50.15]), 1.54056, np.array([1.54056, 1.54439])
    )
    for index, values in enumerate(components):
        outputs[f"components_{index}"] = values
    rng = np.random.default_rng(20260917)
    for index in range(8):
        position = float(rng.uniform(12, 75) if index % 2 else rng.uniform(105, 168))
        gaussian, lorentzian = rng.uniform(0.02, 0.08), rng.uniform(0.001, 0.03)
        width = reference.tch_shape_from_fwhm(gaussian, lorentzian).total_fwhm
        # Fixed geometry, independent of vector trig used to construct test inputs.
        sample, detector = (0.0001, 0.0001) if index % 3 == 0 else (0.0001, 0.0004)
        x = position + width * np.linspace(-4, 4, 401)
        profile = reference.profile_fcj(
            x, position, gaussian, lorentzian, sample, detector, fast_fcj=True
        )
        for field in profile.__dataclass_fields__:
            outputs[f"profile_{index}_{field}"] = getattr(profile, field)
    np.savez(path, **outputs)


def test_reference_geometry_across_numpy_cpu_dispatch(tmp_path):
    paths = [tmp_path / "default.npz", tmp_path / "no-avx512.npz"]
    for disabled, path in zip((False, True), paths, strict=True):
        env = os.environ.copy()
        if disabled:
            env["NPY_DISABLE_CPU_FEATURES"] = ",".join(
                filter(
                    None,
                    (
                        env.get("NPY_DISABLE_CPU_FEATURES", ""),
                        "AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL",
                    ),
                )
            )
        subprocess.run([sys.executable, __file__, str(path)], env=env, check=True)
    with np.load(paths[0]) as default, np.load(paths[1]) as baseline:
        for name in default.files:
            if name.startswith("components_"):
                np.testing.assert_array_equal(default[name], baseline[name])
            else:
                # Match the existing local FCJ derivative envelope. Intrinsic
                # exponential evaluation is still NumPy-vectorized.
                scale = max(float(np.max(np.abs(baseline[name]))), 1.0)
                assert np.max(np.abs(default[name] - baseline[name])) / scale < 2e-9, name


if __name__ == "__main__":
    _write_probe(Path(sys.argv[1]))
