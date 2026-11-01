"""Independent analytic and native/NumPy periodic interpolation comparisons."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from phasesmith.single_crystal.reference.phase_interpolation import interpolate_phase_density


def test_reference_preserves_nyquist_planes_and_periodic_field():
    def field(shape):
        x, y, z = np.meshgrid(*(np.arange(n) / n for n in shape), indexing="ij")
        return (
            1
            + 0.7 * np.cos(2 * np.pi * 4 * x)
            + 0.3 * np.cos(2 * np.pi * 4 * x) * np.cos(2 * np.pi * 5 * y)
            + 0.2 * np.sin(2 * np.pi * (y + 2 * z))
        )

    source, target = (8, 10, 12), (16, 20, 24)
    actual = interpolate_phase_density(field(source), target)
    np.testing.assert_allclose(actual, field(target), atol=2e-14, rtol=0)
    np.testing.assert_array_equal(interpolate_phase_density(field(source), source), field(source))
    with pytest.raises(ValueError):
        interpolate_phase_density(field(source), (8, 8, 12))


@pytest.mark.parametrize(
    "shape,target",
    [
        ((8, 8, 8), (16, 16, 16)),
        ((8, 10, 12), (16, 20, 24)),
        ((8, 10, 12), (8, 20, 12)),
        ((8, 8, 8), (12, 10, 16)),
    ],
)
def test_native_matches_independent_fft_reference(tmp_path, shape, target):
    executable = os.environ.get("PHASESMITH_NATIVE_WORKFLOW")
    if not executable:
        pytest.skip("set PHASESMITH_NATIVE_WORKFLOW for native interpolation comparison")
    executable = Path(executable).resolve()
    if os.name == "nt" and not executable.suffix:
        executable = executable.with_suffix(".exe")
    density = np.random.default_rng(1423).normal(size=shape)
    packet = dict(shape=shape, target=target, density=density.ravel().tolist())
    source, result = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text(json.dumps(packet))
    subprocess.run(
        [str(executable), "--interpolate-phase-density", str(source), str(result)],
        check=True,
        timeout=30,
    )
    actual = np.array(json.loads(result.read_text())["density"]).reshape(target)
    expected = interpolate_phase_density(density, target)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
