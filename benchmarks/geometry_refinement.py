"""Many-reflection fused detector-Jacobian benchmark; optional earlier-core check."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from phasesmith import _core
from phasesmith.experimental.prediction import predict_spots
from phasesmith.experimental.reciprocal import PanelGeometry


def main(output, baseline_core=None):
    rng = np.random.default_rng(20261028)
    g = PanelGeometry.from_poni(
        distance=0.2,
        poni1=0.08,
        poni2=0.08,
        pixel1=75e-6,
        pixel2=75e-6,
        rot1=0.01,
        rot2=-0.03,
        rot3=0.02,
        wavelength=0.4,
        axis=[-1, 0, 0],
    )
    b = np.diag([1 / 4.7, 1 / 6.3, 1 / 8.1])
    h = rng.integers(-8, 9, (50000, 3), dtype=np.int64)
    hint = rng.uniform(-90, 90, len(h))
    pred = predict_spots(h, b, g, hint)
    use = (
        (pred.flags == 0)
        & (pred.values[:, 0] > 0)
        & (pred.values[:, 0] < 2048)
        & (pred.values[:, 1] > 0)
        & (pred.values[:, 1] < 2048)
    )
    h = h[use]
    hint = hint[use]
    results = {}
    for name, fn in [
        ("basis_only", _core.experimental_predict_spots),
        ("with_detector", _core.experimental_predict_spots_with_detector),
    ]:
        times = []
        previous = None
        for _ in range(9):
            start = time.perf_counter()
            r = fn(h, b, hint, g.vectors, g.wavelength, False)
            times.append(time.perf_counter() - start)
            digest = hashlib.sha256(r[0].tobytes() + r[1].tobytes()).hexdigest()
            assert previous is None or previous == digest
            previous = digest
        results[name] = {
            "median_seconds": float(np.median(times[1:])),
            "value_basis_sha256": digest,
            "deterministic_repeats": True,
        }
    assert (
        results["basis_only"]["value_basis_sha256"]
        == results["with_detector"]["value_basis_sha256"]
    )
    if baseline_core:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "arrays.npz"
            np.savez(path, h=h, b=b, hint=hint, g=g.vectors)
            code = """import sys,importlib.util,numpy as np,time,json,hashlib
spec=importlib.util.spec_from_file_location('_core',sys.argv[1]);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core);z=np.load(sys.argv[2]);times=[]
for _ in range(9):
 start=time.perf_counter();r=core.experimental_predict_spots(z['h'],z['b'],z['hint'],z['g'],.4,False);times.append(time.perf_counter()-start)
print(json.dumps(dict(median_seconds=float(np.median(times[1:])),value_basis_sha256=hashlib.sha256(r[0].tobytes()+r[1].tobytes()).hexdigest())))"""
            old = json.loads(
                subprocess.run(
                    [sys.executable, "-c", code, str(baseline_core), str(path)],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )
            assert old["value_basis_sha256"] == results["basis_only"]["value_basis_sha256"]
            results["earlier_core"] = old
            results["basis_only_ratio_to_earlier"] = (
                results["basis_only"]["median_seconds"] / old["median_seconds"]
            )
            results["earlier_core_sha256"] = hashlib.sha256(baseline_core.read_bytes()).hexdigest()
    results["reflections"] = len(h)
    results["core_sha256"] = hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest()
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-core", type=Path)
    args = parser.parse_args()
    main(args.output, args.baseline_core)
