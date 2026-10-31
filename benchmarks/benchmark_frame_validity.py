"""Bounded shadow-mask benchmark and optional black-box Dioptas compatibility.

The fixture has 64 narrow diffraction peaks per exposure on a 512x768 grid,
moving broad seat shadows and deterministic counting noise. No external data
or application dependency is required. The optional MIT upstream adapter only
supplies arrays/readers/resources to the unmodified original module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
from phasesmith.single_crystal.frame_validity import (
    DacApertureMask,
    estimate_seat_shadows,
)
from phasesmith.single_crystal.reciprocal import PanelGeometry


def fixture():
    base = np.full((512, 768), 80.0)
    yy, xx = np.mgrid[-4:5, -4:5]
    spot = 4000 * np.exp(-0.5 * (xx**2 + yy**2))
    for y in np.linspace(32, 480, 8).astype(int):
        for x in np.linspace(32, 736, 8).astype(int):
            base[y - 4 : y + 5, x - 4 : x + 5] += spot
    images = np.repeat(base[None], 40, axis=0)
    for frame in range(10):
        images[frame, :, 440 + frame * 16 :] *= 0.15
        images[39 - frame, :, : 320 - frame * 16] *= 0.15
    return np.random.default_rng(35061).poisson(images).astype(np.uint16)


def upstream_masks(path, images):
    """Execute the original source, adapting only its application-owned imports."""
    prefix = "_phasesmith_dioptas_mask_oracle"
    modules = {}
    for name in (prefix, prefix + ".model", prefix + ".model.util"):
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
    resources = ModuleType(prefix + ".resources")
    resources.get_resources = lambda: SimpleNamespace(image_workers=lambda *a, **kw: 1)
    modules[resources.__name__] = resources
    workers = ModuleType(prefix + ".model.util.bounded_map")
    workers.bounded_map = lambda function, files, *a, **kw: (function(f) for f in files)
    modules[workers.__name__] = workers
    readers = ModuleType(prefix + ".model.rotation_series")
    # Upstream's reader adapter reverses rows; give it that reader convention.
    readers.read_frame = lambda filename: images[int(filename)][::-1]
    modules[readers.__name__] = readers
    sys.modules.update(modules)
    name = prefix + ".model.frame_validity"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        series = SimpleNamespace(files=tuple(map(str, range(len(images)))))
        return module.build_seat_validity(series, module.SeatShadowOptions(), workers=1)
    finally:
        for key in [*modules, name]:
            sys.modules.pop(key, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dioptas-source", type=Path)
    args = parser.parse_args()
    images = fixture()
    record = dict(
        schema="phasesmith.frame-validity-benchmark.v1",
        platform=platform.platform(),
        frames=len(images),
        detector_shape=list(images.shape[1:]),
        peaks_per_frame=64,
        timing="in-memory original-count loaders; three repeats; no file/cache read timing",
    )
    plans, timings = {}, {}
    for workers in (1, 4):
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            plan = estimate_seat_shadows(lambda i: images[i], len(images), workers=workers)
            samples.append(time.perf_counter() - start)
        plans[workers] = plan
        timings[str(workers)] = dict(seconds=samples, median=float(np.median(samples)))
    record["illumination"] = timings
    record["workers_bitwise_equal"] = plans[1].fingerprint() == plans[4].fingerprint()
    assert record["workers_bitwise_equal"]
    record["coarse_storage_bytes"] = plans[1].excluded.nbytes
    record["one_full_resolution_mask_bytes"] = images[0].size
    if args.dioptas_source:
        before = upstream_masks(args.dioptas_source, images)
        record["upstream"] = dict(
            source_sha256=hashlib.sha256(args.dioptas_source.read_bytes()).hexdigest(),
            excluded_bitwise_equal=np.array_equal(before.excluded, plans[1].excluded),
            frame_hashes_equal=before.provenance["image_sha256"]
            == plans[1].provenance["image_sha256"],
        )
        assert record["upstream"]["excluded_bitwise_equal"]
        assert record["upstream"]["frame_hashes_equal"]
    geometry = PanelGeometry(
        [[-0.0288, -0.0192, 0.15], [75e-6, 0, 0], [0, 75e-6, 0], [0, 0, 1], [0, 1, 0]], 0.28457
    )
    bounds = np.c_[np.arange(-30.0, 31.0, 15.0), np.arange(-29.0, 32.0, 15.0)]
    provider = DacApertureMask(
        geometry,
        bounds,
        images.shape[1:],
        axis_at_omega_zero=np.array([0, 0, 1]),
        half_opening_degrees=35,
    )
    start = time.perf_counter()
    hashes = [hashlib.sha256(provider.mask(i).tobytes()).hexdigest() for i in range(len(bounds))]
    record["geometric"] = dict(
        frames=len(bounds),
        seconds=time.perf_counter() - start,
        mask_sha256=hashes,
        tile_rows=provider.tile_rows,
    )
    blocked = DacApertureMask(
        geometry,
        bounds,
        images.shape[1:],
        axis_at_omega_zero=np.array([0, 0, 1]),
        half_opening_degrees=35,
        pixel_block=16,
    )
    start = time.perf_counter()
    coarse = [blocked.mask(i) for i in range(len(bounds))]
    record["geometric_block16"] = dict(
        seconds=time.perf_counter() - start,
        conservative=True,
        extra_excluded_pixels=0,
    )
    for i, mask in enumerate(coarse):
        fine = provider.mask(i)
        assert not np.any(fine & ~mask)
        record["geometric_block16"]["extra_excluded_pixels"] += int((mask & ~fine).sum())
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
