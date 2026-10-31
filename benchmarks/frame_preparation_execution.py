"""Real raw-frame preparation benchmark on frozen target predictions only.

This isolated cohort omits competing-domain exclusions; its hashes compare
execution controls within this cohort, not the full reduction. Each stage also
has a fresh full all-domain workflow checked against retained v11.
"""

import argparse
import hashlib
import inspect
import json
import resource
import time
from pathlib import Path

import numpy as np
from enstatite_raw_workflow import RawFrames, poni_geometry
from phasesmith.experimental.frame_validity import CombinedValidity, DacApertureMask, FrameValidity
from phasesmith.experimental.prediction import ScanPredictions
from phasesmith.experimental.profiles import prepare_profiles
from phasesmith.experimental.reduction_io import load_case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, nargs="+", default=[1, 4, 6, 8, 10])
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    packet = json.loads((a.run / "worker-input.json").read_text())
    protocol = packet["protocol"]
    obs, _ = load_case(a.run / "raw-observations-and-bank.npz")
    t = obs.summation
    frames = RawFrames(packet["raw_path"], protocol["acquisition"], packet["raw_array"])
    geometry = poni_geometry(
        packet["poni_path"], frames.pixel1, frames.pixel2, protocol["acquisition"]["axis"]
    )
    estimated = FrameValidity.load(a.run / "frame-validity.npz")
    geometric = DacApertureMask(
        geometry,
        frames.bounds,
        frames.reader.shape[1:],
        frame_ids=estimated.frame_ids,
        **protocol["shadow_mask"]["geometric"],
    )
    validity = CombinedValidity((estimated, geometric))
    frames.close()
    predictions = ScanPredictions(t.hkl, t.predicted, np.zeros(len(t), np.int64), 2.0)
    records, hashes = [], []
    for workers in a.workers:
        controls = {}
        if "execution" in inspect.signature(prepare_profiles).parameters:
            from phasesmith.experimental.integration import FrameExecution

            controls["execution"] = FrameExecution(workers, 1024**3)
        else:
            assert workers == 1
        seconds = []
        for _ in range(a.repeats):
            frames = RawFrames(
                packet["raw_path"], protocol["acquisition"], packet["raw_array"], validity
            )
            start = time.perf_counter()
            try:
                result = prepare_profiles(
                    frames.load,
                    frames.bounds,
                    predictions,
                    mask_loader=frames.mask,
                    grain_ids=t.grain_ids,
                    options=t.options,
                    scales=obs.scales,
                    planar_transforms=obs.planar_transforms,
                    centres=obs.centres,
                    angular_half_widths=t.angular_half_widths,
                    **controls,
                )
            finally:
                frames.close()
            seconds.append(time.perf_counter() - start)
            h = hashlib.sha256()
            for array in [
                result.samples,
                result.valid,
                result.covariance,
                result.summation.values,
                result.summation.flags,
            ]:
                h.update(array.tobytes())
            hashes.append(h.hexdigest())
        record = dict(
            workers=workers,
            seconds=seconds,
            median_seconds=float(np.median(seconds)),
            sha256=hashes[-1],
            execution=getattr(result.summation, "execution", None),
        )
        records.append(record)
        print(json.dumps(record), flush=True)
    assert len(set(hashes)) == 1
    with a.output.open("x") as stream:
        json.dump(
            dict(
                schema="phasesmith.frame-preparation-execution.v1",
                records=records,
                cohort="7277 frozen target predictions; no competing-domain exclusions",
                all_outputs_bitwise_identical=True,
                maximum_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            ),
            stream,
            indent=2,
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
