"""Alternating synchronized GPU/optimized-CPU kernel timings and parity audit."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from compare import synthetic
from metal_fft import MetalFFT
from phasesmith import _core


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    full = np.random.default_rng(742971).uniform(-1.6, 1.6, (12000, 3))
    phase = full @ np.array([2.71, 3.18, 4.09])
    conditional = full[abs(phase - np.rint(phase)) < 0.08]
    clouds = {"synthetic-two-grain": synthetic(), "diffuse-12000": full, "conditioned": conditional}
    cpu = _core.ExperimentalLatticeEngine(4)
    gpu = MetalFFT()
    rows = []
    for name, q in clouds.items():
        for n in [128, 192, 256]:
            step = min(2 * np.max(abs(q)) / (n - 4), 1 / (2.1 * 15))
            for repeat in range(-1, 9):
                outcomes = {}
                timing = {}
                gpu.records.clear()
                for backend in ["cpu", "gpu"] if repeat % 2 else ["gpu", "cpu"]:
                    engine = cpu if backend == "cpu" else gpu
                    start = perf_counter()
                    outcomes[backend] = engine.windowed_maxima(q, n, step, 1.5, 15, 400)
                    timing[backend] = perf_counter() - start
                x, y = outcomes["cpu"], outcomes["gpu"]
                sx = {tuple(v) for v in x[0]}
                sy = {tuple(v) for v in y[0]}
                rows.append(
                    dict(
                        cloud=name,
                        points=len(q),
                        grid=n,
                        repeat=repeat,
                        timing=timing,
                        gpu_stages=gpu.records[-1],
                        vectors_order_exact=np.array_equal(x[0], y[0]),
                        vectors_set_equal=sx == sy,
                        peak_overlap=len(sx & sy),
                        peaks_cpu=len(x[0]),
                        peaks_gpu=len(y[0]),
                        total_maxima_cpu=x[2],
                        total_maxima_gpu=y[2],
                        scores_bitwise_equal=np.array_equal(x[1], y[1]),
                    )
                )
            print(name, n, "done", flush=True)
    summary = []
    for name in clouds:
        for n in [128, 192, 256]:
            r = [v for v in rows if v["cloud"] == name and v["grid"] == n and v["repeat"] >= 0]
            ct = float(np.median([v["timing"]["cpu"] for v in r]))
            gt = float(np.median([v["timing"]["gpu"] for v in r]))
            summary.append(
                dict(
                    cloud=name,
                    grid=n,
                    cpu_seconds=ct,
                    gpu_seconds=gt,
                    cpu_over_gpu=ct / gt,
                    gpu_prepare_seconds=float(
                        np.median([v["gpu_stages"]["prepare_seconds"] for v in r])
                    ),
                    gpu_submit_wait_compact_seconds=float(
                        np.median([v["gpu_stages"]["gpu_submit_wait_compact_seconds"] for v in r])
                    ),
                    all_peak_sets_equal=all(v["vectors_set_equal"] for v in r),
                    all_peak_order_equal=all(v["vectors_order_exact"] for v in r),
                    all_maxima_counts_equal=all(
                        v["total_maxima_cpu"] == v["total_maxima_gpu"] for v in r
                    ),
                )
            )
    a.output.write_text(json.dumps(dict(rows=rows, summary=summary), indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
