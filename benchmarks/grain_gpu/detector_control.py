"""Unchanged existing strict detector-space control, CPU versus GPU FFT only."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from compare import backend, compare, fingerprint
from grain_remaining import control
from metal_fft import MetalFFT
from phasesmith.single_crystal import grains as G


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    original = G.find_grains_ensemble
    m = MetalFFT()
    report = []
    for repeat in range(a.repeats):
        results = {}
        records = {}
        for kind in ["cpu", "gpu"] if repeat % 2 else ["gpu", "cpu"]:
            captures = []

            def find(*args, _captures=captures, **kw):
                t = perf_counter()
                r = original(*args, **kw)
                _captures.append((r, perf_counter() - t))
                return r

            stages = []
            m.records.clear()
            G.find_grains_ensemble = find
            try:
                t = perf_counter()
                with backend(kind, m, stages):
                    summary = control()
                elapsed = perf_counter() - t
            finally:
                G.find_grains_ensemble = original
            results[kind] = captures
            records[kind] = dict(
                total_seconds=elapsed,
                control=summary,
                fft_seconds=sum(r["seconds"] for r in stages),
                fft_calls=len(stages),
                gpu_stages=list(m.records),
            )
            for native, (result, _seconds) in zip([False, True], captures, strict=True):
                (a.output / f"{repeat}-{kind}-native-{native}-fingerprint.json").write_text(
                    json.dumps(fingerprint(result), indent=2) + "\n"
                )
            print(repeat, kind, "total", elapsed, "fft", records[kind]["fft_seconds"], flush=True)
        row = dict(
            repeat=repeat,
            records=records,
            comparison=[
                dict(native=n, cpu_seconds=x[1], gpu_seconds=y[1], comparison=compare(x[0], y[0]))
                for n, x, y in zip([False, True], results["cpu"], results["gpu"], strict=True)
            ],
        )
        report.append(row)
        (a.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print("comparison", row["comparison"], flush=True)


if __name__ == "__main__":
    main()
