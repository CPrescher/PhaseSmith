"""Freeze native grain inputs, or compare two release coordinator executables.

Build the ignored Rust benchmark in each source tree (see docs/grain-cpu-speed.md).
Timing excludes fixture I/O and bit serialization. Every returned field is
serialized in source order, including NaN payloads and signed zeros.
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import struct
import subprocess
from pathlib import Path

import numpy as np
from grain_cpu_speed import cases
from phasesmith.single_crystal.grains import _rotations
from phasesmith.single_crystal.prediction import predict_spots
from phasesmith.single_crystal.reciprocal import PanelGeometry


def prepare(root):
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    rotations = _rotations(64, 842025)
    records = []

    def geometry(poni):
        return PanelGeometry.from_poni(
            distance=0.2,
            poni1=poni,
            poni2=poni,
            pixel1=8e-5,
            pixel2=8e-5,
            rot1=0.01,
            rot2=-0.03,
            rot3=0.05,
            wavelength=0.4,
            axis=[-0.97, 0.12, 0.2],
        )

    def write(name, rows, panel):
        header = (
            np.r_[
                panel.vectors.reshape(-1),
                panel.wavelength,
                [4.3, 5.1, 7.7, 90, 90, 90],
                rotations.reshape(-1),
            ]
            .astype("<f8")
            .tobytes()
        )
        data = header + struct.pack("<Q", len(rows)) + np.asarray(rows, dtype="<f8").tobytes()
        (inputs / (name + ".bin")).write_bytes(data)
        record = dict(name=name, points=len(rows), sha256=hashlib.sha256(data).hexdigest())
        records.append(record)
        return record

    for name, points in cases().items():
        write("discovery-" + name.removeprefix("synthetic-"), points, geometry(0.08))
    panel = geometry(0.16)
    for target in [600, 1800, 5000]:
        rng = np.random.default_rng(95000 + target)
        parts = []
        for edges in [[4.3, 5.1, 7.7], [5.2, 6.2, 8.3]]:
            basis = np.diag(edges) @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
            h = np.unique(rng.integers(-9, 10, (25000, 3)), axis=0)
            h = h[np.any(h != 0, axis=1)]
            pred = predict_spots(h, np.linalg.inv(basis).T, panel, rng.uniform(-200, -30, len(h)))
            valid = (pred.flags == 0) & np.all(
                (pred.values[:, :2] > 4) & (pred.values[:, :2] < 4092), axis=1
            )
            obs = pred.values[valid]
            size = 2 * target // 5
            if len(obs) < size:
                raise ValueError(f"insufficient predicted crossings: {len(obs)} < {size}")
            obs = obs[rng.choice(len(obs), size, replace=False)].copy()
            obs += rng.normal(size=obs.shape) * [0.08, 0.08, 0.015]
            parts.append(obs)
        observed = np.concatenate(
            [
                *parts,
                rng.uniform([20, 20, -200], [4076, 4076, -30], (target - sum(map(len, parts)), 3)),
            ]
        )
        record = write(f"index-{target}", observed, panel)
        record.update(
            signal_tracks=list(map(len, parts)), noise_tracks=target - sum(map(len, parts))
        )
    (root / "input-manifest.json").write_text(json.dumps(records, indent=2) + "\n")


def run(executable, root, case, label, batch_size):
    output = root / f"{label}-{case}.bits"
    env = dict(
        os.environ,
        PHASESMITH_NATIVE_GRAIN_INPUT=str(root / "inputs" / f"{case}.bin"),
        PHASESMITH_NATIVE_GRAIN_OUTPUT=str(output),
        PHASESMITH_NATIVE_GRAIN_REPEATS=str(batch_size),
    )
    result = subprocess.run(
        [
            str(executable.resolve()),
            "complete_native_grain_benchmark",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    (root / f"{label}-{case}.log").write_text(result.stdout)
    matches = re.findall(r"NATIVE_BENCH seconds=([0-9.]+) (.*)", result.stdout)
    if len(matches) != batch_size:
        raise RuntimeError(result.stdout)
    clocks = [float(seconds) for seconds, _ in matches]
    assert len({message for _, message in matches}) == 1
    data = output.read_bytes()
    return dict(
        seconds=statistics.median(clocks),
        individual_seconds=clocks,
        outcome=matches[0][1],
        sha256=hashlib.sha256(data).hexdigest(),
    ), data


def compare(root, baseline, optimized, repeats, batch_size):
    executable_hashes = {
        arm: hashlib.sha256(path.read_bytes()).hexdigest()
        for arm, path in [("baseline", baseline), ("optimized", optimized)]
    }
    if executable_hashes["baseline"] == executable_hashes["optimized"]:
        raise ValueError(
            "baseline and optimized executables are identical; use separate target dirs"
        )
    manifest = json.loads((root / "input-manifest.json").read_text())
    names = sorted(row["name"] for row in manifest)
    for row in manifest:
        data = (root / "inputs" / (row["name"] + ".bin")).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"], row["name"]
    expected = {}
    # Warm each complete case in both independent executables; compare these too.
    for case in names:
        for arm, exe in [("baseline", baseline), ("optimized", optimized)]:
            _, data = run(exe, root, case, arm + "-warmup", batch_size)
            if case in expected:
                assert data == expected[case], (case, "warmup result differs")
            expected[case] = data
    rows = []
    for repeat in range(repeats):
        for case in names[repeat % len(names) :] + names[: repeat % len(names)]:
            arms = [("baseline", baseline), ("optimized", optimized)]
            if repeat % 2:
                arms.reverse()
            for arm, exe in arms:
                record, data = run(exe, root, case, f"{arm}-{repeat:02}", batch_size)
                assert data == expected[case], (case, arm, repeat, "result bits differ")
                rows.append(dict(case=case, arm=arm, repeat=repeat, **record))
        print(f"{repeat + 1}/{repeats} paired rounds: all results bitwise identical", flush=True)
        (root / "paired-results.json").write_text(json.dumps(rows, indent=2) + "\n")
    summary = []
    for case in names:
        item = dict(case=case)
        for arm in ["baseline", "optimized"]:
            samples = [row["seconds"] for row in rows if row["case"] == case and row["arm"] == arm]
            item[arm] = dict(
                median=statistics.median(samples),
                std=statistics.stdev(samples) if len(samples) > 1 else 0,
                minimum=min(samples),
                maximum=max(samples),
            )
        item["reduction_percent"] = 100 * (
            1 - item["optimized"]["median"] / item["baseline"]["median"]
        )
        item["outcome"] = next(row["outcome"] for row in rows if row["case"] == case)
        summary.append(item)
    result = dict(
        repeats=repeats,
        batch_size=batch_size,
        fingerprinted_executions=(len(rows) + 2 * len(names)) * batch_size,
        before_after_pairs=(repeats + 1) * len(names) * batch_size,
        executables=executable_hashes,
        cases=summary,
    )
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--baseline-exe", type=Path)
    parser.add_argument("--optimized-exe", type=Path)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare(args.output)
    else:
        if (
            not args.baseline_exe
            or not args.optimized_exe
            or min(args.repeats, args.batch_size) < 1
        ):
            parser.error("provide both release executables and a positive repeat count")
        compare(args.output, args.baseline_exe, args.optimized_exe, args.repeats, args.batch_size)


if __name__ == "__main__":
    main()
