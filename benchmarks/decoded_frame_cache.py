"""Three original-count passes with one native source-bound decoded cache.

Time only native read/array-return calls. Independent byte checksums are outside
that timer. Every repeat opens a fresh empty cache; nothing is persisted.
"""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

from phasesmith.io.hdf5 import Hdf5Frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    packet = json.loads((args.run / "worker-input.json").read_text())
    acquisition = packet["protocol"]["acquisition"]
    indices = range(acquisition["first_frame"], acquisition["last_frame_inclusive"] + 1)
    records, hashes = [], []
    for budget in [0, 1024**3]:
        repeats = []
        for _ in range(args.repeats):
            with Hdf5Frames(
                packet["raw_path"], packet["raw_array"]["dataset"], cache_bytes=budget
            ) as reader:
                seconds = []
                checksums = []
                for _ in range(3):
                    elapsed = 0.0
                    h = hashlib.sha256()
                    for index in indices:
                        start = time.perf_counter()
                        frame = reader.read_frame(index)
                        elapsed += time.perf_counter() - start
                        h.update(memoryview(frame).cast("B"))
                    seconds.append(elapsed)
                    checksums.append(h.hexdigest())
                assert len(set(checksums)) == 1
                hashes.extend(checksums)
                repeats.append(
                    dict(
                        pass_seconds=seconds,
                        total_read_seconds=sum(seconds),
                        sha256=checksums[0],
                        cache=reader.cache_stats,
                    )
                )
        record = dict(
            cache_budget_bytes=budget,
            repeats=repeats,
            median_seconds=sorted(r["total_read_seconds"] for r in repeats)[len(repeats) // 2],
        )
        records.append(record)
        print(json.dumps(record), flush=True)
    assert len(set(hashes)) == 1
    with args.output.open("x") as stream:
        json.dump(
            dict(
                schema="phasesmith.decoded-frame-cache.v1",
                records=records,
                all_count_bytes_identical=True,
                maximum_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            ),
            stream,
            indent=2,
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
