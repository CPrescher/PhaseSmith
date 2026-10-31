"""Time lossless writes of the complete real shoebox/profile numeric archive."""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=["numpy", "native"], required=True)
    p.add_argument("--levels", nargs="+", type=int, default=[1, 0])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--discard-verified",
        action="store_true",
        help="Remove this benchmark's output after successful exact read-back",
    )
    a = p.parse_args()
    with np.load(a.case, allow_pickle=False) as archive:
        arrays = dict(archive)
    destination = a.output.with_suffix("")
    destination.mkdir(exist_ok=False)
    records = []
    for level in a.levels if a.backend == "native" else [6]:
        seconds, sizes = [], []
        for i in range(a.repeats):
            path = destination / f"level-{level}-repeat-{i}.npz"
            start = time.perf_counter()
            if a.backend == "numpy":
                with path.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
            else:
                from phasesmith.io.numeric_archive import save_npz

                save_npz(path, arrays, compression_level=level)
            seconds.append(time.perf_counter() - start)
            sizes.append(path.stat().st_size)
            h = hashlib.sha256()
            with np.load(path, allow_pickle=False) as saved:
                assert set(saved.files) == set(arrays)
                for name, expected in arrays.items():
                    value = saved[name]
                    assert value.shape == expected.shape and value.dtype == expected.dtype
                    assert value.tobytes() == expected.tobytes()
                    h.update(value.tobytes())
            with a.output.with_suffix(".journal.jsonl").open("a") as journal:
                journal.write(
                    json.dumps(
                        dict(
                            level=level,
                            repeat=i,
                            seconds=seconds[-1],
                            archive_bytes=sizes[-1],
                            exact_roundtrip=True,
                            numerical_sha256=h.hexdigest(),
                        )
                    )
                    + "\n"
                )
            if a.discard_verified:
                path.unlink()
        record = dict(
            level=level,
            seconds=seconds,
            median_seconds=float(np.median(seconds)),
            archive_bytes=sizes,
            numerical_sha256=h.hexdigest(),
            exact_roundtrip=True,
        )
        records.append(record)
        print(json.dumps(record), flush=True)
    with a.output.open("x") as stream:
        json.dump(
            dict(
                schema="phasesmith.numeric-archive-execution.v1",
                backend=a.backend,
                raw_bytes=sum(v.nbytes for v in arrays.values()),
                records=records,
                maximum_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            ),
            stream,
            indent=2,
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
