#!/usr/bin/env python3
"""Temporarily override rietx's row threshold in a fresh-process QARR audit.

Example (repeat separately for workers/threshold 1/512, 2/1, and 8/1):
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python benchmarks/probe_rietx_thread_threshold.py --threads 8 --threshold 1 \
--json-output validation/results/rietx-20260917-forced-8workers.json

No installed package or production model is modified. Requires the pinned
rietx 1.4.0 benchmark environment and the existing verified QARR data.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import audit_rietx_threads as audit
from phasesmith import ExecutionPolicy
from phasesmith.refinement import rietveld as rv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.threads, args.threshold) < 1:
        parser.error("positive threads and threshold required")
    audit.compiled._THREAD_MIN_ROWS = args.threshold
    original = rv.calculate

    def controlled(*call_args, **kwargs):
        kwargs.setdefault("execution", ExecutionPolicy(threads=1))
        return original(*call_args, **kwargs)

    root = Path(__file__).resolve().parents[1] / "validation/data/iucr-qarr-1g"
    with patch.object(rv, "calculate", controlled):
        result = audit.run(root, args.threads, 5)
    result["shared_phasesmith_preparation_default_workers"] = 1
    result["intervention"] = (
        "temporary in-process row threshold override; installed files unchanged"
    )
    result["notes"][-1] = (
        "Shared PhaseSmith preparation is explicitly restricted to one worker in this experiment."
    )
    args.json_output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
