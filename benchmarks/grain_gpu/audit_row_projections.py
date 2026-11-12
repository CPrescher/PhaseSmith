"""Untimed complete CPU-float64/GPU key audit over the benchmark panel."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from metal_rows import MetalRows
from phasesmith.single_crystal.rows import RowSearchOptions
from rows_compare import ROOT, large_cloud, synthetic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("/Users/clemens/Codex/2026-11-12/phasesmith-row-gpu")
    )
    root = parser.parse_args().output
    root.mkdir(parents=True, exist_ok=True)
    cases = {"synthetic-2674": synthetic()}
    for p in sorted((ROOT / "tests/fixtures/row-search").glob("*.npz")):
        cases[p.stem] = np.load(p)["points"]
    for n in [6000, 12000]:
        cases[f"synthetic-{n}"] = large_cloud(n)
    records = []
    options = json.dumps(RowSearchOptions()._native_options())
    for name, q in cases.items():
        m = MetalRows(gpu_projection=True)
        m.audit_projection = True
        start = perf_counter()
        r = json.loads(m.run(q, options))
        row = dict(
            case=name,
            points=len(q),
            complete=not r["budget_exhausted"] and not r["cancelled"],
            keys_audited=sum(x["keys"] for x in m.projection_audits),
            corrected_pairs=sum(x["corrected_pairs"] for x in m.projection_audits),
            mismatches=sum(x["mismatches"] for x in m.projection_audits),
            seconds_including_expensive_audit=perf_counter() - start,
        )
        assert row["complete"] and row["mismatches"] == 0
        records.append(row)
        print(row, flush=True)
        (root / "projection-audit.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
