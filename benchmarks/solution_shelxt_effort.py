"""Read completed development listings; never infer missing or unprinted cycles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def listing_effort(text):
    """N(iter) is a listed trial budget, not an instrumented executed-cycle count."""
    table = text.split("Try N(iter)", 1)
    trials = []
    if len(table) == 2:
        body = table[1].split("attempts, solution", 1)[0]
        for line in body.splitlines():
            m = re.fullmatch(
                r"\s*(\d+)\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+(\d+)\s+([-\d.]+)\s*",
                line,
            )
            if m:
                x = m.groups()
                trials.append(dict(trial=int(x[0]), listed_n_iter=int(x[1]), cfom=float(x[5])))
    stages = {}
    for name, pattern in {
        "setup": r"Setup:\s*([\d.]+) secs",
        "phasing": r"Structure solution:\s*([\d.]+) secs",
        "symmetry": r"Space group determination:\s*([\d.]+) secs",
        "assignment_isotropic": r"Assign elements and isotropic refinement\s*([\d.]+) secs",
        "total": r"Total time:\s*([\d.]+) secs",
    }.items():
        m = re.search(pattern, text)
        stages[name] = float(m[1]) if m else None
    attempts = re.search(r"(\d+) attempts, solution", text)
    m = re.search(r"Fourier grid:\s*(\d+) x\s*(\d+) x\s*(\d+)", text)
    return dict(
        trials=trials,
        declared_attempts=int(attempts[1]) if attempts else None,
        trial_table_complete=(len(trials) == int(attempts[1])) if attempts else False,
        listed_iteration_sum=sum(t["listed_n_iter"] for t in trials),
        executed_cycle_count=None,
        stages_seconds=stages,
        grid_shape=[int(x) for x in m.groups()] if m else None,
    )


def compare(report, base):
    if report["split"] != "development" or report["holdout_opened"]:
        raise ValueError("development only")
    ps = {r["id"]: r for r in report["runs"] if r["method"] == "phasesmith"}
    xt = {r["id"]: r for r in report["runs"] if r["method"] == "shelxt"}
    if len(ps) != 35 or ps.keys() != xt.keys():
        raise ValueError("all 35 development cases required")
    rows = []
    for key in sorted(ps):
        path = base / xt[key]["directory"] / "sample.lxt"
        data = path.read_bytes() if path.exists() else b""
        listing = listing_effort(data.decode(errors="replace"))
        result = json.loads((base / ps[key]["directory"] / "solution/result.json").read_text())
        rows.append(
            dict(
                id=key,
                shelxt=listing,
                listing_sha256=hashlib.sha256(data).hexdigest(),
                shelxt_hard_timeout=xt[key]["hard_timeout"],
                ps_solver_seconds=ps[key]["kernel_seconds"],
                ps_wall_seconds=ps[key]["wall_seconds_including_startup"],
                shelxt_wall_seconds=xt[key]["wall_seconds_including_startup"],
                ps_trials=ps[key]["effort"]["phasing_starts"],
                ps_cycles=sum(t["completed_cycles"] for t in ps[key]["effort"]["phase_slices"]),
                ps_grid_shape=result["grid_shape"],
            )
        )
    complete = [
        r
        for r in rows
        if r["shelxt"]["stages_seconds"]["total"] is not None and not r["shelxt_hard_timeout"]
    ]
    return dict(
        schema="phasesmith.shelxt-effort.v1",
        development_cases=35,
        holdout_opened=False,
        qualification=(
            "SHELXT internal elapsed totals are printed to milliseconds; "
            "PhaseSmith solve timers exclude worker imports, input reading "
            "and serialization. Algorithms, grids, reflection limits and "
            "isotropic/anisotropic endpoints differ. N(iter) is listed work, "
            "not independently counted executed cycles; incomplete "
            "listings/timeouts remain explicit and are excluded only from "
            "paired internal-time summaries."
        ),
        summary=dict(
            internal_time_pairs=len(complete),
            completed_pair_ps_trials=sum(r["ps_trials"] for r in complete),
            completed_pair_ps_cycles=sum(r["ps_cycles"] for r in complete),
            completed_pair_shelxt_trials=sum(len(r["shelxt"]["trials"]) for r in complete),
            completed_pair_shelxt_listed_iterations=sum(
                r["shelxt"]["listed_iteration_sum"] for r in complete
            ),
            ps_solver_median=float(np.median([r["ps_solver_seconds"] for r in complete])),
            shelxt_internal_median=float(
                np.median([r["shelxt"]["stages_seconds"]["total"] for r in complete])
            ),
            median_paired_solver_ratio=float(
                np.median(
                    [
                        r["ps_solver_seconds"] / r["shelxt"]["stages_seconds"]["total"]
                        for r in complete
                    ]
                )
            ),
            ps_cycles=sum(r["ps_cycles"] for r in rows),
            shelxt_listed_trials=sum(len(r["shelxt"]["trials"]) for r in rows),
            shelxt_listed_iterations=sum(r["shelxt"]["listed_iteration_sum"] for r in rows),
        ),
        cases=rows,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = compare(json.loads(a.report.read_text()), a.base)
    r["parser_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r["summary"], indent=2))
