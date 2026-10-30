"""Every latency and batch prediction must exist before any development reference opens."""

import importlib.util
import json
from pathlib import Path
import sys


def test_all_560_predictions_precede_reference_audits(tmp_path, monkeypatch):
    benches = Path(__file__).resolve().parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(benches))
    spec = importlib.util.spec_from_file_location(
        "multicore_runner", benches / "solution_multicore.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    audited = []
    batches = []
    out = tmp_path / "run"

    def jobs(args):
        args.output.mkdir()
        rows = [{"id": str(i), "method": m} for i in range(35) for m in ["phasesmith", "shelxt"]]
        (args.output / "before-reference-audit.json").write_text(json.dumps({"runs": rows}))
        for row in rows:
            yield row
        yield {"reference_audit_barrier": True}
        assert batches == [1, 2, 4, 8]
        assert (out / "all-predictions-before-reference.json").exists()
        audited.append(args.workers)

    def batch(args, n, base):
        assert not audited
        batches.append(n)
        panel = out / f"batch-w{n}"
        panel.mkdir()
        (panel / "before-reference-audit.json").write_text(json.dumps(base))
        return panel

    calls = []
    monkeypatch.setattr(module, "collect_jobs", jobs)
    monkeypatch.setattr(module, "batch_panel", batch)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: calls.append(a))
    monkeypatch.setattr(sys, "argv", ["runner", "--output", str(out)])
    module.main()
    assert audited == [1, 2, 4, 8] and len(calls) == 8
    events = json.loads((out / "schedule.json").read_text())
    assert len(events) == 284
    assert not any(e.get("reference_audit_barrier") for e in events[:280])
    seal = json.loads((out / "all-predictions-before-reference.json").read_text())
    assert (
        seal["predictions"] == 560 and len(seal["panels"]) == 8 and seal["holdout_opened"] is False
    )


def test_batch_has_one_inner_worker_and_bounded_outer_jobs(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    import time

    benches = Path(__file__).resolve().parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(benches))
    spec = importlib.util.spec_from_file_location(
        "multicore_batch", benches / "solution_multicore.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = {"source_snapshot": "sources", "runs": []}
    active = 0
    maximum = 0
    lock = Lock()
    for i in range(35):
        for method in ["phasesmith", "shelxt"]:
            relative = f"{i}/{method}"
            source = tmp_path / "latency-w1" / relative
            source.mkdir(parents=True)
            for name in ["inputs.json", "sample.hkl", "sample.ins"]:
                (source / name).write_text("input")
            command = (
                ["worker", "--input", str(source / "inputs.json"), "--workers", "1"]
                if method == "phasesmith"
                else ["shelxt", "sample", "-t1"]
            )
            base["runs"].append(
                dict(
                    id=str(i),
                    method=method,
                    directory=relative,
                    input_only_resource_atom_cap=10,
                    expanded_measured_rows_for_resource_cap=100,
                    command=command,
                )
            )

    def launch(command, work, seconds):
        nonlocal active, maximum
        if work.name == "phasesmith":
            assert command[-2:] == ["--workers", "1"]
        else:
            assert "-t1" in command
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(0.002)
            (work / "stdout.txt").write_text("1 threads running in parallel")
            if work.name == "phasesmith":
                (work / "solution").mkdir()
                (work / "solution/worker.json").write_text(json.dumps({"model": None}))
            return dict(
                wall_seconds_including_startup=0.002,
                returncode=0,
                hard_timeout=False,
                command=command,
            )
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(module, "launch", launch)
    monkeypatch.setattr(module, "shelxt_candidate", lambda _: None)
    from types import SimpleNamespace

    panel = module.batch_panel(SimpleNamespace(output=tmp_path, seconds=15), 4, base)
    assert 1 < maximum <= 4 and active == 0
    result = json.loads((panel / "before-reference-audit.json").read_text())
    assert len(result["runs"]) == 70 and result["workers"] == result["shelxt_threads"] == 1
    assert all(
        b["outer_workers"] == 4 and b["inner_workers"] == 1 for b in result["batches"].values()
    )
