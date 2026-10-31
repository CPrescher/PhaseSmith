import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "effort", Path(__file__).resolve().parents[1] / "benchmarks/solution_shelxt_effort.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_listing_cycles_are_not_claimed_as_executed_and_missing_total_is_explicit():
    text = (
        "Fourier grid: 18 x 36 x 40\nTry N(iter)  CC R(weak) CHEM CFOM "
        "best Sig(min) N(P1) Vol/N\n1 100 95.93 .0964 1.0000 .8629 .8629 "
        "1.635 30 16.31\n2 110 96.14 .0945 1.0000 .8669 .8669 2.488 29 "
        "16.87\n2 attempts, solution 2 selected\nStructure solution: .603 "
        "secs\nTotal time: .727 secs"
    )
    r = m.listing_effort(text)
    assert len(r["trials"]) == 2 and r["listed_iteration_sum"] == 210
    assert r["executed_cycle_count"] is None and r["grid_shape"] == [18, 36, 40]
    assert r["stages_seconds"]["phasing"] == 0.603 and r["stages_seconds"]["total"] == 0.727
    r = m.listing_effort(text.split("Total time:")[0])
    assert r["stages_seconds"]["total"] is None and r["listed_iteration_sum"] == 210
    assert not m.listing_effort("truncated")["trials"]
