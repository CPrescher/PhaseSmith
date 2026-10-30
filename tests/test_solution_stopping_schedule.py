"""The four-mode runner counts case/method rounds separately from predictions."""
import importlib.util
import json
import sys
import pytest
from pathlib import Path


@pytest.mark.parametrize("runner",["solution_stopping_interleaved.py","solution_cycle_effort_interleaved.py"])
def test_four_mode_runner_accepts_all_280_predictions_and_four_barriers(tmp_path,monkeypatch,runner):
    benches=Path(__file__).resolve().parents[1]/'benchmarks'
    monkeypatch.syspath_prepend(str(benches))
    spec=importlib.util.spec_from_file_location('stopping_runner',benches/runner)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'cases':[{'id':str(i),'split':'development'} for i in range(35)]}))
    def jobs(args):
        for i in range(35):
            for method in ['phasesmith','shelxt']:
                yield {'id':str(i),'method':method}
        yield {'reference_audit_barrier':True}
    monkeypatch.setattr(module,'collect_jobs',jobs)
    output=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['runner','--manifest',str(manifest),'--output',str(output)])
    module.main()
    events=json.loads((output/'schedule.json').read_text())
    assert len(events)==284
    assert not any(e.get('reference_audit_barrier') for e in events[:280])
    assert all(e.get('reference_audit_barrier') for e in events[280:])
    for mode in module.MODES:
        assert sum(e['mode']==mode and not e.get('reference_audit_barrier') for e in events)==70
