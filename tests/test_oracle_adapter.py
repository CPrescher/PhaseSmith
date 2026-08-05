from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from rietveld.oracle._pinned_probe import PINNED_REVISION, probe_histogram
from rietveld.oracle.gsasii import extract_snapshot


class FakeHistogram:
    name = "PWDR synthetic"
    data: ClassVar[dict[str, object]] = {
        "Instrument Parameters": [{"U": [1.0]}, {}],
        "Sample Parameters": {"Scale": [2.0, False]},
        "Limits": [None, [10.0, 90.0]],
    }

    def getdata(self, kind: str) -> np.ndarray:
        return {
            "X": np.array([1.0, 2.0, 3.0]),
            "Ycalc": np.array([4.0, 5.0, 6.0]),
            "Background": np.ma.array([0.1, 0.2, 0.3]),
        }[kind]

    def reflections(self) -> dict[str, dict[str, object]]:
        return {
            "phase-b": {
                "RefList": np.array([[1, 0, 0, 2.1]]),
                "Type": "PXC",
                "Super": False,
            },
            "phase-a": {
                "RefList": np.array([[0, 0, 1, 3.2]]),
                "Type": "PXC",
                "Super": False,
            },
        }


def test_public_adapter_returns_copied_plain_arrays() -> None:
    histogram = FakeHistogram()
    snapshot = extract_snapshot(histogram)
    assert snapshot.histogram_name == histogram.name
    np.testing.assert_array_equal(snapshot.x, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(snapshot.ycalc, [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(snapshot.background, [0.1, 0.2, 0.3])
    assert [table.phase for table in snapshot.reflections] == ["phase-a", "phase-b"]
    snapshot.x[0] = -100.0
    assert histogram.getdata("X")[0] == 1.0


def test_internal_probe_is_allowlisted_and_pin_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    module = SimpleNamespace(__file__=__file__)
    monkeypatch.setattr(
        "rietveld.oracle._pinned_probe.detected_revision", lambda _module: PINNED_REVISION
    )
    values = probe_histogram(FakeHistogram(), module, "instrument_parameters", "limits")
    assert values["limits"] == [None, [10.0, 90.0]]
    values["limits"][1][0] = -1.0
    assert FakeHistogram.data["Limits"][1][0] == 10.0
    with pytest.raises(KeyError, match="unsupported probe"):
        probe_histogram(FakeHistogram(), module, "entire_data_tree")


def test_internal_probe_rejects_revision_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rietveld.oracle._pinned_probe.detected_revision", lambda _module: "different"
    )
    with pytest.raises(RuntimeError, match=PINNED_REVISION):
        probe_histogram(FakeHistogram(), SimpleNamespace(__file__=__file__), "limits")
