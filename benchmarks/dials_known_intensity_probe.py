"""External DIALS fixed-profile response to frozen known-intensity pixels."""

import json
from pathlib import Path

import numpy as np
from dials.algorithms.integration.fit import ProfileFitter
from dials.array_family import flex

root = Path("runs/profile-bias-investigation-20261025/known-intensity")
source = np.load(root / "oracle-input.npz")
values = []
for data, bg, p in zip(source["data"], source["background"], source["profiles"], strict=True):
    fit = ProfileFitter(
        flex.double(data), flex.double(bg), flex.bool(p > 0), flex.double(p), 1e-8, 100
    )
    values.append([fit.intensity()[0], np.sqrt(fit.variance()[0])])
values = np.array(values)
report = []
for level in np.unique(source["truth"]):
    selected = source["truth"] == level
    v = values[selected]
    delta = v[:, 0] - level
    report.append(
        dict(
            truth=float(level),
            n=int(selected.sum()),
            mean=float(v[:, 0].mean()),
            bias=float(delta.mean()),
            mean_sigma=float(v[:, 1].mean()),
            negative=int(np.count_nonzero(v[:, 0] < 0)),
            mean_pull=float(np.mean(delta / v[:, 1])),
            pull_std=float(np.std(delta / v[:, 1], ddof=1)),
        )
    )
(root / "dials-results.json").write_text(json.dumps(report, indent=2) + "\n")
np.savez_compressed(root / "dials-arrays.npz", truth=source["truth"], values=values)
print(json.dumps(report, indent=2))
