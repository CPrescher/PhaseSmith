"""Render the preserved known-count and conditional ring controls; no new fits."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path("runs/profile-bias-investigation-20261025")
    destination = Path("docs/figures")
    destination.mkdir(exist_ok=True)
    native = json.loads((root / "known-intensity/native-results.json").read_text())
    dials = json.loads((root / "known-intensity/dials-results.json").read_text())
    fig, ax = plt.subplots(figsize=(9, 5.4), layout="constrained")
    methods = [
        (native["native_frozen_strong_shapes"], "Frozen strong-reference shapes", "#2166ac", "o"),
        (native["native_physical_variations"], "Width fitted to each weak spot", "#d95f02", "s"),
        (native["native_narrow_profile_control"], "Deliberately narrow profile", "#b2182b", "^"),
        (
            native["native_background_low_by_one_count"],
            "Background low by 1 count/pixel",
            "#238b45",
            "D",
        ),
    ]
    ax.plot([0, 300], [0, 300], color="0.5", linestyle="--", label="Known intensity")
    for records, label, color, marker in methods:
        weak = [r for r in records if r["truth"] <= 300]
        ax.plot(
            [r["truth"] for r in weak],
            [r["mean"] for r in weak],
            marker=marker,
            color=color,
            label=label,
        )
    weak = [r for r in dials if r["truth"] <= 300]
    ax.scatter(
        [r["truth"] for r in weak],
        [r["mean"] for r in weak],
        marker="x",
        color="black",
        s=65,
        zorder=5,
        label="DIALS, known profile/background",
    )
    ax.set(
        xlabel="True integrated intensity (counts)",
        ylabel="Mean fitted intensity (counts)",
        title="Weak-intensity controls: 500 Poisson trials per level",
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.15)
    fig.savefig(destination / "fe7c3-profile-known-counts.png", dpi=180)
    plt.close(fig)

    ring = json.loads((root / "ring-background/results.json").read_text())["results"]
    fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
    positions = np.arange(2)
    for delta, name, label, color in [
        (-0.18, "plane", "Local plane only", "#d95f02"),
        (0.18, "known_ring_plus_plane", "Known ring shape + local plane", "#2166ac"),
    ]:
        ax.bar(
            positions + delta,
            [r[name]["mean"] for r in ring.values()],
            width=0.34,
            label=label,
            color=color,
        )
    ax.axhline(1000, color="black", linestyle="--", label="True intensity: 1,000")
    ax.set(
        xticks=positions,
        xticklabels=["On ring", "Ring shoulder"],
        ylabel="Mean fitted intensity (counts)",
        title="Conditional ring test: 30 trials; supplied ring shape is exact",
    )
    ax.legend(fontsize=9)
    fig.savefig(destination / "fe7c3-profile-ring-control.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
