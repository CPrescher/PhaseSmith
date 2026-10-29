"""Explicitly verify and seal a corpus before accepting solver outcomes."""

import json
import shutil
from collections import Counter

import gemmi

from .data import assign_splits, common_hkl, inspect_reference, reflection_data, sha
from .run import INPUT_KEYS, ROOT, SCRIPT


def seal(args):
    lock_path = args.manifest.with_suffix(f".benchmark{args.lock_revision}.lock.json")
    if lock_path.exists():
        raise ValueError("corpus lock is immutable; use an explicitly versioned new corpus")
    manifest = json.loads(args.manifest.read_text())
    cases = manifest["cases"]
    recomputed = [{**c} for c in cases]
    assign_splits(recomputed)
    if [(c["id"], c["family"], c["split"]) for c in recomputed] != [
        (c["id"], c["family"], c["split"]) for c in cases
    ]:
        raise ValueError("split/family assignment differs from declared protocol")
    quality = []
    for case in cases:
        for entry in case["files"].values():
            if sha(ROOT / entry["path"]) != entry["sha256"]:
                raise ValueError("case checksum mismatch: " + case["id"])
        info = inspect_reference(ROOT / case["files"]["reference"]["path"], case["metadata"])
        if info != case["curation"]:
            raise ValueError("curation changed: " + case["id"])
        raw, observed = reflection_data(ROOT / case["files"]["observations"]["path"])
        text, quantization = common_hkl(raw)
        if text != (ROOT / case["files"]["common_hkl"]["path"]).read_text():
            raise ValueError("common measured-data translation changed: " + case["id"])
        packet = json.loads((ROOT / case["files"]["inputs"]["path"]).read_text())
        if set(packet) != INPUT_KEYS or any(packet[k] != info[k] for k in INPUT_KEYS - {"id"}):
            raise ValueError("solver packet contains incorrect or private reference fields")
        quality.append(
            {
                "id": case["id"],
                "observations": observed["observations"],
                "quantization": quantization,
                "source_hashes_verified": True,
                "reference_format_symmetry_occupancy_verified": True,
            }
        )
    snapshot = args.manifest.parent / "data/solution-corpus-v1/discovery-v3"
    snapshot.mkdir(exist_ok=True, parents=True)
    for name, expected in manifest["discovery_sha256"].items():
        source = args.discovery / name
        if sha(source) != expected:
            raise ValueError("discovery snapshot changed")
        shutil.copyfile(source, snapshot / name)
    impl = [*sorted((ROOT / "benchmarks/solution_corpus").glob("*.py")), SCRIPT]
    report = {
        "schema": "phasesmith.solution-corpus-lock.v1",
        "manifest_sha256": sha(args.manifest),
        "cases": len(cases),
        "families": len({c["family"] for c in cases}),
        "splits": dict(Counter(c["split"] for c in cases)),
        "crystal_systems": dict(
            Counter(
                gemmi.find_spacegroup_by_name(c["curation"]["space_group"]).crystal_system_str()
                for c in cases
            )
        ),
        "space_group_types": len({c["curation"]["space_group_number"] for c in cases}),
        "chemistry": dict(Counter(c["curation"]["chemistry"] for c in cases)),
        "hydrogen_present": sum(c["curation"]["hydrogen_present"] for c in cases),
        "observation_count_range": [
            min(q["observations"] for q in quality),
            max(q["observations"] for q in quality),
        ],
        "gemmi_version": gemmi.__version__,
        "discovery_snapshot": str(snapshot.relative_to(ROOT)),
        "holdout_solver_outcomes_opened": False,
        "verified_implementation_sha256": {str(p.relative_to(ROOT)): sha(p) for p in impl},
        "quality_checks": quality,
        "qualification": (
            "Format, provenance and supported-model checks do not prove the published "
            "model is physically correct; no solver outcomes used in curation."
        ),
    }
    lock_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ["quality_checks", "verified_implementation_sha256"]
            },
            indent=2,
        )
    )


def fetch_manifest(args):
    """Retrieve pinned raw files and reconstruct only checksum-identical inputs."""
    import hashlib

    from .data import fetch

    manifest = json.loads(args.manifest.read_text())
    for case in manifest["cases"]:
        for role in ["reference", "observations"]:
            entry = case["files"][role]
            target = (ROOT / entry["path"]).resolve()
            if not target.is_relative_to(ROOT / "validation/data"):
                raise ValueError("manifest data path escapes validation/data")
            if target.exists() and sha(target) != entry["sha256"]:
                raise ValueError("existing raw file differs from pinned checksum")
            fetch(entry["url"], target)
            if sha(target) != entry["sha256"]:
                raise ValueError(
                    "download changed; raw source retained for review, manifest not regenerated"
                )
        raw, _ = reflection_data(ROOT / case["files"]["observations"]["path"])
        text, _ = common_hkl(raw)
        packet = {key: case["curation"][key] for key in INPUT_KEYS - {"id"}}
        packet["id"] = case["id"]
        # Serialization order is explicit in the frozen input-packet contract.
        ordered = {
            key: packet[key]
            for key in [
                "cell",
                "space_group",
                "space_group_number",
                "shelxt_space_group",
                "wavelength",
                "elements",
                "id",
            ]
        }
        for role, value in [("common_hkl", text), ("inputs", json.dumps(ordered, indent=2) + "\n")]:
            entry = case["files"][role]
            target = (ROOT / entry["path"]).resolve()
            if not target.is_relative_to(ROOT / "validation/data"):
                raise ValueError("derived path escapes validation/data")
            if hashlib.sha256(value.encode()).hexdigest() != entry["sha256"]:
                raise ValueError("adapter differs from frozen input checksum; no regeneration")
            if target.exists():
                if sha(target) != entry["sha256"]:
                    raise ValueError("existing prepared input differs from checksum")
            else:
                target.parent.mkdir(exist_ok=True, parents=True)
                target.write_text(value)
    print(f"Verified pinned raw and derived files for {len(manifest['cases'])} cases")
