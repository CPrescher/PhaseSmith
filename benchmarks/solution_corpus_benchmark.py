"""Explicit COD corpus acquisition, sealed splits and isolated solver comparisons."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from solution_corpus.data import (
    EXPOSED_IDS,
    assign_splits,
    common_hkl,
    fetch,
    inspect_reference,
    metadata_order,
    reflection_data,
    sha,
)

ROOT = Path(__file__).resolve().parents[1]


def curate(args):
    if args.manifest.exists():
        raise ValueError("manifest is immutable; choose a new path for an explicit new version")
    queries = []
    rows = {}
    for path in sorted(args.discovery.glob("*.json")):
        if path.name.endswith("-source.json"):
            queries.append({"file": path.name, **json.loads(path.read_text())})
            continue
        records = json.loads(path.read_text())
        if isinstance(records, list):
            for row in records:
                rows[row["file"]] = row
    inventory, cases = [], []
    seen_doi, seen_formula = set(), set()
    args.data.mkdir(exist_ok=True, parents=True)
    for row in metadata_order(rows.values()):
        case_id = row["file"]
        record = {"id": case_id, "metadata": row}
        directory = args.data / ("cod-" + case_id)
        try:
            formula = " ".join((row.get("formula") or "").strip(" -").split())
            if formula in {"Br F4 K", "Bi I Se", "C H4 N2 S"}:
                raise ValueError("previously exposed formula family")
            if row.get("doi", "").lower() in seen_doi or formula in seen_formula:
                raise ValueError("duplicate DOI/formula family in first diversity panel")
            url = "https://www.crystallography.net/cod/" + case_id
            cif = directory / (case_id + ".cif")
            record["reference_file"] = fetch(url + ".cif", cif)
            curation = inspect_reference(cif, row)
            hkl = directory / (case_id + ".hkl")
            record["observation_file"] = fetch(url + ".hkl", hkl)
            measured, observation = reflection_data(hkl)
            if len(measured.hkl) < 30:
                raise ValueError("fewer than 30 measured reflections")
            text, rounding = common_hkl(measured)
            prepared = args.data / "prepared-v3" / ("cod-" + case_id)
            prepared.mkdir(exist_ok=True, parents=True)
            common = prepared / "common.hkl"
            common.write_text(text)
            inputs = {
                key: curation[key]
                for key in [
                    "cell",
                    "space_group",
                    "space_group_number",
                    "shelxt_space_group",
                    "wavelength",
                    "elements",
                ]
            }
            inputs["id"] = "cod-" + case_id
            inputs_path = prepared / "inputs.json"
            inputs_path.write_text(json.dumps(inputs, indent=2) + "\n")
            record.update(
                status="accepted",
                curation=curation,
                observation=observation,
                common_input_quantization=rounding,
            )
            cases.append(
                {
                    "id": "cod-" + case_id,
                    "curation": curation,
                    "metadata": row,
                    "observation": observation,
                    "files": {
                        "reference": {
                            **record["reference_file"],
                            "path": str(cif.relative_to(ROOT)),
                        },
                        "observations": {
                            **record["observation_file"],
                            "path": str(hkl.relative_to(ROOT)),
                        },
                        "common_hkl": {
                            "path": str(common.relative_to(ROOT)),
                            "sha256": sha(common),
                        },
                        "inputs": {
                            "path": str(inputs_path.relative_to(ROOT)),
                            "sha256": sha(inputs_path),
                        },
                    },
                    "common_input_quantization": rounding,
                }
            )
            seen_doi.add(curation["doi"])
            seen_formula.add(curation["formula_group"])
            print(
                f"accepted {len(cases)}/{args.target}: {case_id}, "
                f"SG {curation['space_group_number']}, "
                f"{curation['reference_non_h_atoms']} non-H atoms",
                flush=True,
            )
        except Exception as error:
            record.update(status="excluded", reason=f"{type(error).__name__}: {error}")
        inventory.append(record)
        (args.data / "acquisition-progress.json").write_text(json.dumps(inventory, indent=2) + "\n")
        if len(cases) == args.target:
            break
    split_counts = assign_splits(cases)
    report = {
        "schema": "phasesmith.experimental-solution-corpus.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "solver_baseline_commit": "8bc4efc",
        "target_cases": args.target,
        "license": "CC0-1.0; acknowledge original authors and COD",
        "license_url": "https://www.crystallography.net/",
        "api_documentation": "https://wiki.crystallography.net/RESTful_API/",
        "discovery_queries": queries,
        "discovery_sha256": {p.name: sha(p) for p in args.discovery.glob("*.json")},
        "selection": "SG-stratified deterministic SHA order; "
        "one DOI and formula per accepted case; "
        "source/format eligibility only, no solver outcomes",
        "split_rule": "DOI/formula connected components, SHA256('phasesmith-corpus-v1:'+family) "
        "mod 3: zero development, others holdout; immutable version",
        "exposed_development_ids_excluded": sorted(EXPOSED_IDS),
        "contract": {
            "exact_space_group_supplied": True,
            "element_types_only": True,
            "input_atom_counts": False,
            "hydrogens_scored": False,
            "maximum_atoms_upper_bound": 128,
            "runtime_atom_cap_policy": (
                "min(128,ceil(V/8),floor((floor(2000000/M)-7)/5)); observed symmetry-expanded M"
            ),
            "maximum_elements": 8,
            "source_tier": "full-occupancy, untwinned single-crystal X-ray, reference setting",
            "common_hkl_relative_quantization_cap": 5e-4,
        },
        "cases": cases,
        "split_counts": dict(split_counts),
        "space_group_counts": dict(
            Counter(str(c["curation"]["space_group_number"]) for c in cases)
        ),
        "family_count": len({c["family"] for c in cases}),
        "inventory": inventory,
        "acquisition_status": "target_met"
        if len(cases) == args.target
        else "candidate_pool_exhausted",
    }
    args.manifest.parent.mkdir(exist_ok=True, parents=True)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "accepted": len(cases),
                "splits": dict(split_counts),
                "excluded": len(inventory) - len(cases),
                "groups": len(report["space_group_counts"]),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("curate")
    c.add_argument("--discovery", type=Path, required=True)
    c.add_argument("--data", type=Path, default=ROOT / "validation/data/solution-corpus-v1")
    c.add_argument("--manifest", type=Path, default=ROOT / "validation/solution-corpus-v5.json")
    c.add_argument("--target", type=int, default=100)
    seal_parser = commands.add_parser("seal")
    seal_parser.add_argument(
        "--manifest", type=Path, default=ROOT / "validation/solution-corpus-v5.json"
    )
    seal_parser.add_argument("--discovery", type=Path, required=True)
    seal_parser.add_argument("--lock-revision", type=int, default=4)
    fetch_parser = commands.add_parser("fetch")
    fetch_parser.add_argument(
        "--manifest", type=Path, default=ROOT / "validation/solution-corpus-v5.json"
    )
    r = commands.add_parser("run")
    r.add_argument("--manifest", type=Path, default=ROOT / "validation/solution-corpus-v5.json")
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--lock-revision", type=int, default=4)
    r.add_argument("--split", choices=["development", "holdout"], default="development")
    r.add_argument("--open-holdout", action="store_true")
    r.add_argument("--seconds", type=float, default=30)
    r.add_argument(
        "--shelxt",
        type=Path,
        default=Path("/Users/clemens/.local/opt/shelx/versions/2026-10-28/bin/shelxt"),
    )
    w = commands.add_parser("worker")
    w.add_argument("--input", type=Path, required=True)
    w.add_argument("--hkl", type=Path, required=True)
    w.add_argument("--output", type=Path, required=True)
    w.add_argument("--seconds", type=float, required=True)
    w.add_argument("--maximum-atoms", type=int, required=True)
    args = parser.parse_args()
    for name in ["manifest", "discovery", "data", "output", "input", "hkl", "shelxt"]:
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.command == "curate":
        if not 1 <= args.target <= 100:
            parser.error("target must lie in [1,100]")
        curate(args)
    elif args.command == "seal":
        from solution_corpus.seal import seal

        seal(args)
    elif args.command == "fetch":
        from solution_corpus.seal import fetch_manifest

        fetch_manifest(args)
    elif args.command == "worker":
        from solution_corpus.run import worker

        worker(args)
    else:
        if args.split == "holdout" and not args.open_holdout:
            parser.error("holdout stays sealed; --open-holdout explicitly records its opening")
        if not 5 <= args.seconds <= 120:
            parser.error("seconds must lie in [5,120]")
        from solution_corpus.run import run

        run(args)


if __name__ == "__main__":
    main()
