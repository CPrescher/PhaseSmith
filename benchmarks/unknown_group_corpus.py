"""Frozen cross-structure effort/comparator replay; all large artifacts are external.

Prepare before running, run before grading. No reference coordinates/counts enter
the workers. This uses the existing native diagnostic host unchanged; its unused
MgSiO3 HDF5 metadata initialization is included in process wall time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from fractions import Fraction
from pathlib import Path

import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.solution_corpus.run import launch


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def symmetry_instructions(packet):
    """Only Laue rotations and centring, removing all screw/glide translations."""
    group = gemmi.find_spacegroup_by_name(packet["space_group"])
    lattice = {"P": 1, "I": 2, "R": 3, "F": 4, "A": 5, "B": 6, "C": 7}[
        group.operations().find_centering()
    ]
    rotations = set()
    for original in group.operations().sym_ops:
        op = gemmi.Op(original.triplet())
        op.tran = [0, 0, 0]
        if op.det_rot() < 0:
            op = gemmi.Op("-x,-y,-z").combine(op)
        rotations.add(op.triplet())
    return "\n".join(
        [
            "TITL Element-only unknown space-group comparison",
            "CELL " + " ".join(map(str, [packet["wavelength"], *packet["cell"]])),
            f"LATT {lattice}",
            *["SYMM " + op for op in sorted(rotations - {"x,y,z"})],
            "SFAC " + " ".join(packet["elements"]),
            "UNIT " + " ".join("1" for _ in packet["elements"]),
            "HKLF 4",
            "END",
            "",
        ]
    )


def symmetry_prior_from_instructions(text):
    """Read only the same Laue/centring record supplied to SHELXT."""
    lines = text.splitlines()
    lattice = abs(int(next(line.split()[1] for line in lines if line.startswith("LATT "))))
    shifts = {
        1: [(0, 0, 0)],
        2: [(0, 0, 0), (12, 12, 12)],
        3: [(0, 0, 0), (16, 8, 8), (8, 16, 16)],
        4: [(0, 0, 0), (0, 12, 12), (12, 0, 12), (12, 12, 0)],
        5: [(0, 0, 0), (0, 12, 12)],
        6: [(0, 0, 0), (12, 0, 12)],
        7: [(0, 0, 0), (12, 12, 0)],
    }[lattice]
    operations = [gemmi.Op("x,y,z")] + [
        gemmi.Op(line[5:]) for line in lines if line.startswith("SYMM ")
    ]
    rotations = sorted(
        {
            tuple(tuple(sign * v // 24 for v in row) for row in op.rot)
            for op in operations
            for sign in (1, -1)
        }
    )
    translations = [
        [[Fraction(v, 24).numerator, Fraction(v, 24).denominator] for v in shift]
        for shift in shifts
    ]
    return {
        "laue_rotations": rotations,
        "centring_translations": translations,
        "provenance": (
            "Same translation-free Laue rotations and LATT centring supplied to SHELXT; "
            "published intensity-symmetry prior, exact group and counts withheld."
        ),
    }


def prepare(args):
    root = args.output.resolve()
    root.mkdir()
    cases = read(args.manifest)["cases"]
    buckets = {}
    for case in cases:
        system = gemmi.find_spacegroup_by_name(case["curation"]["space_group"]).crystal_system_str()
        buckets.setdefault(system, []).append(case)
    chosen = []
    for system, pool in sorted(buckets.items()):
        ordered = sorted(
            pool,
            key=lambda c: hashlib.sha256(("unknown-effort-v1:" + c["id"]).encode()).hexdigest(),
        )
        first = ordered[0]
        second = next(
            (
                c
                for c in ordered[1:]
                if c["curation"]["chemistry"] != first["curation"]["chemistry"]
            ),
            ordered[1],
        )
        chosen.extend((system, c) for c in (first, second))
    records = []
    for system, case in chosen:
        dest = root / case["id"]
        dest.mkdir()
        files = case["files"]
        for key, name in (
            ("common_hkl", "common.hkl"),
            ("inputs", "source-inputs.json"),
            ("reference", "reference.cif"),
        ):
            source = args.corpus_repository / files[key]["path"]
            if sha(source) != files[key]["sha256"]:
                raise ValueError(f"source hash mismatch: {source}")
            shutil.copyfile(source, dest / name)
        packet = read(dest / "source-inputs.json")
        rows = []
        for line in (dest / "common.hkl").read_text().splitlines():
            hkl = [int(line[i : i + 4]) for i in (0, 4, 8)]
            if hkl != [0, 0, 0]:
                rows.append([hkl, [float(line[12:20]), float(line[20:28])]])
        write(dest / "observations.json", rows)
        (dest / "shelxt.ins").write_text(symmetry_instructions(packet))
        native = read(args.template)
        native["priors"] = {
            "cell_angstrom_degrees": packet["cell"],
            "wavelength_angstrom": packet["wavelength"],
            "elements": packet["elements"],
            "provenance": (
                "Exposed measured COD regression: cell and non-H element types only; no Laue, "
                "exact SG, counts or coordinates. "
            ),
        }
        if args.matched_priors:
            native["priors"]["symmetry"] = symmetry_prior_from_instructions(
                (dest / "shelxt.ins").read_text()
            )
            native["priors"]["provenance"] = (
                "Supplied cell, non-H elements and matched Laue/centring; "
                "no exact space group, atom counts or coordinates."
            )
            native["protocol"]["status_at_freeze"] = (
                "Matched-prior symmetry/acceptance revision; unchanged v40 numerical resource "
                "controls. Inputs and executable frozen before the new comparison."
            )
            native["protocol"]["scope"] = (
                "Exposed intensity-file regression; no frame reduction or atomic reference "
                "input. Same supplied Laue/centring as SHELXT."
            )
            native["protocol"]["space_group_search"]["catalogue"] = (
                "All metric-compatible settings retained; explicit Laue/centring prior "
                "determines processing symmetry and model eligibility. Exact group withheld."
            )
        for mode in ("fast", "auto"):
            native["protocol"]["space_group_search"]["effort"]["mode"] = mode
            write(dest / f"{mode}.json", native)
        records.append(
            {
                "id": case["id"],
                "system": system,
                "chemistry": case["curation"]["chemistry"],
                "rows": len(rows),
                "files": {p.name: sha(p) for p in sorted(dest.iterdir())},
            }
        )
    shutil.copyfile(args.native, root / "native")
    shutil.copymode(args.native, root / "native")
    shutil.copyfile(args.rotations, root / "rotations.json")
    write(
        root / "protocol.json",
        {
            "schema": "phasesmith.unknown-effort-corpus.v2"
            if args.matched_priors
            else "phasesmith.unknown-effort-corpus.v1",
            "matched_symmetry_priors": args.matched_priors,
            "selection": (
                "Two per crystal system: SHA256(unknown-effort-v1:<id>) order, second from "
                "other light/heavy tier if available. Selection never reads solver outcomes. "
            ),
            "cases": records,
            "source_manifest_sha256": sha(args.manifest),
            "native_sha256": sha(root / "native"),
            "shelxt": str(args.shelxt.resolve()),
            "shelxt_sha256": sha(args.shelxt),
            "harness_sha256": sha(__file__),
            "repeats": {"fast": 3, "auto": 1, "shelxt4": 1, "shelxt10": 3},
            "native_controls": (
                "Unchanged v40; four phasing workers, 8 s fast, shared 180 s auto budget, 128 "
                "MiB dense model budget; seed 0. "
            ),
            "comparator_priors": (
                "Both receive identical Laue rotations and centring from the SHELXT INS; "
                "exact space group/counts/coordinates withheld. Native retains all metric "
                "settings with explicit prior-exclusion status."
            )
            if args.matched_priors
            else (
                "SHELXT alone receives published Laue rotations and lattice centring; exact "
                "SG translations, coordinates and counts withheld. No -s. Native infers "
                "symmetry. "
            ),
            "grading": (
                "After all predictions saved: exact non-H species counts, RMS <=0.025 A and "
                "max <=0.05 A after shared origin/inversion; strict max <=0.02 A. Group "
                "number checked separately. No setting-change failure called wrong without "
                "qualification. "
            ),
            "secondary_grading": (
                "Report unlabelled non-H positions with the same total-count, "
                "origin/inversion and distance gates; never select models using this score."
            ),
            "acceptance_policy": (
                "Explicit prior-compatible group; reject measured Laue contradiction, "
                "unresolved higher Laue without a prior, BIC-close different group types, "
                "insufficient common observations and acentric models with consistent inversion "
                "phases. Existing own/common R1 and phase-alpha gates unchanged. When neither "
                "pass is accepted, retain an existing first approximation and archive the other."
            ),
            "hard_seconds": 210,
        },
    )
    print(json.dumps(records, indent=2), flush=True)


def run(args):
    root = args.output.resolve()
    protocol = read(root / "protocol.json")
    if (
        sha(root / "native") != protocol["native_sha256"]
        or sha(protocol["shelxt"]) != protocol["shelxt_sha256"]
    ):
        raise ValueError("executable changed")
    for case in protocol["cases"]:
        source = root / case["id"]
        for name, digest in case["files"].items():
            if sha(source / name) != digest:
                raise ValueError(f"frozen input changed: {name}")
        for mode, repeats in protocol["repeats"].items():
            for repeat in range(1, repeats + 1):
                dest = source / f"{mode}-{repeat}"
                if (dest / "process.json").exists():
                    continue
                dest.mkdir()
                if mode in ("fast", "auto"):
                    command = [
                        str(root / "native"),
                        str(source / f"{mode}.json"),
                        str(root / "rotations.json"),
                        str(dest / "model"),
                        str(source / "observations.json"),
                    ]
                    workers = 4
                else:
                    shutil.copyfile(source / "common.hkl", dest / "case.hkl")
                    shutil.copyfile(source / "shelxt.ins", dest / "case.ins")
                    command = [protocol["shelxt"], "case"]
                    workers = int(mode.removeprefix("shelxt"))
                result = launch(command, dest, protocol["hard_seconds"], threads=workers)
                write(dest / "process.json", result)
                print(
                    case["id"],
                    mode,
                    repeat,
                    round(result["wall_seconds_including_startup"], 4),
                    result["returncode"],
                    flush=True,
                )
    write(root / "predictions-complete.json", {"protocol_sha256": sha(root / "protocol.json")})


def grade(args):
    from benchmarks.solution_corpus.audit import match_geometry, shelxt_candidate
    from benchmarks.solution_corpus.data import atom_block, expanded_sites

    root = args.output.resolve()
    if not (root / "predictions-complete.json").exists():
        raise ValueError("save every prediction before grading")
    protocol = read(root / "protocol.json")
    results = []
    for case in protocol["cases"]:
        source = root / case["id"]
        packet = read(source / "source-inputs.json")
        group = gemmi.find_spacegroup_by_name(packet["space_group"])
        labels, xyz = expanded_sites(atom_block(source / "reference.cif"), group)
        for mode, repeats in protocol["repeats"].items():
            for repeat in range(1, repeats + 1):
                dest = source / f"{mode}-{repeat}"
                record = {**case, "mode": mode, "repeat": repeat, **read(dest / "process.json")}
                record.pop("files")
                try:
                    if mode.startswith("shelxt"):
                        model = shelxt_candidate(dest / "case_a.res")
                    else:
                        search = read(dest / "model/space-group-search.json")
                        record["effort"] = search["effort"]
                        record["selected"] = search["selected"]
                        record["search_termination"] = search.get("termination")
                        if search["selected"] is None:
                            record["model_status"] = "no_model"
                            results.append(record)
                            continue
                        # The full-cell candidate is P1 and preserves the selected
                        # candidate's coordinate origin without assuming Hall choice.
                        block = atom_block(dest / "model/candidate/best.cif")
                        species, points = expanded_sites(block, gemmi.find_spacegroup_by_number(1))
                        model = {
                            "cell": list(
                                gemmi.make_small_structure_from_block(block).cell.parameters
                            ),
                            "species": species,
                            "fractional_xyz": points.tolist(),
                            "space_group_number": search["selected"]["number"],
                        }
                        record["native_search"] = read(dest / "model/search.json")
                    record["model"] = model
                    record["geometry"] = match_geometry(model, labels, xyz, packet["cell"])
                    record["space_group_number_matches"] = (
                        model["space_group_number"] == group.number
                    )
                except Exception as error:
                    record["grading_error"] = f"{type(error).__name__}: {error}"
                results.append(record)
    write(
        root / "graded.json", {"protocol_sha256": sha(root / "protocol.json"), "results": results}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "grade"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--corpus-repository", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--rotations", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--shelxt", type=Path)
    parser.add_argument("--matched-priors", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "grade": grade}[args.command](args)


if __name__ == "__main__":
    main()
