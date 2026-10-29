"""Post-solve structural audits, independent of either solver's fit objective."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import gemmi
import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import atom_block, expanded_sites


def shelxt_candidate(path):
    lines = Path(path).read_text().splitlines()
    text = "\n".join(lines)
    if not re.search(r"(?m)^HKLF\s+4", text) or not re.search(r"(?m)^END\s*$", text):
        raise ValueError("incomplete SHELXT model file")
    match = re.search(r"REM SHELXT solution in (.+?): R1\s+([0-9.]+)", text)
    if not match:
        raise ValueError("missing SHELXT solution/space-group declaration")
    group = gemmi.find_spacegroup_by_name(match[1].replace("(", "").replace(")", ""))
    if group is None:
        raise ValueError("unrecognized output space group")
    # The display name may omit the origin choice. Recover the actual group
    # from output LATT/SYMM instead of guessing an origin from that label.
    lattice_lines = [line.split() for line in lines if line.startswith("LATT ")]
    if lattice_lines:
        lattice = int(lattice_lines[0][1])
        shifts = {
            1: [(0, 0, 0)],
            2: [(0, 0, 0), (12, 12, 12)],
            3: [(0, 0, 0), (16, 8, 8), (8, 16, 16)],
            4: [(0, 0, 0), (0, 12, 12), (12, 0, 12), (12, 12, 0)],
            5: [(0, 0, 0), (0, 12, 12)],
            6: [(0, 0, 0), (12, 0, 12)],
            7: [(0, 0, 0), (12, 12, 0)],
        }[abs(lattice)]
        primitive = [gemmi.Op("x,y,z")] + [
            gemmi.Op(line[5:]) for line in lines if line.startswith("SYMM ")
        ]
        if lattice > 0:
            primitive += [gemmi.Op("-x,-y,-z").combine(op) for op in primitive.copy()]
        full = {}
        for op in primitive:
            for shift in shifts:
                value = gemmi.Op(op.triplet())
                value.tran = [op.tran[i] + shift[i] for i in range(3)]
                full[value.wrap().triplet()] = value
        operations = gemmi.GroupOps(list(full.values()))
        operations.add_missing_elements()
        actual = gemmi.find_spacegroup_by_ops(operations)
        if actual is None or actual.number != group.number:
            raise ValueError("output symmetry operations disagree with its declared space group")
        group = actual
    cell, elements, atoms = None, None, []
    for line in lines:
        words = line.split()
        if not words:
            continue
        if words[0] == "CELL":
            cell = [float(x) for x in words[2:8]]
        elif words[0] == "SFAC":
            elements = words[1:]
        elif words[0] == "HKLF":
            break
        elif re.fullmatch(r"[A-Za-z]{1,3}\d\w*", words[0]) and len(words) >= 7:
            if elements is None:
                raise ValueError("atom record precedes SFAC")
            index = int(words[1]) - 1
            if not 0 <= index < len(elements):
                raise ValueError("atom SFAC index out of range")
            element = gemmi.Element(elements[index]).name
            point = [float(x) for x in words[2:5]]
            sof = float(words[5])
            if 10 <= sof <= 11.00001:
                sof -= 10
            elif not 0 <= sof <= 1.00001:
                raise ValueError("unsupported free-variable site occupancy")
            images = []
            for op in group.operations():
                image = np.mod(op.apply_to_xyz(point), 1)
                if not any(np.linalg.norm((image - p + 0.5) % 1 - 0.5) < 1e-4 for p in images):
                    images.append(image)
            occupancy = sof * len(group.operations()) / len(images)
            if abs(occupancy - 1) > 1e-3:
                raise ValueError("output has partial non-H occupancy")
            if gemmi.Element(element).atomic_number > 1:
                atoms.extend((element, p.tolist()) for p in images)
    if cell is None or not atoms:
        raise ValueError("missing output cell/atoms")
    return {
        "cell": cell,
        "species": [a[0] for a in atoms],
        "fractional_xyz": [a[1] for a in atoms],
        "space_group": group.xhm(),
        "space_group_number": group.number,
        "reported_r1": float(match[2]),
        "source_model": Path(path).name,
    }


def match_geometry(model, reference_species, reference_xyz, cell):
    if Counter(model["species"]) != Counter(reference_species):
        return {"counts_match": False, "correct": False}
    if not np.allclose(model["cell"], cell, rtol=1e-5, atol=1e-4):
        return {
            "counts_match": True,
            "correct": False,
            "status": "unassessed_output_cell_setting_change",
        }
    source = np.asarray(model["fractional_xyz"], float)
    target = np.asarray(reference_xyz, float)
    species = np.asarray(model["species"])
    labels = np.asarray(reference_species)
    if source.shape != target.shape or not np.isfinite(source).all():
        raise ValueError("invalid output model coordinates")
    metric_cell = gemmi.UnitCell(*cell)
    anchor_type = min(Counter(labels), key=lambda label: (np.count_nonzero(labels == label), label))
    anchor = int(np.flatnonzero(species == anchor_type)[0])
    best = None
    for hand in (1, -1):
        for point in target[labels == anchor_type]:
            shift = point - hand * source[anchor]
            for iteration in range(5):
                current = np.mod(hand * source + shift, 1)
                distances, corrections = [], []
                for element in sorted(set(labels)):
                    left, right = (
                        np.flatnonzero(labels == element),
                        np.flatnonzero(species == element),
                    )
                    images = [
                        [
                            metric_cell.find_nearest_pbc_image(
                                gemmi.Fractional(*target[i]), gemmi.Fractional(*current[j]), 0
                            )
                            for j in right
                        ]
                        for i in left
                    ]
                    matrix = np.array([[v.dist() for v in row] for row in images])
                    ii, jj = linear_sum_assignment(matrix**2)
                    distances.extend(matrix[ii, jj])
                    corrections.extend(
                        target[left[i]] - current[right[j]] - np.array(images[i][j].pbc_shift)
                        for i, j in zip(ii, jj, strict=True)
                    )
                delta = np.mean(corrections, axis=0)
                if iteration == 4 or np.linalg.norm(delta) < 1e-10:
                    break
                shift += delta
            value = {
                "rms_error_angstrom": float(np.sqrt(np.mean(np.square(distances)))),
                "maximum_error_angstrom": float(np.max(distances)),
                "hand": hand,
                "shift": np.mod(shift, 1).tolist(),
            }
            if best is None or (value["rms_error_angstrom"], value["maximum_error_angstrom"]) < (
                best["rms_error_angstrom"],
                best["maximum_error_angstrom"],
            ):
                best = value
            if best["maximum_error_angstrom"] < 1e-8:
                break
    return {
        "counts_match": True,
        **best,
        "correct": best["maximum_error_angstrom"] <= 0.05 and best["rms_error_angstrom"] <= 0.025,
        "strict_002_angstrom": best["maximum_error_angstrom"] <= 0.02,
    }


def common_score(model, observations, cell):
    """Both models use the same fixed Uiso and original signed common intensities."""
    import phasesmith as ps
    from phasesmith.crystallography import UnitCell
    from phasesmith.scattering import XrayNonResonant
    from phasesmith.structure import AtomSite, CrystalStructure
    from phasesmith.symmetry import SpaceGroup

    sites = tuple(
        AtomSite(str(i), f"{e}{i}", e, e, tuple(p), u_iso_angstrom2=0.01)
        for i, (e, p) in enumerate(zip(model["species"], model["fractional_xyz"], strict=True))
    )
    structure = CrystalStructure(
        "common", "common fixed Uiso", UnitCell(*cell), SpaceGroup.p1(), sites
    )
    values = ps.calculate_structure_factor_values(
        structure, observations.hkl, np.ones(len(observations.hkl), np.int64), XrayNonResonant()
    ).f_squared
    fo = np.sqrt(np.maximum(observations.intensity, 0))
    fc = np.sqrt(values)
    k = float(fc @ fo / (fc @ fc))
    amplitude_r1 = float(np.abs(k * fc - fo).sum() / fo.sum())
    observed = observations.intensity / np.max(np.abs(observations.intensity))
    calculated = values / np.max(values)
    weight = (np.min(observations.sigma) / observations.sigma) ** 2
    scale = max(0.0, float((weight * calculated) @ observed / ((weight * calculated) @ calculated)))
    objective = float(
        (weight * (scale * calculated - observed) ** 2).sum() / (weight * observed**2).sum()
    )
    return {
        "uiso_angstrom2": 0.01,
        "amplitude_r1": amplitude_r1,
        "signed_intensity_normalized_weighted_sse": objective,
        "qualification": (
            "common nonresonant evaluator, fixed Uiso, independent fitted scales; "
            "not refinement or published R1"
        ),
    }


def audit_case(case, model, root):
    path = root / case["files"]["reference"]["path"]
    group = gemmi.find_spacegroup_by_name(case["curation"]["space_group"])
    species, xyz = expanded_sites(atom_block(path), group)
    result = match_geometry(model, species, xyz, case["curation"]["cell"])
    result["reference_counts"] = dict(Counter(species))
    result["inferred_counts"] = dict(Counter(model["species"]))
    return result
