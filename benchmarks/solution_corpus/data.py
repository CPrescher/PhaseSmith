"""Bounded COD acquisition and input-only experimental diffraction translation."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import gemmi
import numpy as np
from phasesmith.io.hkl import HklData, parse_hkl

MAX_DOWNLOAD = 16 * 1024 * 1024
EXPOSED_IDS = {"1549359", "2010577"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fetch(url, path):
    """Save original bytes atomically, with bounded size/time and no overwrite."""
    path = Path(path)
    if path.exists():
        sidecar = path.with_suffix(path.suffix + ".download.json")
        if sidecar.exists():
            record = json.loads(sidecar.read_text())
            if record["sha256"] != sha(path):
                raise ValueError("cached download checksum mismatch")
            return {**record, "cached": True}
        return {
            "url": url,
            "sha256": sha(path),
            "bytes": path.stat().st_size,
            "cached": True,
            "qualification": "canonical source URL; original mirror route unavailable",
        }
    errors = []
    mirrors = [
        url.replace("https://www.crystallography.net/cod/", base)
        for base in ["https://www.crystallography.net/cod/", "https://qiserver.ugr.es/cod/"]
    ]
    for candidate in mirrors:
        try:
            request = urllib.request.Request(
                candidate, headers={"User-Agent": "PhaseSmith-validation/1.0"}
            )
            with urllib.request.urlopen(request, timeout=12) as response:
                value = response.read(MAX_DOWNLOAD + 1)
                final_url = response.url
            if len(value) > MAX_DOWNLOAD:
                raise ValueError("download exceeds 16 MiB")
            if not value or value.lstrip().lower().startswith((b"<!doctype html", b"<html")):
                raise ValueError("server returned HTML or empty data")
            break
        except Exception as error:
            errors.append({"url": candidate, "error": f"{type(error).__name__}: {error}"})
    else:
        raise OSError("all COD retrieval routes failed: " + json.dumps(errors))
    path.parent.mkdir(exist_ok=True, parents=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(value)
    temporary.replace(path)
    record = {
        "url": url,
        "resolved_url": final_url,
        "sha256": sha(path),
        "bytes": len(value),
        "earlier_retrieval_errors": errors,
    }
    path.with_suffix(path.suffix + ".download.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def number(value):
    return float(gemmi.cif.as_number(value))


def atom_block(path):
    doc = gemmi.cif.read_file(str(path))
    candidates = [b for b in doc if len(b.find_values("_atom_site_fract_x"))]
    if len(candidates) != 1:
        raise ValueError("reference must have one unambiguous atom-coordinate block")
    return candidates[0]


def expanded_sites(block, group, *, tolerance=5e-4):
    structure = gemmi.make_small_structure_from_block(block)
    points, species = [], []
    for site in structure.sites:
        if site.element.atomic_number == 0:
            raise ValueError("unknown reference element")
        if site.element.atomic_number == 1:
            continue
        if not np.isfinite(site.occ) or abs(site.occ - 1) > 1e-5:
            raise ValueError("partial non-H occupancy is outside the clean tier")
        point = np.array([site.fract.x, site.fract.y, site.fract.z])
        if not np.isfinite(point).all():
            raise ValueError("nonfinite reference position")
        images = []
        for op in group.operations():
            image = np.mod(op.apply_to_xyz(point.tolist()), 1)
            if not any(np.linalg.norm((image - p + 0.5) % 1 - 0.5) < tolerance for p in images):
                images.append(image)
        species.extend([site.element.name] * len(images))
        points.extend(images)
    if not points:
        raise ValueError("reference has no non-H atoms")
    return species, np.array(points)


def shelxt_name(group):
    """Convert explicit reference-setting HM tokens to the documented -s spelling."""
    tokens = group.xhm().split(":")[0].split()
    if 3 <= group.number <= 15:
        tokens = [tokens[0]] + [t for t in tokens[1:] if t != "1"]
    return "".join(re.sub(r"([2346])([1-5])", r"\1(\2)", t) for t in tokens).replace("/", "_")


def inspect_reference(path, metadata):
    """Curate independently of outcomes; never return atom coordinates as inputs."""
    block = atom_block(path)
    code = block.find_value("_cod_database_code")
    if code and gemmi.cif.as_string(code) != str(metadata["file"]):
        raise ValueError("reference COD identifier mismatch")
    group = gemmi.find_spacegroup_by_name(metadata["sg"])
    if group is None or not group.is_reference_setting():
        raise ValueError("non-reference space-group setting reserved for a later tier")
    sym = block.find_values("_space_group_symop_operation_xyz")
    if not len(sym):
        sym = block.find_values("_symmetry_equiv_pos_as_xyz")
    if len(sym):
        declared = {gemmi.Op(gemmi.cif.as_string(s)).wrap().triplet() for s in sym}
        expected = {op.wrap().triplet() for op in group.operations()}
        if declared != expected:
            raise ValueError("coordinate-CIF operations disagree with the supplied HM setting")
    cell = [
        number(block.find_value("_cell_" + tag))
        for tag in ["length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma"]
    ]
    if not np.isfinite(cell).all() or min(cell[:3]) <= 0:
        raise ValueError("invalid source cell")
    if not all(40 <= angle <= 140 for angle in cell[3:]):
        raise ValueError("very skew cells reserved for a later tier")
    probe = " ".join(
        str(block.find_value(tag) or "")
        for tag in [
            "_diffrn_radiation_probe",
            "_diffrn_radiation_type",
            "_cod_structure_determination_method",
        ]
    )
    if any(term in probe.lower() for term in ["neutron", "electron", "powder", "theoretical"]):
        raise ValueError("not a supported single-crystal X-ray observation set")
    wavelength_values = [number(v) for v in block.find_values("_diffrn_radiation_wavelength")]
    if (
        not wavelength_values
        or not np.isfinite(wavelength_values).all()
        or np.ptp(wavelength_values) > 1e-8
    ):
        raise ValueError("missing or multiple wavelengths")
    wavelength = wavelength_values[0]
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("missing unambiguous wavelength")
    for tag in ["_atom_site_disorder_group", "_atom_site_disorder_assembly"]:
        if any(gemmi.cif.as_string(v) not in (".", "?", "0", "") for v in block.find_values(tag)):
            raise ValueError("explicit disorder outside clean tier")
    # Twin declarations are retained as exclusions, never removed from the source.
    text = path.read_text(errors="replace")
    embedded = block.find_value("_shelx_res_file") or ""
    if re.search(r"(?im)^\s*(TWIN|BASF)\s", embedded) or re.search(r"(?im)^_twin_", text):
        raise ValueError("declared twinning outside clean tier")
    species, points = expanded_sites(block, group)
    elements = sorted(set(species), key=lambda e: (-gemmi.Element(e).atomic_number, e))
    if len(elements) > 8 or len(points) > 128:
        raise ValueError("exceeds 8 element / 128 full-cell non-H atom benchmark bound")
    doi = (metadata.get("doi") or "").lower().strip().removeprefix("https://doi.org/")
    formula = " ".join((metadata.get("formula") or "").strip(" -").split())
    if not doi or not formula:
        raise ValueError("missing DOI/formula needed for conservative family grouping")
    structure = gemmi.make_small_structure_from_block(block)
    return {
        "cell": cell,
        "space_group": group.xhm(),
        "space_group_number": group.number,
        "shelxt_space_group": shelxt_name(group),
        "wavelength": wavelength,
        "elements": elements,
        "reference_non_h_atoms": len(points),
        "reference_asymmetric_non_h_sites": sum(
            s.element.atomic_number > 1 for s in structure.sites
        ),
        "hydrogen_present": any(s.element.atomic_number == 1 for s in structure.sites),
        "doi": doi,
        "formula_group": formula,
        "chemistry": "light"
        if max(gemmi.Element(e).atomic_number for e in elements) <= 20
        else "heavy",
    }


def reflection_data(path):
    doc = gemmi.cif.read_file(str(path))
    candidates = []
    for block in doc:
        for values, sigma in [
            ("F_squared_meas", "F_squared_sigma"),
            ("intensity_meas", "intensity_sigma"),
        ]:
            table = block.find("_refln_", ["index_h", "index_k", "index_l", values, sigma])
            if len(table):
                candidates.append((block, table, values, sigma))
    if len(candidates) != 1:
        raise ValueError(
            "requires one measured I/F-squared loop with sigma; Fcalc is never substituted"
        )
    block, table, value_tag, sigma_tag = candidates[0]
    rows = np.array([[number(v) for v in row] for row in table])
    if not np.isfinite(rows).all() or np.any(rows[:, :3] != np.rint(rows[:, :3])):
        raise ValueError("nonfinite or fractional measured reflection records")
    if np.any(rows[:, 4] <= 0):
        raise ValueError("nonpositive measured sigma; no silent row rejection")
    data = HklData(rows[:, :3].astype(np.int64), rows[:, 3], rows[:, 4], source_sha256=sha(path))
    return data, {
        "block": block.name,
        "intensity_tag": "_refln_" + value_tag,
        "sigma_tag": "_refln_" + sigma_tag,
        "observations": len(rows),
        "negative_intensities": int(np.count_nonzero(rows[:, 3] < 0)),
    }


def field8(value):
    for precision in range(7, 0, -1):
        text = f"{value:.{precision}G}"
        parts = text.split("E")
        mantissa = parts[0] if "." in parts[0] else parts[0] + "."
        text = mantissa + ("E" + str(int(parts[1])) if len(parts) == 2 else "")
        if len(text) <= 8:
            return text.rjust(8)
    raise ValueError("value not representable in the common HKLF4 input")


def common_hkl(data):
    if np.any(data.hkl < -999) or np.any(data.hkl > 999):
        raise ValueError("HKL index outside common fixed-width bound")
    text = "".join(
        "".join(f"{v:4d}" for v in h) + field8(i) + field8(s) + "\n"
        for h, i, s in zip(data.hkl, data.intensity, data.sigma, strict=True)
    )
    text += f"{0:4d}{0:4d}{0:4d}{0:8.2f}{0:8.2f}\n"
    common = parse_hkl(text)
    if not np.array_equal(common.hkl, data.hkl):
        raise ValueError("common input roundtrip altered reflection indices/order")
    differences = {}
    for name in ("intensity", "sigma"):
        before, after = getattr(data, name), getattr(common, name)
        relative = np.max(np.abs(after - before) / np.maximum(np.abs(before), 1e-100))
        if relative > 5e-4:
            raise ValueError("common input quantization exceeds declared 5e-4 relative cap")
        differences[name + "_maximum_relative_change"] = float(relative)
    return text, differences


def family_components(cases):
    """Connected DOI/formula families: repeated P/T points and polymorphs stay together."""
    parents = {case["id"]: case["id"] for case in cases}

    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    keys = {}
    for case in cases:
        for key in [
            ("doi", case["curation"]["doi"]),
            ("formula", case["curation"]["formula_group"]),
        ]:
            if key in keys:
                left, right = root(case["id"]), root(keys[key])
                parents[max(left, right)] = min(left, right)
            keys[key] = case["id"]
    return {case["id"]: root(case["id"]) for case in cases}


def assign_splits(cases):
    families = family_components(cases)
    for case in cases:
        family = families[case["id"]]
        case["family"] = family
        case["split"] = (
            "development"
            if int(digest("phasesmith-corpus-v1:" + family)[:8], 16) % 3 == 0
            else "holdout"
        )
    return Counter(case["split"] for case in cases)


def metadata_order(rows):
    buckets = defaultdict(list)
    for row in rows:
        if not re.fullmatch(r"[1-9][0-9]{6}", str(row.get("file", ""))):
            continue
        if row["file"] in EXPOSED_IDS or "has Fobs" not in row.get("flags", ""):
            continue
        if any(
            flag in row.get("flags", "").lower() for flag in ("disorder", "error", "theoretical")
        ):
            continue
        buckets[row["sgNumber"]].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: digest("candidate-v1:" + row["file"]))
    result = []
    while any(buckets.values()):
        for key in sorted(buckets, key=int):
            if buckets[key]:
                result.append(buckets[key].pop(0))
    return result
