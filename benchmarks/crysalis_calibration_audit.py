"""Read-only inventory of one archived CrysAlis reduction and its MAR headers.

This is a calibration/conflict audit, not a geometry converter or an intensity
comparison. No source cell, UB, structure or reference HKLs enter an inverse.
Requires the optional image-reader environment (FabIO).
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


def fingerprint(path):
    return dict(
        path=str(path),
        bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def audit(project):
    import fabio

    name = project.name
    summary = project / (name + ".sum")
    text = summary.read_text(errors="replace")

    def record(pattern):
        match = re.search(pattern, text, re.MULTILINE)
        return (
            None
            if match is None
            else dict(text=match.group(0), line=text[: match.start()].count("\n") + 1)
        )

    geometry = dict(
        wavelength=record(r"^WAVELENGTH \(ANG\).*$"),
        detector=record(r"^DETECTOR DISTANCE.*$"),
        pixel=record(r"^\s*ZOOM .*PIXELSIZE.*$"),
        rotation=record(r"^DETECTOR ROTATION.*$"),
        scan=record(r"^Run 1  Phi  scan:.*$"),
        reduction=record(r"^DC PROFFIT.*$"),
    )
    frames = sorted(project.glob(name + "_*.mar3450"))
    representatives = []
    for path in [frames[0], frames[-1]] if len(frames) > 1 else frames:
        with fabio.open(str(path)) as image:
            keys = (
                "WAVELENGTH",
                "DISTANCE",
                "PIXEL_LENGTH",
                "PIXEL_HEIGHT",
                "CENTER_X",
                "CENTER_Y",
                "PHI_START",
                "PHI_END",
                "OMEGA_START",
                "OMEGA_END",
                "TIME",
                "DATE",
            )
            headers = {k: image.header.get(k) for k in keys}
            representatives.append(
                dict(**fingerprint(path), shape=list(image.data.shape), raw_header_fields=headers)
            )
    return dict(
        schema="phasesmith.crysalis-calibration-audit.v1",
        project=str(project),
        reference_files=[
            fingerprint(project / (name + ext))
            for ext in (".sum", ".par", ".hkl", ".eqv", ".run")
            if (project / (name + ext)).exists()
        ],
        recorded_project_geometry=geometry,
        matching_named_frames=len(frames),
        raw_representatives=representatives,
        comparison_ready=False,
        blocking_checks=[
            "Reconcile raw-header wavelength/distance/pixel size with CrysAlis "
            "refined project geometry.",
            "Verify detector orientation, origin and phi-axis mapping using observed positions.",
            "Establish unmerged/merged status, HKL basis, Lorentz/polarization "
            "convention, scale and rejection policy.",
            "Do not reuse the 0.3100 A calibration from the separate Fe7C3 Single "
            "crystal PVT EOS experiment.",
        ],
        interpretation=(
            "reference metadata inspected only; no calibration conversion "
            "or numerical intensity agreement claimed"
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a fresh audit output")
    result = audit(args.project)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("project", "matching_named_frames", "comparison_ready")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
