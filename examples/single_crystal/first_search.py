"""Run native spot detection and a synthetic single-atom HKL search with NumPy only.

This is an installation/API tutorial, not an experimental validation dataset.
The generator's coordinates do not enter the search; the full Fe=1 composition
and P1 symmetry are explicit priors. A one-atom origin is unidentifiable.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from phasesmith.crystallography import UnitCell, calculate_structure_factor_values
from phasesmith.io.hkl import read_hkl
from phasesmith.scattering import XrayNonResonant
from phasesmith.single_crystal.solution import SolutionOptions, solve_structure, write_solution
from phasesmith.single_crystal.spots import detect_spots
from phasesmith.structure import AtomSite, CrystalStructure
from phasesmith.symmetry import SpaceGroup


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    yy, xx = np.indices((48, 48))
    image = 10.0 + 1000 * np.exp(-((xx - 24) ** 2 + (yy - 24) ** 2) / 4)
    spots = detect_spots(image)
    cell = UnitCell(6, 7, 8, 90, 90, 90)
    group = SpaceGroup.p1()
    model = CrystalStructure(
        structure_id="synthetic-tutorial",
        name="Ideal one-atom P1 example, not a measured material",
        cell=cell,
        space_group=group,
        sites=(
            AtomSite(
                site_id="Fe1",
                source_label="Fe1",
                type_symbol="Fe",
                element_symbol="Fe",
                fractional_xyz=(0.17, 0.23, 0.31),
                u_iso_angstrom2=0.02,
            ),
        ),
    )
    hkl = np.array(list(product(range(1, 5), range(-3, 4), range(-3, 4))), dtype=np.int64)
    calculated = calculate_structure_factor_values(
        model, hkl, np.ones(len(hkl), dtype=np.int64), XrayNonResonant()
    )
    # Ideal noiseless |F|²; unit sigma is an illustrative weight, not a calibrated
    # experimental error. The adapter retains the same signed I/sigma convention.
    path = output / "synthetic.hkl"
    np.savetxt(
        path,
        np.column_stack((hkl, calculated.f_squared, np.ones(len(hkl)))),
        fmt=["%d", "%d", "%d", "%.12g", "%.12g"],
    )
    result = solve_structure(
        read_hkl(path),
        cell=cell,
        space_group=group,
        composition={"Fe": 1},
        options=SolutionOptions(
            phasing="dual_space",
            trials=1,
            cycles=64,
            max_seconds=10,
            refinement="none",
            completion=False,
        ),
    )
    write_solution(result, output / "search")
    report = dict(
        kind="synthetic installation/API example, not a structure-recovery benchmark",
        spots=len(spots),
        input_reflections=len(hkl),
        termination=result.termination,
        target_met=result.target_met,
        candidates=len(result.candidates),
        best_r1=None if result.best is None else result.best.r1,
        priors={
            "cell_angstrom_degrees": cell.as_tuple(),
            "space_group": "P1",
            "composition": {"Fe": 1},
        },
        truth_coordinates_supplied_to_search=False,
    )
    (output / "tutorial.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    run(parser.parse_args().output)
