"""Exercise normal installed-wheel single-crystal capabilities without extras."""

import argparse
import importlib.abc
import sys

import numpy as np


class _NoOptionalNumerics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"scipy", "h5py", "hdf5plugin", "gemmi"}:
            raise ImportError(f"wheel smoke forbids optional numerical package: {fullname}")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-fixture")
    args = parser.parse_args()
    sys.meta_path.insert(0, _NoOptionalNumerics())
    from phasesmith import _core, single_crystal

    for name in single_crystal.__all__:
        getattr(single_crystal, name)
    from phasesmith.crystallography import UnitCell
    from phasesmith.io.hdf5 import Hdf5Frames
    from phasesmith.io.hkl import HklData
    from phasesmith.single_crystal.solution import SolutionOptions, solve_structure
    from phasesmith.single_crystal.spots import SpotOptions, available, detect_spots

    assert available()
    assert hasattr(_core, "experimental_row_search")
    assert hasattr(_core, "NativeHdf5Frames")
    yy, xx = np.indices((48, 48))
    image = 10 + 1000 * np.exp(-((xx - 24) ** 2 + (yy - 24) ** 2) / 4)
    spots = detect_spots(image, options=SpotOptions())
    assert len(spots.values) == 1 and spots.values[0, 0] > 0
    hkl = np.array([(h, k, m) for h in range(1, 4) for k in range(3) for m in range(3)])
    result = solve_structure(
        HklData(hkl, np.ones(len(hkl)), np.ones(len(hkl))),
        cell=UnitCell(6, 7, 8, 90, 90, 90),
        composition={"Fe": 1},
        options=SolutionOptions(
            phasing="dual_space",
            trials=1,
            cycles=32,
            max_seconds=5,
            refinement="none",
            completion=False,
        ),
    )
    assert len(result.expanded_hkl) > 0 and np.isfinite(result.amplitudes).all()
    assert result.to_record()["experimental"] is False
    if args.hdf5_fixture:
        with Hdf5Frames(args.hdf5_fixture, "detector/data") as reader:
            frame = reader.read_frame(0)
            assert frame.dtype == np.uint32
            assert frame[0].tolist() == [0, 70000, 2**32 - 2, 2**32 - 1]
    print("Installed single-crystal detection, row search, structure search and HDF5 available")


if __name__ == "__main__":
    main()
