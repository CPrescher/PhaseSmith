#!/usr/bin/env python3
"""Convert the pinned Rowles TOPAS inputs to a neutral GSAS-II/PhaseSmith bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from phasesmith.io import convert_rowles_topas_bundle
from phasesmith.validation import verify_validation_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    verify_validation_dataset("curtin-rowles-qpa-topas", arguments.source)
    manifest = convert_rowles_topas_bundle(arguments.source, arguments.destination)
    print(manifest)


if __name__ == "__main__":
    main()
