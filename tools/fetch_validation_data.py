#!/usr/bin/env python3
"""Explicitly fetch selected checksum-pinned external validation datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from phasesmith.validation import VALIDATION_DATASETS, fetch_validation_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_ids",
        nargs="*",
        choices=tuple(dataset.dataset_id for dataset in VALIDATION_DATASETS),
        help="datasets to fetch; the default fetches every registered dataset",
    )
    parser.add_argument("--data-directory", type=Path, default=Path("validation/data"))
    arguments = parser.parse_args()
    dataset_ids = arguments.dataset_ids or tuple(
        dataset.dataset_id for dataset in VALIDATION_DATASETS
    )
    for dataset_id in dataset_ids:
        destination = arguments.data_directory / dataset_id
        files = fetch_validation_dataset(dataset_id, destination)
        print(f"{dataset_id}: verified {len(files)} files in {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
