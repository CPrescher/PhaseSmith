"""Checksummed external datasets used by opt-in validation workflows."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class ExternalValidationFile:
    """One externally hosted file with immutable retrieval provenance."""

    name: str
    sha256: str
    size_bytes: int
    urls: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or Path(self.name).name != self.name:
            raise ValueError("external validation filename must be a safe basename")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("external validation SHA-256 must be lowercase hexadecimal")
        if self.size_bytes <= 0:
            raise ValueError("external validation size must be positive")
        if not self.urls or any(not url.startswith("https://") for url in self.urls):
            raise ValueError("external validation files require HTTPS URLs")


@dataclass(frozen=True, slots=True)
class ValidationDataset:
    """A citable validation case whose data remains outside the package."""

    dataset_id: str
    title: str
    source_url: str
    citation: str
    license_note: str
    files: tuple[ExternalValidationFile, ...]

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.dataset_id.replace("-", "").isalnum():
            raise ValueError("validation dataset ID must be alphanumeric with hyphens")
        if not self.title or not self.citation or not self.license_note:
            raise ValueError("validation dataset metadata must not be empty")
        if not self.source_url.startswith("https://"):
            raise ValueError("validation dataset source URL must use HTTPS")
        if not self.files or len({file.name for file in self.files}) != len(self.files):
            raise ValueError("validation dataset filenames must be non-empty and unique")


_QARR_MIRROR = (
    "https://raw.githubusercontent.com/EdgarGF93/gsas_tutorial/"
    "78c18d8c2c058067b92e1a8ecb8fa81d32980008"
)
_SUCROSE_SOURCE = (
    "https://raw.githubusercontent.com/AdvancedPhotonSource/GSAS-II-Tutorials/"
    "e2485148a3d7ee4757239b1ba40653f1f715bba5/LeBail/data"
)

VALIDATION_DATASETS: tuple[ValidationDataset, ...] = (
    ValidationDataset(
        dataset_id="iucr-qarr-1g",
        title="IUCr Quantitative Phase Analysis Round Robin sample 1g",
        source_url="https://www.iucr.org/__data/iucr/powder/QARR/data-kit.htm",
        citation=(
            "I. C. Madsen et al., J. Appl. Cryst. 34 (2001) 409-426, doi:10.1107/S0021889801007476"
        ),
        license_note=(
            "External IUCr round-robin data; fetched from a commit-pinned public mirror. "
            "COD-derived CIF headers identify their structural data as public domain/CC0."
        ),
        files=(
            ExternalValidationFile(
                "cpd-1g.prn",
                "afd16d03c8742abf315ab5f94585a6e540ab5c2e2c6df6917ac8dbf6be69d990",
                145_020,
                (f"{_QARR_MIRROR}/cpd-1g.prn",),
            ),
            ExternalValidationFile(
                "Al2O3.cif",
                "bc8b07c4e27fdb9df562c7b72d181a05394df20274a56b7e1ef50f6ec2c85709",
                3_288,
                (f"{_QARR_MIRROR}/Al2O3.cif",),
            ),
            ExternalValidationFile(
                "CaF2.cif",
                "b55fd04a3e344f73d4ece983da412056d04874b8ada7dfd4ecb93118e7905cd9",
                4_361,
                (f"{_QARR_MIRROR}/CaF2.cif",),
            ),
            ExternalValidationFile(
                "ZnO.cif",
                "021db20c9bdabcfd63bc794367fae3536197425ca5826d63541b2417e47d3c1a",
                2_142,
                (f"{_QARR_MIRROR}/ZnO.cif",),
            ),
            ExternalValidationFile(
                "cuka.instprm",
                "aef315a10622fdb1afae5b264d43e7f9dcc053aba80aa3d9fa05067146922a08",
                215,
                (f"{_QARR_MIRROR}/cuka.instprm",),
            ),
        ),
    ),
    ValidationDataset(
        dataset_id="aps-sucrose-11bmb",
        title="APS 11-BM sucrose Le Bail tutorial pattern",
        source_url=(
            "https://github.com/AdvancedPhotonSource/GSAS-II-Tutorials/blob/"
            "e2485148a3d7ee4757239b1ba40653f1f715bba5/LeBail/LeBailSucrose.htm"
        ),
        citation="Advanced Photon Source, GSAS-II Le Bail fitting tutorial, Sucrose",
        license_note=(
            "External tutorial data fetched from the commit-pinned official "
            "AdvancedPhotonSource/GSAS-II-Tutorials repository; not redistributed."
        ),
        files=(
            ExternalValidationFile(
                "11bmb_8716.fxye",
                "0971eebded2fdc6d8ae81a228325b2075ac45f6605da67468750e16ec23ecadd",
                1_725_515,
                (f"{_SUCROSE_SOURCE}/11bmb_8716.fxye",),
            ),
            ExternalValidationFile(
                "11bmb_8716.prm",
                "ef606d5620cc8c8d9e7b712e1ba6113c9628e83f085835a2d477d4cd4d0d312c",
                1_134,
                (f"{_SUCROSE_SOURCE}/11bmb_8716.prm",),
            ),
        ),
    ),
)

_DATASETS_BY_ID = {dataset.dataset_id: dataset for dataset in VALIDATION_DATASETS}


def validation_dataset(dataset_id: str) -> ValidationDataset:
    """Return one registered validation case by stable identifier."""

    try:
        return _DATASETS_BY_ID[dataset_id]
    except KeyError as error:
        available = ", ".join(sorted(_DATASETS_BY_ID))
        message = f"unknown validation dataset {dataset_id!r}; available: {available}"
        raise KeyError(message) from error


def fetch_validation_dataset(
    dataset_id: str,
    destination: str | Path,
    *,
    force: bool = False,
) -> tuple[Path, ...]:
    """Explicitly fetch and hash-verify every file in a registered dataset."""

    if not isinstance(force, bool):
        raise TypeError("force must be a bool")
    dataset = validation_dataset(dataset_id)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for external_file in dataset.files:
        target = root / external_file.name
        if target.exists() and not force:
            _verify_file(target, external_file)
        else:
            _fetch_file(external_file, target)
        paths.append(target)
    return tuple(paths)


def verify_validation_dataset(
    dataset_id: str,
    destination: str | Path,
) -> tuple[Path, ...]:
    """Verify a complete local dataset without performing network access."""

    dataset = validation_dataset(dataset_id)
    root = Path(destination)
    paths = []
    for external_file in dataset.files:
        target = root / external_file.name
        if not target.is_file():
            raise FileNotFoundError(f"missing {external_file.name} for {dataset_id!r} in {root}")
        _verify_file(target, external_file)
        paths.append(target)
    return tuple(paths)


def _fetch_file(external_file: ExternalValidationFile, target: Path) -> None:
    errors: list[str] = []
    for url in external_file.urls:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{external_file.name}.", suffix=".part", dir=target.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                request = Request(url, headers={"User-Agent": "PhaseSmith-validation/0.1"})
                with urlopen(request, timeout=60) as response:
                    if not response.geturl().startswith("https://"):
                        raise ValueError("download redirected away from HTTPS")
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                        if stream.tell() > external_file.size_bytes:
                            raise ValueError("download exceeds pinned byte size")
            _verify_file(temporary, external_file)
            temporary.replace(target)
            return
        except (OSError, URLError, ValueError) as error:
            errors.append(f"{url}: {error}")
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    details = "; ".join(errors)
    raise RuntimeError(f"could not fetch {external_file.name}: {details}")


def _verify_file(path: Path, external_file: ExternalValidationFile) -> None:
    size = path.stat().st_size
    if size != external_file.size_bytes:
        raise ValueError(
            f"size mismatch for {path.name}: expected {external_file.size_bytes}, got {size}"
        )
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != external_file.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {external_file.sha256}, got {digest}"
        )
