"""Checksummed external datasets used by opt-in validation workflows."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal
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
    purpose: Literal["acceptance", "oracle_integrity", "holdout", "capability"] = "acceptance"
    expected_status: Literal["passed", "failed", "blocked"] = "passed"

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.dataset_id.replace("-", "").isalnum():
            raise ValueError("validation dataset ID must be alphanumeric with hyphens")
        if not self.title or not self.citation or not self.license_note:
            raise ValueError("validation dataset metadata must not be empty")
        if not self.source_url.startswith("https://"):
            raise ValueError("validation dataset source URL must use HTTPS")
        if not self.files or len({file.name for file in self.files}) != len(self.files):
            raise ValueError("validation dataset filenames must be non-empty and unique")
        if self.purpose not in {"acceptance", "oracle_integrity", "holdout", "capability"}:
            raise ValueError("invalid validation dataset purpose")
        if self.expected_status not in {"passed", "failed", "blocked"}:
            raise ValueError("invalid expected validation status")


_QARR_MIRROR = (
    "https://raw.githubusercontent.com/EdgarGF93/gsas_tutorial/"
    "78c18d8c2c058067b92e1a8ecb8fa81d32980008"
)
_SUCROSE_SOURCE = (
    "https://raw.githubusercontent.com/AdvancedPhotonSource/GSAS-II-Tutorials/"
    "e2485148a3d7ee4757239b1ba40653f1f715bba5/LeBail/data"
)
_PBSO4_SOURCE = (
    "https://raw.githubusercontent.com/AdvancedPhotonSource/GSAS-II-Tutorials/"
    "e2485148a3d7ee4757239b1ba40653f1f715bba5/PythonScript/data"
)
_ANSTO_ECHIDNA_SOURCE = "https://zenodo.org/records/14286343/files"
_NIST_SRM660C_SOURCE = "https://data.nist.gov/od/ds/mds2-2315"
_POWGEN_TOF_SOURCE = (
    "https://raw.githubusercontent.com/AdvancedPhotonSource/GSAS-II-Tutorials/"
    "e2485148a3d7ee4757239b1ba40653f1f715bba5/TOF%20Calibration/data"
)
_ROWLES_QPA_SOURCE = "https://ddfe.curtin.edu.au/5f44ad65411cc"
_BATH_LTL_SOURCE = "https://researchdata.bath.ac.uk/648"
_XRED_TIO2_SOURCE = (
    "https://raw.githubusercontent.com/WPEM/XRED/"
    "916726657c44ef1ca30c475f136835a9f37393c4/biphase/TiO2Rutile%20Anatase"
)
_IUCR_SILICON_SOURCE = "https://journals.iucr.org/e/issues/2017/02/00/wm5358"

VALIDATION_DATASETS: tuple[ValidationDataset, ...] = (
    ValidationDataset(
        dataset_id="iucr-dicesium-citrate-si-standard",
        title="Dicesium hydrogen citrate with NIST SRM 640b silicon internal standard",
        source_url="https://doi.org/10.1107/S2056989017000792",
        citation=(
            "A. Rammohan, A. A. Sarjeant and J. A. Kaduk, Acta Cryst. E73 "
            "(2017) 231-234, "
            "doi:10.1107/S2056989017000792"
        ),
        license_note=(
            "External IUCr supplementary CIF containing laboratory counts, structures, "
            "and deposited legacy-GSAS results; the article is CC BY 4.0. The file is "
            "checksum-pinned and not redistributed."
        ),
        files=(
            ExternalValidationFile(
                "wm5358sup1.cif",
                "c0a14d617874bc649c922313164a12542fd2327072be88cc2e7ac7a6dedbc78d",
                228_093,
                (f"{_IUCR_SILICON_SOURCE}/wm5358sup1.cif",),
            ),
        ),
        purpose="capability",
        expected_status="passed",
    ),
    ValidationDataset(
        dataset_id="xred-tio2-anatase-rutile",
        title="XRED experimental anatase/rutile laboratory X-ray pattern",
        source_url=(
            "https://github.com/WPEM/XRED/tree/"
            "916726657c44ef1ca30c475f136835a9f37393c4/biphase/TiO2Rutile%20Anatase"
        ),
        citation=(
            "B. Cao, X-Ray phase Identification public Experimental Dataset (XRED), "
            "commit 916726657c44ef1ca30c475f136835a9f37393c4"
        ),
        license_note=(
            "External experimental pattern under the repository MIT license; COD CIFs "
            "declare public-domain structural data. Files are checksum-pinned and not "
            "redistributed."
        ),
        files=(
            ExternalValidationFile(
                "data.csv",
                "33bc06f9a9c6ec94643c83fc3fef33ef4909641c9562c7a47b0c1d1b7ba9eb4b",
                23_770,
                (f"{_XRED_TIO2_SOURCE}/data.csv",),
            ),
            ExternalValidationFile(
                "anatase.cif",
                "4fc026291a789910481782d09b13b291fde2ab6b0166249e007ee0eb93b9b915",
                2_452,
                (f"{_XRED_TIO2_SOURCE}/1010942.cif",),
            ),
            ExternalValidationFile(
                "rutile.cif",
                "e0bbe47fbb0a23051cf6416cb6beab12d1abe53b26f2694e505cc53b972ffe34",
                2_099,
                (f"{_XRED_TIO2_SOURCE}/1530150.cif",),
            ),
        ),
        purpose="capability",
        expected_status="passed",
    ),
    ValidationDataset(
        dataset_id="bath-ltl-lab-xray",
        title="Bath K/Li/Cs-exchanged zeolite L laboratory X-ray refinements",
        source_url="https://doi.org/10.15125/BATH-00648",
        citation=(
            "University of Bath Research Data Archive, The Effect of Cation Exchange on "
            "the Pore Geometry of Zeolite L, doi:10.15125/BATH-00648"
        ),
        license_note=(
            "External Rigaku SmartLab scans, final CIFs and legacy GSAS projects under "
            "Creative Commons Attribution 4.0; checksum-pinned archives are not redistributed."
        ),
        files=(
            ExternalValidationFile(
                "patterns.zip",
                "5d4ab92de3d64eb1e346e65b2bd38109c3fe17bd6e1c167738b089708d52c580",
                117_902,
                (f"{_BATH_LTL_SOURCE}/2/XRD_Patterns.zip",),
            ),
            ExternalValidationFile(
                "cifs.zip",
                "9bcf4d4190f9482e84ac0f8eea6370e744974d7bef91716fec12574d440facf2",
                74_982,
                (
                    f"{_BATH_LTL_SOURCE}/3/"
                    "CIF%20files-%20final%20refined%20structures%20all%20zeolites.zip",
                ),
            ),
            ExternalValidationFile(
                "gsas.zip",
                "a2f2c61383ffd459e18ba9381675f166b385d5b56fdabed6bc566c5bb820e394",
                3_269_620,
                (f"{_BATH_LTL_SOURCE}/1/GSAS%20refinement%20files.zip",),
            ),
            ExternalValidationFile(
                "README.txt",
                "4dd3afedba584630cb812b494851834fa2e69e89c8dc7bc0336afe62ecaa150b",
                1_586,
                (f"{_BATH_LTL_SOURCE}/4/README.txt",),
            ),
        ),
        purpose="capability",
        expected_status="failed",
    ),
    ValidationDataset(
        dataset_id="curtin-rowles-qpa-topas",
        title="Rowles laboratory X-ray QPA robustness study, TOPAS inputs",
        source_url="https://doi.org/10.25917/5f44ad65411cc",
        citation=(
            "M. R. Rowles, J. Appl. Cryst. 54 (2021) 626-635, "
            "doi:10.1107/S160057672100371X; dataset doi:10.25917/5f44ad65411cc"
        ),
        license_note=(
            "External Curtin University laboratory XRD data and TOPAS v6 inputs under "
            "Creative Commons Attribution 4.0; selected files are checksum-pinned and "
            "not redistributed."
        ),
        files=(
            ExternalValidationFile(
                "1a_1000000_0-010_n001.xy",
                "04643a24457bdefadb44b3432c569d353c49e777d1b3089e42b91a23db7c8a84",
                264_883,
                (f"{_ROWLES_QPA_SOURCE}/diffraction_data/1a_1000000_0-010_n001.xy",),
            ),
            ExternalValidationFile(
                "1e_1000000_0-010_n001.xy",
                "d12cf03415965666eabed014f4519360a9dd8a806b2d5ff09a0a31f36d0a2487",
                272_757,
                (f"{_ROWLES_QPA_SOURCE}/diffraction_data/1e_1000000_0-010_n001.xy",),
            ),
            ExternalValidationFile(
                "robustness2_1a_4.INP",
                "87d00f688194d0b498687cc7030425f20dff3841b9bb87028c4a890f0d107d9f",
                17_948,
                (f"{_ROWLES_QPA_SOURCE}/topas_files/robustness2_1a_4.INP",),
            ),
            ExternalValidationFile(
                "robustness2_1e_4.INP",
                "85f5e899ca527d3407296eed2bcb8e891e3fe4ce02c43de4afaddb5720495b1c",
                17_921,
                (f"{_ROWLES_QPA_SOURCE}/topas_files/robustness2_1e_4.INP",),
            ),
            ExternalValidationFile(
                "row119.inc",
                "f1b13d60c6b73b3139c984e99ba50c2880a0dea5e1d2ce1e81873da56c9c2bd3",
                212_531,
                (f"{_ROWLES_QPA_SOURCE}/topas_files/row119.inc",),
            ),
        ),
        purpose="capability",
        expected_status="passed",
    ),
    ValidationDataset(
        dataset_id="ansto-echidna-lab6-cw-neutron",
        title="ANSTO Echidna LaB6 constant-wavelength neutron calibration pattern",
        source_url="https://doi.org/10.5281/zenodo.14286343",
        citation=(
            "M. Avdeev and J. R. Hester, Echidna Ge(115) monochromator calibration data, "
            "doi:10.5281/zenodo.14286343"
        ),
        license_note=(
            "External ANSTO calibration data from an immutable Zenodo record; the record "
            "declares Creative Commons Attribution 4.0. Files are not redistributed."
        ),
        files=(
            ExternalValidationFile(
                "ECH0034258_LaB6.xyd",
                "09950aaf3596518a40b2c5173ffdfd870043f55dcd3390fb83dc9f64e348e115",
                112_390,
                (f"{_ANSTO_ECHIDNA_SOURCE}/ECH0034258_LaB6.xyd?download=1",),
            ),
            ExternalValidationFile(
                "ECH0034258_LaB6.cif",
                "0fa02945683ea39d56dfb81d0f99de629e4e336abab85c6d92b3689b42d130d1",
                112_390,
                (f"{_ANSTO_ECHIDNA_SOURCE}/ECH0034258_LaB6.cif?download=1",),
            ),
        ),
    ),
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
    ValidationDataset(
        dataset_id="gsasii-pbso4-cw",
        title="GSAS-II PbSO4 combined constant-wavelength refinement tutorial",
        source_url=(
            "https://github.com/AdvancedPhotonSource/GSAS-II-Tutorials/blob/"
            "e2485148a3d7ee4757239b1ba40653f1f715bba5/"
            "CWCombined/Combined%20refinement.htm"
        ),
        citation="Advanced Photon Source, GSAS-II combined X-ray/neutron refinement tutorial",
        license_note=(
            "External tutorial data fetched from the commit-pinned official "
            "AdvancedPhotonSource/GSAS-II-Tutorials repository; not redistributed."
        ),
        files=(
            ExternalValidationFile(
                "PBSO4.XRA",
                "ca2da02fc7e17d2fc912de22a979f8101ba6140f2e8104f9d3aa64f030ca58bb",
                49_445,
                (f"{_PBSO4_SOURCE}/PBSO4.XRA",),
            ),
            ExternalValidationFile(
                "PBSO4.CWN",
                "59462ba6d7c72c9800b0b6bf41903f9933ed5ac5adab101d27022d5d25cf18ba",
                24_190,
                (f"{_PBSO4_SOURCE}/PBSO4.CWN",),
            ),
            ExternalValidationFile(
                "PbSO4-Wyckoff.cif",
                "9bc19d0995561afd78a5f0563599c032751621da994e36c7570b4f515d9f0e2f",
                1_516,
                (f"{_PBSO4_SOURCE}/PbSO4-Wyckoff.cif",),
            ),
            ExternalValidationFile(
                "INST_XRY.PRM",
                "e59413059bc3b12a51f1c6eebc5d581470320d05a31f11a41dbb32e16d1d16b9",
                794,
                (f"{_PBSO4_SOURCE}/INST_XRY.PRM",),
            ),
            ExternalValidationFile(
                "inst_d1a.prm",
                "a1031174a9b509f889377752f5c095a1dad49a25fb0a72c057f184f7b321fcea",
                971,
                (f"{_PBSO4_SOURCE}/inst_d1a.prm",),
            ),
        ),
    ),
    ValidationDataset(
        dataset_id="iucr-qarr-1h",
        title="IUCr Quantitative Phase Analysis Round Robin sample 1h holdout",
        source_url="https://www.iucr.org/resources/commissions/powder-diffraction/projects/qarr/data",
        citation=(
            "I. C. Madsen et al., J. Appl. Cryst. 34 (2001) 409-426, doi:10.1107/S0021889801007476"
        ),
        license_note=(
            "External IUCr round-robin data; fetched from a commit-pinned public mirror. "
            "COD-derived CIF headers identify their structural data as public domain/CC0."
        ),
        files=(
            ExternalValidationFile(
                "cpd-1h.prn",
                "0c36af18d7a341f03f4d8f24972608791351c45c951c431dd93fb54eb8e56da3",
                145_020,
                (f"{_QARR_MIRROR}/cpd-1h.prn",),
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
        purpose="holdout",
        expected_status="failed",
    ),
    ValidationDataset(
        dataset_id="nist-srm660c-lab6-xray",
        title="NIST SRM 660c LaB6 line-position and line-shape certification scans",
        source_url="https://catalog.data.gov/dataset/diffraction-data-for-srm-660c",
        citation=(
            "D. R. Black et al., Powder Diffraction 35 (2020) 17-22, doi:10.1017/S0885715620000068"
        ),
        license_note=(
            "Public NIST certification data under the NIST open-data license; the original "
            "archive is checksum-pinned and is not redistributed."
        ),
        files=(
            ExternalValidationFile(
                "srm_660c_cifs_20201029_081700.zip",
                "92034d06498161db365420831fe35d91ec18bad4ed17385329a1761658da2c42",
                1_719_632,
                (f"{_NIST_SRM660C_SOURCE}/srm_660c_cifs_20201029_081700.zip",),
            ),
        ),
        purpose="oracle_integrity",
        expected_status="passed",
    ),
    ValidationDataset(
        dataset_id="powgen-lab6-tof-calibration",
        title="POWGEN NIST LaB6 time-of-flight calibration tutorial",
        source_url=(
            "https://github.com/AdvancedPhotonSource/GSAS-II-Tutorials/blob/"
            "e2485148a3d7ee4757239b1ba40653f1f715bba5/TOF%20Calibration/"
            "Calibration%20of%20a%20TOF%20powder%20diffractometer.htm"
        ),
        citation="Advanced Photon Source, GSAS-II TOF powder diffractometer calibration tutorial",
        license_note=(
            "External tutorial data fetched from the commit-pinned official "
            "AdvancedPhotonSource/GSAS-II-Tutorials repository; not redistributed."
        ),
        files=(
            ExternalValidationFile(
                "PG3_17541.gsa",
                "ff7a408451e75d23e828ab2bb35a061a53517ff3430331bffeed21fcbc87d69c",
                539_682,
                (f"{_POWGEN_TOF_SOURCE}/PG3_17541.gsa",),
            ),
            ExternalValidationFile(
                "PGHR_60-2015A.prm",
                "1a098c260555d27642ab0501708c5d9058c5836fb201dec5bc7ab9880cea1cb8",
                7_295,
                (f"{_POWGEN_TOF_SOURCE}/PGHR_60-2015A.prm",),
            ),
        ),
        purpose="acceptance",
        expected_status="passed",
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
