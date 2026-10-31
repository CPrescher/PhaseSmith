import zipfile

import numpy as np
import pytest
from phasesmith.io.numeric_archive import save_npz


@pytest.mark.parametrize("level", [0, 1, 6, 9])
def test_native_numeric_npz_exact_bytes_shapes_endianness_and_exclusive_creation(tmp_path, level):
    arrays = dict(
        doubles=np.array([0.0, -0.0, np.nan, np.inf, -np.inf]),
        integers=np.array([0, -1, 2**62], dtype="<i8"),
        unsigned=np.array([0, 2**32 - 1], dtype="<u4"),
        big=np.arange(6, dtype=">f8").reshape(2, 3),
        mask=np.array([True, False]),
        metadata_json=np.frombuffer(b'{"schema": 1}', dtype=np.uint8),
        empty=np.empty((2, 0, 3)),
        scalar=np.array(-0.0),
        noncontiguous=np.arange(35, dtype=np.float32).reshape(5, 7)[::2, ::-1],
    )
    arrays["doubles"].view(np.uint64)[2] = 0x7FF8000000000123
    path = tmp_path / "native.npz"
    save_npz(path, arrays, compression_level=level)
    with np.load(path, allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays)
        for name, expected in arrays.items():
            assert loaded[name].shape == expected.shape
            assert loaded[name].dtype == expected.dtype
            assert loaded[name].tobytes() == expected.tobytes()
    other = tmp_path / "repeat.npz"
    save_npz(other, arrays, compression_level=level)
    assert path.read_bytes() == other.read_bytes()
    with zipfile.ZipFile(path) as archive:
        assert all(info.compress_type == (0 if level == 0 else 8) for info in archive.infolist())
    before = path.read_bytes()
    with pytest.raises(OSError):
        save_npz(path, arrays, compression_level=level)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "arrays",
    [
        {"../escape": np.ones(1)},
        {"bad": np.array(["text"])},
        {"bad": np.array([object()])},
        {"bad": np.array([1j])},
    ],
)
def test_invalid_archives_create_no_destination(tmp_path, arrays):
    path = tmp_path / "invalid.npz"
    with pytest.raises(ValueError):
        save_npz(path, arrays)
    assert not path.exists()
