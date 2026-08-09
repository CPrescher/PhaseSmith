from pathlib import Path

import numpy as np
import pytest
from phasesmith.io import PowderReadLimits, read_powder_data


def test_reads_two_column_pattern_from_text() -> None:
    data = read_powder_data("# 2theta intensity\n5.000 174\n5.020 176\n")

    assert data.format == "columns"
    assert data.source_name is None
    assert data.uncertainty is None
    np.testing.assert_array_equal(data.x, [5.0, 5.02])
    np.testing.assert_array_equal(data.observed_y, [174.0, 176.0])
    assert not data.x.flags.writeable
    assert not data.observed_y.flags.writeable
    pattern = data.to_pattern()
    np.testing.assert_array_equal(pattern.observed_y, data.observed_y)


def test_reads_three_column_pattern_from_path(tmp_path: Path) -> None:
    path = tmp_path / "pattern.xye"
    path.write_text("1, 10, 2\n2, 11, 3 # comment\n", encoding="utf-8")

    data = read_powder_data(path)

    assert data.source_name == str(path)
    np.testing.assert_array_equal(data.uncertainty, [2.0, 3.0])
    assert data.uncertainty is not None and not data.uncertainty.flags.writeable


def test_reads_selected_gsas_fxye_bank_and_converts_centidegrees() -> None:
    text = """Example
BANK 1 2 2 CONS 50.0 2.0 0 0 FXYE
 500.0 100.0 10.0
 502.0 121.0 11.0
BANK 2 1 1 CONS 1000.0 1.0 0 0 FXYE
 1000.0 9.0 3.0
"""

    data = read_powder_data(text, bank=2)

    assert data.format == "gsas_fxye"
    assert data.bank == 2
    np.testing.assert_array_equal(data.x, [10.0])
    np.testing.assert_array_equal(data.observed_y, [9.0])


def test_bom_prefixed_fxye_zero_esd_is_exposed_as_an_exclusion_mask() -> None:
    data = read_powder_data(
        "\ufeffExample\nBANK 1 2 2 CONS 50.0 2.0 0 0 FXYE\n500.0 0.0 0.0\n502.0 121.0 11.0\n"
    )

    np.testing.assert_array_equal(data.uncertainty, [1.0, 11.0])
    np.testing.assert_array_equal(data.mask, [False, True])
    np.testing.assert_array_equal(data.to_pattern().mask, data.mask)
    assert data.mask is not None and not data.mask.flags.writeable


def test_reads_packed_constant_step_gsas_std_bank() -> None:
    text = """Packed example
BANK 1 3 1 CONST 1000 2.5 0 0 STD
     100 2    50     121
"""

    data = read_powder_data(text)

    assert data.format == "gsas_std"
    assert data.bank == 1
    np.testing.assert_allclose(data.x, [10.0, 10.025, 10.05], atol=1.0e-15)
    np.testing.assert_array_equal(data.observed_y, [100.0, 50.0, 121.0])
    np.testing.assert_allclose(data.uncertainty, [10.0, 5.0, 11.0])


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("1 2\n1 3\n", "strictly increasing"),
        ("1 2 0\n2 3 1\n", "positive"),
        ("1 2\n2 3 4\n", "expected 2 columns"),
        ("BANK 1 2 2 CONS 1 1 0 0 ESD\n1 2 3\n", "packed constant-step"),
        ("BANK 1 1 1 TIME_MAP 1 1 0 0 STD\n 1    10\n", "constant-step"),
        ("BANK 1 1 1 CONST 1 1 0 0 STD ESD\n     10\n", "packed constant-step"),
        ("BANK 1 1 1 CONST 1 1 0 0 STD\nxx    10\n", "fixed-width"),
    ],
)
def test_rejects_invalid_powder_data(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_powder_data(text)


def test_enforces_row_and_byte_limits() -> None:
    with pytest.raises(ValueError, match="max_rows"):
        read_powder_data("1 2\n2 3\n", limits=PowderReadLimits(max_rows=1))
    with pytest.raises(ValueError, match="max_bytes"):
        read_powder_data("1 2\n", limits=PowderReadLimits(max_bytes=2))


def test_reports_missing_gsas_bank() -> None:
    with pytest.raises(ValueError, match="available banks: 1"):
        read_powder_data("BANK 1 1 1 CONS 1 1 0 0 FXYE\n100 2 1\n", bank=2)
