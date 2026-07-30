import io
import pandas as pd
import pytest

from services.file_loader import load_file, list_worksheets, get_extension


def _csv_bytes(text: str) -> io.BytesIO:
    buf = io.BytesIO(text.encode("utf-8"))
    buf.name = "test.csv"
    return buf


def test_get_extension():
    assert get_extension("data.csv") == "csv"
    assert get_extension("data.XLSX") == "xlsx"
    assert get_extension("no_extension") == ""


def test_load_valid_csv():
    buf = _csv_bytes("a,b,c\n1,2,3\n4,5,6\n")
    result = load_file(buf, "test.csv")
    assert result.ok
    assert result.df.shape == (2, 3)


def test_load_empty_csv():
    buf = _csv_bytes("")
    result = load_file(buf, "empty.csv")
    assert not result.ok
    assert "empty" in result.error.lower()


def test_load_unsupported_extension():
    buf = _csv_bytes("a,b\n1,2\n")
    result = load_file(buf, "test.txt")
    assert not result.ok
    assert "unsupported" in result.error.lower()


def test_load_csv_no_data_rows():
    buf = _csv_bytes("a,b,c\n")
    result = load_file(buf, "headers_only.csv")
    assert not result.ok


def test_load_excel_with_worksheets(tmp_path):
    path = tmp_path / "multi_sheet.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"x": [1, 2, 3]}).to_excel(writer, sheet_name="Sheet1", index=False)
        pd.DataFrame({"y": [4, 5, 6]}).to_excel(writer, sheet_name="Sheet2", index=False)

    with open(path, "rb") as f:
        sheets = list_worksheets(f)
    assert sheets == ["Sheet1", "Sheet2"]

    with open(path, "rb") as f:
        result = load_file(f, "multi_sheet.xlsx", sheet_name="Sheet2")
    assert result.ok
    assert result.selected_sheet == "Sheet2"
    assert list(result.df.columns) == ["y"]


def test_load_excel_missing_sheet_name(tmp_path):
    path = tmp_path / "single.xlsx"
    pd.DataFrame({"x": [1]}).to_excel(path, index=False)
    with open(path, "rb") as f:
        result = load_file(f, "single.xlsx", sheet_name="DoesNotExist")
    assert not result.ok
