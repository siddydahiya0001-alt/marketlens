"""Loads uploaded CSV / Excel files and reports basic file-level information.

All functions here are defensive: they never raise raw exceptions up to the
Streamlit UI. Instead they return a `LoadResult` with `.ok` and `.error`
so the calling page can show a friendly message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


class FileLoadError(Exception):
    """Raised for any problem loading a file, with a user-friendly message."""


@dataclass
class LoadResult:
    ok: bool
    df: Optional[pd.DataFrame] = None
    sheet_names: list = field(default_factory=list)
    selected_sheet: Optional[str] = None
    error: Optional[str] = None


SUPPORTED_EXTENSIONS = {"csv", "xlsx", "xls"}


def get_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def list_worksheets(file_obj) -> list:
    """Return worksheet names for an Excel file-like object."""
    try:
        excel_file = pd.ExcelFile(file_obj)
        return excel_file.sheet_names
    except Exception as exc:  # noqa: BLE001
        raise FileLoadError(
            "This Excel file could not be opened. It may be corrupted or in an unsupported format."
        ) from exc


def load_file(file_obj, filename: str, sheet_name: Optional[str] = None) -> LoadResult:
    """Load a CSV or Excel file (single sheet) into a DataFrame.

    For Excel files, `sheet_name` must be provided by the caller after the
    user has picked a worksheet (see `list_worksheets`).
    """
    extension = get_extension(filename)

    if extension not in SUPPORTED_EXTENSIONS:
        return LoadResult(
            ok=False,
            error=(
                f"Unsupported file type '.{extension}'. Please upload a .csv, .xlsx or .xls file."
            ),
        )

    try:
        if extension == "csv":
            df = _read_csv(file_obj)
            sheet_names = []
        else:
            sheet_names = list_worksheets(file_obj)
            if not sheet_names:
                return LoadResult(ok=False, error="This Excel file does not contain any worksheets.")
            chosen_sheet = sheet_name or sheet_names[0]
            if chosen_sheet not in sheet_names:
                return LoadResult(ok=False, error=f"Worksheet '{chosen_sheet}' was not found in this file.")
            df = pd.read_excel(file_obj, sheet_name=chosen_sheet)
            sheet_name = chosen_sheet
    except FileLoadError as exc:
        return LoadResult(ok=False, error=str(exc))
    except pd.errors.EmptyDataError:
        return LoadResult(ok=False, error="This file appears to be empty.")
    except Exception as exc:  # noqa: BLE001
        return LoadResult(
            ok=False,
            error=(
                "This file could not be read. It may be corrupted, password protected, "
                "or not a genuine CSV/Excel file."
            ),
        )

    validation_error = _validate_dataframe(df)
    if validation_error:
        return LoadResult(ok=False, error=validation_error, sheet_names=sheet_names, selected_sheet=sheet_name)

    return LoadResult(ok=True, df=df, sheet_names=sheet_names, selected_sheet=sheet_name)


def _read_csv(file_obj) -> pd.DataFrame:
    try:
        return pd.read_csv(file_obj)
    except UnicodeDecodeError:
        file_obj.seek(0)
        return pd.read_csv(file_obj, encoding="latin-1")
    except pd.errors.ParserError as exc:
        raise FileLoadError("This CSV file could not be parsed. Please check its formatting.") from exc


def _validate_dataframe(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.shape[0] == 0:
        return "The selected sheet or file contains no data rows."
    if df.shape[1] == 0:
        return "The selected sheet or file contains no columns."

    # Drop fully-empty columns before deciding whether anything useful remains.
    useful_columns = df.dropna(axis=1, how="all")
    if useful_columns.shape[1] == 0:
        return "This file has no columns with usable data."
    return None


def build_load_summary(df: pd.DataFrame, filename: str, sheet_name: Optional[str]) -> dict:
    return {
        "file_name": filename,
        "sheet_name": sheet_name,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_missing_cells": int(df.isna().sum().sum()),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
    }
