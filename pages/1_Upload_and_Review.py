"""Page 1: Upload a file, pick a worksheet, review the data, and confirm column meanings."""

import io

import streamlit as st
import pandas as pd

import numpy as np

from utils.session_state import init_session_state, reset_for_new_file
from utils.sidebar import render_global_sidebar
from utils.constants import ALL_ROLES, CURRENCY_CHOICES
from services.file_loader import load_file, list_worksheets, build_load_summary, get_extension, FileLoadError
from services.column_detector import profile_columns
from visualisations.metric_cards import data_overview_metrics

COLUMN_TYPE_OPTIONS = ["Text", "Whole number", "Decimal number", "Percentage", "Currency", "Date", "Yes/No"]


def _build_editor_column_config(column_display_formats: dict) -> dict:
    """Translate saved display-format choices into st.data_editor column_config entries."""
    config = {}
    for col, fmt in column_display_formats.items():
        col_type = fmt.get("type")
        if col_type == "currency":
            config[col] = st.column_config.NumberColumn(col, format=f'{fmt.get("symbol", "$")} %.2f')
        elif col_type == "percentage":
            config[col] = st.column_config.NumberColumn(col, format="%.1f%%", help="Enter as a number, e.g. 10 for 10%")
        elif col_type == "whole_number":
            config[col] = st.column_config.NumberColumn(col, format="%d", step=1)
        elif col_type == "decimal_number":
            config[col] = st.column_config.NumberColumn(col, format="%.2f")
        elif col_type == "date":
            config[col] = st.column_config.DateColumn(col)
        elif col_type == "yesno":
            config[col] = st.column_config.SelectboxColumn(col, options=["Yes", "No"])
        # "text" needs no special column_config - TextColumn is the default.
    return config


def _default_value_for_type(col_type: str):
    if col_type in ("Whole number", "Decimal number", "Percentage", "Currency"):
        return np.nan
    if col_type == "Date":
        return pd.NaT
    if col_type == "Yes/No":
        return "No"
    return ""

st.set_page_config(page_title="Upload and Review - MarketLens", page_icon="📤", layout="wide")
init_session_state()
render_global_sidebar()

st.title("📤 Upload and review your data")
st.write("Upload a spreadsheet to get started. Nothing is modified in your original file.")


@st.cache_data(show_spinner=False)
def _cached_list_worksheets(file_bytes: bytes) -> list:
    return list_worksheets(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def _cached_load_file(file_bytes: bytes, filename: str, sheet_name):
    # Returns the same LoadResult shape as services.file_loader.load_file.
    return load_file(io.BytesIO(file_bytes), filename, sheet_name=sheet_name)


uploaded_file = st.file_uploader("Upload a .csv, .xlsx or .xls file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    is_new_file = st.session_state.get("file_name") != uploaded_file.name
    extension = get_extension(uploaded_file.name)
    file_bytes = uploaded_file.getvalue()  # already fully buffered in memory - no re-parsing here

    selected_sheet = None
    if extension in ("xlsx", "xls"):
        try:
            with st.spinner("Reading worksheet names - this can take a moment for large files..."):
                sheet_names = _cached_list_worksheets(file_bytes)
        except FileLoadError as exc:
            st.error(str(exc))
            sheet_names = []

        if sheet_names:
            default_index = 0
            if st.session_state.get("selected_sheet") in sheet_names and not is_new_file:
                default_index = sheet_names.index(st.session_state["selected_sheet"])
            selected_sheet = st.selectbox("This file has multiple worksheets. Select one:", sheet_names, index=default_index)

    if extension in ("xlsx", "xls") and not selected_sheet:
        st.stop()

    needs_reload = (
        is_new_file
        or st.session_state.get("selected_sheet") != selected_sheet
        or st.session_state.get("raw_df") is None
    )

    if needs_reload:
        with st.spinner("Loading your file - large files can take a little while..."):
            result = _cached_load_file(file_bytes, uploaded_file.name, selected_sheet)

        if not result.ok:
            st.error(f"⚠️ {result.error}")
            st.stop()

        if is_new_file or st.session_state.get("selected_sheet") != result.selected_sheet:
            reset_for_new_file()

        st.session_state["raw_df"] = result.df
        st.session_state["file_name"] = uploaded_file.name
        st.session_state["sheet_names"] = result.sheet_names
        st.session_state["selected_sheet"] = result.selected_sheet
        st.session_state["load_summary"] = build_load_summary(result.df, uploaded_file.name, result.selected_sheet)

if st.session_state.get("raw_df") is None:
    st.info("No file uploaded yet. Try the sample dataset in `sample_data/marketing_sample.xlsx` if you'd like to explore first.")
    st.stop()

df = st.session_state["raw_df"]
summary = st.session_state["load_summary"]

st.success(f"Loaded **{summary['file_name']}**" + (f" - worksheet **{summary['sheet_name']}**" if summary["sheet_name"] else ""))

st.markdown("### Data overview")
data_overview_metrics(summary["n_rows"], summary["n_columns"], summary["n_missing_cells"], summary["n_duplicate_rows"])

tab_preview, tab_types, tab_missing = st.tabs(["Data preview", "Data types", "Missing values"])

EDITABLE_ROW_LIMIT = 2000

with tab_preview:
    if summary["n_rows"] <= EDITABLE_ROW_LIMIT:
        st.caption(
            "Values are editable directly in the table below - edits apply to your working data for "
            "this session only, never to the original uploaded file. Add or delete rows using the "
            "controls at the bottom-right of the table."
        )
        column_display_formats = st.session_state.setdefault("column_display_formats", {})
        edited_df = st.data_editor(
            df, use_container_width=True, num_rows="dynamic", key="raw_data_editor",
            column_config=_build_editor_column_config(column_display_formats),
        )
        if not edited_df.equals(df):
            st.session_state["raw_df"] = edited_df
            df = edited_df
            summary = build_load_summary(df, summary["file_name"], summary["sheet_name"])
            st.session_state["load_summary"] = summary
        st.caption(f"{len(df):,} rows shown.")

        with st.expander("➕ Add a new column"):
            ac1, ac2, ac3, ac4 = st.columns([1.3, 1, 1, 0.7])
            with ac1:
                new_col_name = st.text_input("Column name", key="new_col_name")
            with ac2:
                new_col_type = st.selectbox("Column type", COLUMN_TYPE_OPTIONS, key="new_col_type")
            with ac3:
                new_col_currency = None
                if new_col_type == "Currency":
                    currency_label = st.selectbox(
                        "Currency", [label for _, _, label in CURRENCY_CHOICES], key="new_col_currency",
                    )
                    new_col_currency = next(symbol for code, symbol, label in CURRENCY_CHOICES if label == currency_label)
            with ac4:
                st.write("")
                st.write("")
                add_clicked = st.button("Add column", key="add_column_btn")

            st.caption(
                "This choice controls how the column is formatted and edited in this table (e.g. as "
                "$ 12.50 or 12.5%) - you can still set its business meaning in Column Understanding below."
            )

            if add_clicked:
                cleaned_name = new_col_name.strip()
                if not cleaned_name:
                    st.warning("Please enter a column name.")
                elif cleaned_name in df.columns:
                    st.warning(f"A column named '{cleaned_name}' already exists.")
                else:
                    new_df = df.copy()
                    new_df[cleaned_name] = _default_value_for_type(new_col_type)
                    type_key = {
                        "Text": "text", "Whole number": "whole_number", "Decimal number": "decimal_number",
                        "Percentage": "percentage", "Currency": "currency", "Date": "date", "Yes/No": "yesno",
                    }[new_col_type]
                    fmt = {"type": type_key}
                    if type_key == "currency":
                        fmt["symbol"] = new_col_currency
                    st.session_state["column_display_formats"][cleaned_name] = fmt
                    st.session_state["raw_df"] = new_df
                    st.session_state["load_summary"] = build_load_summary(new_df, summary["file_name"], summary["sheet_name"])
                    st.session_state["column_profile"] = None  # force Column Understanding to include the new column
                    st.success(f"Added column '{cleaned_name}'.")
                    st.rerun()
    else:
        st.dataframe(df.head(50), use_container_width=True)
        st.caption(
            f"Showing a read-only preview of the first 50 of {summary['n_rows']:,} rows. Inline "
            "editing is disabled for datasets this large for performance - use the cleaning tools on "
            "the Data Quality page instead."
        )

with tab_types:
    dtype_table = pd.DataFrame({"Column": df.columns, "Pandas data type": df.dtypes.astype(str).values})
    st.dataframe(dtype_table, use_container_width=True, hide_index=True)

with tab_missing:
    missing_table = pd.DataFrame({
        "Column": df.columns,
        "Missing values": df.isna().sum().values,
        "Missing %": (df.isna().mean().values * 100).round(1),
    }).sort_values("Missing %", ascending=False)
    st.dataframe(missing_table, use_container_width=True, hide_index=True)
    if summary["n_duplicate_rows"] > 0:
        st.warning(f"{summary['n_duplicate_rows']} fully duplicate rows were found. You can remove these on the Data Quality page.")

st.markdown("---")
st.markdown("### Column understanding")
st.write(
    "MarketLens has guessed what each column means for your business. Review and correct these "
    "before moving on - your choices are used throughout the rest of the analysis."
)

if st.session_state.get("column_profile") is None:
    st.session_state["column_profile"] = profile_columns(df)

edited_profile = st.data_editor(
    st.session_state["column_profile"],
    use_container_width=True,
    hide_index=True,
    column_config={
        "User role": st.column_config.SelectboxColumn("User role", options=ALL_ROLES, required=True),
        "Missing %": st.column_config.NumberColumn("Missing %", format="%.1f%%"),
    },
    disabled=["Column", "Detected data type", "Suggested business meaning", "Missing %", "Unique values"],
    key="column_profile_editor",
)

st.session_state["column_profile"] = edited_profile
st.session_state["roles"] = dict(zip(edited_profile["Column"], edited_profile["User role"]))

st.markdown("---")
st.page_link("pages/2_Data_Quality.py", label="Continue to Data Quality →", icon="➡️")
