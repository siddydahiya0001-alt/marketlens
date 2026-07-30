"""Page 2: Data-quality findings in plain language, with user-chosen cleaning actions."""

import streamlit as st

from utils.session_state import init_session_state
from utils.sidebar import render_global_sidebar
from services.data_cleaner import (
    generate_quality_findings, apply_cleaning_actions, recommend_missing_value_action,
    MISSING_ACTION_OPTIONS, MISSING_ACTION_KEEP, MISSING_ACTION_REMOVE, MISSING_ACTION_IMPUTE,
)
from utils.download_helpers import lazy_download_button
from visualisations.result_panels import render_finding_cards

st.set_page_config(page_title="Data Quality - MarketLens", page_icon="🧹", layout="wide")
init_session_state()
render_global_sidebar()

st.title("🧹 Data quality")

if st.session_state.get("raw_df") is None:
    st.warning("Please upload a file on the **Upload and Review** page first.")
    st.page_link("pages/1_Upload_and_Review.py", label="Go to Upload and Review", icon="⬅️")
    st.stop()

df = st.session_state["raw_df"]
roles = st.session_state.get("roles", {})

st.write("Here is what MarketLens noticed about your data, in plain language.")

findings = generate_quality_findings(df, roles)
st.session_state["quality_findings"] = findings

if not findings:
    st.success("No major data-quality issues were found.")
else:
    render_finding_cards(findings)

st.markdown("---")
st.markdown("### Choose how to handle these issues")
st.caption("Nothing happens to your original file - a cleaned copy is created for analysis.")

missing_recommendation = recommend_missing_value_action(df, roles)
recommended_action = missing_recommendation["action"]

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Missing values**")
    if recommended_action != MISSING_ACTION_KEEP:
        st.info(f"💡 **MarketLens recommends: {recommended_action}.** {missing_recommendation['reason']}")
    else:
        st.caption(missing_recommendation["reason"])

    missing_choice = st.radio(
        "How should missing values be handled?",
        MISSING_ACTION_OPTIONS,
        index=MISSING_ACTION_OPTIONS.index(recommended_action),
        format_func=lambda opt: f"⭐ {opt} (Recommended)" if opt == recommended_action else opt,
        label_visibility="collapsed",
    )

    st.markdown("**Duplicate rows**")
    remove_duplicates = st.checkbox("Remove fully duplicate rows", value=bool(df.duplicated().sum()))

with col2:
    st.markdown("**Inconsistent labels**")
    standardise_labels = st.checkbox(
        "Standardise similar labels (e.g. 'Yes', 'Y', 'yes' → 'Yes')", value=True,
    )

    st.markdown("**Columns to exclude**")
    high_missing_columns = missing_recommendation["high_missing_columns"]
    if high_missing_columns:
        st.caption(
            f"MarketLens pre-selected {', '.join(f'{c!r}' for c in high_missing_columns)} because "
            f"{'it is' if len(high_missing_columns) == 1 else 'they are'} missing a large share of "
            "values. Remove any you'd like to keep."
        )
    excluded_columns = st.multiselect(
        "Exclude these columns from analysis", options=list(df.columns), default=high_missing_columns,
    )

actions = {
    "remove_missing_rows": missing_choice == MISSING_ACTION_REMOVE,
    "fill_numeric_median": missing_choice == MISSING_ACTION_IMPUTE,
    "fill_categorical_mode": missing_choice == MISSING_ACTION_IMPUTE,
    "remove_duplicates": remove_duplicates,
    "standardise_labels": standardise_labels,
    "excluded_columns": excluded_columns,
}

if st.button("Apply and continue with cleaned data", type="primary"):
    cleaned_df, log = apply_cleaning_actions(df, actions, roles)
    st.session_state["cleaned_df"] = cleaned_df
    st.session_state["cleaning_log"] = log
    st.session_state["rows_before_cleaning"] = len(df)
    st.session_state["rows_after_cleaning"] = len(cleaned_df)
    for col in excluded_columns:
        st.session_state["roles"].pop(col, None)
    st.session_state.pop("_export_bytes_cleaned_dataset", None)  # invalidate any stale cached export
    st.success("Cleaned dataset created.")

if st.session_state.get("cleaned_df") is not None:
    st.markdown("---")
    st.markdown("### Before and after")
    before, after = st.session_state["rows_before_cleaning"], st.session_state["rows_after_cleaning"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows before", before)
    c2.metric("Rows after", after)
    c3.metric("Rows removed", before - after)

    with st.expander("What changed"):
        for entry in st.session_state["cleaning_log"]:
            st.write(f"- {entry}")

    st.dataframe(st.session_state["cleaned_df"].head(20), use_container_width=True)

    st.markdown("---")
    st.page_link("pages/3_Choose_Question.py", label="Continue to Choose a Question →", icon="➡️")

    st.markdown("---")
    st.markdown("### Download")
    if len(st.session_state["cleaned_df"]) > 50_000:
        st.caption(
            f"This dataset has {len(st.session_state['cleaned_df']):,} rows, so building the Excel "
            "file may take a little while. It only needs to be prepared once."
        )
    from services.export_service import dataframe_to_excel_bytes
    lazy_download_button(
        "Download cleaned dataset as Excel",
        lambda: dataframe_to_excel_bytes(st.session_state["cleaned_df"], "Cleaned data"),
        file_name="marketlens_cleaned_data.xlsx",
        key="cleaned_dataset",
    )
else:
    st.info("Choose your cleaning options above, then click **Apply and continue** to proceed. "
            "You can also continue without changes.")
    if st.button("Continue without changes"):
        st.session_state["cleaned_df"] = df.copy()
        st.session_state["cleaning_log"] = ["Continued without changes."]
        st.session_state["rows_before_cleaning"] = len(df)
        st.session_state["rows_after_cleaning"] = len(df)
        st.rerun()
