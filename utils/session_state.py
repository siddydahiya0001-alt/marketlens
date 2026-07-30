"""Central definition and helpers for st.session_state so state survives page navigation."""

import streamlit as st

DEFAULTS = {
    # File / data
    "raw_df": None,
    "file_name": None,
    "sheet_names": [],
    "selected_sheet": None,
    "load_summary": None,
    # Column understanding
    "column_profile": None,  # DataFrame: column, detected_type, suggested_role, missing_pct, unique, user_role
    "roles": {},  # column -> role (user editable)
    "column_display_formats": {},  # column -> {"type": ..., "symbol": ...} for user-added columns
    # Data quality / cleaning
    "quality_findings": [],
    "cleaned_df": None,
    "cleaning_log": [],
    "rows_before_cleaning": None,
    "rows_after_cleaning": None,
    # Business question
    "business_question": None,
    "question_config": {},
    # Settings (flexible interface)
    "settings": {
        "confidence_threshold": 0.05,
        "n_clusters": None,
        "chart_type": "Bar chart",
        "time_period": "Monthly",
        "business_priority": "Increase sales",
        "audience": "Marketing manager",
        "explanation_level": "Business level",
        "currency": "$",
        "number_format": "1,234.56",
        "show_technical": False,
    },
    # Results
    "analysis_result": None,
}


def init_session_state():
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, (dict, list)) else value


def get_setting(name):
    return st.session_state.get("settings", {}).get(name, DEFAULTS["settings"].get(name))


def set_setting(name, value):
    st.session_state.setdefault("settings", {})[name] = value


def reset_for_new_file():
    """Clear downstream state when a new file is uploaded."""
    for key in [
        "column_profile", "roles", "column_display_formats", "quality_findings", "cleaned_df", "cleaning_log",
        "rows_before_cleaning", "rows_after_cleaning", "business_question",
        "question_config", "analysis_result",
    ]:
        st.session_state[key] = DEFAULTS[key].copy() if isinstance(DEFAULTS[key], (dict, list)) else DEFAULTS[key]


def has_data() -> bool:
    return st.session_state.get("raw_df") is not None


def has_cleaned_data() -> bool:
    return st.session_state.get("cleaned_df") is not None


def active_df():
    """Return the cleaned dataset if available, otherwise the raw dataset."""
    if st.session_state.get("cleaned_df") is not None:
        return st.session_state["cleaned_df"]
    return st.session_state.get("raw_df")
