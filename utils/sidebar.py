"""Shared sidebar rendered on every page: status, navigation hints, settings, technical toggle."""

from __future__ import annotations

import streamlit as st

from utils.constants import (
    AUDIENCE_OPTIONS, EXPLANATION_LEVELS, BUSINESS_PRIORITIES, CURRENCY_OPTIONS, NUMBER_FORMATS,
)
from utils.session_state import get_setting, set_setting, has_data, has_cleaned_data
from utils.theme import inject_custom_css, progress_item_html


def render_global_sidebar():
    inject_custom_css()
    with st.sidebar:
        st.markdown("## 🔎 MarketLens")
        st.caption("Marketing analytics without the statistics jargon.")

        st.markdown("---")
        st.markdown("#### Workflow status")
        steps_done = [
            has_data(),
            st.session_state.get("cleaning_log") not in (None, []),
            st.session_state.get("business_question") is not None,
            st.session_state.get("analysis_result") is not None,
        ]
        st.progress(sum(steps_done) / len(steps_done))
        _status_line("1. Upload data", steps_done[0])
        _status_line("2. Review data quality", steps_done[1])
        _status_line("3. Choose a question", steps_done[2])
        _status_line("4. View results", steps_done[3])

        if st.session_state.get("file_name"):
            chip = f"📄 {st.session_state['file_name']}"
            if st.session_state.get("selected_sheet"):
                chip += f"<br>📑 {st.session_state['selected_sheet']}"
            st.markdown(f'<div class="ml-file-chip">{chip}</div>', unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("⚙️ Settings", expanded=False):
            set_setting("business_priority", st.selectbox(
                "Business priority", BUSINESS_PRIORITIES,
                index=BUSINESS_PRIORITIES.index(get_setting("business_priority")),
                help="Recommendations adjust based on this priority.",
            ))
            set_setting("audience", st.selectbox(
                "Audience", AUDIENCE_OPTIONS, index=AUDIENCE_OPTIONS.index(get_setting("audience")),
            ))
            set_setting("explanation_level", st.selectbox(
                "Explanation level", EXPLANATION_LEVELS,
                index=EXPLANATION_LEVELS.index(get_setting("explanation_level")),
            ))
            set_setting("currency", st.selectbox(
                "Currency", CURRENCY_OPTIONS, index=CURRENCY_OPTIONS.index(get_setting("currency")),
            ))
            set_setting("number_format", st.selectbox(
                "Number format", NUMBER_FORMATS, index=NUMBER_FORMATS.index(get_setting("number_format")),
            ))
            set_setting("confidence_threshold", st.slider(
                "Confidence threshold (lower = stricter)", 0.01, 0.10,
                float(get_setting("confidence_threshold")), 0.01,
            ))

        st.markdown("---")
        set_setting("show_technical", st.toggle(
            "Show technical details", value=get_setting("show_technical"),
            help="Reveal statistical terms, coefficients, p-values and diagnostics.",
        ))


def _status_line(label: str, done: bool):
    st.markdown(progress_item_html(label, done), unsafe_allow_html=True)
