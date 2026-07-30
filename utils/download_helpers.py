"""Streamlit helpers that make expensive exports (e.g. Excel writes on large
datasets) opt-in and cached, instead of being recomputed on every page rerun.

`st.download_button(..., data=expensive_call())` evaluates `expensive_call()`
eagerly on every script run, even ones triggered by an unrelated widget
elsewhere on the page. For a dataset with hundreds of thousands of rows,
writing it to Excel via openpyxl can take long enough to make the rest of
the page - including navigation links below it - appear stuck. This module
defers that work until the user explicitly asks for it, and caches the
result so it isn't rebuilt on subsequent reruns.
"""

from __future__ import annotations

import streamlit as st


def lazy_download_button(label: str, build_fn, file_name: str, key: str,
                          mime: str | None = None, help_text: str | None = None):
    """Render a 'Prepare' button that computes `build_fn()` once on click, then
    shows the real download button. Cached in session state by `key` so it is
    not recomputed on unrelated reruns.
    """
    state_key = f"_export_bytes_{key}"
    cached = st.session_state.get(state_key)

    if cached is None:
        if st.button(f"Prepare: {label}", key=f"prep_{key}", help=help_text):
            with st.spinner("Preparing file - this can take a moment for large datasets..."):
                cached = build_fn()
                st.session_state[state_key] = cached

    if cached is not None:
        st.download_button(label, cached, file_name=file_name, mime=mime, key=f"dl_{key}")
