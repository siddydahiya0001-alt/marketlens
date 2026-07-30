"""Page 4: Results - dispatches to the right analysis module and renders its plain-language output."""

import streamlit as st

from utils.session_state import init_session_state, active_df
from utils.sidebar import render_global_sidebar
from utils.constants import QUESTION_EXPLORE, QUESTION_CUSTOM
from services import analysis_router
from visualisations.metric_cards import data_overview_metrics

st.set_page_config(page_title="Results - MarketLens", page_icon="📊", layout="wide")
init_session_state()
render_global_sidebar()

st.title("📊 Results")

if st.session_state.get("raw_df") is None:
    st.warning("Please upload a file first.")
    st.page_link("pages/1_Upload_and_Review.py", label="Go to Upload and Review", icon="⬅️")
    st.stop()

if not st.session_state.get("business_question"):
    st.warning("Please choose a business question first.")
    st.page_link("pages/3_Choose_Question.py", label="Go to Choose a Question", icon="⬅️")
    st.stop()

df = active_df()
roles = st.session_state.get("roles", {})
question = st.session_state["business_question"]
config = st.session_state.get("question_config", {})
settings = st.session_state.get("settings", {})

with st.expander("Data overview", expanded=False):
    data_overview_metrics(len(df), df.shape[1], int(df.isna().sum().sum()), int(df.duplicated().sum()))

st.markdown(f"## {question}")

if question in (QUESTION_EXPLORE, QUESTION_CUSTOM):
    from analyses.descriptive_analysis import render_explore
    render_explore(df, roles, settings)
else:
    result = analysis_router.run(question, df, roles, config, settings)
    st.session_state["analysis_result"] = result

    renderer_map = {}
    from utils.constants import (
        QUESTION_SALES_DRIVERS, QUESTION_PURCHASE_LIKELIHOOD, QUESTION_SEGMENTATION,
        QUESTION_CAMPAIGN, QUESTION_CHURN, QUESTION_CUSTOMER_VALUE, QUESTION_CHANNEL,
        QUESTION_TIME_SERIES,
    )

    if question == QUESTION_SALES_DRIVERS:
        from analyses import sales_driver_analysis as module
    elif question == QUESTION_PURCHASE_LIKELIHOOD:
        from analyses import purchase_prediction as module
    elif question == QUESTION_SEGMENTATION:
        from analyses import customer_segmentation as module
    elif question == QUESTION_CAMPAIGN:
        from analyses import campaign_comparison as module
    elif question == QUESTION_CHURN:
        from analyses import churn_analysis as module
    elif question == QUESTION_CUSTOMER_VALUE:
        from analyses import customer_value as module
    elif question == QUESTION_CHANNEL:
        from analyses import channel_performance as module
    elif question == QUESTION_TIME_SERIES:
        from analyses import time_series_analysis as module
    else:
        module = None

    if module is not None:
        module.render(result, df, roles, config, settings)
    else:
        st.info("This analysis is not available yet.")

st.markdown("---")
st.markdown("### Full report")
if st.button("Generate summary report (HTML)"):
    from services.export_service import build_html_report
    html = build_html_report({
        "file_name": st.session_state.get("file_name"),
        "sheet_name": st.session_state.get("selected_sheet"),
        "question": question,
        "columns_used": list(config.values()) if config else [],
        "quality_findings": st.session_state.get("quality_findings", []),
        "main_answer": "See the results above for the full plain-language explanation.",
        "explanation": "",
        "confidence": "",
        "recommendations": [],
        "limitations": st.session_state.get("analysis_result", {}).get("limitations", []) if isinstance(st.session_state.get("analysis_result"), dict) else [],
    })
    st.download_button("Download HTML report", html, file_name="marketlens_report.html", mime="text/html")

    from services.export_service import html_report_to_pdf_bytes
    pdf_bytes = html_report_to_pdf_bytes(html)
    if pdf_bytes:
        st.download_button("Download PDF report", pdf_bytes, file_name="marketlens_report.pdf", mime="application/pdf")
