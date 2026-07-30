"""MarketLens - entry point and landing page.

Run with: streamlit run app.py
"""

import streamlit as st

from utils.session_state import init_session_state, has_data
from utils.sidebar import render_global_sidebar
from utils.constants import APP_NAME, APP_TAGLINE, BUSINESS_QUESTIONS, QUESTION_DESCRIPTIONS
from utils.theme import CATEGORICAL, step_card_html, question_card_html

st.set_page_config(page_title=APP_NAME, page_icon="🔎", layout="wide")

init_session_state()
render_global_sidebar()

st.markdown(
    f"""
    <div class="ml-hero">
        <div class="ml-hero-eyebrow">Marketing analytics, plainly explained</div>
        <div class="ml-hero-title">🔎 {APP_NAME}</div>
        <div class="ml-hero-sub">{APP_TAGLINE} Upload a spreadsheet, ask a business question in plain
        English, and get a visual, jargon-free answer - regression, classification and clustering run
        quietly underneath, explained the way you'd explain it to a colleague.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if has_data():
    st.success(f"A file is already loaded: **{st.session_state.get('file_name')}**. Continue in the sidebar pages.")
else:
    st.info("Get started by opening **1 Upload and Review** in the sidebar page list.")

st.markdown("### How it works")
steps = [
    ("1", "Upload", "Upload an Excel or CSV file. Pick a worksheet if needed.", CATEGORICAL[0]),
    ("2", "Review", "Check data quality and clean up issues in plain language.", CATEGORICAL[1]),
    ("3", "Ask", "Pick what you want to understand - no statistics vocabulary required.", CATEGORICAL[2]),
    ("4", "Learn", "Get a plain-language answer, confidence level, and recommended action.", CATEGORICAL[6]),
]
cols = st.columns(4)
for col, (number, title, desc, color) in zip(cols, steps):
    with col:
        st.markdown(step_card_html(number, title, desc, color), unsafe_allow_html=True)

st.markdown("### What MarketLens can answer for you")
QUESTION_ICONS = {
    "What is influencing sales?": "📈",
    "Who is most likely to purchase?": "🎯",
    "Which customers behave similarly?": "🧩",
    "Did a campaign perform better?": "🆚",
    "Which customers may stop buying?": "⚠️",
    "What is the value of each customer?": "💎",
    "Which marketing channel performs best?": "📡",
    "How are sales changing over time?": "📆",
    "Explore the data visually": "🔍",
    "Create a custom analysis": "🛠️",
}
q_cols = st.columns(2)
for i, question in enumerate(BUSINESS_QUESTIONS):
    color = CATEGORICAL[i % len(CATEGORICAL)]
    icon = QUESTION_ICONS.get(question, "❓")
    with q_cols[i % 2]:
        st.markdown(
            question_card_html(icon, question, QUESTION_DESCRIPTIONS[question], color),
            unsafe_allow_html=True,
        )

st.caption(
    "MarketLens uses real statistical techniques internally (regression, classification, clustering) "
    "but explains everything in plain business language by default. Turn on 'Show technical details' "
    "in the sidebar at any time to see the underlying statistics."
)
