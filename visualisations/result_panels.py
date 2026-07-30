"""Standard result-panel layout used by every analysis module.

Every analysis answers the same eight questions (see requirements), rendered
with a consistent card-based visual language (see utils.theme) so
pages/4_Results.py stays thin and every module looks like one system.
"""

from __future__ import annotations

import html

import streamlit as st

from utils.theme import confidence_badge_html


def _card(body_html: str, css_class: str = "ml-card"):
    st.markdown(f'<div class="{css_class}">{body_html}</div>', unsafe_allow_html=True)


def render_finding_cards(findings: list, icon: str = "⚠️"):
    """Themed card list for plain-language findings (e.g. data-quality issues)."""
    for finding in findings:
        _card(f'<div class="ml-card-body">{icon} {html.escape(finding)}</div>', "ml-limitation")


def render_main_answer(main_answer: str, explanation: str):
    st.markdown("### What we found")
    _card(f'<div class="ml-card-body">{html.escape(main_answer)}</div>', "ml-card ml-answer")
    st.markdown("### What it means")
    st.write(explanation)


def render_confidence(confidence: str, confidence_sentence: str | None = None):
    st.markdown("### How confident are we?")
    st.markdown(confidence_badge_html(confidence), unsafe_allow_html=True)
    if confidence_sentence:
        st.caption(confidence_sentence)


def render_reasoning_trail(steps: list):
    """The 'how did MarketLens work this out' trail - plain language, always
    visible, distinct from the raw statistics behind the technical toggle.
    """
    if not steps:
        return
    st.markdown("### How we got this answer")
    items_html = "".join(f"<li>{html.escape(step)}</li>" for step in steps)
    _card(f'<div class="ml-card-title">Reasoning</div><ol>{items_html}</ol>', "ml-card ml-reasoning")


def render_factor_ranking(fig, explanation: str):
    st.markdown("### Which factors matter most?")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(explanation)


def render_recommendations(recommendations: list):
    st.markdown("### Recommended action")
    if not recommendations:
        st.info("No specific action can be confidently recommended from this result yet.")
        return
    for rec in recommendations:
        _card(f'<div class="ml-card-body">👉 {html.escape(rec)}</div>', "ml-recommendation")


def render_limitations(limitations: list, do_not_conclude: list | None = None):
    st.markdown("### Risks and limitations")
    if limitations:
        for warning in limitations:
            _card(f'<div class="ml-card-body">⚠️ {html.escape(warning)}</div>', "ml-limitation")
    if do_not_conclude:
        with st.expander("What should not be concluded"):
            for item in do_not_conclude:
                st.write(f"- {item}")


def render_missing_data_suggestion(suggestions: list):
    if suggestions:
        with st.expander("What additional data would improve this answer?"):
            for item in suggestions:
                st.write(f"- {item}")


def render_technical_details(render_fn, show_technical: bool):
    """render_fn: a callable that renders the technical widgets when invoked."""
    with st.expander("Show technical details", expanded=show_technical):
        if show_technical:
            render_fn()
        else:
            st.caption("Turn on 'Show technical details' in the sidebar to see statistical output.")


def render_result_structure(result: dict, show_technical: bool):
    """Render the full standard result structure from a result dict.

    Expected keys: main_answer, explanation, confidence, confidence_sentence,
    reasoning_steps, ranking_fig, ranking_explanation, recommendations,
    limitations, do_not_conclude, additional_data_suggestions, technical_render_fn
    """
    render_main_answer(result.get("main_answer", ""), result.get("explanation", ""))
    render_confidence(result.get("confidence", "Not enough evidence"), result.get("confidence_sentence"))
    render_reasoning_trail(result.get("reasoning_steps", []))

    if result.get("ranking_fig") is not None:
        render_factor_ranking(result["ranking_fig"], result.get("ranking_explanation", ""))

    render_recommendations(result.get("recommendations", []))
    render_limitations(result.get("limitations", []), result.get("do_not_conclude"))
    render_missing_data_suggestion(result.get("additional_data_suggestions"))

    if result.get("technical_render_fn"):
        render_technical_details(result["technical_render_fn"], show_technical)
