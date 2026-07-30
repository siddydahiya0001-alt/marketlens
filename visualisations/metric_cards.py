"""Reusable stat-tile layouts, styled consistently with the rest of the app (see utils.theme)."""

from __future__ import annotations

import streamlit as st

from utils.helpers import format_currency, format_number, format_percent
from utils.theme import CATEGORICAL, stat_tile_html


def render_metric_row(metrics: list, columns_per_row: int = 4):
    """metrics: list of dicts {label, value, delta (optional, unused), help (optional, unused)}"""
    cols = st.columns(min(columns_per_row, max(len(metrics), 1)))
    for i, metric in enumerate(metrics):
        color = CATEGORICAL[i % len(CATEGORICAL)]
        with cols[i % len(cols)]:
            st.markdown(stat_tile_html(str(metric["value"]), metric["label"], color), unsafe_allow_html=True)


def data_overview_metrics(n_rows: int, n_cols: int, n_missing: int, n_duplicates: int):
    render_metric_row([
        {"label": "Rows", "value": format_number(n_rows)},
        {"label": "Columns", "value": format_number(n_cols)},
        {"label": "Missing values", "value": format_number(n_missing)},
        {"label": "Duplicate rows", "value": format_number(n_duplicates)},
    ])


def business_summary_metrics(n_rows, n_customers, total_sales, avg_sales, currency="$", number_format="1,234.56"):
    metrics = [{"label": "Total records", "value": format_number(n_rows, number_format)}]
    if n_customers is not None:
        metrics.append({"label": "Total customers", "value": format_number(n_customers, number_format)})
    if total_sales is not None:
        metrics.append({"label": "Total sales", "value": format_currency(total_sales, currency, number_format)})
    if avg_sales is not None:
        metrics.append({"label": "Average sales", "value": format_currency(avg_sales, currency, number_format)})
    render_metric_row(metrics)
