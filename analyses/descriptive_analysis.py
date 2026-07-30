"""Module A: Explore the data visually.

Provides summary statistics plus a free-form chart builder the user drives
from the Results page. Kept mostly stateless/pure so it is easy to test.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from utils.constants import ROLE_REVENUE, ROLE_SALES, ROLE_PROFIT, ROLE_CUSTOMER_ID, ROLE_DATE, ROLE_CATEGORY


def compute_summary(df: pd.DataFrame, roles: dict) -> dict:
    summary = {"n_rows": len(df)}

    id_cols = [c for c, r in roles.items() if r == ROLE_CUSTOMER_ID and c in df.columns]
    summary["n_customers"] = int(df[id_cols[0]].nunique()) if id_cols else None

    sales_cols = [c for c, r in roles.items() if r in (ROLE_SALES, ROLE_REVENUE) and c in df.columns]
    if sales_cols:
        sales_series = df[sales_cols[0]]
        summary["sales_column"] = sales_cols[0]
        summary["total_sales"] = float(sales_series.sum())
        summary["avg_sales"] = float(sales_series.mean())
        summary["median_sales"] = float(sales_series.median())
    else:
        summary["sales_column"] = None
        summary["total_sales"] = summary["avg_sales"] = summary["median_sales"] = None

    profit_cols = [c for c, r in roles.items() if r == ROLE_PROFIT and c in df.columns]
    summary["total_profit"] = float(df[profit_cols[0]].sum()) if profit_cols else None

    summary["missing_cells"] = int(df.isna().sum().sum())
    summary["missing_pct"] = float(df.isna().mean().mean() * 100)
    return summary


def category_distribution(df: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = df[column].value_counts(dropna=False).reset_index()
    counts.columns = [column, "Count"]
    return counts


def numeric_distribution_stats(df: pd.DataFrame, column: str) -> dict:
    series = df[column].dropna()
    return {
        "mean": float(series.mean()) if len(series) else None,
        "median": float(series.median()) if len(series) else None,
        "std": float(series.std()) if len(series) else None,
        "min": float(series.min()) if len(series) else None,
        "max": float(series.max()) if len(series) else None,
    }


def top_performing_categories(df: pd.DataFrame, category_col: str, value_col: str, top_n: int = 5) -> pd.DataFrame:
    grouped = df.groupby(category_col, dropna=False)[value_col].sum().sort_values(ascending=False).head(top_n)
    return grouped.reset_index()


def suggest_time_column(roles: dict) -> str | None:
    date_cols = [c for c, r in roles.items() if r == ROLE_DATE]
    return date_cols[0] if date_cols else None


def suggest_category_columns(roles: dict) -> list:
    return [c for c, r in roles.items() if r == ROLE_CATEGORY]


def suggest_numeric_columns(df: pd.DataFrame, roles: dict) -> list:
    from utils.constants import NUMERIC_ROLES
    return [c for c in df.select_dtypes(include="number").columns if c in df.columns]


def render_explore(df: pd.DataFrame, roles: dict, settings: dict):
    """Interactive free-form exploration screen (Module A)."""
    import streamlit as st
    from visualisations import charts
    from visualisations.metric_cards import business_summary_metrics
    from utils.constants import CHART_TYPES
    from utils.helpers import format_number, format_percent

    summary = compute_summary(df, roles)
    currency = settings.get("currency", "$")
    number_format = settings.get("number_format", "1,234.56")

    st.markdown("### Overview")
    business_summary_metrics(
        summary["n_rows"], summary["n_customers"], summary["total_sales"], summary["avg_sales"],
        currency, number_format,
    )
    if summary["total_profit"] is not None:
        st.metric("Total profit", f"{currency}{format_number(summary['total_profit'], number_format)}")
    st.caption(f"Missing data across the dataset: {format_percent(summary['missing_pct'])}")

    st.markdown("---")
    st.markdown("### Build your own chart")

    all_cols = list(df.columns)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    category_cols = suggest_category_columns(roles) or [c for c in all_cols if df[c].dtype == object]

    col1, col2, col3 = st.columns(3)
    with col1:
        chart_type = st.selectbox("Chart type", CHART_TYPES, index=CHART_TYPES.index(settings.get("chart_type", "Bar chart")) if settings.get("chart_type") in CHART_TYPES else 0)
    with col2:
        x_col = st.selectbox("X-axis / grouping column", all_cols)
    with col3:
        y_options = numeric_cols if numeric_cols else all_cols
        y_col = st.selectbox("Y-axis / value column", y_options) if chart_type not in ("Histogram", "Correlation heatmap") else None

    fig, explanation = None, ""
    try:
        if chart_type == "Bar chart":
            fig, explanation = charts.bar_chart(df, x_col, y_col, f"{y_col} by {x_col}")
        elif chart_type == "Line chart":
            fig, explanation = charts.line_chart(df, x_col, y_col, f"{y_col} over {x_col}")
        elif chart_type == "Histogram":
            fig, explanation = charts.histogram_chart(df, x_col, f"Distribution of {x_col}")
        elif chart_type == "Box plot":
            fig, explanation = charts.box_plot(df, y_col, x_col, f"Spread of {y_col} by {x_col}")
        elif chart_type == "Scatter plot":
            fig, explanation = charts.scatter_plot(df, x_col, y_col, f"{y_col} vs {x_col}")
        elif chart_type == "Pie chart":
            fig, explanation = charts.pie_chart(df, x_col, y_col, f"Share of {y_col} by {x_col}")
        elif chart_type == "Funnel chart":
            counts = df[x_col].value_counts().sort_values(ascending=False)
            fig, explanation = charts.funnel_chart(counts.index.tolist(), counts.values.tolist(), f"Funnel by {x_col}")
        elif chart_type == "Correlation heatmap":
            if len(numeric_cols) < 2:
                st.info("Need at least two numeric columns for a correlation heatmap.")
            else:
                selected = st.multiselect("Columns to include", numeric_cols, default=numeric_cols[:8])
                if len(selected) >= 2:
                    fig, explanation = charts.correlation_heatmap(df, selected)
    except Exception:
        st.error("This chart could not be built with the selected columns. Try a different combination.")

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.caption(explanation)
        from services.export_service import chart_to_png_bytes
        png = chart_to_png_bytes(fig)
        if png:
            st.download_button("Download chart as PNG", png, file_name="marketlens_chart.png", mime="image/png")

    if suggest_time_column(roles):
        st.markdown("---")
        st.markdown("### Trend over time")
        time_col = suggest_time_column(roles)
        value_col = numeric_cols[0] if numeric_cols else None
        if value_col:
            trend_df = df[[time_col, value_col]].dropna().sort_values(time_col)
            trend_fig, trend_expl = charts.line_chart(trend_df, time_col, value_col, f"{value_col} over time")
            st.plotly_chart(trend_fig, use_container_width=True)
            st.caption(trend_expl)

    if category_cols and numeric_cols:
        st.markdown("---")
        st.markdown("### Top-performing categories")
        cat_col = st.selectbox("Category", category_cols, key="top_cat_col")
        val_col = st.selectbox("Value", numeric_cols, key="top_val_col")
        top_df = top_performing_categories(df, cat_col, val_col)
        top_fig, top_expl = charts.bar_chart(df[df[cat_col].isin(top_df[cat_col])], cat_col, val_col, f"Top {cat_col} by {val_col}")
        st.plotly_chart(top_fig, use_container_width=True)
        st.caption(top_expl)
