"""Plotly chart builders. Every chart has a title, axis labels, hover info and
returns a matching plain-language explanation string alongside the figure.

Colors are drawn from the shared, colorblind-validated design system in
`utils.theme` - categorical hues in fixed order, single-hue sequential ramps
for magnitude, a diverging blue<->red pair for polarity, and reserved status
colors (good/warning/serious/critical) for direction and risk cues.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils import theme

TEMPLATE = "plotly_white"
COLOR_SEQUENCE = theme.CATEGORICAL


def _apply_chart_theme(fig, title: str):
    """Shared chrome: font, ink, gridlines, surface - applied to every chart."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color=theme.INK_PRIMARY, family=theme.FONT_FAMILY)),
        font=dict(family=theme.FONT_FAMILY, color=theme.INK_SECONDARY, size=13),
        paper_bgcolor=theme.CHART_SURFACE,
        plot_bgcolor=theme.CHART_SURFACE,
        margin=dict(t=56, l=10, r=10, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=theme.INK_SECONDARY)),
        hoverlabel=dict(bgcolor=theme.CHART_SURFACE, font=dict(family=theme.FONT_FAMILY, color=theme.INK_PRIMARY)),
    )
    fig.update_xaxes(gridcolor=theme.GRIDLINE, linecolor=theme.BASELINE, tickfont=dict(color=theme.INK_MUTED))
    fig.update_yaxes(gridcolor=theme.GRIDLINE, linecolor=theme.BASELINE, tickfont=dict(color=theme.INK_MUTED))
    return fig


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, agg: str = "sum"):
    grouped = df.groupby(x, dropna=False)[y].agg(agg).reset_index().sort_values(y, ascending=False)
    fig = px.bar(
        grouped, x=x, y=y, template=TEMPLATE, text_auto=".2s",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_layout(xaxis_title=x, yaxis_title=f"{agg.capitalize()} of {y}")
    fig.update_traces(hovertemplate=f"{x}=%{{x}}<br>{y}=%{{y:,.2f}}<extra></extra>", marker_line_width=0)
    _apply_chart_theme(fig, title)
    explanation = f"This chart shows the {agg} of {y} for each {x}, ordered from highest to lowest."
    return fig, explanation


def line_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None):
    fig = px.line(
        df.sort_values(x), x=x, y=y, color=color, template=TEMPLATE, markers=True,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_layout(xaxis_title=x, yaxis_title=y)
    fig.update_traces(line=dict(width=2), marker=dict(size=8))
    _apply_chart_theme(fig, title)
    explanation = f"This chart shows how {y} changes over {x}."
    return fig, explanation


def histogram_chart(df: pd.DataFrame, x: str, title: str, nbins: int = 30):
    fig = px.histogram(df, x=x, template=TEMPLATE, nbins=nbins, color_discrete_sequence=COLOR_SEQUENCE)
    fig.update_layout(xaxis_title=x, yaxis_title="Number of records")
    fig.update_traces(marker_line_width=0)
    _apply_chart_theme(fig, title)
    explanation = f"This chart shows how {x} values are distributed across all records."
    return fig, explanation


def box_plot(df: pd.DataFrame, y: str, x: str | None, title: str):
    fig = px.box(df, y=y, x=x, template=TEMPLATE, color=x, color_discrete_sequence=COLOR_SEQUENCE)
    fig.update_layout(yaxis_title=y, xaxis_title=x or "", showlegend=False)
    _apply_chart_theme(fig, title)
    explanation = (
        f"This chart shows the spread of {y}"
        + (f" for each {x}" if x else "")
        + ", including typical range and unusual values (dots)."
    )
    return fig, explanation


def scatter_plot(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None):
    fig = px.scatter(
        df, x=x, y=y, color=color, template=TEMPLATE, opacity=0.75,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_layout(xaxis_title=x, yaxis_title=y)
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color=theme.CHART_SURFACE)))
    _apply_chart_theme(fig, title)
    explanation = f"This chart shows the relationship between {x} and {y}. Each point is one record."
    return fig, explanation


def pie_chart(df: pd.DataFrame, names: str, values: str, title: str):
    grouped = df.groupby(names, dropna=False)[values].sum().reset_index()
    fig = px.pie(
        grouped, names=names, values=values, template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(
        textinfo="percent+label", hovertemplate=f"{names}=%{{label}}<br>{values}=%{{value:,.2f}}<extra></extra>",
        marker=dict(line=dict(color=theme.CHART_SURFACE, width=2)),
    )
    _apply_chart_theme(fig, title)
    explanation = f"This chart shows what share of total {values} comes from each {names}."
    return fig, explanation


def funnel_chart(stage_labels: list, stage_values: list, title: str):
    fig = go.Figure(go.Funnel(
        y=stage_labels, x=stage_values, textinfo="value+percent initial",
        marker=dict(color=COLOR_SEQUENCE[: len(stage_labels)]),
        connector=dict(line=dict(color=theme.GRIDLINE, width=1)),
    ))
    _apply_chart_theme(fig, title)
    explanation = "This chart shows how many records make it through each stage, in order."
    return fig, explanation


def correlation_heatmap(df: pd.DataFrame, columns: list, title: str = "How closely factors move together"):
    numeric_df = df[columns].select_dtypes(include="number")
    corr = numeric_df.corr().round(2)
    diverging_scale = [[stop, color] for stop, color in theme.DIVERGING]
    fig = px.imshow(
        corr, text_auto=True, template=TEMPLATE, color_continuous_scale=diverging_scale, zmin=-1, zmax=1,
        aspect="auto",
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Strength"))
    fig.update_traces(xgap=2, ygap=2)
    _apply_chart_theme(fig, title)
    explanation = (
        "Values close to 1 mean two factors move up and down together. Values close to -1 mean "
        "one goes up when the other goes down. Values near 0 mean little relationship."
    )
    return fig, explanation


def horizontal_ranking_chart(ranked_factors: list, title: str = "What matters most"):
    """ranked_factors: list of {name, direction, strength, score}"""
    names = [f["name"] for f in ranked_factors][::-1]
    scores = [f["score"] for f in ranked_factors][::-1]
    colors = [theme.STATUS["good"] if f["direction"] == "positive" else theme.STATUS["critical"]
              for f in ranked_factors][::-1]
    labels = [f"{f['strength']} {f['direction']} influence" for f in ranked_factors][::-1]

    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation="h", marker_color=colors, marker_line_width=0,
        text=labels, textposition="outside", textfont=dict(color=theme.INK_SECONDARY),
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Relative influence", yaxis_title="", showlegend=False)
    _apply_chart_theme(fig, title)
    explanation = (
        f"Green bars (●) push the outcome up, red bars (●) push it down - not a color choice, a fixed "
        "meaning used everywhere in MarketLens. Longer bars matter more."
    )
    return fig, explanation


def trend_with_forecast_chart(actual_df: pd.DataFrame, x: str, y: str, trend_df: pd.DataFrame | None,
                               forecast_df: pd.DataFrame | None, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_df[x], y=actual_df[y], mode="lines+markers", name="Actual",
                              line=dict(color=theme.CATEGORICAL[0], width=2), marker=dict(size=7)))
    if trend_df is not None:
        fig.add_trace(go.Scatter(x=trend_df[x], y=trend_df[y], mode="lines", name="Trend",
                                  line=dict(color=theme.CATEGORICAL[6], width=2, dash="dot")))
    if forecast_df is not None:
        fig.add_trace(go.Scatter(x=forecast_df[x], y=forecast_df[y], mode="lines+markers", name="Forecast",
                                  line=dict(color=theme.CATEGORICAL[1], width=2, dash="dash"),
                                  marker=dict(size=7)))
    fig.update_layout(xaxis_title=x, yaxis_title=y)
    _apply_chart_theme(fig, title)
    explanation = "Solid blue is actual history, dotted violet is the underlying trend, and dashed orange is the forecast."
    return fig, explanation


def confusion_matrix_chart(matrix, labels, title: str = "Prediction accuracy check"):
    fig = px.imshow(
        matrix, x=labels, y=labels, text_auto=True, template=TEMPLATE,
        color_continuous_scale=theme.SEQUENTIAL_BLUE,
    )
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    fig.update_traces(xgap=2, ygap=2)
    _apply_chart_theme(fig, title)
    explanation = "The diagonal shows correct predictions. Off-diagonal cells show mistakes the model made."
    return fig, explanation


def roc_curve_chart(fpr, tpr, auc_score, title: str = "Model's ability to separate buyers from non-buyers"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model", line=dict(color=theme.CATEGORICAL[0], width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                              line=dict(color=theme.INK_MUTED, dash="dash", width=1.5)))
    fig.update_layout(xaxis_title="False alarm rate", yaxis_title="True detection rate")
    _apply_chart_theme(fig, f"{title} (AUC={auc_score:.2f})")
    explanation = "A curve further to the top-left means the model separates buyers from non-buyers better than chance."
    return fig, explanation


def residual_chart(fitted, residuals, title: str = "Residual check (technical)"):
    fig = px.scatter(x=fitted, y=residuals, template=TEMPLATE,
                      labels={"x": "Predicted value", "y": "Residual (actual - predicted)"})
    fig.update_traces(marker=dict(color=theme.CATEGORICAL[0], size=7, opacity=0.7,
                                   line=dict(width=1, color=theme.CHART_SURFACE)))
    fig.add_hline(y=0, line_dash="dash", line_color=theme.INK_MUTED)
    _apply_chart_theme(fig, title)
    explanation = "Residuals scattered randomly around zero suggest the model's assumptions are reasonable."
    return fig, explanation
