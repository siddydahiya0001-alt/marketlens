"""MarketLens design tokens: a validated, colorblind-safe palette applied
consistently across every chart and every custom UI card in the app.

Categorical hues are assigned in a fixed order (never cycled or reassigned
per-chart) - the ordering itself is the colorblind-safety mechanism. Sequential
and diverging scales use single/two-hue ramps rather than rainbows. Status
colors are reserved for confidence/risk indicators and never reused as a
generic series color.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Categorical palette - fixed order, used for chart series and campaign/
# channel/segment identity. Never reorder per-chart.
# ---------------------------------------------------------------------------
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Sequential (single hue, light -> dark) for magnitude encodings.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Diverging pair (blue <-> red) with a neutral gray midpoint, for polarity
# encodings such as the correlation heatmap (-1 to +1).
DIVERGING = [
    [0.0, "#0d366b"], [0.25, "#3987e5"], [0.5, "#f0efec"], [0.75, "#eb6834"], [1.0, "#8a1f1f"],
]

# Status colors - fixed, never themed, always paired with an icon + label.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# Chart chrome / ink
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e6e9f0"
BASELINE = "#c3c2b7"
# Cards/charts render on pure white so they visibly lift off the tinted page
# canvas below - flat, same-tone surfaces are what read as "dull".
CHART_SURFACE = "#ffffff"
PAGE_PLANE = "#eef2fa"
SIDEBAR_PLANE = "#e7edf8"
BORDER = "rgba(42,120,214,0.16)"
SHADOW = "0 1px 2px rgba(16,40,80,0.04), 0 4px 14px rgba(16,40,80,0.07)"

FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', sans-serif"

# Confidence label -> status color, mirrored across every result panel.
CONFIDENCE_STATUS = {
    "Very high confidence": STATUS["good"],
    "High confidence": STATUS["good"],
    "Moderate confidence": STATUS["warning"],
    "Low confidence": STATUS["serious"],
    "Not enough evidence": STATUS["critical"],
}
CONFIDENCE_ICON = {
    "Very high confidence": "✓",
    "High confidence": "✓",
    "Moderate confidence": "!",
    "Low confidence": "!",
    "Not enough evidence": "✕",
}


def inject_custom_css():
    """Injects the shared design system once per page. Idempotent per-render."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, {PAGE_PLANE} 0%, {PAGE_PLANE} 320px, #f5f8fc 320px, #f5f8fc 100%);
        }}
        section[data-testid="stSidebar"] {{
            background: {SIDEBAR_PLANE};
            border-right: 1px solid {BORDER};
        }}
        .block-container {{ padding-top: 2rem; }}

        .ml-card {{
            background: {CHART_SURFACE};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 16px 20px;
            margin: 10px 0;
            box-shadow: {SHADOW};
        }}
        .ml-card-title {{
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            color: {INK_MUTED};
            margin-bottom: 6px;
        }}
        .ml-card-body {{
            color: {INK_PRIMARY};
            font-size: 1.02rem;
            line-height: 1.5;
        }}

        .ml-answer {{
            border-left: 4px solid {CATEGORICAL[0]};
            background: linear-gradient(90deg, {CATEGORICAL[0]}1f, {CATEGORICAL[0]}05 55%, {CHART_SURFACE} 100%);
            box-shadow: {SHADOW};
        }}

        .ml-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border-radius: 999px;
            padding: 6px 14px;
            font-weight: 600;
            font-size: 0.92rem;
            border: 1.5px solid currentColor;
            background: {CHART_SURFACE};
            box-shadow: {SHADOW};
        }}
        .ml-badge-dot {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 18px;
            height: 18px;
            border-radius: 999px;
            color: white;
            font-size: 0.72rem;
            font-weight: 700;
        }}

        .ml-reasoning {{
            border-left: 4px solid {CATEGORICAL[6]};
            background: linear-gradient(90deg, {CATEGORICAL[6]}1f, {CATEGORICAL[6]}05 55%, {CHART_SURFACE} 100%);
            box-shadow: {SHADOW};
        }}
        .ml-reasoning ol {{
            margin: 4px 0 0 0;
            padding-left: 1.2rem;
        }}
        .ml-reasoning li {{
            margin-bottom: 6px;
            color: {INK_SECONDARY};
            line-height: 1.5;
        }}

        .ml-recommendation {{
            border-left: 4px solid {STATUS["good"]};
            background: linear-gradient(90deg, {STATUS["good"]}1f, {STATUS["good"]}05 55%, {CHART_SURFACE} 100%);
            padding: 12px 18px;
            margin: 8px 0;
            border-radius: 8px;
            box-shadow: {SHADOW};
        }}

        .ml-limitation {{
            border-left: 4px solid {STATUS["serious"]};
            background: linear-gradient(90deg, {STATUS["serious"]}1f, {STATUS["serious"]}05 55%, {CHART_SURFACE} 100%);
            padding: 12px 18px;
            margin: 8px 0;
            border-radius: 8px;
            box-shadow: {SHADOW};
        }}

        /* Hero band */
        .ml-hero {{
            background:
                radial-gradient(1200px 240px at 15% -40%, rgba(255,255,255,0.28), transparent 60%),
                linear-gradient(120deg, {CATEGORICAL[0]} 0%, {CATEGORICAL[6]} 65%, #351f7a 100%);
            border-radius: 16px;
            padding: 40px 44px;
            margin-bottom: 20px;
            color: #ffffff;
            box-shadow: 0 12px 30px rgba(42,60,140,0.28);
        }}
        .ml-hero-eyebrow {{
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.85;
            margin-bottom: 8px;
        }}
        .ml-hero-title {{
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 10px;
        }}
        .ml-hero-sub {{
            font-size: 1.05rem;
            max-width: 640px;
            opacity: 0.95;
            line-height: 1.55;
        }}

        /* Numbered step cards */
        .ml-step-card {{
            background: {CHART_SURFACE};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 18px 18px 16px 18px;
            height: 100%;
            border-top: 4px solid var(--step-color, {CATEGORICAL[0]});
            box-shadow: {SHADOW};
        }}
        .ml-step-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 999px;
            background: var(--step-color, {CATEGORICAL[0]});
            color: #fff;
            font-weight: 700;
            font-size: 0.85rem;
            margin-bottom: 10px;
        }}
        .ml-step-title {{
            font-weight: 700;
            font-size: 1rem;
            color: {INK_PRIMARY};
            margin-bottom: 4px;
        }}
        .ml-step-desc {{
            font-size: 0.88rem;
            color: {INK_SECONDARY};
            line-height: 1.45;
        }}

        /* Question gallery cards */
        .ml-q-card {{
            background: linear-gradient(90deg, var(--q-color, {CATEGORICAL[0]})14, {CHART_SURFACE} 45%);
            border: 1px solid {BORDER};
            border-left: 3px solid var(--q-color, {CATEGORICAL[0]});
            border-radius: 10px;
            padding: 12px 16px;
            height: 100%;
            margin-bottom: 4px;
            box-shadow: {SHADOW};
        }}
        .ml-q-title {{
            font-weight: 650;
            font-size: 0.92rem;
            color: {INK_PRIMARY};
            margin-bottom: 2px;
        }}
        .ml-q-desc {{
            font-size: 0.82rem;
            color: {INK_SECONDARY};
        }}

        /* Stat tiles */
        .ml-stat-tile {{
            background: {CHART_SURFACE};
            border: 1px solid {BORDER};
            border-top: 4px solid var(--stat-color, {CATEGORICAL[0]});
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 6px;
            box-shadow: {SHADOW};
        }}
        .ml-stat-value {{
            font-size: 1.6rem;
            font-weight: 750;
            color: {INK_PRIMARY};
            font-variant-numeric: tabular-nums;
            line-height: 1.2;
        }}
        .ml-stat-label {{
            font-size: 0.8rem;
            color: {INK_MUTED};
            margin-top: 2px;
        }}

        .ml-progress-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.86rem;
            padding: 3px 0;
        }}
        .ml-progress-dot {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 18px;
            height: 18px;
            border-radius: 999px;
            font-size: 0.65rem;
            font-weight: 700;
            flex-shrink: 0;
        }}
        .ml-file-chip {{
            background: {CHART_SURFACE};
            border: 1px solid {BORDER};
            border-left: 3px solid {CATEGORICAL[0]};
            border-radius: 8px;
            padding: 8px 12px;
            margin: 6px 0;
            font-size: 0.82rem;
            color: {INK_SECONDARY};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def progress_item_html(label: str, done: bool) -> str:
    if done:
        dot = f'<span class="ml-progress-dot" style="background:{STATUS["good"]}; color:#fff;">✓</span>'
        text_style = f"color:{INK_PRIMARY}; font-weight:600;"
    else:
        dot = f'<span class="ml-progress-dot" style="background:transparent; border:1.5px solid {BASELINE}; color:{INK_MUTED};"></span>'
        text_style = f"color:{INK_MUTED};"
    return f'<div class="ml-progress-item"><span style="{text_style}">{dot} {label}</span></div>'


def step_card_html(number: int, title: str, description: str, color: str) -> str:
    return (
        f'<div class="ml-step-card" style="--step-color:{color};">'
        f'<div class="ml-step-badge">{number}</div>'
        f'<div class="ml-step-title">{title}</div>'
        f'<div class="ml-step-desc">{description}</div>'
        f'</div>'
    )


def question_card_html(icon: str, title: str, description: str, color: str) -> str:
    return (
        f'<div class="ml-q-card" style="--q-color:{color};">'
        f'<div class="ml-q-title">{icon} {title}</div>'
        f'<div class="ml-q-desc">{description}</div>'
        f'</div>'
    )


def stat_tile_html(value: str, label: str, color: str) -> str:
    return (
        f'<div class="ml-stat-tile" style="--stat-color:{color};">'
        f'<div class="ml-stat-value">{value}</div>'
        f'<div class="ml-stat-label">{label}</div>'
        f'</div>'
    )


def confidence_badge_html(confidence: str) -> str:
    color = CONFIDENCE_STATUS.get(confidence, STATUS["warning"])
    icon = CONFIDENCE_ICON.get(confidence, "!")
    return (
        f'<span class="ml-badge" style="color:{color}; border-color:{color}80; background:{color}17;">'
        f'<span class="ml-badge-dot" style="background:{color};">{icon}</span>{confidence}</span>'
    )
