"""Module D: Which customers behave similarly? (K-means clustering under the hood)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from services import validation_service as vs

GROUP_NAME_LIBRARY = [
    "Premium Loyalists", "Discount Seekers", "At-Risk Customers", "New Explorers",
    "Steady Regulars", "High-Value Occasionals", "Budget Browsers", "Rising Spenders",
]


def suggest_n_clusters(X_scaled: np.ndarray, k_range=range(2, 9)) -> tuple[int, dict]:
    scores = {}
    for k in k_range:
        if k >= len(X_scaled):
            continue
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            continue
        scores[k] = float(silhouette_score(X_scaled, labels))
    if not scores:
        return 2, scores
    best_k = max(scores, key=scores.get)
    return best_k, scores


def run_analysis(df: pd.DataFrame, cluster_cols: list, roles: dict, n_clusters: int = 4,
                  auto_suggest: bool = True) -> dict:
    warnings = []
    warnings += vs.check_dataset_size(df)

    clean_cols, id_warnings = vs.clean_predictor_list(cluster_cols, roles)
    warnings += id_warnings
    clean_cols = [c for c in clean_cols if pd.api.types.is_numeric_dtype(df[c])]

    if len(clean_cols) < 2:
        return {"ok": False, "reason": "Please select at least two numeric columns describing customer behaviour.",
                "limitations": warnings}

    working = df[clean_cols].copy()
    for col in clean_cols:
        working[col] = working[col].fillna(working[col].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(working)

    silhouette_scores = {}
    if auto_suggest:
        suggested_k, silhouette_scores = suggest_n_clusters(X_scaled)
        k = suggested_k
    else:
        k = max(2, min(8, n_clusters))

    k = min(k, len(working) - 1) if len(working) > 2 else 2
    k = max(k, 2)

    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)
    sil_score = float(silhouette_score(X_scaled, labels)) if len(set(labels)) > 1 else 0.0

    assigned = df.copy()
    assigned["Cluster"] = labels

    group_summaries = []
    overall_means = working.mean()
    for cluster_id in sorted(set(labels)):
        group_df = working[labels == cluster_id]
        group_means = group_df.mean()
        size = len(group_df)

        # Characteristics: which columns are notably above/below overall average
        diffs = ((group_means - overall_means) / overall_means.replace(0, np.nan).abs()).fillna(0)
        top_traits = diffs.abs().sort_values(ascending=False).head(3).index.tolist()
        characteristics = [
            f"{col} is {'higher' if diffs[col] > 0 else 'lower'} than average" for col in top_traits
        ]

        suggested_name = GROUP_NAME_LIBRARY[cluster_id % len(GROUP_NAME_LIBRARY)]
        suggested_action = _suggest_action(diffs, top_traits)

        group_summaries.append({
            "cluster_id": int(cluster_id),
            "size": int(size),
            "size_pct": round(size / len(working) * 100, 1),
            "averages": group_means.to_dict(),
            "characteristics": characteristics,
            "suggested_name": suggested_name,
            "suggested_action": suggested_action,
        })

    warnings += vs.check_model_performance(sil_score, "group separation score", 0.15)

    return {
        "ok": True,
        "cluster_cols": clean_cols,
        "n_groups": k,
        "silhouette_score": sil_score,
        "silhouette_scores_by_k": silhouette_scores,
        "group_summaries": group_summaries,
        "assigned_df": assigned,
        "limitations": warnings,
    }


def _suggest_action(diffs: pd.Series, top_traits: list) -> str:
    if not top_traits:
        return "Monitor this group's behaviour over time."
    trait = top_traits[0]
    trait_lower = trait.lower()
    going_up = diffs[trait] > 0
    if "recency" in trait_lower or "days_since" in trait_lower:
        return "Re-engage with a win-back offer." if going_up else "Reward frequent recent engagement with loyalty perks."
    if "spend" in trait_lower or "revenue" in trait_lower or "value" in trait_lower or "order" in trait_lower:
        return "Offer premium or exclusive perks to protect this valuable relationship." if going_up else \
            "Introduce entry-level offers to encourage larger purchases."
    if "discount" in trait_lower or "promo" in trait_lower:
        return "Continue targeted promotions, but monitor margin impact." if going_up else \
            "Test small incentives to see if they influence this group."
    if "frequency" in trait_lower or "visit" in trait_lower:
        return "Nurture with regular engagement content." if going_up else "Encourage repeat visits with reminders."
    if "complaint" in trait_lower:
        return "Prioritise service recovery for this group." if going_up else "Low-risk group - maintain standard service."
    return "Tailor messaging based on this group's distinct profile."


def render(result: dict, df: pd.DataFrame, roles: dict, config: dict, settings: dict):
    import streamlit as st
    from services.interpretation_service import get_interpreter
    from services.recommendation_engine import recommend_from_segmentation
    from visualisations import charts, result_panels

    if not result.get("ok"):
        st.error(result.get("reason", "This analysis could not be run with the selected columns."))
        return

    interpreter = get_interpreter()
    interpretation = interpreter.interpret_segmentation(result)
    recommendations = recommend_from_segmentation(result["group_summaries"], settings.get("business_priority"))

    result_panels.render_main_answer(interpretation["main_answer"], interpretation["explanation"])
    result_panels.render_confidence(interpretation["confidence"])
    result_panels.render_reasoning_trail(interpretation.get("reasoning_steps", []))

    st.markdown("### Customer groups")
    cols = st.columns(min(4, len(result["group_summaries"])) or 1)
    for i, group in enumerate(result["group_summaries"]):
        with cols[i % len(cols)]:
            st.markdown(f"#### {group['suggested_name']}")
            st.metric("Customers", f"{group['size']} ({group['size_pct']}%)")
            for trait in group["characteristics"]:
                st.caption(f"• {trait}")
            st.info(group["suggested_action"])

    st.markdown("### Group comparison")
    compare_df = pd.DataFrame([
        {"Group": g["suggested_name"], **{k: round(v, 2) for k, v in g["averages"].items()}}
        for g in result["group_summaries"]
    ])
    metric_cols = [c for c in compare_df.columns if c != "Group"]
    if metric_cols:
        long_df = compare_df.melt(id_vars="Group", value_vars=metric_cols, var_name="Measure", value_name="Value")
        fig, explanation = charts.bar_chart(long_df, "Group", "Value", "Average behaviour by group", agg="sum")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

    result_panels.render_recommendations(recommendations)
    result_panels.render_limitations(
        result.get("limitations", []),
        do_not_conclude=["Group membership describes current behaviour, not a fixed customer identity."],
    )

    st.markdown("### Download")
    from services.export_service import dataframe_to_excel_bytes
    from utils.download_helpers import lazy_download_button
    lazy_download_button(
        "Download customer group assignments",
        lambda: dataframe_to_excel_bytes(result["assigned_df"], "Customer groups"),
        file_name="customer_segments.xlsx",
        key="customer_segments",
    )

    with st.expander("Show technical details", expanded=settings.get("show_technical", False)):
        if settings.get("show_technical"):
            st.write(f"Silhouette score: {result['silhouette_score']:.3f}")
            if result["silhouette_scores_by_k"]:
                sil_df = pd.DataFrame(
                    list(result["silhouette_scores_by_k"].items()), columns=["Number of groups (k)", "Silhouette score"],
                )
                st.dataframe(sil_df, use_container_width=True, hide_index=True)
        else:
            st.caption("Turn on 'Show technical details' in the sidebar to see statistical output.")
