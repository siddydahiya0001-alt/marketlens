"""Rule-based recommendation generation.

Recommendations are derived from analysis findings and change based on the
user-selected business priority (increase sales, retain customers, etc.).
"""

from __future__ import annotations

PRIORITY_FRAMING = {
    "Increase sales": "to grow overall sales",
    "Increase profit": "to protect or improve profit margin",
    "Retain customers": "to keep existing customers engaged",
    "Reduce marketing cost": "to spend marketing budget more efficiently",
    "Reduce risk": "to reduce exposure to unreliable patterns",
    "Improve campaign performance": "to improve future campaign results",
}


def recommend_from_sales_drivers(ranked_factors: list, priority: str) -> list:
    if not ranked_factors:
        return ["No strong factors were identified, so no specific action is recommended yet."]

    # Only act on factors that cleared the statistical significance bar - a factor that merely
    # ranks first by effect size isn't necessarily a reliable pattern to recommend acting on.
    significant_factors = [f for f in ranked_factors if f.get("significant")]
    if not significant_factors:
        return [
            "None of the tested factors showed a pattern reliable enough to act on with confidence yet. "
            "Treat the ranking above as exploratory rather than a basis for action - more data or "
            "different factors may give a clearer picture."
        ]

    recommendations = []
    positive = [f for f in significant_factors if f["direction"] == "positive"][:2]
    negative = [f for f in significant_factors if f["direction"] == "negative"][:2]

    framing = PRIORITY_FRAMING.get(priority, "to support the business goal")

    for factor in positive:
        recommendations.append(
            f"Invest further in '{factor['name']}' - it shows a {factor['strength'].lower()} positive "
            f"connection with the outcome, which can help {framing}."
        )
    for factor in negative:
        recommendations.append(
            f"Investigate and reduce '{factor['name']}' - it shows a {factor['strength'].lower()} negative "
            f"connection with the outcome."
        )
    if not recommendations:
        recommendations.append("No factor showed a strong enough effect to base a confident action on yet.")
    return recommendations


def recommend_from_purchase_prediction(top_signals: list, priority: str, accuracy: float | None = None) -> list:
    if not top_signals:
        return ["No strong purchase signals were identified yet."]
    # A model that barely beats a coin flip isn't a reliable basis for targeting decisions.
    if accuracy is not None and accuracy < 0.60:
        return [
            "The model's accuracy is too close to guessing for its signals to be used as a targeting "
            "basis yet. Consider adding more relevant behaviour columns or gathering more data before "
            "acting on this ranking.",
        ]
    signal_text = " and ".join(top_signals[:2])
    base = f"Create a high-intent audience using {signal_text}."
    if priority == "Reduce marketing cost":
        base += " Focus spend on this audience instead of broadcasting to everyone."
    elif priority == "Increase sales":
        base += " Prioritise this audience in the next campaign to convert more likely buyers."
    return [base]


def recommend_from_segmentation(group_summaries: list, priority: str) -> list:
    recommendations = []
    for group in group_summaries:
        name = group.get("suggested_name", "This group")
        action = group.get("suggested_action", "Monitor this group over time.")
        recommendations.append(f"{name}: {action}")
    return recommendations


def recommend_from_campaign_comparison(better_campaign: str, roas_diff: float | None, priority: str,
                                        p_value: float | None = None) -> list:
    # A numerically higher conversion rate is not the same as a confirmed difference - only
    # recommend scaling once the evidence clears a reasonable bar (p < 0.10).
    if p_value is not None and p_value >= 0.10:
        return [
            f"The difference favouring '{better_campaign}' is not strong enough to confidently act on yet - "
            "it could plausibly be due to chance. Consider running the comparison for longer or with more "
            "customers before shifting budget.",
        ]

    recs = [f"Scale {better_campaign} gradually and continue monitoring profitability."]
    if roas_diff is not None and roas_diff < 0:
        recs.append(
            f"{better_campaign} converted more customers but returned less revenue per unit spent. "
            "Confirm the cost is justified before committing further budget."
        )
    if priority == "Reduce marketing cost":
        recs.append("Consider testing a lower budget on the weaker campaign before cutting it fully.")
    return recs


def recommend_from_churn(risk_reasons: list, priority: str) -> list:
    if not risk_reasons:
        return ["No consistent risk pattern was found yet."]
    top_reason = risk_reasons[0]
    rec = f"Prioritise retention outreach for customers flagged with '{top_reason}'."
    if priority == "Retain customers":
        rec += " Consider a proactive check-in or loyalty offer for this group."
    return [rec]


def recommend_from_channel_performance(best_channel: str, priority: str) -> list:
    recs = [
        f"'{best_channel}' currently shows the strongest overall performance. Consider a modest, gradual "
        "budget increase while monitoring results, rather than moving the entire budget at once."
    ]
    if priority == "Reduce marketing cost":
        recs.append("Test reducing spend on the weakest channel before cutting it entirely.")
    return recs


def recommend_from_customer_value(top_share_pct: float, priority: str) -> list:
    rec = [
        f"The top customers generate {top_share_pct:.0f}% of total value. Prioritise retention for this group, "
        "since losing them would have an outsized impact."
    ]
    if priority == "Retain customers":
        rec.append("Build a loyalty or early-access programme targeted at the highest-value customers.")
    return rec
