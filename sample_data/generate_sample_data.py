"""Generates sample_data/marketing_sample.xlsx: 500+ fictional customers with
realistic relationships plus intentional data-quality issues (missing values,
duplicates, outliers, inconsistent labels) so MarketLens's data-quality
features can be demonstrated.

Run with: python sample_data/generate_sample_data.py
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 620  # a bit above 500 so removing duplicates/outliers still leaves 500+


def generate():
    customer_id = [f"CUST{1000 + i}" for i in range(N)]
    age = RNG.integers(18, 75, N)
    gender = RNG.choice(["Male", "Female"], N, p=[0.49, 0.51])
    city = RNG.choice(
        ["Mumbai", "Delhi", "Bangalore", "Chennai", "Pune", "Hyderabad", "Kolkata"], N,
    )
    income = RNG.normal(65000, 22000, N).clip(15000, None).round(0)

    channel = RNG.choice(["Email", "Social Media", "Search", "Referral", "Direct"], N,
                          p=[0.25, 0.3, 0.2, 0.1, 0.15])
    campaign = RNG.choice(["Campaign A", "Campaign B", "Campaign C"], N, p=[0.4, 0.35, 0.25])

    website_visits = RNG.poisson(4, N) + 1
    days_since_last_purchase = RNG.exponential(45, N).round(0).astype(int)
    purchase_frequency = RNG.poisson(3, N)
    discount_pct = RNG.choice([0, 5, 10, 15, 20, 25], N, p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05])
    complaints = RNG.poisson(0.3, N)
    delivery_delay = RNG.exponential(1.5, N).round(0).astype(int)

    # Purchase likelihood depends on visits, frequency, discount and recency (business-realistic signal)
    purchase_score = (
        0.35 * (website_visits / website_visits.max())
        + 0.30 * (purchase_frequency / max(purchase_frequency.max(), 1))
        + 0.15 * (discount_pct / 25)
        - 0.25 * (days_since_last_purchase / days_since_last_purchase.max())
        - 0.10 * (complaints / max(complaints.max(), 1))
        + RNG.normal(0, 0.12, N)
    )
    purchase_prob = 1 / (1 + np.exp(-8 * (purchase_score - purchase_score.mean())))
    purchased = RNG.random(N) < purchase_prob

    avg_order_value = RNG.normal(2500, 800, N).clip(300, None).round(0)
    base_revenue = np.where(purchased, avg_order_value * (1 + purchase_frequency / 4), 0)
    revenue = (base_revenue * (1 - discount_pct / 100)).round(0)

    marketing_cost = (RNG.normal(350, 120, N).clip(20, None) + discount_pct * 5).round(0)
    profit = (revenue * 0.35 - marketing_cost).round(0)

    start_date = pd.Timestamp("2024-01-01")
    purchase_date = start_date + pd.to_timedelta(RNG.integers(0, 545, N), unit="D")

    df = pd.DataFrame({
        "Customer_ID": customer_id,
        "Age": age,
        "Gender": gender,
        "City": city,
        "Income": income,
        "Marketing_Channel": channel,
        "Campaign": campaign,
        "Website_Visits": website_visits,
        "Days_Since_Last_Purchase": days_since_last_purchase,
        "Purchase_Frequency": purchase_frequency,
        "Average_Order_Value": avg_order_value,
        "Discount_Percentage": discount_pct,
        "Customer_Service_Complaints": complaints,
        "Delivery_Delay_Days": delivery_delay,
        "Purchased": np.where(purchased, "Yes", "No"),
        "Revenue": revenue,
        "Marketing_Cost": marketing_cost,
        "Profit": profit,
        "Purchase_Date": purchase_date,
    })

    df = _introduce_messiness(df)
    return df


def _introduce_messiness(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    # Missing values in a few columns
    for col, frac in [("Income", 0.04), ("Discount_Percentage", 0.03), ("Purchased", 0.02), ("City", 0.015)]:
        idx = RNG.choice(n, size=int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    # Duplicate rows
    dup_idx = RNG.choice(n, size=15, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # Outliers in Revenue and Income
    outlier_idx = RNG.choice(df.index, size=8, replace=False)
    df.loc[outlier_idx, "Revenue"] = df.loc[outlier_idx, "Revenue"] * RNG.uniform(6, 10, size=8)
    outlier_idx2 = RNG.choice(df.index, size=5, replace=False)
    df.loc[outlier_idx2, "Income"] = df.loc[outlier_idx2, "Income"] * RNG.uniform(4, 6, size=5)

    # Inconsistent category labels
    purchased_variants = {"Yes": ["Yes", "Y", "yes", "Purchased"], "No": ["No", "N", "no"]}
    def _vary_purchased(value):
        if pd.isna(value):
            return value
        options = purchased_variants.get(value, [value])
        return RNG.choice(options)
    variant_idx = RNG.choice(df.index, size=int(len(df) * 0.15), replace=False)
    df.loc[variant_idx, "Purchased"] = df.loc[variant_idx, "Purchased"].apply(_vary_purchased)

    gender_variants = {"Male": ["Male", "M"], "Female": ["Female", "F"]}
    variant_idx2 = RNG.choice(df.index, size=int(len(df) * 0.1), replace=False)
    df.loc[variant_idx2, "Gender"] = df.loc[variant_idx2, "Gender"].apply(
        lambda v: RNG.choice(gender_variants.get(v, [v])) if pd.notna(v) else v
    )

    return df.sample(frac=1, random_state=1).reset_index(drop=True)


def main():
    df = generate()
    notes = pd.DataFrame({
        "About this file": [
            "Fictional marketing dataset generated for MarketLens demonstration purposes.",
            f"{len(df)} rows across customers, with intentional missing values, duplicates, "
            "outliers and inconsistent labels to demonstrate the Data Quality page.",
            "No real customers are represented in this data.",
        ]
    })

    with pd.ExcelWriter("sample_data/marketing_sample.xlsx", engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Customer_Data", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)

    print(f"Generated sample_data/marketing_sample.xlsx with {len(df)} rows.")


if __name__ == "__main__":
    main()
