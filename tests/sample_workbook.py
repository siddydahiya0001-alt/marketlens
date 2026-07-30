"""Access to the shipped sample workbook for tests.

Tests that use this run against the exact file a first-time user uploads -
messy labels, missing values, duplicates and all - which is where bugs that
clean synthetic fixtures cannot reach tend to hide.
"""

import os

import pandas as pd
import pytest

SAMPLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sample_data", "marketing_sample.xlsx",
)


def load_sample_customer_data() -> pd.DataFrame:
    """Load the Customer_Data sheet exactly as uploaded, with no cleaning applied."""
    if not os.path.exists(SAMPLE_PATH):
        pytest.skip("sample_data/marketing_sample.xlsx is not present")
    return pd.read_excel(SAMPLE_PATH, sheet_name="Customer_Data")
