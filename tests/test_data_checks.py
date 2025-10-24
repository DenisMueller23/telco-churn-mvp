"""Basic tests for data validation."""
import pandas as pd
import pytest
from pathlib import Path


def test_data_exists():
    """Check if raw data file exists."""
    data_path = Path("data/raw/telco.csv")
    assert data_path.exists(), "telco.csv not found in data/raw/"


def test_data_shape():
    """Check basic data structure."""
    df = pd.read_csv("data/raw/telco.csv")
    assert len(df) > 1000, "Dataset too small"
    assert 'Churn' in df.columns, "Churn column missing"
    assert 'customerID' in df.columns, "customerID column missing"


def test_churn_values():
    """Check Churn column has expected values."""
    df = pd.read_csv("data/raw/telco.csv")
    unique_vals = set(df['Churn'].unique())
    assert unique_vals.issubset({'Yes', 'No'}), "Unexpected Churn values"