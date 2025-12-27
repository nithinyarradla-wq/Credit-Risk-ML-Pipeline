"""Tests for data preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import DataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "SK_ID_CURR": range(100),
        "TARGET": np.random.randint(0, 2, 100),
        "AMT_CREDIT": np.random.uniform(100000, 1000000, 100),
        "AMT_INCOME_TOTAL": np.random.uniform(50000, 500000, 100),
        "DAYS_BIRTH": np.random.randint(-25000, -7000, 100),
        "DAYS_EMPLOYED": np.random.randint(-5000, 0, 100),
        "CODE_GENDER": np.random.choice(["M", "F", "XNA"], 100),
        "NAME_INCOME_TYPE": np.random.choice(["Working", "Commercial associate", "Pensioner"], 100),
    })


@pytest.fixture
def preprocessor():
    """Create preprocessor instance."""
    return DataPreprocessor()


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""

    def test_clean_application_data(self, preprocessor, sample_data):
        """Test that cleaning handles XNA values."""
        cleaned = preprocessor.clean_application_data(sample_data)

        # XNA should be replaced with NaN
        assert "XNA" not in cleaned["CODE_GENDER"].dropna().values

    def test_handle_missing_values_numeric(self, preprocessor, sample_data):
        """Test missing value handling for numeric columns."""
        # Introduce missing values
        sample_data.loc[0:10, "AMT_CREDIT"] = np.nan

        handled = preprocessor.handle_missing_values(sample_data)

        # Should have no missing values in numeric columns
        assert handled["AMT_CREDIT"].isnull().sum() == 0

    def test_handle_missing_values_categorical(self, preprocessor, sample_data):
        """Test missing value handling for categorical columns."""
        # Introduce missing values
        sample_data.loc[0:10, "CODE_GENDER"] = np.nan

        handled = preprocessor.handle_missing_values(sample_data)

        # Should have no missing values
        assert handled["CODE_GENDER"].isnull().sum() == 0

    def test_encode_categorical_one_hot(self, preprocessor, sample_data):
        """Test one-hot encoding for low cardinality columns."""
        encoded = preprocessor.encode_categorical(sample_data, columns=["CODE_GENDER"])

        # Original column should be replaced with dummy columns
        assert "CODE_GENDER" not in encoded.columns
        assert any(col.startswith("CODE_GENDER_") for col in encoded.columns)

    def test_remove_outliers(self, preprocessor, sample_data):
        """Test outlier handling."""
        # Add outliers
        sample_data.loc[0, "AMT_CREDIT"] = 1e12

        handled = preprocessor.remove_outliers(sample_data, columns=["AMT_CREDIT"])

        # Outlier should be capped
        assert handled["AMT_CREDIT"].max() < 1e12

    def test_preserves_id_and_target(self, preprocessor, sample_data):
        """Test that ID and target columns are preserved."""
        handled = preprocessor.handle_missing_values(sample_data)

        assert "SK_ID_CURR" in handled.columns
        assert "TARGET" in handled.columns


class TestMissingValueThreshold:
    """Tests for missing value threshold handling."""

    def test_drops_high_missing_columns(self):
        """Test that columns with high missing rate are dropped."""
        preprocessor = DataPreprocessor()

        data = pd.DataFrame({
            "SK_ID_CURR": range(100),
            "TARGET": [0] * 100,
            "good_col": np.random.uniform(0, 1, 100),
            "bad_col": [np.nan] * 80 + list(range(20)),  # 80% missing
        })

        handled = preprocessor.handle_missing_values(data, missing_threshold=0.7)

        # bad_col should be dropped
        assert "bad_col" not in handled.columns
        assert "good_col" in handled.columns
