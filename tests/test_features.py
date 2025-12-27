"""Tests for feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import FeatureEngineer


@pytest.fixture
def sample_application_data():
    """Create sample application data."""
    np.random.seed(42)
    return pd.DataFrame({
        "SK_ID_CURR": range(100),
        "TARGET": np.random.randint(0, 2, 100),
        "AMT_CREDIT": np.random.uniform(100000, 1000000, 100),
        "AMT_INCOME_TOTAL": np.random.uniform(50000, 500000, 100),
        "AMT_ANNUITY": np.random.uniform(10000, 50000, 100),
        "AMT_GOODS_PRICE": np.random.uniform(80000, 800000, 100),
        "DAYS_BIRTH": np.random.randint(-25000, -7000, 100),
        "DAYS_EMPLOYED": np.random.randint(-5000, 0, 100),
        "CNT_FAM_MEMBERS": np.random.randint(1, 6, 100),
        "EXT_SOURCE_1": np.random.uniform(0, 1, 100),
        "EXT_SOURCE_2": np.random.uniform(0, 1, 100),
        "EXT_SOURCE_3": np.random.uniform(0, 1, 100),
    })


@pytest.fixture
def sample_bureau_data():
    """Create sample bureau data."""
    np.random.seed(42)
    return pd.DataFrame({
        "SK_ID_CURR": np.random.choice(range(100), 500),
        "SK_ID_BUREAU": range(500),
        "CREDIT_ACTIVE": np.random.choice(["Active", "Closed", "Sold"], 500),
        "CREDIT_TYPE": np.random.choice(["Consumer credit", "Credit card", "Car loan"], 500),
        "DAYS_CREDIT": np.random.randint(-3000, 0, 500),
        "CREDIT_DAY_OVERDUE": np.random.randint(0, 100, 500),
        "AMT_CREDIT_SUM": np.random.uniform(10000, 500000, 500),
        "AMT_CREDIT_SUM_DEBT": np.random.uniform(0, 200000, 500),
    })


@pytest.fixture
def engineer():
    """Create feature engineer instance."""
    return FeatureEngineer()


class TestApplicationFeatures:
    """Tests for application feature engineering."""

    def test_credit_income_ratio(self, engineer, sample_application_data):
        """Test credit to income ratio calculation."""
        features = engineer.create_application_features(sample_application_data)

        assert "CREDIT_INCOME_RATIO" in features.columns
        # Check calculation
        expected = sample_application_data["AMT_CREDIT"] / sample_application_data["AMT_INCOME_TOTAL"]
        pd.testing.assert_series_equal(
            features["CREDIT_INCOME_RATIO"],
            expected,
            check_names=False,
        )

    def test_age_calculation(self, engineer, sample_application_data):
        """Test age in years calculation."""
        features = engineer.create_application_features(sample_application_data)

        assert "AGE_YEARS" in features.columns
        # All ages should be positive (DAYS_BIRTH is negative)
        assert (features["AGE_YEARS"] > 0).all()
        # Ages should be reasonable (18-100 years for this dataset)
        assert features["AGE_YEARS"].min() >= 18
        assert features["AGE_YEARS"].max() <= 100

    def test_employment_years(self, engineer, sample_application_data):
        """Test employment years calculation."""
        features = engineer.create_application_features(sample_application_data)

        assert "EMPLOYMENT_YEARS" in features.columns
        # Employment years should be non-negative where valid
        valid_employment = features["EMPLOYMENT_YEARS"].dropna()
        assert (valid_employment >= 0).all()

    def test_external_source_aggregations(self, engineer, sample_application_data):
        """Test external source feature aggregations."""
        features = engineer.create_application_features(sample_application_data)

        assert "EXT_SOURCE_MEAN" in features.columns
        assert "EXT_SOURCE_STD" in features.columns
        assert "EXT_SOURCE_PROD" in features.columns

    def test_loan_term_calculation(self, engineer, sample_application_data):
        """Test loan term calculation."""
        features = engineer.create_application_features(sample_application_data)

        assert "LOAN_TERM" in features.columns
        # Loan term is credit / annuity (in months conceptually)

    def test_preserves_original_columns(self, engineer, sample_application_data):
        """Test that original columns are preserved."""
        features = engineer.create_application_features(sample_application_data)

        assert "SK_ID_CURR" in features.columns
        assert "TARGET" in features.columns


class TestBureauFeatures:
    """Tests for bureau feature engineering."""

    def test_bureau_aggregations(self, engineer, sample_bureau_data):
        """Test bureau data aggregation."""
        bureau_features = engineer.create_bureau_features(sample_bureau_data)

        # Should have one row per customer
        assert bureau_features["SK_ID_CURR"].nunique() == len(bureau_features)

        # Should have aggregated columns
        assert "BUREAU_COUNT" in bureau_features.columns
        assert any("DAYS_CREDIT" in col for col in bureau_features.columns)
        assert any("AMT_CREDIT_SUM" in col for col in bureau_features.columns)

    def test_bureau_credit_active_counts(self, engineer, sample_bureau_data):
        """Test bureau credit status counts."""
        bureau_features = engineer.create_bureau_features(sample_bureau_data)

        # Should have columns for each credit status
        credit_cols = [col for col in bureau_features.columns if "CREDIT_Active" in col or "CREDIT_Closed" in col]
        assert len(credit_cols) > 0


class TestFeatureMerging:
    """Tests for feature merging."""

    def test_merge_features(self, engineer, sample_application_data, sample_bureau_data):
        """Test merging multiple feature sets."""
        app_features = engineer.create_application_features(sample_application_data)
        bureau_features = engineer.create_bureau_features(sample_bureau_data)

        merged = engineer.merge_features(app_features, bureau_features)

        # Should have all application rows
        assert len(merged) == len(app_features)

        # Should have columns from both sources
        assert "CREDIT_INCOME_RATIO" in merged.columns
        assert "BUREAU_COUNT" in merged.columns

    def test_add_timestamp(self, engineer, sample_application_data):
        """Test timestamp addition."""
        features = engineer.create_application_features(sample_application_data)
        features = engineer.add_timestamp(features)

        assert "event_timestamp" in features.columns
        # Check for datetime type (works with both ns and us precision)
        assert "datetime64" in str(features["event_timestamp"].dtype)
