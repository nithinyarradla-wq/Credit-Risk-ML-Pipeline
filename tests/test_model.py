"""Tests for model training module."""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from src.models.trainer import ModelTrainer


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 1000

    # Create features correlated with target for realistic testing
    target = np.random.randint(0, 2, n_samples)

    return pd.DataFrame({
        "SK_ID_CURR": range(n_samples),
        "TARGET": target,
        "feature_1": np.random.randn(n_samples) + target * 0.5,
        "feature_2": np.random.randn(n_samples) - target * 0.3,
        "feature_3": np.random.uniform(0, 1, n_samples),
        "feature_4": np.random.randn(n_samples) + target * 0.2,
        "feature_5": np.random.uniform(-1, 1, n_samples),
    })


@pytest.fixture
def trainer():
    """Create trainer instance."""
    return ModelTrainer(model_type="random_forest")


class TestModelTrainer:
    """Tests for ModelTrainer class."""

    def test_prepare_data(self, trainer, sample_training_data):
        """Test data preparation."""
        X_train, X_test, y_train, y_test = trainer.prepare_data(sample_training_data)

        # Check shapes
        assert len(X_train) + len(X_test) == len(sample_training_data)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

        # Check that ID and target are not in features
        assert "SK_ID_CURR" not in X_train.columns
        assert "TARGET" not in X_train.columns

    def test_train_random_forest(self, trainer, sample_training_data):
        """Test random forest training."""
        X_train, X_test, y_train, y_test = trainer.prepare_data(sample_training_data)
        trainer.train(X_train, y_train)

        assert trainer.model is not None
        assert trainer.training_date is not None

    def test_evaluate(self, trainer, sample_training_data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test = trainer.prepare_data(sample_training_data)
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)

        # Check all metrics are present
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics

        # Check metrics are in valid range
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"{metric_name} should be between 0 and 1"

    def test_cross_validate(self, trainer, sample_training_data):
        """Test cross-validation."""
        X_train, X_test, y_train, y_test = trainer.prepare_data(sample_training_data)
        trainer.train(X_train, y_train)

        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test])

        cv_results = trainer.cross_validate(X_full, y_full, n_folds=3)

        assert "cv_mean_roc_auc" in cv_results
        assert "cv_std_roc_auc" in cv_results
        assert "cv_scores" in cv_results
        assert len(cv_results["cv_scores"]) == 3

    def test_feature_importance(self, trainer, sample_training_data):
        """Test feature importance extraction."""
        X_train, X_test, y_train, y_test = trainer.prepare_data(sample_training_data)
        trainer.train(X_train, y_train)

        importance_df = trainer.get_feature_importance()

        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert len(importance_df) == len(trainer.feature_columns)
        assert importance_df["importance"].sum() > 0

    def test_save_and_load_model(self, trainer, sample_training_data):
        """Test model saving and loading."""
        X_train, X_test, y_train, y_test = trainer.prepare_data(sample_training_data)
        trainer.train(X_train, y_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = trainer.save_model(model_dir=tmpdir, model_name="test_model")

            # Create new trainer and load
            new_trainer = ModelTrainer()
            new_trainer.load_model(model_path)

            assert new_trainer.model is not None
            assert new_trainer.feature_columns == trainer.feature_columns

            # Predictions should match
            original_preds = trainer.model.predict(X_test)
            loaded_preds = new_trainer.model.predict(X_test)
            np.testing.assert_array_equal(original_preds, loaded_preds)


class TestDifferentModels:
    """Tests for different model types."""

    @pytest.mark.parametrize("model_type", ["random_forest", "gradient_boosting", "logistic_regression"])
    def test_train_different_models(self, model_type, sample_training_data):
        """Test training with different model types."""
        trainer = ModelTrainer(model_type=model_type)
        X_train, X_test, y_train, y_test = trainer.prepare_data(sample_training_data)
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)

        assert trainer.model is not None
        assert metrics["roc_auc"] >= 0.5  # Should be better than random


class TestEdgeCases:
    """Tests for edge cases."""

    def test_missing_values_in_features(self, trainer):
        """Test handling of missing values."""
        data = pd.DataFrame({
            "SK_ID_CURR": range(100),
            "TARGET": np.random.randint(0, 2, 100),
            "feature_1": [np.nan] * 10 + list(np.random.randn(90)),
            "feature_2": np.random.randn(100),
        })

        X_train, X_test, y_train, y_test = trainer.prepare_data(data)

        # Should handle missing values
        assert not X_train.isnull().any().any()

    def test_imbalanced_data(self, trainer):
        """Test handling of imbalanced data."""
        np.random.seed(42)
        n_samples = 1000

        # 90% class 0, 10% class 1
        target = np.array([0] * 900 + [1] * 100)

        # Add some signal to features so model can learn
        data = pd.DataFrame({
            "SK_ID_CURR": range(n_samples),
            "TARGET": target,
            "feature_1": np.random.randn(n_samples) + target * 0.5,
            "feature_2": np.random.randn(n_samples) - target * 0.3,
        })

        X_train, X_test, y_train, y_test = trainer.prepare_data(data)
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)

        # Model should produce valid metrics on imbalanced data
        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["accuracy"] <= 1
