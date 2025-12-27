"""Model training module for credit risk prediction."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Handles model training and evaluation for credit risk prediction."""

    SUPPORTED_MODELS = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "logistic_regression": LogisticRegression,
    }

    def __init__(self, model_type: Optional[str] = None):
        """
        Initialize the model trainer.

        Args:
            model_type: Type of model to train. If None, uses config default.
        """
        self.config = get_config()
        self.model_type = model_type or self.config.get("model.algorithm", "random_forest")
        self.id_column = self.config.get("features.id_column", "SK_ID_CURR")
        self.target_column = self.config.get("features.target_column", "TARGET")

        self.model = None
        self.feature_columns: List[str] = []
        self.metrics: Dict[str, float] = {}
        self.training_date: Optional[datetime] = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training.

        Args:
            df: Feature DataFrame with target column.
            test_size: Fraction of data for testing.
            stratify: Whether to stratify the split.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        logger.info("Preparing data for training")

        # Identify feature columns
        exclude_cols = [self.id_column, self.target_column, "event_timestamp"]
        self.feature_columns = [
            col for col in df.columns
            if col not in exclude_cols
            and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]

        logger.info(f"Using {len(self.feature_columns)} features")

        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()

        # Handle missing values
        X = X.fillna(X.median())

        # Replace infinities
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        # Split data
        random_state = self.config.get("model.random_state", 42)
        stratify_y = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y,
        )

        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Target distribution - Train: {y_train.mean():.4f}, Test: {y_test.mean():.4f}")

        return X_train, X_test, y_train, y_test

    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters from configuration."""
        params = self.config.get(f"model.{self.model_type}", {})

        if not params:
            # Default parameters
            if self.model_type == "random_forest":
                params = {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "class_weight": "balanced",
                    "n_jobs": -1,
                    "random_state": 42,
                }
            elif self.model_type == "gradient_boosting":
                params = {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "learning_rate": 0.1,
                    "random_state": 42,
                }
            elif self.model_type == "logistic_regression":
                params = {
                    "max_iter": 1000,
                    "class_weight": "balanced",
                    "random_state": 42,
                }

        return params

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Train the model.

        Args:
            X_train: Training features.
            y_train: Training target.
            params: Optional model parameters. If None, uses config.
        """
        logger.info(f"Training {self.model_type} model")

        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        model_class = self.SUPPORTED_MODELS[self.model_type]
        model_params = params or self.get_model_params()

        self.model = model_class(**model_params)
        self.model.fit(X_train, y_train)
        self.training_date = datetime.now()

        logger.info("Model training complete")

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """
        Evaluate the trained model.

        Args:
            X_test: Test features.
            y_test: Test target.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model has not been trained")

        logger.info("Evaluating model")

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        # Log metrics
        logger.info("Model Evaluation Results:")
        for metric, value in self.metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # Log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")

        return self.metrics

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
    ) -> Dict[str, float]:
        """
        Perform cross-validation.

        Args:
            X: Feature data.
            y: Target data.
            n_folds: Number of folds.

        Returns:
            Dictionary with CV scores.
        """
        logger.info(f"Running {n_folds}-fold cross-validation")

        if self.model is None:
            model_class = self.SUPPORTED_MODELS[self.model_type]
            model_params = self.get_model_params()
            model = model_class(**model_params)
        else:
            model = self.model

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

        cv_results = {
            "cv_mean_roc_auc": scores.mean(),
            "cv_std_roc_auc": scores.std(),
            "cv_scores": scores.tolist(),
        }

        logger.info(f"CV ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

        return cv_results

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.

        Returns:
            DataFrame with feature importances.
        """
        if self.model is None:
            raise ValueError("Model has not been trained")

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            raise ValueError("Model does not support feature importance")

        importance_df = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        return importance_df

    def save_model(
        self,
        model_dir: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> str:
        """
        Save the trained model and metadata.

        Args:
            model_dir: Directory to save the model. If None, uses config.
            model_name: Name for the model file. If None, auto-generated.

        Returns:
            Path to the saved model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained")

        config = get_config()
        if model_dir is None:
            model_dir = config.get_project_root() / config.get("paths.models", "models")
        else:
            model_dir = Path(model_dir)

        model_dir.mkdir(parents=True, exist_ok=True)

        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"credit_risk_{self.model_type}_{timestamp}"

        # Save model
        model_path = model_dir / f"{model_name}.joblib"
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Save metadata
        metadata = {
            "model_type": self.model_type,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
            "training_date": self.training_date.isoformat() if self.training_date else None,
            "model_params": self.get_model_params(),
        }

        metadata_path = model_dir / f"{model_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")

        # Save feature importance
        try:
            importance_df = self.get_feature_importance()
            importance_path = model_dir / f"{model_name}_feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to {importance_path}")
        except ValueError:
            logger.warning("Could not save feature importance")

        return str(model_path)

    def load_model(self, model_path: str) -> None:
        """
        Load a trained model and metadata.

        Args:
            model_path: Path to the model file.
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Load metadata if available
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.model_type = metadata.get("model_type", self.model_type)
            self.feature_columns = metadata.get("feature_columns", [])
            self.metrics = metadata.get("metrics", {})
            logger.info("Model metadata loaded")


def train_model(
    df: pd.DataFrame,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    save_model: bool = True,
) -> Tuple[ModelTrainer, Dict[str, float]]:
    """
    Convenience function to train a credit risk model.

    Args:
        df: Feature DataFrame with target.
        model_type: Type of model to train.
        test_size: Fraction for test set.
        save_model: Whether to save the trained model.

    Returns:
        Tuple of (trainer instance, metrics dictionary).
    """
    trainer = ModelTrainer(model_type)

    X_train, X_test, y_train, y_test = trainer.prepare_data(df, test_size)
    trainer.train(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)

    if save_model:
        trainer.save_model()

    return trainer, metrics
