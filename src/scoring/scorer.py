"""Scoring module for credit risk prediction."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CreditRiskScorer:
    """Handles scoring for credit risk prediction."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the scorer.

        Args:
            model_path: Path to the trained model. If None, loads latest model.
        """
        self.config = get_config()
        self.id_column = self.config.get("features.id_column", "SK_ID_CURR")

        self.model = None
        self.feature_columns: List[str] = []
        self.model_metadata: Dict = {}

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """
        Load a trained model.

        Args:
            model_path: Path to the model file.
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Load metadata
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.model_metadata = json.load(f)
            self.feature_columns = self.model_metadata.get("feature_columns", [])
            logger.info(f"Loaded metadata with {len(self.feature_columns)} features")

    def load_latest_model(self, model_dir: Optional[str] = None) -> str:
        """
        Load the most recent model from the model directory.

        Args:
            model_dir: Directory containing models. If None, uses config.

        Returns:
            Path to the loaded model.
        """
        if model_dir is None:
            model_dir = self.config.get_project_root() / self.config.get("paths.models", "models")
        else:
            model_dir = Path(model_dir)

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Find the most recent model file
        model_files = list(model_dir.glob("*.joblib"))
        if not model_files:
            raise FileNotFoundError("No model files found")

        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        self.load_model(str(latest_model))

        return str(latest_model)

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for scoring.

        Args:
            df: DataFrame with features.

        Returns:
            DataFrame ready for prediction.
        """
        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            logger.warning(f"Missing features, filling with 0: {missing_features}")
            for feat in missing_features:
                df[feat] = 0

        # Select and order features
        X = df[self.feature_columns].copy()

        # Handle missing values
        X = X.fillna(X.median())

        # Handle infinities
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        return X

    def score(
        self,
        df: pd.DataFrame,
        include_probability: bool = True,
    ) -> pd.DataFrame:
        """
        Score a batch of applications.

        Args:
            df: DataFrame with features.
            include_probability: Whether to include prediction probability.

        Returns:
            DataFrame with scores.
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")

        logger.info(f"Scoring {len(df)} applications")

        # Prepare features
        X = self.prepare_features(df)

        # Make predictions
        predictions = self.model.predict(X)

        # Create result DataFrame
        result = pd.DataFrame({
            self.id_column: df[self.id_column].values,
            "prediction": predictions,
        })

        if include_probability:
            probabilities = self.model.predict_proba(X)[:, 1]
            result["probability"] = probabilities
            result["risk_score"] = (probabilities * 1000).astype(int)  # 0-1000 scale

        result["scored_at"] = datetime.now()

        logger.info(f"Scoring complete. Default rate: {predictions.mean():.4f}")

        return result

    def score_single(
        self,
        features: Dict[str, float],
        customer_id: Optional[int] = None,
    ) -> Dict[str, Union[int, float, str]]:
        """
        Score a single application.

        Args:
            features: Dictionary of feature values.
            customer_id: Optional customer ID.

        Returns:
            Dictionary with prediction results.
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")

        # Create DataFrame with single row
        df = pd.DataFrame([features])
        if customer_id is not None:
            df[self.id_column] = customer_id
        else:
            df[self.id_column] = 0

        # Prepare features
        X = self.prepare_features(df)

        # Make prediction
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0, 1]

        result = {
            "customer_id": customer_id,
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_score": int(probability * 1000),
            "risk_category": self._get_risk_category(probability),
            "scored_at": datetime.now().isoformat(),
        }

        return result

    def _get_risk_category(self, probability: float) -> str:
        """
        Categorize risk based on probability.

        Args:
            probability: Predicted probability of default.

        Returns:
            Risk category string.
        """
        if probability < 0.1:
            return "LOW"
        elif probability < 0.3:
            return "MEDIUM"
        elif probability < 0.5:
            return "HIGH"
        else:
            return "VERY_HIGH"

    def batch_score_from_file(
        self,
        input_path: str,
        output_path: str,
        chunk_size: int = 10000,
    ) -> str:
        """
        Score applications from a file in batches.

        Args:
            input_path: Path to input file (CSV or Parquet).
            output_path: Path to save results.
            chunk_size: Number of rows to process at a time.

        Returns:
            Path to the output file.
        """
        logger.info(f"Batch scoring from {input_path}")

        input_path = Path(input_path)
        output_path = Path(output_path)

        # Determine file format
        if input_path.suffix == ".parquet":
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)

        # Score in chunks
        results = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            chunk_results = self.score(chunk)
            results.append(chunk_results)
            logger.info(f"Processed {min(i + chunk_size, len(df))}/{len(df)} rows")

        # Combine results
        all_results = pd.concat(results, ignore_index=True)

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".parquet":
            all_results.to_parquet(output_path, index=False)
        else:
            all_results.to_csv(output_path, index=False)

        logger.info(f"Results saved to {output_path}")

        return str(output_path)


class OnlineScorer:
    """Handles real-time scoring with feature store integration."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_store_path: Optional[str] = None,
    ):
        """
        Initialize the online scorer.

        Args:
            model_path: Path to the trained model.
            feature_store_path: Path to the feature store.
        """
        self.scorer = CreditRiskScorer(model_path)
        self.feature_store = None

        if feature_store_path:
            from src.features.store import FeatureStore
            self.feature_store = FeatureStore(feature_store_path)

    def score_with_features(
        self,
        customer_id: int,
        feature_views: Optional[List[str]] = None,
    ) -> Dict[str, Union[int, float, str]]:
        """
        Score using features from the feature store.

        Args:
            customer_id: Customer ID to score.
            feature_views: Feature views to retrieve.

        Returns:
            Scoring result dictionary.
        """
        if self.feature_store is None:
            raise ValueError("Feature store not configured")

        if feature_views is None:
            feature_views = ["credit_features"]

        # Get features from store
        features = self.feature_store.get_online_features(
            entity_ids=[customer_id],
            feature_views=feature_views,
        )

        if customer_id not in features:
            raise ValueError(f"Features not found for customer: {customer_id}")

        customer_features = features[customer_id]

        # Score
        return self.scorer.score_single(customer_features, customer_id)


def score_applications(
    df: pd.DataFrame,
    model_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to score applications.

    Args:
        df: DataFrame with features.
        model_path: Optional path to model. Uses latest if None.

    Returns:
        DataFrame with predictions.
    """
    scorer = CreditRiskScorer()

    if model_path:
        scorer.load_model(model_path)
    else:
        scorer.load_latest_model()

    return scorer.score(df)
