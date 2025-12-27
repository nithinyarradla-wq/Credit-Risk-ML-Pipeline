"""Feature selection module for identifying important features."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """Handles feature selection for model training."""

    def __init__(self):
        self.config = get_config()
        self.id_column = self.config.get("features.id_column", "SK_ID_CURR")
        self.target_column = self.config.get("features.target_column", "TARGET")
        self.selected_features: List[str] = []
        self.feature_importances: Optional[pd.DataFrame] = None

    def remove_low_variance_features(
        self, df: pd.DataFrame, threshold: float = 0.01
    ) -> pd.DataFrame:
        """
        Remove features with very low variance.

        Args:
            df: Feature DataFrame.
            threshold: Minimum variance threshold.

        Returns:
            DataFrame with low variance features removed.
        """
        logger.info(f"Removing features with variance < {threshold}")
        df = df.copy()

        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = [self.id_column, self.target_column, "event_timestamp"]
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        # Calculate variance
        variances = df[feature_cols].var()
        low_var_cols = variances[variances < threshold].index.tolist()

        if low_var_cols:
            logger.info(f"Removing {len(low_var_cols)} low variance features")
            df = df.drop(columns=low_var_cols)

        return df

    def remove_correlated_features(
        self, df: pd.DataFrame, threshold: float = 0.95
    ) -> pd.DataFrame:
        """
        Remove highly correlated features to reduce redundancy.

        Args:
            df: Feature DataFrame.
            threshold: Correlation threshold above which features are removed.

        Returns:
            DataFrame with highly correlated features removed.
        """
        logger.info(f"Removing features with correlation > {threshold}")
        df = df.copy()

        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = [self.id_column, self.target_column, "event_timestamp"]
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        if len(feature_cols) == 0:
            return df

        # Calculate correlation matrix
        corr_matrix = df[feature_cols].corr().abs()

        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation above threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        if to_drop:
            logger.info(f"Removing {len(to_drop)} highly correlated features")
            df = df.drop(columns=to_drop)

        return df

    def select_by_importance(
        self,
        df: pd.DataFrame,
        n_features: int = 50,
        importance_threshold: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select features based on Random Forest feature importance.

        Args:
            df: Feature DataFrame with target column.
            n_features: Number of top features to select.
            importance_threshold: Alternative to n_features; select features above threshold.

        Returns:
            Tuple of (selected features DataFrame, feature importances DataFrame).
        """
        logger.info("Calculating feature importances using Random Forest")
        df = df.copy()

        # Prepare data
        exclude_cols = [self.id_column, self.target_column, "event_timestamp"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Keep only numeric features for importance calculation
        X = df[feature_cols].select_dtypes(include=[np.number])
        feature_cols = X.columns.tolist()

        if self.target_column not in df.columns:
            logger.warning("Target column not found. Returning original DataFrame.")
            return df, pd.DataFrame()

        y = df[self.target_column]

        # Handle missing values for training
        X = X.fillna(X.median())

        # Train a Random Forest for feature importance
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        rf.fit(X, y)

        # Get feature importances
        importances = pd.DataFrame(
            {"feature": feature_cols, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)

        self.feature_importances = importances

        # Select features
        if importance_threshold is not None:
            selected = importances[importances["importance"] >= importance_threshold][
                "feature"
            ].tolist()
        else:
            selected = importances.head(n_features)["feature"].tolist()

        self.selected_features = selected
        logger.info(f"Selected {len(selected)} features")

        # Keep essential columns
        keep_cols = [self.id_column] + selected
        if self.target_column in df.columns:
            keep_cols.append(self.target_column)
        if "event_timestamp" in df.columns:
            keep_cols.append("event_timestamp")

        return df[keep_cols], importances

    def select_by_mutual_information(
        self, df: pd.DataFrame, n_features: int = 50
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select features based on mutual information with target.

        Args:
            df: Feature DataFrame with target column.
            n_features: Number of top features to select.

        Returns:
            Tuple of (selected features DataFrame, mutual information scores).
        """
        logger.info("Calculating mutual information scores")
        df = df.copy()

        # Prepare data
        exclude_cols = [self.id_column, self.target_column, "event_timestamp"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].select_dtypes(include=[np.number])
        feature_cols = X.columns.tolist()

        if self.target_column not in df.columns:
            logger.warning("Target column not found. Returning original DataFrame.")
            return df, pd.DataFrame()

        y = df[self.target_column]

        # Handle missing values
        X = X.fillna(X.median())

        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)

        # Create scores DataFrame
        mi_df = pd.DataFrame(
            {"feature": feature_cols, "mutual_info": mi_scores}
        ).sort_values("mutual_info", ascending=False)

        # Select top features
        selected = mi_df.head(n_features)["feature"].tolist()
        self.selected_features = selected

        logger.info(f"Selected {len(selected)} features by mutual information")

        # Keep essential columns
        keep_cols = [self.id_column] + selected
        if self.target_column in df.columns:
            keep_cols.append(self.target_column)
        if "event_timestamp" in df.columns:
            keep_cols.append("event_timestamp")

        return df[keep_cols], mi_df

    def get_feature_list(self) -> List[str]:
        """Get the list of selected features."""
        return self.selected_features.copy()


def select_features(
    df: pd.DataFrame,
    n_features: int = 50,
    method: str = "importance",
    remove_low_variance: bool = True,
    remove_correlated: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to run feature selection pipeline.

    Args:
        df: Feature DataFrame.
        n_features: Number of features to select.
        method: Selection method ('importance' or 'mutual_info').
        remove_low_variance: Whether to remove low variance features first.
        remove_correlated: Whether to remove highly correlated features.

    Returns:
        Tuple of (selected features DataFrame, feature scores DataFrame).
    """
    selector = FeatureSelector()

    if remove_low_variance:
        df = selector.remove_low_variance_features(df)

    if remove_correlated:
        df = selector.remove_correlated_features(df)

    if method == "importance":
        return selector.select_by_importance(df, n_features)
    else:
        return selector.select_by_mutual_information(df, n_features)
