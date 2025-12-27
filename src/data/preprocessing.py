"""Data preprocessing module for cleaning and preparing raw data."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Handles data cleaning and preprocessing operations."""

    def __init__(self):
        self.config = get_config()
        self.id_column = self.config.get("features.id_column", "SK_ID_CURR")
        self.target_column = self.config.get("features.target_column", "TARGET")

    def clean_application_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the main application DataFrame.

        Args:
            df: Raw application DataFrame.

        Returns:
            Cleaned DataFrame.
        """
        logger.info("Starting application data cleaning")
        df = df.copy()

        # Handle anomalous values in DAYS columns (convert positive to negative)
        days_columns = [col for col in df.columns if col.startswith("DAYS_")]
        for col in days_columns:
            if col in df.columns:
                # Replace anomalous positive values (like 365243) with NaN
                df.loc[df[col] > 0, col] = np.nan

        # Handle XNA values in categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            df[col] = df[col].replace("XNA", np.nan)

        # Handle infinity values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        logger.info(f"Cleaned data: {len(df)} rows")
        return df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
        missing_threshold: float = 0.7,
    ) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.

        Args:
            df: DataFrame with missing values.
            numeric_strategy: Strategy for numeric columns ('mean', 'median', 'zero').
            categorical_strategy: Strategy for categorical columns ('mode', 'unknown').
            missing_threshold: Drop columns with more than this fraction of missing values.

        Returns:
            DataFrame with handled missing values.
        """
        logger.info("Handling missing values")
        df = df.copy()

        # Drop columns with too many missing values
        missing_ratio = df.isnull().sum() / len(df)
        cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()

        # Keep essential columns
        essential_cols = [self.id_column, self.target_column]
        cols_to_drop = [col for col in cols_to_drop if col not in essential_cols]

        if cols_to_drop:
            logger.info(f"Dropping {len(cols_to_drop)} columns with >{missing_threshold*100}% missing values")
            df = df.drop(columns=cols_to_drop)

        # Handle numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in essential_cols]

        for col in numeric_cols:
            if df[col].isnull().any():
                if numeric_strategy == "median":
                    fill_value = df[col].median()
                elif numeric_strategy == "mean":
                    fill_value = df[col].mean()
                else:
                    fill_value = 0
                df[col] = df[col].fillna(fill_value)

        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            if df[col].isnull().any():
                if categorical_strategy == "mode":
                    mode_value = df[col].mode()
                    fill_value = mode_value[0] if len(mode_value) > 0 else "Unknown"
                else:
                    fill_value = "Unknown"
                df[col] = df[col].fillna(fill_value)

        logger.info(f"Missing value handling complete. Remaining columns: {len(df.columns)}")
        return df

    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        max_cardinality: int = 10,
    ) -> pd.DataFrame:
        """
        Encode categorical variables.

        Args:
            df: DataFrame with categorical columns.
            columns: List of columns to encode. If None, encodes all object columns.
            max_cardinality: Maximum unique values for one-hot encoding.
                            Columns with more unique values use label encoding.

        Returns:
            DataFrame with encoded categorical variables.
        """
        logger.info("Encoding categorical variables")
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=["object"]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                continue

            n_unique = df[col].nunique()

            if n_unique <= max_cardinality:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
                logger.debug(f"One-hot encoded {col} ({n_unique} categories)")
            else:
                # Label encoding for high cardinality
                df[col] = df[col].astype("category").cat.codes
                logger.debug(f"Label encoded {col} ({n_unique} categories)")

        logger.info(f"Encoding complete. Total columns: {len(df.columns)}")
        return df

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> pd.DataFrame:
        """
        Remove or cap outliers in numeric columns.

        Args:
            df: DataFrame with potential outliers.
            columns: Columns to check for outliers. If None, checks all numeric columns.
            method: Method for outlier detection ('iqr' or 'zscore').
            threshold: Threshold for outlier detection.

        Returns:
            DataFrame with outliers handled.
        """
        logger.info(f"Handling outliers using {method} method")
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude ID and target columns
            columns = [
                col
                for col in columns
                if col not in [self.id_column, self.target_column]
            ]

        for col in columns:
            if col not in df.columns:
                continue

            if method == "iqr":
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
            else:  # zscore
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std

            # Cap outliers instead of removing rows
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        logger.info("Outlier handling complete")
        return df


def preprocess_data(
    df: pd.DataFrame,
    is_training: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to run full preprocessing pipeline.

    Args:
        df: Raw DataFrame.
        is_training: Whether this is training data.

    Returns:
        Preprocessed DataFrame.
    """
    preprocessor = DataPreprocessor()

    df = preprocessor.clean_application_data(df)
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.encode_categorical(df)

    return df
