"""Feature engineering module for creating derived features."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Creates derived features from raw application data."""

    def __init__(self):
        self.config = get_config()
        self.id_column = self.config.get("features.id_column", "SK_ID_CURR")
        self.target_column = self.config.get("features.target_column", "TARGET")

    def create_application_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from the main application data.

        Args:
            df: Application DataFrame.

        Returns:
            DataFrame with new features added.
        """
        logger.info("Creating application features")
        df = df.copy()

        # Credit to income ratio
        if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
            df["CREDIT_INCOME_RATIO"] = df["CREDIT_INCOME_RATIO"].replace(
                [np.inf, -np.inf], np.nan
            )

        # Annuity to income ratio
        if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
            df["ANNUITY_INCOME_RATIO"] = df["ANNUITY_INCOME_RATIO"].replace(
                [np.inf, -np.inf], np.nan
            )

        # Loan term (credit amount / annuity)
        if "AMT_CREDIT" in df.columns and "AMT_ANNUITY" in df.columns:
            df["LOAN_TERM"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
            df["LOAN_TERM"] = df["LOAN_TERM"].replace([np.inf, -np.inf], np.nan)

        # Goods price to credit ratio
        if "AMT_GOODS_PRICE" in df.columns and "AMT_CREDIT" in df.columns:
            df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
            df["GOODS_CREDIT_RATIO"] = df["GOODS_CREDIT_RATIO"].replace(
                [np.inf, -np.inf], np.nan
            )

        # Age in years
        if "DAYS_BIRTH" in df.columns:
            df["AGE_YEARS"] = df["DAYS_BIRTH"] / -365
            df["AGE_CATEGORY"] = pd.cut(
                df["AGE_YEARS"],
                bins=[0, 25, 35, 45, 55, 65, 100],
                labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
            )

        # Employment years
        if "DAYS_EMPLOYED" in df.columns:
            df["EMPLOYMENT_YEARS"] = df["DAYS_EMPLOYED"] / -365
            # Handle anomalous values (e.g., 365243 days = pensioners/unemployed)
            df.loc[df["EMPLOYMENT_YEARS"] < 0, "EMPLOYMENT_YEARS"] = np.nan

        # Employment to age ratio
        if "EMPLOYMENT_YEARS" in df.columns and "AGE_YEARS" in df.columns:
            df["EMPLOYMENT_AGE_RATIO"] = df["EMPLOYMENT_YEARS"] / df["AGE_YEARS"]
            df["EMPLOYMENT_AGE_RATIO"] = df["EMPLOYMENT_AGE_RATIO"].replace(
                [np.inf, -np.inf], np.nan
            )

        # Income per family member
        if "AMT_INCOME_TOTAL" in df.columns and "CNT_FAM_MEMBERS" in df.columns:
            df["INCOME_PER_FAMILY_MEMBER"] = (
                df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
            )
            df["INCOME_PER_FAMILY_MEMBER"] = df["INCOME_PER_FAMILY_MEMBER"].replace(
                [np.inf, -np.inf], np.nan
            )

        # External source average
        ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        existing_ext = [col for col in ext_cols if col in df.columns]
        if existing_ext:
            df["EXT_SOURCE_MEAN"] = df[existing_ext].mean(axis=1)
            df["EXT_SOURCE_STD"] = df[existing_ext].std(axis=1)
            df["EXT_SOURCE_PROD"] = df[existing_ext].prod(axis=1)

        # Document flags aggregation
        doc_cols = [col for col in df.columns if col.startswith("FLAG_DOCUMENT_")]
        if doc_cols:
            df["DOCUMENTS_PROVIDED_COUNT"] = df[doc_cols].sum(axis=1)

        # Contact flags aggregation (only numeric FLAG columns)
        contact_cols = [
            col
            for col in df.columns
            if col.startswith("FLAG_") and "DOCUMENT" not in col
            and df[col].dtype in [np.int64, np.int32, np.float64, np.float32]
        ]
        if contact_cols:
            df["FLAGS_COUNT"] = df[contact_cols].sum(axis=1)

        logger.info(f"Created application features. Total columns: {len(df.columns)}")
        return df

    def create_bureau_features(
        self, bureau_df: pd.DataFrame, bureau_balance_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create aggregated features from bureau credit history.

        Args:
            bureau_df: Bureau DataFrame.
            bureau_balance_df: Optional bureau balance DataFrame.

        Returns:
            DataFrame with aggregated bureau features per customer.
        """
        logger.info("Creating bureau features")
        bureau = bureau_df.copy()

        # Aggregations for bureau data
        agg_funcs = {
            "DAYS_CREDIT": ["min", "max", "mean"],
            "CREDIT_DAY_OVERDUE": ["max", "mean"],
            "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
            "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"],
            "CNT_CREDIT_PROLONG": ["sum"],
            "AMT_CREDIT_SUM": ["sum", "mean", "max"],
            "AMT_CREDIT_SUM_DEBT": ["sum", "mean", "max"],
            "AMT_CREDIT_SUM_OVERDUE": ["sum", "mean"],
            "AMT_ANNUITY": ["sum", "mean"],
        }

        # Filter to existing columns
        agg_funcs = {k: v for k, v in agg_funcs.items() if k in bureau.columns}

        bureau_agg = bureau.groupby("SK_ID_CURR").agg(agg_funcs)
        bureau_agg.columns = ["BUREAU_" + "_".join([str(c) for c in col]).upper() for col in bureau_agg.columns]
        bureau_agg = bureau_agg.reset_index()

        # Count of bureau records
        bureau_count = bureau.groupby("SK_ID_CURR").size().reset_index(name="BUREAU_COUNT")
        bureau_agg = bureau_agg.merge(bureau_count, on="SK_ID_CURR", how="left")

        # Count by credit type
        if "CREDIT_TYPE" in bureau.columns:
            credit_type_counts = (
                bureau.groupby(["SK_ID_CURR", "CREDIT_TYPE"])
                .size()
                .unstack(fill_value=0)
            )
            credit_type_counts.columns = [
                f"BUREAU_CREDIT_TYPE_{col}" for col in credit_type_counts.columns
            ]
            credit_type_counts = credit_type_counts.reset_index()
            bureau_agg = bureau_agg.merge(credit_type_counts, on="SK_ID_CURR", how="left")

        # Active vs closed credits
        if "CREDIT_ACTIVE" in bureau.columns:
            active_counts = (
                bureau.groupby(["SK_ID_CURR", "CREDIT_ACTIVE"])
                .size()
                .unstack(fill_value=0)
            )
            active_counts.columns = [
                f"BUREAU_CREDIT_{col}" for col in active_counts.columns
            ]
            active_counts = active_counts.reset_index()
            bureau_agg = bureau_agg.merge(active_counts, on="SK_ID_CURR", how="left")

        logger.info(f"Created bureau features: {len(bureau_agg.columns)} columns")
        return bureau_agg

    def create_previous_application_features(
        self, prev_app_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create aggregated features from previous applications.

        Args:
            prev_app_df: Previous applications DataFrame.

        Returns:
            DataFrame with aggregated previous application features.
        """
        logger.info("Creating previous application features")
        prev = prev_app_df.copy()

        agg_funcs = {
            "AMT_ANNUITY": ["min", "max", "mean"],
            "AMT_APPLICATION": ["min", "max", "mean", "sum"],
            "AMT_CREDIT": ["min", "max", "mean", "sum"],
            "AMT_DOWN_PAYMENT": ["min", "max", "mean"],
            "AMT_GOODS_PRICE": ["min", "max", "mean"],
            "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
            "RATE_DOWN_PAYMENT": ["min", "max", "mean"],
            "DAYS_DECISION": ["min", "max", "mean"],
            "CNT_PAYMENT": ["mean", "sum"],
        }

        # Filter to existing columns
        agg_funcs = {k: v for k, v in agg_funcs.items() if k in prev.columns}

        prev_agg = prev.groupby("SK_ID_CURR").agg(agg_funcs)
        prev_agg.columns = ["PREV_" + "_".join([str(c) for c in col]).upper() for col in prev_agg.columns]
        prev_agg = prev_agg.reset_index()

        # Count of previous applications
        prev_count = prev.groupby("SK_ID_CURR").size().reset_index(name="PREV_APP_COUNT")
        prev_agg = prev_agg.merge(prev_count, on="SK_ID_CURR", how="left")

        # Application status counts
        if "NAME_CONTRACT_STATUS" in prev.columns:
            status_counts = (
                prev.groupby(["SK_ID_CURR", "NAME_CONTRACT_STATUS"])
                .size()
                .unstack(fill_value=0)
            )
            status_counts.columns = [
                f"PREV_STATUS_{col}" for col in status_counts.columns
            ]
            status_counts = status_counts.reset_index()
            prev_agg = prev_agg.merge(status_counts, on="SK_ID_CURR", how="left")

        logger.info(f"Created previous application features: {len(prev_agg.columns)} columns")
        return prev_agg

    def create_payment_features(self, installments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregated features from installment payments.

        Args:
            installments_df: Installments payments DataFrame.

        Returns:
            DataFrame with aggregated payment features.
        """
        logger.info("Creating payment features")
        inst = installments_df.copy()

        # Calculate payment difference (late/early payment)
        if "DAYS_INSTALMENT" in inst.columns and "DAYS_ENTRY_PAYMENT" in inst.columns:
            inst["PAYMENT_DIFF"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
            inst["PAYMENT_LATE"] = (inst["PAYMENT_DIFF"] > 0).astype(int)

        # Calculate payment amount difference
        if "AMT_INSTALMENT" in inst.columns and "AMT_PAYMENT" in inst.columns:
            inst["AMT_PAYMENT_DIFF"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]
            inst["PAYMENT_RATIO"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"]
            inst["PAYMENT_RATIO"] = inst["PAYMENT_RATIO"].replace([np.inf, -np.inf], np.nan)

        agg_funcs = {}
        if "PAYMENT_DIFF" in inst.columns:
            agg_funcs["PAYMENT_DIFF"] = ["mean", "max", "sum"]
        if "PAYMENT_LATE" in inst.columns:
            agg_funcs["PAYMENT_LATE"] = ["sum", "mean"]
        if "AMT_PAYMENT_DIFF" in inst.columns:
            agg_funcs["AMT_PAYMENT_DIFF"] = ["mean", "max", "sum"]
        if "PAYMENT_RATIO" in inst.columns:
            agg_funcs["PAYMENT_RATIO"] = ["mean", "min"]
        if "AMT_PAYMENT" in inst.columns:
            agg_funcs["AMT_PAYMENT"] = ["sum", "mean"]

        if not agg_funcs:
            logger.warning("No columns available for payment aggregation")
            return pd.DataFrame(columns=["SK_ID_CURR"])

        inst_agg = inst.groupby("SK_ID_CURR").agg(agg_funcs)
        inst_agg.columns = ["INST_" + "_".join([str(c) for c in col]).upper() for col in inst_agg.columns]
        inst_agg = inst_agg.reset_index()

        # Count of installments
        inst_count = inst.groupby("SK_ID_CURR").size().reset_index(name="INST_COUNT")
        inst_agg = inst_agg.merge(inst_count, on="SK_ID_CURR", how="left")

        logger.info(f"Created payment features: {len(inst_agg.columns)} columns")
        return inst_agg

    def merge_features(
        self,
        application_df: pd.DataFrame,
        bureau_features: Optional[pd.DataFrame] = None,
        prev_app_features: Optional[pd.DataFrame] = None,
        payment_features: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Merge all feature sets into a single DataFrame.

        Args:
            application_df: Main application features.
            bureau_features: Aggregated bureau features.
            prev_app_features: Aggregated previous application features.
            payment_features: Aggregated payment features.

        Returns:
            Merged DataFrame with all features.
        """
        logger.info("Merging all feature sets")
        result = application_df.copy()

        if bureau_features is not None and len(bureau_features) > 0:
            result = result.merge(bureau_features, on="SK_ID_CURR", how="left")
            logger.info(f"Merged bureau features. Columns: {len(result.columns)}")

        if prev_app_features is not None and len(prev_app_features) > 0:
            result = result.merge(prev_app_features, on="SK_ID_CURR", how="left")
            logger.info(f"Merged previous application features. Columns: {len(result.columns)}")

        if payment_features is not None and len(payment_features) > 0:
            result = result.merge(payment_features, on="SK_ID_CURR", how="left")
            logger.info(f"Merged payment features. Columns: {len(result.columns)}")

        logger.info(f"Final merged dataset: {len(result)} rows, {len(result.columns)} columns")
        return result

    def add_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add event timestamp for feature store compatibility.

        Args:
            df: Feature DataFrame.

        Returns:
            DataFrame with event_timestamp column.
        """
        df = df.copy()
        df["event_timestamp"] = pd.Timestamp.now()
        return df


def engineer_features(
    application_df: pd.DataFrame,
    bureau_df: Optional[pd.DataFrame] = None,
    prev_app_df: Optional[pd.DataFrame] = None,
    installments_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Convenience function to run full feature engineering pipeline.

    Args:
        application_df: Main application data.
        bureau_df: Bureau credit history data.
        prev_app_df: Previous applications data.
        installments_df: Installment payments data.

    Returns:
        DataFrame with all engineered features.
    """
    engineer = FeatureEngineer()

    # Create application features
    app_features = engineer.create_application_features(application_df)

    # Create bureau features if data is provided
    bureau_features = None
    if bureau_df is not None and len(bureau_df) > 0:
        bureau_features = engineer.create_bureau_features(bureau_df)

    # Create previous application features if data is provided
    prev_features = None
    if prev_app_df is not None and len(prev_app_df) > 0:
        prev_features = engineer.create_previous_application_features(prev_app_df)

    # Create payment features if data is provided
    payment_features = None
    if installments_df is not None and len(installments_df) > 0:
        payment_features = engineer.create_payment_features(installments_df)

    # Merge all features
    final_features = engineer.merge_features(
        app_features, bureau_features, prev_features, payment_features
    )

    # Add timestamp for feature store
    final_features = engineer.add_timestamp(final_features)

    return final_features
