"""Data ingestion module for loading raw credit risk data."""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:
    """Handles loading and basic validation of raw data files."""

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the data ingestion handler.

        Args:
            data_dir: Path to directory containing raw data files.
                     If None, uses path from config.
        """
        self.config = get_config()
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = self.config.get_data_path("raw_data")

    def load_application_train(self) -> pd.DataFrame:
        """Load the main training application data."""
        file_name = self.config.get("data.train_file", "application_train.csv")
        return self._load_csv(file_name)

    def load_application_test(self) -> pd.DataFrame:
        """Load the test application data."""
        file_name = self.config.get("data.test_file", "application_test.csv")
        return self._load_csv(file_name)

    def load_bureau(self) -> pd.DataFrame:
        """Load bureau credit history data."""
        file_name = self.config.get("data.bureau_file", "bureau.csv")
        return self._load_csv(file_name)

    def load_bureau_balance(self) -> pd.DataFrame:
        """Load bureau balance data."""
        file_name = self.config.get("data.bureau_balance_file", "bureau_balance.csv")
        return self._load_csv(file_name)

    def load_previous_application(self) -> pd.DataFrame:
        """Load previous application data."""
        file_name = self.config.get(
            "data.previous_application_file", "previous_application.csv"
        )
        return self._load_csv(file_name)

    def load_pos_cash_balance(self) -> pd.DataFrame:
        """Load POS cash balance data."""
        file_name = self.config.get("data.pos_cash_file", "POS_CASH_balance.csv")
        return self._load_csv(file_name)

    def load_installments_payments(self) -> pd.DataFrame:
        """Load installments payments data."""
        file_name = self.config.get(
            "data.installments_file", "installments_payments.csv"
        )
        return self._load_csv(file_name)

    def load_credit_card_balance(self) -> pd.DataFrame:
        """Load credit card balance data."""
        file_name = self.config.get("data.credit_card_file", "credit_card_balance.csv")
        return self._load_csv(file_name)

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available data files.

        Returns:
            Dictionary mapping data names to DataFrames.
        """
        data = {}
        loaders = {
            "application_train": self.load_application_train,
            "application_test": self.load_application_test,
            "bureau": self.load_bureau,
            "bureau_balance": self.load_bureau_balance,
            "previous_application": self.load_previous_application,
            "pos_cash_balance": self.load_pos_cash_balance,
            "installments_payments": self.load_installments_payments,
            "credit_card_balance": self.load_credit_card_balance,
        }

        for name, loader in loaders.items():
            try:
                data[name] = loader()
                logger.info(f"Loaded {name}: {len(data[name])} rows")
            except FileNotFoundError:
                logger.warning(f"File for {name} not found, skipping")

        return data

    def _load_csv(self, file_name: str) -> pd.DataFrame:
        """
        Load a CSV file from the data directory.

        Args:
            file_name: Name of the CSV file to load.

        Returns:
            DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = self.data_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

        return df

    def validate_data(self, df: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate that a DataFrame contains required columns.

        Args:
            df: DataFrame to validate.
            required_columns: List of column names that must be present.

        Returns:
            True if validation passes.

        Raises:
            ValueError: If required columns are missing.
        """
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True


def ingest_training_data(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to load training data.

    Args:
        data_dir: Optional path to data directory.

    Returns:
        Training DataFrame.
    """
    ingestion = DataIngestion(data_dir)
    return ingestion.load_application_train()


def ingest_test_data(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to load test data.

    Args:
        data_dir: Optional path to data directory.

    Returns:
        Test DataFrame.
    """
    ingestion = DataIngestion(data_dir)
    return ingestion.load_application_test()
