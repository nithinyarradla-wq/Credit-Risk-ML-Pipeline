"""Feature store operations for managing credit risk features."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureStore:
    """Manages feature storage and retrieval operations."""

    def __init__(self, feature_store_path: Optional[str] = None):
        """
        Initialize the feature store.

        Args:
            feature_store_path: Path to the feature store directory.
        """
        self.config = get_config()
        if feature_store_path:
            self.store_path = Path(feature_store_path)
        else:
            self.store_path = self.config.get_project_root() / "feature_store"

        self.data_path = self.store_path / "data"
        self.data_path.mkdir(parents=True, exist_ok=True)

        self._feast_store = None

    def _get_feast_store(self):
        """Get or initialize the Feast feature store."""
        if self._feast_store is None:
            try:
                from feast import FeatureStore as FeastStore
                self._feast_store = FeastStore(repo_path=str(self.store_path))
            except Exception as e:
                logger.warning(f"Could not initialize Feast store: {e}")
                logger.info("Using local file-based storage instead")
        return self._feast_store

    def save_features(
        self,
        df: pd.DataFrame,
        feature_set_name: str,
        id_column: str = "SK_ID_CURR",
    ) -> str:
        """
        Save features to the feature store.

        Args:
            df: DataFrame containing features.
            feature_set_name: Name of the feature set.
            id_column: Name of the ID column.

        Returns:
            Path to the saved feature file.
        """
        logger.info(f"Saving feature set: {feature_set_name}")

        # Ensure timestamp column exists
        if "event_timestamp" not in df.columns:
            df = df.copy()
            df["event_timestamp"] = pd.Timestamp.now()

        # Save to parquet for Feast compatibility
        features_dir = self.data_path / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        file_path = features_dir / f"{feature_set_name}.parquet"
        df.to_parquet(file_path, index=False)

        logger.info(f"Saved {len(df)} rows to {file_path}")
        return str(file_path)

    def load_features(
        self,
        feature_set_name: str,
        entity_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Load features from the feature store.

        Args:
            feature_set_name: Name of the feature set to load.
            entity_ids: Optional list of entity IDs to filter.

        Returns:
            DataFrame containing the features.
        """
        logger.info(f"Loading feature set: {feature_set_name}")

        file_path = self.data_path / "features" / f"{feature_set_name}.parquet"

        if not file_path.exists():
            raise FileNotFoundError(f"Feature set not found: {file_path}")

        df = pd.read_parquet(file_path)

        if entity_ids is not None:
            df = df[df["SK_ID_CURR"].isin(entity_ids)]

        logger.info(f"Loaded {len(df)} rows from {feature_set_name}")
        return df

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_views: List[str],
    ) -> pd.DataFrame:
        """
        Get historical features for a set of entities.

        Args:
            entity_df: DataFrame with entity IDs and event timestamps.
            feature_views: List of feature view names to retrieve.

        Returns:
            DataFrame with all requested features joined.
        """
        logger.info(f"Retrieving historical features from: {feature_views}")

        store = self._get_feast_store()

        if store is not None:
            try:
                # Use Feast for historical feature retrieval
                feature_refs = []
                for view in feature_views:
                    # Get all features from the view
                    feature_refs.append(f"{view}:*")

                training_df = store.get_historical_features(
                    entity_df=entity_df,
                    features=feature_refs,
                ).to_df()

                return training_df
            except Exception as e:
                logger.warning(f"Feast retrieval failed: {e}")
                logger.info("Falling back to local file retrieval")

        # Fallback: load from local parquet files and join
        result = entity_df.copy()

        for view_name in feature_views:
            try:
                features = self.load_features(view_name)
                # Drop timestamp if it exists to avoid conflicts
                if "event_timestamp" in features.columns:
                    features = features.drop(columns=["event_timestamp"])

                result = result.merge(
                    features,
                    on="SK_ID_CURR",
                    how="left",
                )
            except FileNotFoundError:
                logger.warning(f"Feature set {view_name} not found")

        return result

    def get_online_features(
        self,
        entity_ids: List[int],
        feature_views: List[str],
    ) -> Dict[int, Dict[str, float]]:
        """
        Get online features for real-time scoring.

        Args:
            entity_ids: List of entity IDs to retrieve features for.
            feature_views: List of feature view names.

        Returns:
            Dictionary mapping entity IDs to their features.
        """
        logger.info(f"Retrieving online features for {len(entity_ids)} entities")

        store = self._get_feast_store()

        if store is not None:
            try:
                # Use Feast for online feature retrieval
                entity_rows = [{"SK_ID_CURR": eid} for eid in entity_ids]

                feature_refs = []
                for view in feature_views:
                    feature_refs.append(f"{view}:*")

                online_features = store.get_online_features(
                    entity_rows=entity_rows,
                    features=feature_refs,
                ).to_dict()

                # Convert to per-entity dictionary
                result = {}
                for i, eid in enumerate(entity_ids):
                    result[eid] = {
                        k: v[i] for k, v in online_features.items() if k != "SK_ID_CURR"
                    }

                return result
            except Exception as e:
                logger.warning(f"Feast online retrieval failed: {e}")

        # Fallback: load from parquet and filter
        result = {}
        all_features = pd.DataFrame({"SK_ID_CURR": entity_ids})

        for view_name in feature_views:
            try:
                features = self.load_features(view_name, entity_ids)
                all_features = all_features.merge(features, on="SK_ID_CURR", how="left")
            except FileNotFoundError:
                logger.warning(f"Feature set {view_name} not found")

        for _, row in all_features.iterrows():
            eid = row["SK_ID_CURR"]
            result[eid] = {
                k: v for k, v in row.to_dict().items()
                if k not in ["SK_ID_CURR", "event_timestamp"]
            }

        return result

    def materialize_features(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> None:
        """
        Materialize features to the online store.

        Args:
            start_date: Start date for materialization window.
            end_date: End date for materialization window.
        """
        logger.info("Materializing features to online store")

        store = self._get_feast_store()

        if store is None:
            logger.warning("Feast store not available. Skipping materialization.")
            return

        try:
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = datetime(2020, 1, 1)

            store.materialize(
                start_date=start_date,
                end_date=end_date,
            )
            logger.info("Materialization complete")
        except Exception as e:
            logger.error(f"Materialization failed: {e}")

    def apply_feature_definitions(self) -> None:
        """Apply feature definitions to the feature store registry."""
        logger.info("Applying feature definitions to registry")

        store = self._get_feast_store()

        if store is None:
            logger.warning("Feast store not available. Skipping apply.")
            return

        try:
            store.apply([])  # Apply all objects in the feature repo
            logger.info("Feature definitions applied successfully")
        except Exception as e:
            logger.error(f"Apply failed: {e}")

    def list_feature_sets(self) -> List[str]:
        """List all available feature sets."""
        features_dir = self.data_path / "features"
        if not features_dir.exists():
            return []

        return [f.stem for f in features_dir.glob("*.parquet")]


def save_features_to_store(
    df: pd.DataFrame,
    feature_set_name: str,
    store_path: Optional[str] = None,
) -> str:
    """
    Convenience function to save features.

    Args:
        df: Feature DataFrame.
        feature_set_name: Name for the feature set.
        store_path: Optional path to feature store.

    Returns:
        Path to saved features.
    """
    store = FeatureStore(store_path)
    return store.save_features(df, feature_set_name)


def load_features_from_store(
    feature_set_name: str,
    entity_ids: Optional[List[int]] = None,
    store_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to load features.

    Args:
        feature_set_name: Name of the feature set.
        entity_ids: Optional entity IDs to filter.
        store_path: Optional path to feature store.

    Returns:
        Feature DataFrame.
    """
    store = FeatureStore(store_path)
    return store.load_features(feature_set_name, entity_ids)
