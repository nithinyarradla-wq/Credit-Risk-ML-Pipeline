"""Main entry point for the credit risk ML pipeline."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.utils.config_loader import get_config
from src.utils.logger import setup_logger, get_logger


def run_ingestion(data_dir: Optional[str] = None) -> None:
    """Run data ingestion step."""
    from src.data.ingestion import DataIngestion

    logger = get_logger(__name__)
    logger.info("Running data ingestion")

    ingestion = DataIngestion(data_dir)
    data = ingestion.load_all_data()

    for name, df in data.items():
        logger.info(f"  {name}: {len(df)} rows, {len(df.columns)} columns")


def run_preprocessing() -> str:
    """Run data preprocessing step."""
    from src.data.ingestion import DataIngestion
    from src.data.preprocessing import DataPreprocessor

    logger = get_logger(__name__)
    config = get_config()

    logger.info("Running data preprocessing")

    ingestion = DataIngestion()
    preprocessor = DataPreprocessor()

    # Load and preprocess training data
    train_df = ingestion.load_application_train()
    train_df = preprocessor.clean_application_data(train_df)
    train_df = preprocessor.handle_missing_values(train_df)
    train_df = preprocessor.encode_categorical(train_df)

    # Save preprocessed data
    output_path = config.get_data_path("processed_data")
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "train_preprocessed.parquet"
    train_df.to_parquet(output_file, index=False)

    logger.info(f"Preprocessed data saved to {output_file}")
    return str(output_file)


def run_feature_engineering() -> str:
    """Run feature engineering step."""
    import pandas as pd
    from src.data.ingestion import DataIngestion
    from src.features.engineering import engineer_features
    from src.features.store import FeatureStore

    logger = get_logger(__name__)
    config = get_config()

    logger.info("Running feature engineering")

    ingestion = DataIngestion()

    # Load data
    train_df = ingestion.load_application_train()

    # Try to load additional data
    bureau_df = None
    prev_app_df = None
    installments_df = None

    try:
        bureau_df = ingestion.load_bureau()
    except FileNotFoundError:
        logger.warning("Bureau data not available")

    try:
        prev_app_df = ingestion.load_previous_application()
    except FileNotFoundError:
        logger.warning("Previous application data not available")

    try:
        installments_df = ingestion.load_installments_payments()
    except FileNotFoundError:
        logger.warning("Installments data not available")

    # Engineer features
    features_df = engineer_features(
        train_df,
        bureau_df=bureau_df,
        prev_app_df=prev_app_df,
        installments_df=installments_df,
    )

    # Save features
    feature_path = config.get_data_path("feature_data")
    feature_path.mkdir(parents=True, exist_ok=True)
    output_file = feature_path / "train_features.parquet"
    features_df.to_parquet(output_file, index=False)

    # Update feature store
    feature_store = FeatureStore()
    feature_store.save_features(features_df, "credit_features")

    logger.info(f"Features saved to {output_file}")
    logger.info(f"Total features: {len(features_df.columns)}")

    return str(output_file)


def run_training(model_type: str = "random_forest") -> str:
    """Run model training step."""
    import pandas as pd
    from src.models.trainer import ModelTrainer
    from src.features.selection import select_features

    logger = get_logger(__name__)
    config = get_config()

    logger.info(f"Running model training with {model_type}")

    # Load features
    feature_path = config.get_data_path("feature_data") / "train_features.parquet"
    if not feature_path.exists():
        logger.error("Features not found. Run feature engineering first.")
        sys.exit(1)

    features_df = pd.read_parquet(feature_path)

    # Feature selection
    features_df, importance_df = select_features(
        features_df,
        n_features=50,
        method="importance",
    )

    # Train model
    trainer = ModelTrainer(model_type)
    X_train, X_test, y_train, y_test = trainer.prepare_data(features_df)
    trainer.train(X_train, y_train)

    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)

    # Cross-validate
    cv_results = trainer.cross_validate(
        pd.concat([X_train, X_test]),
        pd.concat([y_train, y_test]),
    )

    # Save model
    model_path = trainer.save_model()

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"CV ROC-AUC: {cv_results['cv_mean_roc_auc']:.4f} (+/- {cv_results['cv_std_roc_auc']:.4f})")

    return model_path


def run_scoring(model_path: Optional[str] = None, output_path: Optional[str] = None) -> str:
    """Run batch scoring step."""
    import pandas as pd
    from src.data.ingestion import DataIngestion
    from src.features.engineering import FeatureEngineer
    from src.scoring.scorer import CreditRiskScorer
    from datetime import datetime

    logger = get_logger(__name__)
    config = get_config()

    logger.info("Running batch scoring")

    # Load test data
    ingestion = DataIngestion()
    try:
        test_df = ingestion.load_application_test()
    except FileNotFoundError:
        logger.error("Test data not found")
        sys.exit(1)

    # Engineer features
    engineer = FeatureEngineer()
    test_features = engineer.create_application_features(test_df)
    test_features = engineer.add_timestamp(test_features)

    # Score
    scorer = CreditRiskScorer()
    if model_path:
        scorer.load_model(model_path)
    else:
        scorer.load_latest_model()

    results = scorer.score(test_features)

    # Save results
    if output_path is None:
        predictions_dir = config.get_project_root() / "data" / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(predictions_dir / f"predictions_{timestamp}.csv")

    results.to_csv(output_path, index=False)

    logger.info(f"Predictions saved to {output_path}")
    logger.info(f"Total predictions: {len(results)}")
    logger.info(f"Predicted default rate: {results['prediction'].mean():.4f}")

    return output_path


def run_full_pipeline() -> None:
    """Run the complete ML pipeline."""
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("Starting Credit Risk ML Pipeline")
    logger.info("=" * 60)

    # Step 1: Ingestion
    logger.info("\n[Step 1/4] Data Ingestion")
    run_ingestion()

    # Step 2: Feature Engineering
    logger.info("\n[Step 2/4] Feature Engineering")
    run_feature_engineering()

    # Step 3: Model Training
    logger.info("\n[Step 3/4] Model Training")
    run_training()

    # Step 4: Scoring
    logger.info("\n[Step 4/4] Batch Scoring")
    try:
        run_scoring()
    except Exception as e:
        logger.warning(f"Scoring skipped: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete")
    logger.info("=" * 60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Credit Risk ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline commands")

    # Full pipeline command
    subparsers.add_parser("run", help="Run the full pipeline")

    # Individual step commands
    ingest_parser = subparsers.add_parser("ingest", help="Run data ingestion")
    ingest_parser.add_argument("--data-dir", help="Path to raw data directory")

    subparsers.add_parser("preprocess", help="Run data preprocessing")

    subparsers.add_parser("features", help="Run feature engineering")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--model-type",
        choices=["random_forest", "gradient_boosting", "logistic_regression"],
        default="random_forest",
        help="Model type to train",
    )

    score_parser = subparsers.add_parser("score", help="Run batch scoring")
    score_parser.add_argument("--model-path", help="Path to model file")
    score_parser.add_argument("--output", help="Path to save predictions")

    args = parser.parse_args()

    # Setup logging
    setup_logger(log_level=args.log_level)
    logger = get_logger(__name__)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "run":
            run_full_pipeline()
        elif args.command == "ingest":
            run_ingestion(args.data_dir)
        elif args.command == "preprocess":
            run_preprocessing()
        elif args.command == "features":
            run_feature_engineering()
        elif args.command == "train":
            run_training(args.model_type)
        elif args.command == "score":
            run_scoring(args.model_path, args.output)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
