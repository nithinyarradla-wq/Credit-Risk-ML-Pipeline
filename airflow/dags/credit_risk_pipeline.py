"""
Apache Airflow DAG for Credit Risk ML Pipeline.

This DAG orchestrates the following steps:
1. Data ingestion and validation
2. Feature engineering
3. Feature store update
4. Model training (optional, on schedule)
5. Batch scoring
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule


# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Default DAG arguments
default_args = {
    "owner": "data-science",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def ingest_data(**context):
    """Load and validate raw data."""
    from src.data.ingestion import DataIngestion
    from src.utils.logger import setup_logger, get_logger

    setup_logger()
    logger = get_logger(__name__)

    logger.info("Starting data ingestion task")

    ingestion = DataIngestion()

    try:
        # Load main application data
        train_df = ingestion.load_application_train()

        # Validate data
        required_columns = ["SK_ID_CURR", "TARGET", "AMT_CREDIT", "AMT_INCOME_TOTAL"]
        ingestion.validate_data(train_df, required_columns)

        # Store row count for downstream tasks
        context["ti"].xcom_push(key="train_rows", value=len(train_df))

        logger.info(f"Data ingestion complete: {len(train_df)} rows")
        return True

    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")
        raise


def preprocess_data(**context):
    """Clean and preprocess raw data."""
    from src.data.ingestion import DataIngestion
    from src.data.preprocessing import DataPreprocessor
    from src.utils.config_loader import get_config
    from src.utils.logger import setup_logger, get_logger

    setup_logger()
    logger = get_logger(__name__)

    logger.info("Starting data preprocessing task")

    config = get_config()
    ingestion = DataIngestion()
    preprocessor = DataPreprocessor()

    # Load data
    train_df = ingestion.load_application_train()

    # Clean data
    train_df = preprocessor.clean_application_data(train_df)
    train_df = preprocessor.handle_missing_values(train_df)
    train_df = preprocessor.encode_categorical(train_df)

    # Save preprocessed data
    output_path = config.get_data_path("processed_data")
    output_path.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(output_path / "train_preprocessed.parquet", index=False)

    logger.info(f"Preprocessing complete: {len(train_df)} rows, {len(train_df.columns)} columns")
    return str(output_path / "train_preprocessed.parquet")


def engineer_features(**context):
    """Create derived features from preprocessed data."""
    import pandas as pd
    from src.data.ingestion import DataIngestion
    from src.features.engineering import FeatureEngineer
    from src.utils.config_loader import get_config
    from src.utils.logger import setup_logger, get_logger

    setup_logger()
    logger = get_logger(__name__)

    logger.info("Starting feature engineering task")

    config = get_config()
    ingestion = DataIngestion()
    engineer = FeatureEngineer()

    # Load preprocessed data
    processed_path = config.get_data_path("processed_data") / "train_preprocessed.parquet"
    if processed_path.exists():
        train_df = pd.read_parquet(processed_path)
    else:
        train_df = ingestion.load_application_train()

    # Load additional data for feature engineering
    bureau_df = None
    prev_app_df = None
    installments_df = None

    try:
        bureau_df = ingestion.load_bureau()
    except FileNotFoundError:
        logger.warning("Bureau data not found, skipping")

    try:
        prev_app_df = ingestion.load_previous_application()
    except FileNotFoundError:
        logger.warning("Previous application data not found, skipping")

    try:
        installments_df = ingestion.load_installments_payments()
    except FileNotFoundError:
        logger.warning("Installments data not found, skipping")

    # Create features
    train_features = engineer.create_application_features(train_df)

    if bureau_df is not None:
        bureau_features = engineer.create_bureau_features(bureau_df)
        train_features = train_features.merge(
            bureau_features, on="SK_ID_CURR", how="left"
        )

    if prev_app_df is not None:
        prev_features = engineer.create_previous_application_features(prev_app_df)
        train_features = train_features.merge(
            prev_features, on="SK_ID_CURR", how="left"
        )

    if installments_df is not None:
        payment_features = engineer.create_payment_features(installments_df)
        train_features = train_features.merge(
            payment_features, on="SK_ID_CURR", how="left"
        )

    # Add timestamp
    train_features = engineer.add_timestamp(train_features)

    # Save features
    feature_path = config.get_data_path("feature_data")
    feature_path.mkdir(parents=True, exist_ok=True)
    train_features.to_parquet(feature_path / "train_features.parquet", index=False)

    logger.info(f"Feature engineering complete: {len(train_features.columns)} features")
    context["ti"].xcom_push(key="feature_count", value=len(train_features.columns))

    return str(feature_path / "train_features.parquet")


def update_feature_store(**context):
    """Update the feature store with new features."""
    import pandas as pd
    from src.features.store import FeatureStore
    from src.utils.config_loader import get_config
    from src.utils.logger import setup_logger, get_logger

    setup_logger()
    logger = get_logger(__name__)

    logger.info("Starting feature store update task")

    config = get_config()
    feature_store = FeatureStore()

    # Load engineered features
    feature_path = config.get_data_path("feature_data") / "train_features.parquet"
    features_df = pd.read_parquet(feature_path)

    # Save to feature store
    feature_store.save_features(features_df, "credit_features")

    logger.info("Feature store updated successfully")
    return True


def check_training_schedule(**context):
    """Determine if model training should run."""
    from src.utils.logger import setup_logger, get_logger

    setup_logger()
    logger = get_logger(__name__)

    # Check execution date - train weekly on Sundays
    execution_date = context["execution_date"]
    should_train = execution_date.weekday() == 6  # Sunday

    # Can also force training via DAG config
    dag_conf = context.get("dag_run").conf or {}
    if dag_conf.get("force_training", False):
        should_train = True

    logger.info(f"Training check: should_train={should_train}")

    if should_train:
        return "train_model"
    else:
        return "skip_training"


def train_model(**context):
    """Train the credit risk model."""
    import pandas as pd
    from src.models.trainer import ModelTrainer
    from src.features.selection import FeatureSelector
    from src.utils.config_loader import get_config
    from src.utils.logger import setup_logger, get_logger

    setup_logger()
    logger = get_logger(__name__)

    logger.info("Starting model training task")

    config = get_config()

    # Load features
    feature_path = config.get_data_path("feature_data") / "train_features.parquet"
    features_df = pd.read_parquet(feature_path)

    # Feature selection
    selector = FeatureSelector()
    features_df = selector.remove_low_variance_features(features_df)
    features_df = selector.remove_correlated_features(features_df)

    # Train model
    trainer = ModelTrainer(model_type="random_forest")
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

    logger.info(f"Model training complete. ROC-AUC: {metrics['roc_auc']:.4f}")

    # Push metrics to XCom
    context["ti"].xcom_push(key="model_metrics", value=metrics)
    context["ti"].xcom_push(key="model_path", value=model_path)

    return model_path


def batch_score(**context):
    """Score new applications in batch."""
    import pandas as pd
    from src.scoring.scorer import CreditRiskScorer
    from src.data.ingestion import DataIngestion
    from src.features.engineering import FeatureEngineer
    from src.utils.config_loader import get_config
    from src.utils.logger import setup_logger, get_logger

    setup_logger()
    logger = get_logger(__name__)

    logger.info("Starting batch scoring task")

    config = get_config()
    ingestion = DataIngestion()
    engineer = FeatureEngineer()

    # Load test data
    try:
        test_df = ingestion.load_application_test()
    except FileNotFoundError:
        logger.warning("Test data not found, skipping batch scoring")
        return None

    # Engineer features for test data
    test_features = engineer.create_application_features(test_df)
    test_features = engineer.add_timestamp(test_features)

    # Load model and score
    scorer = CreditRiskScorer()
    scorer.load_latest_model()

    results = scorer.score(test_features)

    # Save results
    output_path = config.get_project_root() / "data" / "predictions"
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"predictions_{timestamp}.parquet"
    results.to_parquet(results_file, index=False)

    logger.info(f"Batch scoring complete: {len(results)} predictions saved to {results_file}")

    return str(results_file)


def validate_results(**context):
    """Validate scoring results and generate summary."""
    import pandas as pd
    from src.utils.config_loader import get_config
    from src.utils.logger import setup_logger, get_logger

    setup_logger()
    logger = get_logger(__name__)

    logger.info("Starting results validation task")

    config = get_config()
    predictions_dir = config.get_project_root() / "data" / "predictions"

    if not predictions_dir.exists():
        logger.warning("No predictions directory found")
        return

    # Find latest predictions
    prediction_files = list(predictions_dir.glob("predictions_*.parquet"))
    if not prediction_files:
        logger.warning("No prediction files found")
        return

    latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
    results = pd.read_parquet(latest_file)

    # Generate summary
    summary = {
        "total_scored": len(results),
        "predicted_defaults": int(results["prediction"].sum()),
        "default_rate": float(results["prediction"].mean()),
        "avg_probability": float(results["probability"].mean()),
        "high_risk_count": int((results["probability"] > 0.5).sum()),
    }

    logger.info(f"Validation Summary: {summary}")

    # Push to XCom
    context["ti"].xcom_push(key="scoring_summary", value=summary)

    return summary


# Define the DAG
with DAG(
    dag_id="credit_risk_pipeline",
    default_args=default_args,
    description="Credit Risk ML Pipeline with Feature Store",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "credit-risk", "feature-store"],
) as dag:

    # Task definitions
    start = EmptyOperator(task_id="start")

    ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    features = PythonOperator(
        task_id="engineer_features",
        python_callable=engineer_features,
    )

    store_features = PythonOperator(
        task_id="update_feature_store",
        python_callable=update_feature_store,
    )

    check_training = BranchPythonOperator(
        task_id="check_training_schedule",
        python_callable=check_training_schedule,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    skip_training = EmptyOperator(task_id="skip_training")

    # Join point after training decision
    training_complete = EmptyOperator(
        task_id="training_complete",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    score = PythonOperator(
        task_id="batch_score",
        python_callable=batch_score,
    )

    validate = PythonOperator(
        task_id="validate_results",
        python_callable=validate_results,
    )

    end = EmptyOperator(task_id="end")

    # Define task dependencies
    start >> ingest >> preprocess >> features >> store_features >> check_training
    check_training >> [train, skip_training]
    [train, skip_training] >> training_complete >> score >> validate >> end
