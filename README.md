# Credit Risk ML Pipeline

A production-ready machine learning pipeline for credit risk prediction, featuring data ingestion, feature engineering, a feature store, model training, and Apache Airflow orchestration.

## Project Overview

This project simulates a credit risk ML pipeline where:

- Raw application and payment data is ingested and transformed
- Features are stored in an offline/online feature store (Feast)
- A model is trained, deployed, and reused for real-time or batch scoring
- The entire flow is orchestrated using Apache Airflow

## Tech Stack

| Layer | Tools |
|-------|-------|
| Ingestion / ETL | Python, Pandas |
| Orchestration | Apache Airflow |
| Feature Store | Feast (local file-based) |
| Modeling | scikit-learn (Random Forest, Gradient Boosting, Logistic Regression) |
| Storage | Local disk / Parquet files |

## Project Structure

```
Ml_pipeline/
├── airflow/
│   └── dags/
│       └── credit_risk_pipeline.py    # Airflow DAG definition
├── config/
│   └── config.yaml                     # Pipeline configuration
├── data/
│   ├── raw/                            # Raw data files (CSV)
│   ├── processed/                      # Preprocessed data
│   ├── features/                       # Engineered features
│   └── predictions/                    # Model predictions
├── feature_store/
│   ├── feature_store.yaml              # Feast configuration
│   └── features.py                     # Feature definitions
├── models/                             # Saved model artifacts
├── src/
│   ├── data/
│   │   ├── ingestion.py               # Data loading
│   │   └── preprocessing.py           # Data cleaning
│   ├── features/
│   │   ├── engineering.py             # Feature creation
│   │   ├── selection.py               # Feature selection
│   │   └── store.py                   # Feature store operations
│   ├── models/
│   │   └── trainer.py                 # Model training
│   ├── scoring/
│   │   └── scorer.py                  # Batch and online scoring
│   ├── utils/
│   │   ├── config_loader.py           # Configuration management
│   │   └── logger.py                  # Logging utilities
│   └── main.py                        # CLI entry point
├── tests/                             # Unit tests
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup
└── README.md
```

## Installation

1. Clone the repository and navigate to the project directory:

```bash
cd Ml_pipeline
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install the package in development mode:

```bash
pip install -e .
```

## Dataset

This pipeline uses the [Home Credit Default Risk Dataset](https://www.kaggle.com/c/home-credit-default-risk/data) from Kaggle.

Download and place the following files in `data/raw/`:

- `application_train.csv` - Main training data
- `application_test.csv` - Test data for scoring
- `bureau.csv` - Credit bureau data (optional)
- `previous_application.csv` - Previous loan applications (optional)
- `installments_payments.csv` - Payment history (optional)

## Usage

### Command Line Interface

Run the full pipeline:

```bash
python -m src.main run
```

Run individual steps:

```bash
# Data ingestion
python -m src.main ingest --data-dir data/raw

# Feature engineering
python -m src.main features

# Model training
python -m src.main train --model-type random_forest

# Batch scoring
python -m src.main score --output predictions.csv
```

### Using Airflow

1. Initialize Airflow:

```bash
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
```

2. Start Airflow:

```bash
airflow webserver --port 8080 &
airflow scheduler &
```

3. Access the Airflow UI at `http://localhost:8080` and trigger the `credit_risk_pipeline` DAG.

### Python API

```python
from src.data.ingestion import DataIngestion
from src.features.engineering import engineer_features
from src.models.trainer import train_model
from src.scoring.scorer import score_applications

# Load data
ingestion = DataIngestion()
train_df = ingestion.load_application_train()

# Engineer features
features_df = engineer_features(train_df)

# Train model
trainer, metrics = train_model(features_df, model_type="random_forest")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

# Score new data
test_df = ingestion.load_application_test()
predictions = score_applications(test_df)
```

## Feature Store

The pipeline uses Feast for feature management. Features are organized into views:

- `credit_features` - Financial ratios and demographic features
- `bureau_features` - Aggregated credit bureau history
- `payment_features` - Payment behavior features

### Saving Features

```python
from src.features.store import FeatureStore

store = FeatureStore()
store.save_features(features_df, "credit_features")
```

### Retrieving Features

```python
# Historical features for training
features = store.get_historical_features(
    entity_df=entity_df,
    feature_views=["credit_features", "bureau_features"]
)

# Online features for real-time scoring
features = store.get_online_features(
    entity_ids=[12345],
    feature_views=["credit_features"]
)
```

## Configuration

Edit `config/config.yaml` to customize:

- Data file paths
- Feature columns
- Model parameters
- Feast settings
- Airflow schedule

## Running Tests

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ -v --cov=src --cov-report=html
```

## Pipeline Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Ingest    │────>│  Preprocess  │────>│ Feature Engineer│
│    Data     │     │    Data      │     │                 │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                  │
                                                  v
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Batch     │<────│    Train     │<────│  Feature Store  │
│   Score     │     │    Model     │     │     (Feast)     │
└─────────────┘     └──────────────┘     └─────────────────┘
```

## Model Training

The pipeline supports multiple model types:

- **Random Forest** (default) - Good balance of performance and interpretability
- **Gradient Boosting** - Often higher accuracy
- **Logistic Regression** - Fast, interpretable, good baseline

Models are evaluated using:
- ROC-AUC
- Precision, Recall, F1-Score
- 5-fold Cross-Validation

## Features

Key engineered features include:

| Feature | Description |
|---------|-------------|
| CREDIT_INCOME_RATIO | Credit amount / Income |
| ANNUITY_INCOME_RATIO | Monthly payment / Income |
| LOAN_TERM | Credit amount / Annuity |
| AGE_YEARS | Applicant age |
| EMPLOYMENT_YEARS | Years employed |
| EXT_SOURCE_MEAN | Mean of external data sources |
| BUREAU_COUNT | Number of previous credits |

## License

MIT License
