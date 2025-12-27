.PHONY: install install-dev test lint format clean data run train score airflow-init airflow-start airflow-stop

# Python interpreter
PYTHON := python

# Install production dependencies
install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

# Install development dependencies
install-dev:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e ".[dev]"

# Run tests
test:
	$(PYTHON) -m pytest tests/ -v

# Run tests with coverage
test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Generate sample data
data:
	$(PYTHON) scripts/generate_sample_data.py

# Run the full pipeline
run:
	$(PYTHON) -m src.main run

# Run data ingestion
ingest:
	$(PYTHON) -m src.main ingest

# Run feature engineering
features:
	$(PYTHON) -m src.main features

# Train the model
train:
	$(PYTHON) -m src.main train

# Run batch scoring
score:
	$(PYTHON) -m src.main score

# Initialize Airflow
airflow-init:
	export AIRFLOW_HOME=$(PWD)/airflow && \
	airflow db init && \
	airflow users create \
		--username admin \
		--password admin \
		--firstname Admin \
		--lastname User \
		--role Admin \
		--email admin@example.com

# Start Airflow webserver and scheduler
airflow-start:
	export AIRFLOW_HOME=$(PWD)/airflow && \
	airflow webserver --port 8080 --daemon && \
	airflow scheduler --daemon
	@echo "Airflow started. Access UI at http://localhost:8080"

# Stop Airflow
airflow-stop:
	@pkill -f "airflow webserver" || true
	@pkill -f "airflow scheduler" || true
	@echo "Airflow stopped"

# Apply Feast feature definitions
feast-apply:
	cd feature_store && feast apply

# Materialize features to online store
feast-materialize:
	cd feature_store && feast materialize-incremental $(shell date -u +"%Y-%m-%dT%H:%M:%S")

# Clean generated files
clean:
	rm -rf data/processed/*
	rm -rf data/features/*
	rm -rf data/predictions/*
	rm -rf models/*.joblib
	rm -rf models/*.json
	rm -rf models/*.csv
	rm -rf feature_store/data/*
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Clean everything including raw data
clean-all: clean
	rm -rf data/raw/*

# Format code
format:
	$(PYTHON) -m black src/ tests/ scripts/
	$(PYTHON) -m isort src/ tests/ scripts/

# Lint code
lint:
	$(PYTHON) -m flake8 src/ tests/ scripts/
	$(PYTHON) -m mypy src/

# Show help
help:
	@echo "Available commands:"
	@echo "  make install        - Install production dependencies"
	@echo "  make install-dev    - Install development dependencies"
	@echo "  make data           - Generate sample data"
	@echo "  make run            - Run the full pipeline"
	@echo "  make ingest         - Run data ingestion"
	@echo "  make features       - Run feature engineering"
	@echo "  make train          - Train the model"
	@echo "  make score          - Run batch scoring"
	@echo "  make test           - Run tests"
	@echo "  make test-cov       - Run tests with coverage"
	@echo "  make airflow-init   - Initialize Airflow"
	@echo "  make airflow-start  - Start Airflow"
	@echo "  make airflow-stop   - Stop Airflow"
	@echo "  make feast-apply    - Apply Feast feature definitions"
	@echo "  make clean          - Clean generated files"
	@echo "  make format         - Format code"
	@echo "  make lint           - Lint code"
