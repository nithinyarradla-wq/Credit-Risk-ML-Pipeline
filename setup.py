from setuptools import setup, find_packages

setup(
    name="credit_risk_pipeline",
    version="1.0.0",
    description="Credit Risk ML Pipeline with Feature Store and Airflow Orchestration",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "feast>=0.35.0",
        "apache-airflow>=2.7.0",
        "pyyaml>=6.0.0",
        "loguru>=0.7.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "credit-pipeline=src.main:main",
        ],
    },
)
