"""Feast feature definitions for credit risk pipeline."""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Float64, Int64


# Define the customer entity
customer = Entity(
    name="customer",
    join_keys=["SK_ID_CURR"],
    description="Customer ID for credit applications",
)

# Define the file source for features
credit_features_source = FileSource(
    name="credit_features_source",
    path="data/features/credit_features.parquet",
    timestamp_field="event_timestamp",
)

# Define the main credit features view
credit_features_view = FeatureView(
    name="credit_features",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        # Financial ratios
        Field(name="CREDIT_INCOME_RATIO", dtype=Float64),
        Field(name="ANNUITY_INCOME_RATIO", dtype=Float64),
        Field(name="LOAN_TERM", dtype=Float64),
        Field(name="GOODS_CREDIT_RATIO", dtype=Float64),

        # Demographics
        Field(name="AGE_YEARS", dtype=Float64),
        Field(name="EMPLOYMENT_YEARS", dtype=Float64),
        Field(name="EMPLOYMENT_AGE_RATIO", dtype=Float64),
        Field(name="INCOME_PER_FAMILY_MEMBER", dtype=Float64),

        # External sources
        Field(name="EXT_SOURCE_MEAN", dtype=Float64),
        Field(name="EXT_SOURCE_STD", dtype=Float64),
        Field(name="EXT_SOURCE_PROD", dtype=Float64),
        Field(name="EXT_SOURCE_1", dtype=Float64),
        Field(name="EXT_SOURCE_2", dtype=Float64),
        Field(name="EXT_SOURCE_3", dtype=Float64),

        # Application amounts
        Field(name="AMT_INCOME_TOTAL", dtype=Float64),
        Field(name="AMT_CREDIT", dtype=Float64),
        Field(name="AMT_ANNUITY", dtype=Float64),
        Field(name="AMT_GOODS_PRICE", dtype=Float64),

        # Document and flag counts
        Field(name="DOCUMENTS_PROVIDED_COUNT", dtype=Int64),
        Field(name="FLAGS_COUNT", dtype=Int64),

        # Region info
        Field(name="REGION_RATING_CLIENT", dtype=Int64),
    ],
    source=credit_features_source,
    online=True,
    description="Credit risk features derived from application data",
)

# Define bureau features source
bureau_features_source = FileSource(
    name="bureau_features_source",
    path="data/features/bureau_features.parquet",
    timestamp_field="event_timestamp",
)

# Define bureau features view
bureau_features_view = FeatureView(
    name="bureau_features",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="BUREAU_COUNT", dtype=Int64),
        Field(name="BUREAU_DAYS_CREDIT_MIN", dtype=Float64),
        Field(name="BUREAU_DAYS_CREDIT_MAX", dtype=Float64),
        Field(name="BUREAU_DAYS_CREDIT_MEAN", dtype=Float64),
        Field(name="BUREAU_AMT_CREDIT_SUM_SUM", dtype=Float64),
        Field(name="BUREAU_AMT_CREDIT_SUM_MEAN", dtype=Float64),
        Field(name="BUREAU_AMT_CREDIT_SUM_DEBT_SUM", dtype=Float64),
        Field(name="BUREAU_AMT_CREDIT_SUM_DEBT_MEAN", dtype=Float64),
        Field(name="BUREAU_CREDIT_DAY_OVERDUE_MAX", dtype=Float64),
        Field(name="BUREAU_CREDIT_DAY_OVERDUE_MEAN", dtype=Float64),
    ],
    source=bureau_features_source,
    online=True,
    description="Aggregated bureau credit history features",
)

# Define payment features source
payment_features_source = FileSource(
    name="payment_features_source",
    path="data/features/payment_features.parquet",
    timestamp_field="event_timestamp",
)

# Define payment features view
payment_features_view = FeatureView(
    name="payment_features",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="INST_COUNT", dtype=Int64),
        Field(name="INST_PAYMENT_DIFF_MEAN", dtype=Float64),
        Field(name="INST_PAYMENT_DIFF_MAX", dtype=Float64),
        Field(name="INST_PAYMENT_LATE_SUM", dtype=Float64),
        Field(name="INST_PAYMENT_LATE_MEAN", dtype=Float64),
        Field(name="INST_AMT_PAYMENT_SUM", dtype=Float64),
        Field(name="INST_AMT_PAYMENT_MEAN", dtype=Float64),
        Field(name="INST_PAYMENT_RATIO_MEAN", dtype=Float64),
    ],
    source=payment_features_source,
    online=True,
    description="Aggregated installment payment features",
)
