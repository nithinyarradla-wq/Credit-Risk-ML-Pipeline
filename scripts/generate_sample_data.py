"""Generate sample data for testing the credit risk pipeline."""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_application_data(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic application data similar to Home Credit dataset."""
    np.random.seed(seed)

    # Generate target with ~8% default rate (similar to real data)
    target = np.random.binomial(1, 0.08, n_samples)

    # Generate features correlated with target
    data = {
        "SK_ID_CURR": range(100000, 100000 + n_samples),
        "TARGET": target,

        # Contract type
        "NAME_CONTRACT_TYPE": np.random.choice(
            ["Cash loans", "Revolving loans"], n_samples, p=[0.9, 0.1]
        ),

        # Gender
        "CODE_GENDER": np.random.choice(["M", "F"], n_samples, p=[0.35, 0.65]),

        # Car and realty ownership
        "FLAG_OWN_CAR": np.random.choice(["Y", "N"], n_samples, p=[0.34, 0.66]),
        "FLAG_OWN_REALTY": np.random.choice(["Y", "N"], n_samples, p=[0.69, 0.31]),

        # Family
        "CNT_CHILDREN": np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.7, 0.15, 0.1, 0.04, 0.01]),
        "CNT_FAM_MEMBERS": np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.2, 0.35, 0.25, 0.15, 0.05]),

        # Income
        "AMT_INCOME_TOTAL": np.random.lognormal(11.5, 0.8, n_samples),
        "NAME_INCOME_TYPE": np.random.choice(
            ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
            n_samples, p=[0.52, 0.23, 0.18, 0.06, 0.01]
        ),

        # Education
        "NAME_EDUCATION_TYPE": np.random.choice(
            ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"],
            n_samples, p=[0.71, 0.24, 0.03, 0.015, 0.005]
        ),

        # Family status
        "NAME_FAMILY_STATUS": np.random.choice(
            ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
            n_samples, p=[0.63, 0.15, 0.1, 0.07, 0.05]
        ),

        # Housing
        "NAME_HOUSING_TYPE": np.random.choice(
            ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"],
            n_samples, p=[0.88, 0.05, 0.03, 0.02, 0.01, 0.01]
        ),

        # Region rating
        "REGION_RATING_CLIENT": np.random.choice([1, 2, 3], n_samples, p=[0.1, 0.6, 0.3]),
        "REGION_RATING_CLIENT_W_CITY": np.random.choice([1, 2, 3], n_samples, p=[0.1, 0.6, 0.3]),

        # Days (negative values representing days before application)
        "DAYS_BIRTH": np.random.randint(-25000, -7000, n_samples),
        "DAYS_EMPLOYED": np.random.randint(-15000, 0, n_samples),
        "DAYS_REGISTRATION": np.random.randint(-25000, 0, n_samples),
        "DAYS_ID_PUBLISH": np.random.randint(-7000, 0, n_samples),

        # Credit amount
        "AMT_CREDIT": np.random.lognormal(13, 0.8, n_samples),
        "AMT_ANNUITY": np.random.lognormal(10, 0.7, n_samples),
        "AMT_GOODS_PRICE": np.random.lognormal(12.8, 0.8, n_samples),

        # External sources (important predictors)
        "EXT_SOURCE_1": np.random.beta(2, 5, n_samples),
        "EXT_SOURCE_2": np.random.beta(3, 3, n_samples),
        "EXT_SOURCE_3": np.random.beta(2, 4, n_samples),

        # Document flags
        "FLAG_DOCUMENT_2": np.random.binomial(1, 0.01, n_samples),
        "FLAG_DOCUMENT_3": np.random.binomial(1, 0.7, n_samples),
        "FLAG_DOCUMENT_4": np.random.binomial(1, 0.01, n_samples),
        "FLAG_DOCUMENT_5": np.random.binomial(1, 0.01, n_samples),
        "FLAG_DOCUMENT_6": np.random.binomial(1, 0.08, n_samples),
        "FLAG_DOCUMENT_7": np.random.binomial(1, 0.01, n_samples),
        "FLAG_DOCUMENT_8": np.random.binomial(1, 0.08, n_samples),

        # Contact flags
        "FLAG_MOBIL": np.ones(n_samples, dtype=int),
        "FLAG_EMP_PHONE": np.random.binomial(1, 0.8, n_samples),
        "FLAG_WORK_PHONE": np.random.binomial(1, 0.2, n_samples),
        "FLAG_CONT_MOBILE": np.random.binomial(1, 0.99, n_samples),
        "FLAG_PHONE": np.random.binomial(1, 0.3, n_samples),
        "FLAG_EMAIL": np.random.binomial(1, 0.06, n_samples),

        # Occupation
        "OCCUPATION_TYPE": np.random.choice(
            ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers", "High skill tech staff",
             "Medicine staff", "Security staff", "Cooking staff", "Cleaning staff", np.nan],
            n_samples, p=[0.18, 0.1, 0.1, 0.08, 0.08, 0.05, 0.03, 0.03, 0.02, 0.02, 0.31]
        ),
    }

    # Adjust external sources based on target (lower for defaults)
    df = pd.DataFrame(data)
    df.loc[df["TARGET"] == 1, "EXT_SOURCE_1"] *= 0.7
    df.loc[df["TARGET"] == 1, "EXT_SOURCE_2"] *= 0.8
    df.loc[df["TARGET"] == 1, "EXT_SOURCE_3"] *= 0.75

    # Add some noise and missing values
    for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        mask = np.random.random(n_samples) < 0.3
        df.loc[mask, col] = np.nan

    return df


def generate_bureau_data(customer_ids: list, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic bureau credit history data."""
    np.random.seed(seed)

    records = []
    bureau_id = 5000000

    for cust_id in customer_ids:
        # Each customer has 0-10 bureau records
        n_records = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], p=[0.1, 0.2, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.03, 0.02, 0.01])

        for _ in range(n_records):
            records.append({
                "SK_ID_CURR": cust_id,
                "SK_ID_BUREAU": bureau_id,
                "CREDIT_ACTIVE": np.random.choice(["Active", "Closed", "Sold", "Bad debt"], p=[0.3, 0.6, 0.05, 0.05]),
                "CREDIT_CURRENCY": "currency 1",
                "DAYS_CREDIT": np.random.randint(-3000, 0),
                "CREDIT_DAY_OVERDUE": np.random.choice([0, 1, 5, 10, 30, 60, 90], p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02]),
                "DAYS_CREDIT_ENDDATE": np.random.randint(-1000, 1000),
                "DAYS_ENDDATE_FACT": np.random.randint(-2000, 0),
                "AMT_CREDIT_MAX_OVERDUE": np.random.lognormal(8, 2) if np.random.random() > 0.5 else 0,
                "CNT_CREDIT_PROLONG": np.random.choice([0, 1, 2, 3], p=[0.85, 0.1, 0.03, 0.02]),
                "AMT_CREDIT_SUM": np.random.lognormal(12, 1.5),
                "AMT_CREDIT_SUM_DEBT": np.random.lognormal(11, 1.5) if np.random.random() > 0.4 else 0,
                "AMT_CREDIT_SUM_LIMIT": np.random.lognormal(11, 1.5) if np.random.random() > 0.6 else 0,
                "AMT_CREDIT_SUM_OVERDUE": np.random.lognormal(8, 2) if np.random.random() > 0.9 else 0,
                "CREDIT_TYPE": np.random.choice(
                    ["Consumer credit", "Credit card", "Car loan", "Mortgage", "Microloan"],
                    p=[0.5, 0.25, 0.1, 0.1, 0.05]
                ),
                "DAYS_CREDIT_UPDATE": np.random.randint(-500, 0),
                "AMT_ANNUITY": np.random.lognormal(9, 1.5) if np.random.random() > 0.3 else np.nan,
            })
            bureau_id += 1

    return pd.DataFrame(records)


def generate_previous_application_data(customer_ids: list, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic previous application data."""
    np.random.seed(seed)

    records = []
    prev_id = 1000000

    for cust_id in customer_ids:
        n_records = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.1, 0.07, 0.03])

        for _ in range(n_records):
            records.append({
                "SK_ID_PREV": prev_id,
                "SK_ID_CURR": cust_id,
                "NAME_CONTRACT_TYPE": np.random.choice(["Cash loans", "Consumer loans", "Revolving loans"], p=[0.4, 0.5, 0.1]),
                "AMT_ANNUITY": np.random.lognormal(10, 0.8),
                "AMT_APPLICATION": np.random.lognormal(12, 1),
                "AMT_CREDIT": np.random.lognormal(12.2, 1),
                "AMT_DOWN_PAYMENT": np.random.lognormal(9, 1.5) if np.random.random() > 0.3 else 0,
                "AMT_GOODS_PRICE": np.random.lognormal(12, 1),
                "WEEKDAY_APPR_PROCESS_START": np.random.choice(["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]),
                "HOUR_APPR_PROCESS_START": np.random.randint(0, 24),
                "FLAG_LAST_APPL_PER_CONTRACT": np.random.choice(["Y", "N"], p=[0.8, 0.2]),
                "NFLAG_LAST_APPL_IN_DAY": np.random.choice([0, 1], p=[0.95, 0.05]),
                "RATE_DOWN_PAYMENT": np.random.uniform(0, 0.3) if np.random.random() > 0.3 else np.nan,
                "NAME_CONTRACT_STATUS": np.random.choice(["Approved", "Refused", "Canceled", "Unused offer"], p=[0.6, 0.2, 0.15, 0.05]),
                "DAYS_DECISION": np.random.randint(-3000, 0),
                "NAME_PAYMENT_TYPE": np.random.choice(["Cash through the bank", "XNA", "Non-cash from account"], p=[0.7, 0.2, 0.1]),
                "CNT_PAYMENT": np.random.choice([6, 12, 18, 24, 36, 48, 60], p=[0.1, 0.25, 0.2, 0.2, 0.15, 0.07, 0.03]),
            })
            prev_id += 1

    return pd.DataFrame(records)


def generate_installments_data(customer_ids: list, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic installments payments data."""
    np.random.seed(seed)

    records = []

    for cust_id in customer_ids:
        # Skip some customers
        if np.random.random() < 0.3:
            continue

        n_prev = np.random.randint(1, 4)

        for prev_num in range(n_prev):
            prev_id = cust_id * 10 + prev_num
            n_installments = np.random.choice([6, 12, 18, 24, 36], p=[0.2, 0.3, 0.25, 0.15, 0.1])

            for inst_num in range(n_installments):
                days_instalment = -n_installments * 30 + inst_num * 30
                days_entry = days_instalment + np.random.randint(-10, 30)

                records.append({
                    "SK_ID_PREV": prev_id,
                    "SK_ID_CURR": cust_id,
                    "NUM_INSTALMENT_VERSION": 1,
                    "NUM_INSTALMENT_NUMBER": inst_num + 1,
                    "DAYS_INSTALMENT": days_instalment,
                    "DAYS_ENTRY_PAYMENT": days_entry,
                    "AMT_INSTALMENT": np.random.lognormal(9, 0.5),
                    "AMT_PAYMENT": np.random.lognormal(9, 0.5) * (1 + np.random.uniform(-0.1, 0.1)),
                })

    return pd.DataFrame(records)


def main():
    """Generate and save all sample data files."""
    output_dir = Path(__file__).parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating sample data...")

    # Generate application data
    print("  Generating application_train.csv...")
    train_df = generate_application_data(n_samples=10000, seed=42)
    train_df.to_csv(output_dir / "application_train.csv", index=False)

    print("  Generating application_test.csv...")
    test_df = generate_application_data(n_samples=2000, seed=123)
    test_df = test_df.drop(columns=["TARGET"])  # No target in test
    test_df.to_csv(output_dir / "application_test.csv", index=False)

    # Get customer IDs
    customer_ids = train_df["SK_ID_CURR"].tolist()

    # Generate bureau data
    print("  Generating bureau.csv...")
    bureau_df = generate_bureau_data(customer_ids, seed=42)
    bureau_df.to_csv(output_dir / "bureau.csv", index=False)

    # Generate previous application data
    print("  Generating previous_application.csv...")
    prev_app_df = generate_previous_application_data(customer_ids, seed=42)
    prev_app_df.to_csv(output_dir / "previous_application.csv", index=False)

    # Generate installments data
    print("  Generating installments_payments.csv...")
    installments_df = generate_installments_data(customer_ids, seed=42)
    installments_df.to_csv(output_dir / "installments_payments.csv", index=False)

    print(f"\nSample data generated in {output_dir}")
    print(f"  application_train.csv: {len(train_df)} rows")
    print(f"  application_test.csv: {len(test_df)} rows")
    print(f"  bureau.csv: {len(bureau_df)} rows")
    print(f"  previous_application.csv: {len(prev_app_df)} rows")
    print(f"  installments_payments.csv: {len(installments_df)} rows")


if __name__ == "__main__":
    main()
