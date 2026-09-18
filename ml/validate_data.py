from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/tickets_v2.csv")

VALID_CATEGORIES = {
    "account_access",
    "billing",
    "technical_issue",
    "feature_request",
}

VALID_PRIORITIES = {
    "low",
    "medium",
    "high",
    "critical",
}


def validate_dataset() -> None:
    df = pd.read_csv(DATA_PATH)

    required_columns = {"text", "category", "priority"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if df[list(required_columns)].isnull().any().any():
        raise ValueError("The dataset contains missing values.")

    duplicate_count = df["text"].str.lower().duplicated().sum()

    invalid_categories = set(df["category"]) - VALID_CATEGORIES
    invalid_priorities = set(df["priority"]) - VALID_PRIORITIES

    if invalid_categories:
        raise ValueError(f"Invalid categories: {invalid_categories}")

    if invalid_priorities:
        raise ValueError(f"Invalid priorities: {invalid_priorities}")

    short_tickets = df[df["text"].str.len() < 5]

    print(f"Rows: {len(df)}")
    print(f"Duplicate tickets: {duplicate_count}")
    print(f"Very short tickets: {len(short_tickets)}")
    print("\nCategory distribution:")
    print(df["category"].value_counts())
    print("\nPriority distribution:")
    print(df["priority"].value_counts())


if __name__ == "__main__":
    validate_dataset()