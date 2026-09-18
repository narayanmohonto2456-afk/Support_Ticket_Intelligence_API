import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT_DIR / "artifacts"
RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_target(dataframe: pd.DataFrame, target: str) -> tuple[Pipeline, dict]:
    x_train, x_test, y_train, y_test = train_test_split(
        dataframe["text"],
        dataframe[target],
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=dataframe[target],
    )
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "test_samples": len(y_test),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }
    return pipeline, metrics


def validate_training_data(dataframe: pd.DataFrame) -> None:
    required_columns = {"text", "category", "priority"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    if dataframe[list(required_columns)].isnull().any().any():
        raise ValueError("Training data contains missing values.")

    normalized_text = dataframe["text"].astype(str).str.strip().str.lower()

    empty_text_count = normalized_text.eq("").sum()

    if empty_text_count:
        raise ValueError(f"Training data contains {empty_text_count} empty tickets.")

    duplicate_count = normalized_text.duplicated().sum()

    if duplicate_count:
        raise ValueError(f"Training data contains {duplicate_count} duplicate tickets.")

    for target in ("category", "priority"):
        class_counts = dataframe[target].value_counts()

        if len(class_counts) < 2:
            raise ValueError(f"The '{target}' target needs at least two classes.")

        if class_counts.min() < 2:
            raise ValueError(
                f"Every '{target}' class needs at least two examples. "
                f"Current counts: {class_counts.to_dict()}"
            )

    if len(dataframe) < 200:
        warnings.warn(
            f"The dataset contains only {len(dataframe)} tickets. "
            "Training is allowed for development, but the model evaluation "
            "is not reliable yet. Aim for at least 200 tickets.",
            stacklevel=2,
        )


def train_models() -> dict:
    data_path = ROOT_DIR / "data" / "tickets_v2.csv"
    dataframe = pd.read_csv(data_path)

    validate_training_data(dataframe)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = {
        "dataset": {
            "samples": len(dataframe),
            "source": "manually_labeled_v2",
            "file": data_path.name,
        }
    }

    for target in ("category", "priority"):
        model, metrics = train_target(dataframe, target)

        model_path = ARTIFACT_DIR / f"{target}_model.joblib"
        joblib.dump(model, model_path)

        all_metrics[target] = metrics

    metrics_path = ARTIFACT_DIR / "metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)

    return all_metrics


if __name__ == "__main__":
    results = train_models()
    print(json.dumps(results, indent=2))
