import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.features.preprocess import get_preprocessor
from src.models.evaluate import evaluate_model
from src.utils.config import (
    MAX_ITER,
    MODEL_NAME,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)


def load_training_data():
    """
    Load processed train and validation data.
    """
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    X_val = pd.read_csv(PROCESSED_DATA_DIR / "X_val.csv")

    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").squeeze()
    y_val = pd.read_csv(PROCESSED_DATA_DIR / "y_val.csv").squeeze()

    return X_train, X_val, y_train, y_val


def build_model_pipeline():
    """
    Build a preprocessing + model pipeline.
    """
    preprocessor = get_preprocessor()

    model = LogisticRegression(
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def save_metrics(metrics: dict, output_path: Path):
    """
    Save metrics to a JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_metrics = {
        key: value.tolist() if hasattr(value, "tolist") else value
        for key, value in metrics.items()
    }

    with open(output_path, "w") as f:
        json.dump(serializable_metrics, f, indent=4)


def save_markdown_report(metrics: dict, output_path: Path):
    """
    Save a human-readable markdown report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = metrics["confusion_matrix"]

    report = f"""# Baseline Logistic Regression Results

## Metrics
- Accuracy: {metrics["accuracy"]:.4f}
- Precision: {metrics["precision"]:.4f}
- Recall: {metrics["recall"]:.4f}
- F1: {metrics["f1"]:.4f}
- ROC-AUC: {metrics["roc_auc"]:.4f}
- Training Time (seconds): {metrics["training_time_seconds"]:.4f}

## Confusion Matrix
{cm}

## Interpretation
This is the first baseline model for the loan risk project.
We will compare it with tree-based models next.
"""

    with open(output_path, "w") as f:
        f.write(report)


def main():
    # 1. Load data
    X_train, X_val, y_train, y_val = load_training_data()

    # 2. Build pipeline
    pipeline = build_model_pipeline()

    # 3. Start timer
    start_time = time.perf_counter()

    # 4. Train model
    pipeline.fit(X_train, y_train)

    # 5. Stop timer
    end_time = time.perf_counter()
    training_time = end_time - start_time

    # 6. Predict on validation set
    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    # 7. Evaluate
    metrics = evaluate_model(y_val, y_pred, y_prob)
    metrics["training_time_seconds"] = training_time

    print("\nValidation Metrics:")
    for key, value in metrics.items():
        if key not in ["confusion_matrix", "classification_report"]:
            print(f"{key}: {value}")

    # 8. Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{MODEL_NAME}.joblib"
    joblib.dump(pipeline, model_path)

    # 9. Save metrics
    metrics_path = REPORTS_DIR / f"{MODEL_NAME}_results.json"
    save_metrics(metrics, metrics_path)

    # 10. Save markdown report
    report_path = REPORTS_DIR / f"{MODEL_NAME}_results.md"
    save_markdown_report(metrics, report_path)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved markdown report to: {report_path}")
    print(f"Training Time: {training_time:.4f} seconds")


if __name__ == "__main__":
    main()