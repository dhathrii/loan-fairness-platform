import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.load_data import load_data
from src.features.preprocess import get_preprocessor
from src.models.evaluate import evaluate_model
from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR


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

    model = LogisticRegression(max_iter=1000, random_state=42)

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


def main():
    # 1. Load data
    X_train, X_val, y_train, y_val = load_training_data()

    # 2. Build pipeline
    pipeline = build_model_pipeline()

    # 3. Train model
    pipeline.fit(X_train, y_train)

    # 4. Predict on validation set
    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    # 5. Evaluate
    metrics = evaluate_model(y_val, y_pred, y_prob)

    print("\nValidation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # 6. Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "logistic_baseline.joblib"
    joblib.dump(pipeline, model_path)

    # 7. Save metrics
    metrics_path = REPORTS_DIR / "baseline_results.json"
    save_metrics(metrics, metrics_path)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()