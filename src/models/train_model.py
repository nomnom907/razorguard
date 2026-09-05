import os

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "data/raw/transactions.csv"
MODEL_PATH = "src/models/fraud_model.joblib"


def load_data():
    df = pd.read_csv(DATA_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Amount-based feature
    df["amount_log"] = (df["amount"] + 1).apply(lambda x: __import__("math").log(x))

    features = [
        "amount",
        "amount_log",
        "hour",
        "day_of_week",
        "payment_method",
        "transaction_status",
    ]

    X = df[features]
    y = df["is_fraud"]

    return X, y


def train():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    categorical_features = [
        "payment_method",
        "transaction_status",
    ]

    numeric_features = [
        "amount",
        "amount_log",
        "hour",
        "day_of_week",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    print("=" * 50)
    print("RAZORGUARD MODEL EVALUATION")
    print("=" * 50)

    print(f"Training samples: {len(X_train)}")
    print(f"Held-out test samples: {len(X_test)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    os.makedirs("src/models", exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()