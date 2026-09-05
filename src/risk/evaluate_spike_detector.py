import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix


DATA_PATH = "data/raw/transactions.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    return df


def build_hourly_data(df):
    hourly = (
        df.groupby(["merchant_id", "hour"])
        .agg(
            transaction_count=("transaction_id", "count"),
            fraud_count=("is_fraud", "sum"),
        )
        .reset_index()
    )

    hourly["fraud_rate"] = (
        hourly["fraud_count"] /
        hourly["transaction_count"]
    )

    return hourly


def evaluate():
    df = load_data()
    hourly = build_hourly_data(df)

    results = []

    for merchant in hourly["merchant_id"].unique():

        merchant_data = hourly[
            hourly["merchant_id"] == merchant
        ].copy()

        baseline_volume = merchant_data[
            "transaction_count"
        ].median()

        baseline_fraud_rate = merchant_data[
            "fraud_rate"
        ].median()

        for _, row in merchant_data.iterrows():

            volume_ratio = (
                row["transaction_count"] /
                baseline_volume
            )

            if baseline_fraud_rate == 0:
                fraud_ratio = (
                    1.0
                    if row["fraud_rate"] == 0
                    else 2.0
                )
            else:
                fraud_ratio = (
                    row["fraud_rate"] /
                    baseline_fraud_rate
                )

            volume_score = min(
                max((volume_ratio - 1) / 2, 0),
                1
            )

            fraud_score = min(
                max((fraud_ratio - 1) / 5, 0),
                1
            )

            risk_score = (
                0.35 * volume_score +
                0.65 * fraud_score
            ) * 100

            predicted_spike = risk_score >= 70

            # Ground truth:
            # A merchant-hour is considered a real fraud spike
            # when the fraud rate is at least 10%.
            actual_spike = row["fraud_rate"] >= 0.10

            results.append(
                {
                    "merchant_id": merchant,
                    "hour": int(row["hour"]),
                    "risk_score": risk_score,
                    "predicted_spike": predicted_spike,
                    "actual_spike": actual_spike,
                }
            )

    results_df = pd.DataFrame(results)

    y_true = results_df["actual_spike"]
    y_pred = results_df["predicted_spike"]

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    print("=" * 60)
    print("RAZORGUARD FRAUD-SPIKE EVALUATION")
    print("=" * 60)

    print(f"Merchant-hours evaluated: {len(results_df)}")
    print()
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    print()
    print("Confusion Matrix:")
    print(matrix)

    print()
    print("Interpretation:")
    print(f"True fraud spikes: {y_true.sum()}")
    print(f"Detected fraud spikes: {y_pred.sum()}")
    print(
        f"False alarms: "
        f"{((y_pred == 1) & (y_true == 0)).sum()}"
    )
    print(
        f"Missed fraud spikes: "
        f"{((y_pred == 0) & (y_true == 1)).sum()}"
    )


if __name__ == "__main__":
    evaluate()