import pandas as pd


DATA_PATH = "data/raw/transactions.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    return df


def detect_spikes(df):
    hourly = (
        df.groupby(["merchant_id", "hour"])
        .agg(
            transaction_count=("transaction_id", "count"),
            fraud_count=("is_fraud", "sum"),
            average_amount=("amount", "mean"),
        )
        .reset_index()
    )

    hourly["fraud_rate"] = (
        hourly["fraud_count"] / hourly["transaction_count"]
    )

    results = []

    for merchant in hourly["merchant_id"].unique():

        merchant_data = hourly[
            hourly["merchant_id"] == merchant
        ].copy()

        baseline_volume = merchant_data["transaction_count"].median()
        baseline_fraud_rate = merchant_data["fraud_rate"].median()

        for _, row in merchant_data.iterrows():

            volume_ratio = (
                row["transaction_count"] / baseline_volume
            )

            if baseline_fraud_rate == 0:
                fraud_ratio = 1.0 if row["fraud_rate"] == 0 else 2.0
            else:
                fraud_ratio = (
                    row["fraud_rate"] / baseline_fraud_rate
                )

            # Cap each component separately.
            volume_score = min(
                max((volume_ratio - 1) / 2, 0),
                1
            )

            fraud_score = min(
                max((fraud_ratio - 1) / 5, 0),
                1
            )

            # Fraud behavior is more important than volume alone.
            risk_score = (
                0.35 * volume_score +
                0.65 * fraud_score
            ) * 100

            if risk_score >= 70:
                alert = "FRAUD SPIKE"
            elif risk_score >= 40:
                alert = "SUSPICIOUS"
            else:
                alert = "NORMAL"

            results.append(
                {
                    "merchant_id": merchant,
                    "hour": int(row["hour"]),
                    "transactions": int(row["transaction_count"]),
                    "baseline_transactions": round(
                        baseline_volume, 2
                    ),
                    "volume_ratio": round(
                        volume_ratio, 2
                    ),
                    "fraud_rate": round(
                        row["fraud_rate"] * 100, 2
                    ),
                    "baseline_fraud_rate": round(
                        baseline_fraud_rate * 100, 2
                    ),
                    "fraud_ratio": round(
                        fraud_ratio, 2
                    ),
                    "risk_score": round(
                        risk_score, 2
                    ),
                    "alert": alert,
                }
            )

    return pd.DataFrame(results)


def main():
    df = load_data()
    results = detect_spikes(df)

    print("=" * 75)
    print("RAZORGUARD MERCHANT FRAUD-SPIKE DETECTOR")
    print("=" * 75)

    print(f"Merchants analyzed: {df['merchant_id'].nunique()}")
    print(f"Hours analyzed: {df['hour'].nunique()}")
    print()

    alerts = results[
        results["alert"] != "NORMAL"
    ].sort_values(
        "risk_score",
        ascending=False
    )

    if alerts.empty:
        print("No suspicious activity detected.")
        return

    print("Detected alerts:")
    print()

    print(
        alerts.head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()