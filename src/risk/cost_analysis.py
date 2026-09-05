import pandas as pd


DATA_PATH = "data/raw/transactions.csv"

# Illustrative operational cost for investigating one false alarm.
# This is an assumption for the benchmark, not a Razorpay fee.
FALSE_POSITIVE_COST = 50


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    return df


def evaluate_cost(df):

    hourly = (
        df.groupby(["merchant_id", "hour"])
        .agg(
            transaction_count=("transaction_id", "count"),
            fraud_count=("is_fraud", "sum"),
        )
        .reset_index()
    )

    # Calculate total fraudulent transaction value per merchant-hour.
    fraud_amounts = (
        df[df["is_fraud"] == 1]
        .groupby(["merchant_id", "hour"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "fraud_amount"})
    )

    hourly = hourly.merge(
        fraud_amounts,
        on=["merchant_id", "hour"],
        how="left"
    )

    hourly["fraud_amount"] = (
        hourly["fraud_amount"].fillna(0)
    )

    hourly["fraud_rate"] = (
        hourly["fraud_count"] /
        hourly["transaction_count"]
    )

    predictions = []

    for merchant in hourly["merchant_id"].unique():

        merchant_data = hourly[
            hourly["merchant_id"] == merchant
        ]

        baseline_volume = (
            merchant_data["transaction_count"].median()
        )

        baseline_fraud_rate = (
            merchant_data["fraud_rate"].median()
        )

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

            # Benchmark definition:
            # fraud rate >= 10% = actual fraud spike.
            actual_spike = row["fraud_rate"] >= 0.10

            predictions.append(
                {
                    "predicted": predicted_spike,
                    "actual": actual_spike,
                    "fraud_amount": row["fraud_amount"],
                }
            )

    results = pd.DataFrame(predictions)

    false_positives = results[
        (results["predicted"] == True) &
        (results["actual"] == False)
    ]

    false_negatives = results[
        (results["predicted"] == False) &
        (results["actual"] == True)
    ]

    fp_count = len(false_positives)
    fn_count = len(false_negatives)

    false_positive_cost = (
        fp_count * FALSE_POSITIVE_COST
    )

    missed_fraud_value = (
        false_negatives["fraud_amount"].sum()
    )

    total_cost = (
        false_positive_cost +
        missed_fraud_value
    )

    print("=" * 60)
    print("RAZORGUARD FALSE-POSITIVE COST ANALYSIS")
    print("=" * 60)

    print()
    print(f"False positives: {fp_count}")
    print(
        f"False-positive investigation cost: "
        f"Rs. {false_positive_cost:,.2f}"
    )

    print()
    print(f"False negatives: {fn_count}")
    print(
        f"Fraud value missed: "
        f"Rs. {missed_fraud_value:,.2f}"
    )

    print()
    print(
        f"Estimated total benchmark cost: "
        f"Rs. {total_cost:,.2f}"
    )

    print()
    print("NOTE:")
    print(
        "Rs. 50 per false positive is an illustrative "
        "operational-cost assumption."
    )
    print(
        "Missed fraud value is calculated from the "
        "synthetic dataset."
    )


if __name__ == "__main__":
    df = load_data()
    evaluate_cost(df)