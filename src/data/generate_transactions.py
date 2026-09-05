import random
import uuid
from datetime import datetime, timedelta

import pandas as pd


random.seed(42)


NUM_MERCHANTS = 20
HOURS = 24
TRANSACTIONS_PER_HOUR = 50


PAYMENT_METHODS = ["upi", "card", "netbanking", "wallet"]


def generate_transaction(
    transaction_id,
    merchant_id,
    timestamp,
    scenario="normal"
):
    """Generate one synthetic transaction."""

    if scenario == "normal":
        amount = random.uniform(100, 5000)
        is_fraud = 1 if random.random() < 0.02 else 0

    elif scenario == "fraud_spike":
        amount = random.uniform(500, 15000)
        is_fraud = 1 if random.random() < 0.20 else 0

    elif scenario == "legitimate_spike":
        amount = random.uniform(100, 6000)
        is_fraud = 1 if random.random() < 0.02 else 0

    elif scenario == "high_velocity":
        amount = random.uniform(200, 10000)
        is_fraud = 1 if random.random() < 0.15 else 0

    else:
        amount = random.uniform(100, 5000)
        is_fraud = 0

    customer_id = f"C{random.randint(1, 5000):05d}"
    device_id = f"D{random.randint(1, 3000):05d}"

    payment_method = random.choice(PAYMENT_METHODS)

    if is_fraud and scenario in ["fraud_spike", "high_velocity"]:
        transaction_status = random.choices(
            ["success", "failed"],
            weights=[0.65, 0.35]
        )[0]
    else:
        transaction_status = random.choices(
            ["success", "failed"],
            weights=[0.95, 0.05]
        )[0]

    return {
        "transaction_id": transaction_id,
        "merchant_id": merchant_id,
        "timestamp": timestamp,
        "amount": round(amount, 2),
        "payment_method": payment_method,
        "customer_id": customer_id,
        "device_id": device_id,
        "transaction_status": transaction_status,
        "is_fraud": is_fraud,
    }


def generate_dataset():
    rows = []

    start_time = datetime(2026, 1, 1, 0, 0)

    for merchant_number in range(1, NUM_MERCHANTS + 1):

        merchant_id = f"M{merchant_number:03d}"

        for hour in range(HOURS):

            timestamp_base = start_time + timedelta(hours=hour)

            # Different merchants experience different scenarios.
            if merchant_number % 4 == 0 and 12 <= hour <= 14:
                scenario = "fraud_spike"

            elif merchant_number % 4 == 1 and 18 <= hour <= 20:
                scenario = "legitimate_spike"

            elif merchant_number % 4 == 2 and 8 <= hour <= 10:
                scenario = "high_velocity"

            else:
                scenario = "normal"

            transactions_this_hour = TRANSACTIONS_PER_HOUR

            if scenario in ["fraud_spike", "legitimate_spike"]:
                transactions_this_hour = 120

            elif scenario == "high_velocity":
                transactions_this_hour = 150

            for _ in range(transactions_this_hour):

                transaction_id = str(uuid.uuid4())

                seconds_offset = random.randint(0, 3599)

                timestamp = (
                    timestamp_base
                    + timedelta(seconds=seconds_offset)
                )

                row = generate_transaction(
                    transaction_id,
                    merchant_id,
                    timestamp,
                    scenario
                )

                rows.append(row)

    return pd.DataFrame(rows)


def main():
    df = generate_dataset()

    output_path = "data/raw/transactions.csv"

    df.to_csv(output_path, index=False)

    print("Dataset generated successfully.")
    print(f"Rows: {len(df)}")
    print(f"Saved to: {output_path}")
    print()
    print("Fraud distribution:")
    print(df["is_fraud"].value_counts())


if __name__ == "__main__":
    main()