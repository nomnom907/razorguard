import json
import os

import pandas as pd

from risk.fraud_spike_detector import detect_spikes, load_data


RAZORPAY_DATA_PATH = "data/raw/razorpay_payments.json"


def show_razorpay_data():
    print("=" * 70)
    print("RAZORPAY TEST MODE")
    print("=" * 70)

    if not os.path.exists(RAZORPAY_DATA_PATH):
        print("No Razorpay payment data found.")
        print("Run the Razorpay fetcher first.")
        return

    with open(RAZORPAY_DATA_PATH, "r") as file:
        data = json.load(file)

    payments = data.get("items", [])

    print(f"Payment attempts fetched: {len(payments)}")
    print()

    if not payments:
        print("No test payments available.")
        return

    print("Recent payment activity:")

    for payment in payments[:10]:
        amount = payment.get("amount", 0) / 100
        status = payment.get("status", "unknown")
        method = payment.get("method", "unknown")

        print(
            f"  {status.upper():8} | "
            f"Rs. {amount:,.2f} | "
            f"{method}"
        )

    print()


def run_razorguard():
    print("=" * 70)
    print("RAZORGUARD MERCHANT RISK ENGINE")
    print("=" * 70)

    df = load_data()

    results = detect_spikes(df)

    alerts = results[
        results["alert"] == "FRAUD SPIKE"
    ].sort_values(
        "risk_score",
        ascending=False
    )

    print(f"Merchants monitored: {df['merchant_id'].nunique()}")
    print(f"Merchant-hours analyzed: {len(results)}")
    print()

    if alerts.empty:
        print("STATUS: NORMAL")
        return

    top = alerts.iloc[0]

    print("🚨 FRAUD SPIKE DETECTED")
    print()
    print(f"Merchant:           {top['merchant_id']}")
    print(f"Hour:               {top['hour']}:00")
    print(
        f"Transactions:       "
        f"{top['transactions']}"
    )
    print(
        f"Normal baseline:    "
        f"{top['baseline_transactions']}"
    )
    print(
        f"Volume increase:    "
        f"{top['volume_ratio']}x"
    )
    print(
        f"Fraud rate:         "
        f"{top['fraud_rate']}%"
    )
    print(
        f"Normal fraud rate:  "
        f"{top['baseline_fraud_rate']}%"
    )
    print(
        f"Fraud-rate change:  "
        f"{top['fraud_ratio']}x"
    )
    print(
        f"Risk score:         "
        f"{top['risk_score']}/100"
    )
    print()
    print("ACTION: Investigate merchant activity")


def main():
    print()
    print("RAZORGUARD")
    print("Merchant Fraud-Spike Detection System")
    print()

    show_razorpay_data()
    run_razorguard()

    print()
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()