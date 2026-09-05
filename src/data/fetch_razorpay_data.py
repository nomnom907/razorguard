import json
import os

from dotenv import load_dotenv
import razorpay


load_dotenv()


def create_client():
    key_id = os.getenv("RAZORPAY_KEY_ID")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET")

    if not key_id or not key_secret:
        raise RuntimeError("Razorpay credentials not found.")

    return razorpay.Client(
        auth=(key_id, key_secret)
    )


def fetch_payments():
    client = create_client()

    payments = client.payment.all({
        "count": 100
    })

    return payments


def main():
    payments = fetch_payments()

    os.makedirs("data/raw", exist_ok=True)

    with open(
        "data/raw/razorpay_payments.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(payments, file, indent=2)

    print("Razorpay payment data fetched successfully.")
    print(f"Payments found: {payments.get('count', 0)}")
    print("Saved to: data/raw/razorpay_payments.json")


if __name__ == "__main__":
    main()