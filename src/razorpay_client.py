import os

import razorpay
from dotenv import load_dotenv


load_dotenv()


def create_client():
    key_id = os.getenv("RAZORPAY_KEY_ID")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET")

    if not key_id or not key_secret:
        raise RuntimeError(
            "Razorpay API credentials are missing. "
            "Check your .env file."
        )

    return razorpay.Client(
        auth=(key_id, key_secret)
    )