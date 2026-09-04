from razorpay_client import create_client


def main():
    client = create_client()

    orders = client.order.all({
        "count": 1
    })

    print("Razorpay API connection successful.")
    print(orders)


if __name__ == "__main__":
    main()