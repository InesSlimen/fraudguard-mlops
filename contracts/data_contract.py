from pandera import Check, Column, DataFrameSchema


def get_data_schema() -> DataFrameSchema:
    return DataFrameSchema(
        {
            "TransactionID": Column(int),
            "TransactionAmt": Column(
                float,
                Check.ge(0),
                nullable=False,
            ),
            "ProductCD": Column(
                str,
                Check.isin(["W", "C", "R", "H", "S"]),
                nullable=True,
            ),
            "card1": Column(int),
            "card2": Column(float, nullable=True),
            "card3": Column(float, nullable=True),
            "card4": Column(
                str,
                nullable=True,
            ),
            "card5": Column(float, nullable=True),
            "card6": Column(
                str,
                nullable=True,
            ),
            "addr1": Column(float, nullable=True),
            "addr2": Column(float, nullable=True),
            "P_emaildomain": Column(str, nullable=True),
            "R_emaildomain": Column(str, nullable=True),
            "DeviceType": Column(str, nullable=True),
            "DeviceInfo": Column(str, nullable=True),
            "isFraud": Column(
                int,
                Check.isin([0, 1]),
                nullable=False,
            ),
        },
        strict=False,
    )
