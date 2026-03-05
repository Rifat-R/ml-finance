import pandas as pd
from backend.data import tiingo_client


def download_data(
    ticker: str, start: str = "2020-01-01", end: str | None = None
) -> pd.DataFrame:
    data = tiingo_client.get_dataframe(
        ticker,
        startDate=start,
        endDate=end,
        frequency="daily",
    )

    if not isinstance(data, pd.DataFrame):
        raise RuntimeError(f"Data downloaded for {ticker} is not a DataFrame")

    return data
