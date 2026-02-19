from typing import Optional

import pandas as pd

from tiingo import TiingoClient

from dotenv import load_dotenv
import os

load_dotenv()

TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")

config = {}
config["session"] = True
config["api_key"] = TIINGO_API_KEY
tiingo_client = TiingoClient()


# TODO: Use "incremental loading", where we store the last time we
# download data and use that for start day
# May be an issue due to adjusted close column
def download_data(
    ticker: str, start: str = "2020-01-01", end: Optional[str] = None
) -> pd.DataFrame:
    data = tiingo_client.get_dataframe(
        ticker,
        startDate=start,
        endDate=end,
        frequency="daily",
    )
    if data is None or data.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")

    return data  # type: ignore
