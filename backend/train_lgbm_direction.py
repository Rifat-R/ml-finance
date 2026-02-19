from typing import Optional

import pandas as pd

from backend.data import tiingo_client


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
