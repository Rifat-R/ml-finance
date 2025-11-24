from typing import Optional

import pandas as pd
import yfinance as yf

from .features import clean_name, create_features


def download_data(
    ticker: str, start: str = "2015-01-01", end: Optional[str] = None
) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end)
    if data is None or data.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")

    # Ensure columns are strings and cleaned
    data.columns = [clean_name(c) for c in data.columns]
    return data
