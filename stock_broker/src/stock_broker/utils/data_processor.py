import os
import pandas as pd
def save_intraday(symbol: str, series: dict):
    """append latest intraday bars to csv,
    avoiding duplicates."""
    df = pd.Dataframe.from_dict(series, orient = "index")
    df.index.name = "timestamp"
    df.columns = ["open","high","low","close","volume"]
    path = os.path.join("data", "historical", f"{symbol}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        existing = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
        df = df[~df.index.isin(existing.index)]
        df = pd.concat([existing, df])
    df.sort_index(inplace=True)
    df.to_csv(path)
    return len(df)