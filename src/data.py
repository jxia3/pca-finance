from dataclasses import dataclass
import numpy as np

from src.config import DAILY_RETURNS_FILE, TICKERS_FILE

@dataclass(kw_only=True)
class DailyReturns:
    dates: np.ndarray
    stock_ids: np.ndarray
    returns: np.ndarray

def load_daily_returns() -> DailyReturns:
    with open(DAILY_RETURNS_FILE) as file:
        header = file.readline().strip().split(",")
    stock_ids = np.array(header[1:], dtype=np.int32)

    dates = np.loadtxt(
        fname=DAILY_RETURNS_FILE,
        dtype=np.datetime64,
        delimiter=",",
        skiprows=1,
        usecols=0,
    )
    returns = np.loadtxt(
        fname=DAILY_RETURNS_FILE,
        dtype=np.float64,
        delimiter=",",
        skiprows=1,
        usecols=range(1, len(header)),
    )

    return DailyReturns(
        dates=dates,
        stock_ids=stock_ids,
        returns=returns,
    )

def load_tickers() -> dict[int, str]:
    tickers = {}
    with open(TICKERS_FILE) as file:
        next(file)
        for line in file.readlines():
            parts = line.strip().split(",")
            tickers[int(parts[0])] = parts[1]
    return tickers