import numpy as np
import datetime as dt
from data_service import DataService
from finance import portfolio_returns, portfolio_sd, sharpe_fun

SYMBOLS = [
    "AAPL", "BRK-A", "COST", "GOOGL", "KO", "MCD", "MSFT", "VZ"
]

START = dt.datetime(2013, 3, 1)
END = dt.datetime(2023, 3, 1)

data_service = DataService(symbols=SYMBOLS, start_date=START, end_date=END)
daily_returns = data_service.get_daily_returns()

TARGET_RANGE = np.linspace(
    start=0.10,
    stop=0.30,
    num=100
)

CML_SD = np.linspace(
    start=0,
    stop=0.30,
    num=100
)

MAX_SHARPE_DICT = {
    "fun": sharpe_fun,
    "constraints": ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}),
    "bounds": tuple(
        (0, 1) for w in range(len(SYMBOLS))
    ),
    "equal_weights": np.array(
        [1 / len(SYMBOLS)] * len(SYMBOLS)
    ),
    "args": daily_returns
}

MIN_VOLATILITY_DICT = {
    "fun": portfolio_sd,
    "constraints": ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}),
    "bounds": tuple(
        (0, 1) for w in range(len(SYMBOLS))
    ),
    "equal_weights": np.array(
        [1 / len(SYMBOLS)] * len(SYMBOLS)
    ),
    "args": daily_returns
}

EFFICIENT_FRONTIER_DICT = {
    "fun": portfolio_sd,
    "bounds": tuple(
        (0, 1) for w in range(len(SYMBOLS))
    ),
    "equal_weights": np.array(
        [1 / len(SYMBOLS)] * len(SYMBOLS)
    ),
    "args": daily_returns
}

TANGENCY_DICT = {
    "fun": portfolio_sd,
    "bounds": tuple(
        (0, 1) for w in range(len(SYMBOLS))
    ),
    "equal_weights": np.array(
        [1 / len(SYMBOLS)] * len(SYMBOLS)
    ),
    "args": daily_returns
}