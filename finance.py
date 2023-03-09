import numpy as np


def portfolio_returns(weights, daily_returns):
    return (np.sum(daily_returns.mean() * weights)) * 253


def portfolio_sd(weights, daily_returns):
    return np.sqrt(np.dot(weights.T, np.dot(daily_returns.cov() * 252, weights)))


# Negative sign to compute the negative value of Sharpe ratio
def sharpe_fun(weights, daily_returns):
    return - (portfolio_returns(weights, daily_returns) / portfolio_sd(weights, daily_returns))