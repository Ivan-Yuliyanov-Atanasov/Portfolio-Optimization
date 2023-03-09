import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from finance import portfolio_returns, portfolio_sd


class MonteCarlo:
    def __init__(self, data, n_iter, symbols):
        self._RF = 0
        self._symbols = symbols
        self._number_of_portfolios = n_iter
        self._list_portfolio_returns = []
        self._list_portfolio_risks = []
        self._list_portfolio_weights = []
        self._sharpe_ratios = []
        self._daily_returns = data
        self._post_init()

    def _post_init(self):
        for portfolio in range(self._number_of_portfolios):
            # Generate random portfolio weights
            weights = self._generate_weights()
            self._list_portfolio_weights.append(weights)
            # Generate annualized return
            annualized_return = portfolio_returns(weights, self._daily_returns)
            self._list_portfolio_returns.append(annualized_return)
            # Matrix of covariance and risk calculation
            portfolio_standard_deviation = portfolio_sd(weights, self._daily_returns)
            self._list_portfolio_risks.append(portfolio_standard_deviation)
            # Sharpe ratio
            sharpe_ratio = (annualized_return - self._RF) / portfolio_standard_deviation
            self._sharpe_ratios.append(sharpe_ratio)

        self._create_portfolio_df()

    def _create_portfolio_df(self):
        portfolio_metrics = [self._list_portfolio_returns, self._list_portfolio_risks, self._sharpe_ratios,
                             self._list_portfolio_weights]
        self.portfolio_df = pd.DataFrame(portfolio_metrics).T
        self.portfolio_df.columns = ['Return', 'Risk', 'Sharpe', 'Weights']

    def _generate_weights(self):
        weights = np.random.random_sample(len(self._symbols))
        weights = np.round(weights / np.sum(weights), 3)
        return weights

    def get_portfolio_standard_deviation_df(self):
        return np.array(self._list_portfolio_risks)

    def get_portfolio_return(self):
        return np.array(self._list_portfolio_returns)

    def _print_metrics(self, df, name):
        print(
            f"{name}\n\tReturn: {np.round(df[0], 4)}\n\tRisk: {np.round(df[1], 4)}\n\tSharpe ratio: {np.round(df[2], 4)}\n\tWeights: {list(zip(self._symbols, list(np.round(df[3], 3))))}")

    # plot_monte_carlo(port_sd, port_returns, number_of_portfolios)
    def print_min_risk(self):
        df = self.portfolio_df.iloc[self.portfolio_df['Risk'].astype(float).idxmin()]
        self._print_metrics(df, "Min Volatility Portfolio")

    def print_highest_return(self):
        df = self.portfolio_df.iloc[self.portfolio_df['Return'].astype(float).idxmax()]
        self._print_metrics(df, "Highest Return Portfolio")

    def print_highest_sharpe_ratio(self):
        df = self.portfolio_df.iloc[self.portfolio_df['Sharpe'].astype(float).idxmax()]
        self._print_metrics(df, "Highest Sharpe Ratio Portfolio")

    def plot_monte_carlo(self):
        ptf_stds = self.get_portfolio_standard_deviation_df()
        ptf_rs = self.get_portfolio_return()

        plt.figure(figsize=(12, 6))
        plt.scatter(ptf_stds, ptf_rs, c=ptf_rs / ptf_stds, marker='o')
        plt.grid(True)
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.colorbar(label='Sharpe Ratio')
        plt.title(f'{self._number_of_portfolios} Randomly Generated Portfolios In The Risk-Return Space')
        plt.show()
