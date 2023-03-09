import constants
import numpy as np
import pandas as pd

from finance import portfolio_returns
from monte_carlo import MonteCarlo
from solver import MarketLineSolver
from optimizer import Optimizer, EfficientFrontier
from visuazlization import display

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# Monte Carlo Simulation
monte_carlo = MonteCarlo(data=constants.daily_returns, n_iter=3000, symbols=constants.SYMBOLS)

port_sd = monte_carlo.get_portfolio_standard_deviation_df()
port_returns = monte_carlo.get_portfolio_return()

monte_carlo.print_min_risk()
monte_carlo.print_highest_return()
monte_carlo.print_highest_sharpe_ratio()

# The Optimal Portfolios

# Maximum Sharpe Ratio
max_sharpe_optimizer = Optimizer(constants.MAX_SHARPE_DICT)
max_sharpe_port_return, max_sharpe_port_sd, max_sharpe_port_sharpe, max_sharpe_port_weights = max_sharpe_optimizer.create_metrics()
max_sharpe_optimizer.print_metrics("Max Sharpe Optimized Portfolio", max_sharpe_port_return, max_sharpe_port_sd, max_sharpe_port_sharpe, max_sharpe_port_weights)

# Minimum Variance Portfolio
# Minimize sd
min_sd_optimizer = Optimizer(constants.MIN_VOLATILITY_DICT)
min_sd_port_return, min_sd_port_sd, min_sd_port_sharpe, min_sd_port_weights = min_sd_optimizer.create_metrics()
min_sd_optimizer.print_metrics("Minimal Volatility Portfolio", min_sd_port_return, min_sd_port_sd, min_sd_port_sharpe, min_sd_port_weights)

# Efficient frontier
efficient_frontier = EfficientFrontier(constants.EFFICIENT_FRONTIER_DICT)
obj_sd = efficient_frontier.create_obj_sd()

# Capital Market Line
solver = MarketLineSolver(data=obj_sd, target_range=constants.TARGET_RANGE, cml_sd=constants.CML_SD)
sol_set = solver.solve()

# Tangency Portfolio
# Constraints
# The target return is now f(x) where x is taken from the solution set above
constants.TANGENCY_DICT["constraints"] = (
    {'type': 'eq', 'fun': lambda x: portfolio_returns(x, constants.daily_returns) - solver.f(x=sol_set[2])},
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
)
# Optimize
tangency_optimizer = Optimizer(constants.TANGENCY_DICT)
tangency_port_return, tangency_port_sd, tangency_port_sharpe, tangency_port_weights = tangency_optimizer.create_metrics()
tangency_optimizer.print_metrics("Tangency Portfolio", tangency_port_return, tangency_port_sd, tangency_port_sharpe, tangency_port_weights)

# Capital market line
cml_exp_returns = solver.cml()

# Plot result
display(min_sd_port_sd, min_sd_port_return, max_sharpe_port_sd, tangency_port_sd, tangency_port_return, port_sd, port_returns, obj_sd, cml_exp_returns)


