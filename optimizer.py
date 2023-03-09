from abc import ABC, abstractmethod
import numpy as np
import scipy.optimize as sco

from finance import portfolio_returns, portfolio_sd
import constants


class Minimizer(ABC):

    @abstractmethod
    def __init__(self, params_dict):
        self._params_dict = params_dict

    def _minimize(self):
        return sco.minimize(
            # Objective function
            fun=self._params_dict["fun"],
            # Initial guess, which is the equal weight array
            x0=self._params_dict["equal_weights"],
            args=self._params_dict["args"],
            method='SLSQP',
            bounds=self._params_dict["bounds"],
            constraints=self._params_dict["constraints"]
        )


class Optimizer(Minimizer):
    def __init__(self, params_dict):
        super().__init__(params_dict)
        self._data = self._minimize()

    def create_metrics(self):
        port_weights = self._data["x"]
        port_return = round(portfolio_returns(port_weights, constants.daily_returns), 4)
        port_sd = round(portfolio_sd(port_weights, constants.daily_returns), 4)
        port_sharpe = round(port_return / port_sd, 4)
        asset_weight = list(zip(constants.SYMBOLS, list(np.round(port_weights, 3))))
        return port_return, port_sd, port_sharpe, asset_weight

    @staticmethod
    def print_metrics(name, port_return, port_sd, port_sharpe, asset_weight):
        print(
            f"{name}\n\tReturn: {port_return}\n\tRisk: {port_sd}\n\tSharpe ratio: {port_sharpe}\n\tWeights: {asset_weight}")


class EfficientFrontier(Minimizer):
    def __init__(self, params_dict):
        super().__init__(params_dict)
        self.obj_sd = []

    def create_obj_sd(self):
        for target in constants.TARGET_RANGE:
            self._params_dict["constraints"] = (
                {'type': 'eq', 'fun': lambda x: portfolio_returns(x, constants.daily_returns) - target},
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            )
            min_result_object = self._minimize()

            # Extract the objective value and append it to the output container
            self.obj_sd.append(min_result_object['fun'])

        return np.array(self.obj_sd)
