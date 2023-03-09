import numpy as np
import scipy.optimize as sco
import scipy.interpolate as sci


# Capital Market Line
class MarketLineSolver:
    def __init__(self, data, target_range, cml_sd):
        self._data = data
        self._target_range = target_range
        self._cml_sd = cml_sd
        self._min_index = np.argmin(self._data)
        self._efficient_returns = self._target_range[self._min_index:]
        self._efficient_sd = self._data[self._min_index:]
        self._tck = self._calculate_tck()

    # Cubic spline interpolation
    def _calculate_tck(self):
        return sci.splrep(np.sort(self._efficient_sd), self._efficient_returns)

    # Define functional approximation of efficient frontier
    def f(self, x):
        return sci.splev(
            x=x,
            tck=self._tck,
            der=0
        )

    # Define first derivative of the efficient frontier
    def df(self, x):
        return sci.splev(
            x=x,
            tck=self._tck,
            der=1
        )

    # System of equations
    def _sys_eq(self, p, r_f=0.01):
        # Equations
        eq1 = r_f - p[0]
        eq2 = p[1] - self.df(p[2])
        eq3 = r_f + self.df(p[2]) * p[2] - self.f(p[2])
        # Output values
        return eq1, eq2, eq3

        # Sanity check

    def _sanity_check(self, sol_set):
        check = np.round(
            self._sys_eq(
                p=sol_set
            ),
            4
        )
        print(check)

    def solve(self):
        sol_set = sco.fsolve(self._sys_eq, [0.01, 1, 0.15])
        self._sanity_check(sol_set)
        return sol_set

    def cml(self):
        sol_set = self.solve()
        return sol_set[0] + sol_set[1] * self._cml_sd
