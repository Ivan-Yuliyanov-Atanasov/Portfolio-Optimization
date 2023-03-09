import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


class DataService:
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.asset_data = pd.DataFrame()
        self._post_init()

    def _post_init(self):
        for sym in self.symbols:
            # Each run of the loop returns a pandas data frame
            self.asset_data[sym] = (yf.download(sym, self.start_date, self.end_date))['Adj Close']
        # Set column indices
        self.asset_data.columns = self.symbols

    def get_daily_returns(self):
        return (
            self.asset_data.pct_change()
            .dropna(
                # Drop the first row since we have NaN's
                # The first date 2013-03-01 does not have a value since it is our cut-off date
                axis=0,
                how='any',
                inplace=False
            )
        )

    def plot_daily_returns(self):
        normalised = self.asset_data / self.asset_data.iloc[0] * 100
        normalised.plot(figsize=(10, 5))
        plt.title('Stock Time Series 2013 - 2023', fontsize=16)
        plt.show()



