import pandas as pd
import numpy as np
import warnings

class ETFTrick:
    """
    A utility class to transform a weighted basket of assets into a "virtual ETF"
    by applying weights to asset returns instead of price levels.

    Parameters
    ----------
    prices : np.ndarray or pd.DataFrame
        Asset price matrix of shape (T, N). If DataFrame, keeps index/columns metadata.
    weights : np.ndarray or pd.Series
        Portfolio weights of length N. Can be normalized automatically.
    normalize_weights : bool, default=True
        If True, normalize weights to sum to 1 when they do not.
    """

    def __init__(self, prices, weights, normalize_weights=True):
        
        self._prices_type = type(prices)
        if not isinstance (prices, (np.ndarray, pd.DataFrame)):
            raise ValueError("Prices must be NumPy ndarray or Pandas DataFrame")
            
        self._prices = prices
        if isinstance(prices, pd.DataFrame):
            self._columns_names = prices.columns
            self._index = prices.index
            self._prices = prices.values

        self._weights = weights
        if not isinstance (weights, (np.ndarray, pd.Series)):
            raise ValueError("Weights must be Numpy ndarray or Pandas Series")
        if isinstance(weights, pd.Series):
            self._weights = weights.values

        if not self._weights.shape[0] == self._prices.shape[-1]:
            raise ValueError("Weights and prices shapes does not match")

        total_weights = self._weights.sum()
        if not np.isclose(total_weights, 1.0):
            if normalize_weights:
                self._weights = self._weights / total_weights
                warnings.warn(
                    f"Weights normalized (sum was {total_weights:.4f}). "
                    f"New weights: {np.array2string(self._weights, precision=4, separator=', ', threshold=10)}",
                    UserWarning
                )
            else:
                warnings.warn(
                    f"Sum of weights is {total_weights:.4f} (not 1.0)", UserWarning
                )
        

    def fit(self):
        """
        Compute asset returns, ETF returns, and cumulative ETF returns
        based on the provided prices and weights.
        """
        self._returns = (self._prices[1:] / self._prices[:-1]) - 1
        if self._prices_type == pd.DataFrame:
            self._cumulative_returns = (pd.DataFrame(
                index = self._index[1:],
                columns = self._columns_names,
                data = self._returns
            ) + 1).cumprod().values
        else:
            self._cumulative_returns = np.cumprod(self._returns + 1, axis=0)
        self._etf_returns = self._returns @ self._weights
        self._etf_cumulative_returns = np.cumprod(self._etf_returns+1)
        

    @property
    def etf_returns(self):
        """
        Daily virtual ETF returns.

        Returns
        -------
        np.ndarray or pd.Series
            Array (or Series if DataFrame input) of length T-1.
        """
        if not hasattr(self, '_etf_returns'):
            raise KeyError('No ETF returns found')
        if self._prices_type == pd.DataFrame:
            return pd.Series(
                data = self._etf_returns,
                index = self._index[1:]
            )
        return self._etf_returns
        

    @property
    def etf_cumulative_returns(self):
        """
        Cumulative virtual ETF value series (starts at 1).

        Returns
        -------
        np.ndarray or pd.Series
            Array (or Series if DataFrame input) of length T-1.
        """
        if not hasattr(self, '_etf_cumulative_returns'):
            raise KeyError('No ETF cumulative returns found')
        if self._prices_type == pd.DataFrame:
            return pd.Series(
                data = self._etf_cumulative_returns,
                index = self._index[1:]
            )
        return self._etf_cumulative_returns
        

    @property
    def returns(self):
        """
        Daily asset returns.

        Returns
        -------
        np.ndarray or pd.DataFrame
            Matrix of daily returns (T-1, N). If DataFrame input, preserves index/columns.
        """
        if not hasattr(self, '_returns'):
            raise KeyError('No returns found')
        if self._prices_type == pd.DataFrame:
            return pd.DataFrame(
                index = self._index[1:],
                columns = self._columns_names,
                data = self._returns
            )
        return self._returns

    @property
    def cumulative_returns(self):
        """
        Cumulative asset returns.

        Returns
        -------
        np.ndarray or pd.DataFrame
            Matrix of cumulative returns (T-1, N). If DataFrame input, preserves index/columns.
        """
        if not hasattr(self, '_cumulative_returns'):
            raise KeyError('No cumulative returns found')
        if self._prices_type == pd.DataFrame:
            return pd.DataFrame(
                index = self._index[1:],
                columns = self._columns_names,
                data = self._cumulative_returns
            )
        return self._cumulative_returns

        
    @property
    def columns_names(self):
        """
        Asset column names if DataFrame input.

        Returns
        -------
        list-like
        """
        if not hasattr(self, '_columns_names'):
            raise KeyError('No columns names found')
        return self._columns_names