import pandas as pd
import numpy as np

class CumSumFilter:
    """
    Cumulative Sum (CUSUM) filter for event-based sampling of time series.

    Detects significant positive/negative cumulative changes in log-returns 
    exceeding a given threshold `h`. Returns the indices (or timestamps, if available) 
    of detected events.
    """

    def __init__(self, log_returns, h):
        """
        Parameters
        ----------
        log_returns : np.ndarray, pd.Series, or single-column pd.DataFrame
            Input log-returns to filter.
        h : float
            Threshold for cumulative sum to trigger an event.
        """
        if not isinstance(log_returns, (np.ndarray, pd.Series, pd.DataFrame)):
            illegal_type = type(log_returns)
            raise ValueError(
                f"log returns must be NumPy array, Pandas Series or single-column DataFrame; got {illegal_type}"
            )

        if isinstance(log_returns, pd.Series):
            self._index = log_returns.index
            self._data = log_returns.values.ravel()

        elif isinstance(log_returns, pd.DataFrame):
            if log_returns.shape[1] != 1:
                raise ValueError("log_returns DataFrame must have exactly one column")
            self._index = log_returns.index
            self._data = log_returns.values.ravel()

        else:  # np.ndarray
            self._data = log_returns.ravel()

        self._h = h
        self._events = []

    def filter(self):
        """
        Run the CUSUM filter.

        Returns
        -------
        list of int
            Indices of detected events.
        """
        s_plus, s_minus = 0, 0
        self._events = []

        for i, v in enumerate(self._data):
            s_plus = max(0, s_plus + v)
            s_minus = min(0, s_minus + v)

            if s_plus >= self._h:
                self._events.append(i)
                s_plus, s_minus = 0, 0

            if abs(s_minus) >= self._h:
                self._events.append(i)
                s_plus, s_minus = 0, 0

        return self._events

    @property
    def filtered_events(self):
        """Return the values of log-returns at detected event indices."""
        return self._data[self._events]

    @property
    def events_index(self):
        """Return integer indices of detected events."""
        return self._events

    @property
    def index(self):
        """Return original index if input was a Series/DataFrame."""
        if not hasattr(self, '_index'):
            raise KeyError('No index was found')
        return self._index
