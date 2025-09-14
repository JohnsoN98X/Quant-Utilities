import pandas as pd
import numpy as np
from sklearn.model_selection import BaseCrossValidator

class TimeSeriesEmbargoCV(BaseCrossValidator):
    """
    Time Series cross-validator with embargo period to prevent leakage.

    This cross-validator splits time series data into sequential train and test folds,
    introducing an embargo period (a gap) between train and test sets to prevent data leakage.
    The embargo_size specifies the number of samples to skip between the train and test sets
    in each fold. Designed for use-cases where temporal dependency may lead to lookahead bias,
    such as in financial modeling or forecasting tasks.

    Parameters
    ----------
    cv : int, default=5
        Number of folds.

    embargo_size : int, default=0
        Number of samples to exclude (the embargo period) between train and test sets
        in each fold.

    Examples
    --------
    >>> cv = TimeSeriesEmbargoCV(cv=5, embargo_size=3)
    >>> for train_idx, test_idx in cv.split(X):
    ...     print("TRAIN:", train_idx, "TEST:", test_idx)
    """

    def __init__(self, cv=5, embargo_size=0):
        """
        Initialize the cross-validator.

        Parameters
        ----------
        cv : int, default=5
            Number of folds.

        embargo_size : int, default=0
            Number of samples to exclude (the embargo period) between train and test sets.
        """
        if not isinstance(cv, int):
            raise ValueError("'cv' must be int")
        if not isinstance(embargo_size, int):
            raise ValueError("'embargo_size' must be int")
        self.n_splits = cv
        self.embargo_size = embargo_size

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like, shape (n_samples, ...)
            Always ignored, exists for compatibility.

        y : array-like, shape (n_samples, ...)
            Always ignored, exists for compatibility.

        groups : array-like, with shape (n_samples,), optional
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Number of folds/splits.
        """
        n_samples = len(X)
        fold_size = int(n_samples / self.n_splits)
        n_possible_splits = (n_samples - self.embargo_size) // fold_size
        return min(self.n_splits, n_possible_splits)

    def split(self, X=None, y=None, groups=None):
        """
        Generate indices to split data into training and test set for each fold.

        Parameters
        ----------
        X : array-like, shape (n_samples, ...)
            The data to split.

        y : array-like, shape (n_samples, ...)
            Always ignored, exists for compatibility.

        groups : array-like, with shape (n_samples,), optional
            Always ignored, exists for compatibility.

        Yields
        ------
        train_indices : ndarray
            The training set indices for that split.

        test_indices : ndarray
            The testing set indices for that split.
        """
        n_samples = len(X)
        fold_size = int(n_samples / self.n_splits)
        n_actual_splits = self.get_n_splits(X)
        for i in range(n_actual_splits):
            train_start_idx = i * fold_size
            train_end_idx = i * fold_size + fold_size
            test_start_idx = train_end_idx + self.embargo_size
            test_end_idx = test_start_idx + fold_size - self.embargo_size
            if test_end_idx > n_samples:
                break
            train_indices = np.arange(train_start_idx, train_end_idx)
            test_indices = np.arange(test_start_idx, test_end_idx)
            yield train_indices, test_indices