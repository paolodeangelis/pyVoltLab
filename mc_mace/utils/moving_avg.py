from collections import deque

import numpy as np


class ForgetfulMovingAvg:
    """
    Implements a forgetful moving average and variance calculator with a dynamic window size.
    The window size grows based on the number of samples seen, controlled by a dropout parameter
    that determines what fraction of the history to maintain. Uses numerically stable updates
    based on the algorithm from https://doi.org/10.1103/PhysRevE.105.045311


    Attributes:
        dropout (float): Fraction of history to use, between 0.0 (forget everything) and 1.0 (keep everything).
        _n (int): Total number of samples seen.
        _L (int): Current effective window size.
        _dropped (int): Number of samples dropped from the beginning of the sequence.
        _sum (float): Running sum for mean calculation.
        _mean (float): Current running mean.
        _M2 (float): Sum of squared differences for variance calculation.
        _var (float): Current running variance.
        _last_value (float): Most recently added sample.
        _buffer (deque): Buffer containing the current window of samples.

    Notes:
        Time complexity: O(n^0.01)
        Memory complexity: O(n^(0.85 - 0.22*dropout))
    """

    def __init__(self, dropout: float = 1.0) -> None:
        """
        Initialize the ForgetfulMovingAvg with a specified dropout rate.

        Args:
            dropout (float): Fraction of history to maintain, between 0.0 and 1.0.
                           1.0 means keep all history, 0.0 means only keep latest value.

        Raises:
            ValueError: If dropout is not between 0.0 and 1.0.
        """
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("Dropout must be between 0.0 and 1.0")

        self.dropout = dropout
        self._n: int = 0
        self._L: int = 0
        self._dropped: int = 0
        self._sum: float = 0.0
        self._mean: float = 0.0
        self._M2: float = 0.0
        self._var: float = 0.0
        self._last_value: float = np.nan
        self._buffer: deque[float] = deque()

    def _get_dropped(self) -> bool:
        """
        Calculate if a sample should be dropped based on the current window size.

        Returns:
            bool: True if a sample should be dropped from the window, False otherwise.
        """
        new_dropped = np.ceil(self._n * self.dropout)
        changed = bool(new_dropped == self._dropped + 1)
        self._dropped = new_dropped
        return changed

    def _get_window_size(self) -> int:
        """
        Calculate the current window size based on total samples and dropout rate.

        Returns:
            int: Current effective window size (n - dropped + 1).
        """
        return self._n - self._dropped + 1

    def add_sample(self, new_sample: float) -> None:
        """
        Add a new sample and update running statistics using numerically stable formulas.
        Implementation follows Annex A from https://doi.org/10.1103/PhysRevE.105.045311

        Args:
            new_sample (float): The new data point to add to the moving average.
        """
        if np.isnan(new_sample):
            return

        self._n += 1
        self._last_value = new_sample

        if self._get_dropped() and self._n > 1:
            # Case B: Window moves forward, dropping oldest sample
            self._L = self._get_window_size()
            old_value = self._buffer.popleft()
            self._sum += self._last_value - old_value
            mean_new = self._sum / self._L
            M2_delta = (new_sample - old_value) * (new_sample - mean_new + old_value - self._mean)
            self._M2 += M2_delta
        else:
            # Case A: Window expands to include new sample
            self._L = self._get_window_size()
            self._sum += self._last_value
            mean_new = self._sum / self._L
            M2_delta = (new_sample - self._mean) * (new_sample - mean_new)
            self._M2 += M2_delta

        self._buffer.append(new_sample)

    def get_mean(self) -> float:
        """
        Calculate the current mean value.

        Returns:
            float: The mean of the samples in the current window.
        """
        self._mean = float(self._sum / self._L)
        return self._mean

    def get_variance(self) -> float:
        """
        Calculate the current variance value.

        Returns:
            float: The variance of the samples in the current window.
        """
        self._var = float(self._M2 / self._L)
        return self._var

    def reset(self) -> None:
        """
        Reset all statistics and buffer to their initial state.
        """
        self._n = 0
        self._L = 0
        self._dropped = 0
        self._mean = 0.0
        self._sum = 0.0
        self._var = 0.0
        self._M2 = 0.0
        self._last_value = np.nan
        self._buffer.clear()

    def get_last(self) -> float:
        """
        Get the most recently added sample.

        Returns:
            float: The last added sample, or np.nan if no valid samples exist.
        """
        return self._last_value

    def get_buffer(self) -> list[float]:
        """
        Get the current contents of the sample buffer.

        Returns:
            list[float]: List containing the samples in the current window.
        """
        return list(self._buffer)

    def get_window_size(self) -> int:
        """
        Get the current effective window size.

        Returns:
            int: The number of samples currently in the window.
        """
        return self._get_window_size()
