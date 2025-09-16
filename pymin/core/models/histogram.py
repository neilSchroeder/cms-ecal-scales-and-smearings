from dataclasses import dataclass
from typing import Optional

import numpy as np
import numba
from scipy import stats

from pymin.config.defaults import (
    UNSET,
    HIST_MIN,
    HIST_MAX,
)


@numba.njit(cache=True)
def get_bin_edges(a_min, a_max, bins):
    """
    Create bin edges without using np.linspace to avoid dtype parameter issue
    """
    bin_edges = np.zeros(bins + 1, dtype=np.float64)
    delta = (a_max - a_min) / bins

    for i in range(bins + 1):
        bin_edges[i] = a_min + i * delta

    # Ensure last bin edge is exactly a_max to avoid floating point issues
    bin_edges[bins] = a_max
    return bin_edges


@numba.njit(cache=True)
def compute_bin(x, a_min, a_max, bins):
    """
    Compute bin index more efficiently
    """
    # Handle edge cases
    if x < a_min:
        return -1  # Out of range

    if x >= a_max:
        return bins - 1  # Last bin includes a_max

    # Fast bin calculation
    bin_idx = int(bins * (x - a_min) / (a_max - a_min))

    # Ensure bin index is in valid range
    if bin_idx >= bins:
        return bins - 1

    return bin_idx


@numba.njit(cache=True)
def numba_histogram(a, bin_edges):
    """
    Optimized version of numba_histogram compatible with existing code
    """
    # Get min and max values
    a_min, a_max = HIST_MIN, HIST_MAX

    # Small epsilon to ensure max value is included
    a_max = a_max * (1.0 + 1e-10)

    # Pre-allocate histogram arrays
    num_bins = len(bin_edges) - 1
    hist = np.zeros(num_bins, dtype=np.int64)

    # Fill histogram
    for i in range(len(a)):
        bin_idx = compute_bin(a[i], a_min, a_max, num_bins)
        if bin_idx >= 0:
            hist[bin_idx] += 1

    return hist, bin_edges


@numba.njit(cache=True)
def numba_weighted_histogram(a, weights, bins):
    """
    Optimized version of weighted histogram compatible with existing code
    """
    # Get min and max values
    a_min = np.min(a)
    a_max = np.max(a)

    # Small epsilon to ensure max value is included
    a_max = a_max * (1.0 + 1e-10)

    # Pre-allocate histogram arrays
    hist = np.zeros(bins, dtype=np.float32)
    bin_edges = get_bin_edges(a_min, a_max, bins)

    # Fill histogram with weights
    for i in range(len(a)):
        bin_idx = compute_bin(a[i], a_min, a_max, bins)
        if bin_idx >= 0:
            hist[bin_idx] += weights[i]

    return hist, bin_edges


@dataclass
class Histogram:
    """
    Class for representing a histogram.
    """

    bins: np.ndarray
    values: np.ndarray
    errors: np.ndarray
    min_value: float
    max_value: float
    auto_bin: bool = True
    bin_width: float = 0.25
    num_bins: int = 80
    parallel: bool = False

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        weights: Optional[np.ndarray | object] = UNSET,
        bins: Optional[np.ndarray | object] = UNSET,
        auto_bin: Optional[bool] = True,
        bin_width: Optional[float | object] = UNSET,
        num_bins: Optional[int] = 80,
        parallel: Optional[bool] = False,
    ):
        """
        Create a histogram from data.

        Parameters
        ----------
        name : str
            Name of the histogram.
        data : np.ndarray
            Data to histogram.
        weights : np.ndarray | None
            Weights for the data points.
        bins : int
            Number of bins.
        auto_bin : bool
            Whether to automatically determine bin edges.
        bin_width : float
            Width of each bin if not auto-binning.

        Returns
        -------
        Histogram
            The created histogram.
        """

        if bins is not UNSET:
            # user provided bin edges, use them directly
            bins, values = (
                numba_histogram(data, len(bins) - 1)
                if weights is UNSET
                else numba_weighted_histogram(data, weights, len(bin_edges) - 1)
            )

    @classmethod
    def determine_bin_size(cls, dist1: np.ndarray, dist2: np.ndarray) -> float:
        """
        Compute the bin size.

        Returns
        -------
        float
            The bin size.
        """
        width1 = (
            2
            * stats.iqr(dist1, rng=(25, 75), nan_policy="omit")
            / np.power(len(dist1), 1.0 / 3.0)
        )
        width2 = (
            2
            * stats.iqr(dist2, rng=(25, 75), nan_policy="omit")
            / np.power(len(dist2), 1.0 / 3.0)
        )

        return max(width1, width2)  # always choose the larger binning scheme

    def check_invalid(self) -> bool:
        """
        Check if the histogram is valid.

        Returns
        -------
        bool
            True if the histogram is valid, False otherwise.
        """
        if self.bins is None or self.values is None or self.errors is None:
            return False
        if len(self.bins) < 2 or len(self.values) < 1 or len(self.errors) < 1:
            return False
        if (
            len(self.bins) != len(self.values) + 1
            or len(self.bins) != len(self.errors) + 1
        ):
            return False
        return True
