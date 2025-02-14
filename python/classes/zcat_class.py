import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numba
import numpy as np
from scipy import stats

import python.tools.numba_hist as numba_hist
from python.classes.constant_classes import CategoryConstants as cc

# Constants to avoid magic numbers
EPSILON = 1e-15
DEFAULT_HIST_MIN = 80.0
DEFAULT_HIST_MAX = 100.0
DEFAULT_BIN_SIZE = 0.25
MAX_HISTORY_SIZE = 10000  # Limit history size to prevent memory issues


@dataclass
class HistogramParams:
    """Data class for histogram parameters"""

    hist_min: float = DEFAULT_HIST_MIN
    hist_max: float = DEFAULT_HIST_MAX
    bin_size: float = DEFAULT_BIN_SIZE
    auto_bin: bool = True


@numba.njit
def _update_arrays_numba(
    data: np.ndarray,
    mc: np.ndarray,
    weights: np.ndarray,
    temp_data: np.ndarray,
    temp_mc: np.ndarray,
    temp_weights: np.ndarray,
    data_mask: np.ndarray,
    mc_mask: np.ndarray,
    hist_min: float,
    hist_max: float,
) -> Tuple[int, int]:
    """
    Optimized array update function separate from class.

    Args:
        data: Input data array
        mc: Monte Carlo simulation array
        weights: Weights array for MC
        temp_data: Pre-allocated array for data
        temp_mc: Pre-allocated array for MC
        temp_weights: Pre-allocated array for weights
        data_mask: Boolean mask for valid data
        mc_mask: Boolean mask for valid MC
        hist_min: Minimum histogram value
        hist_max: Maximum histogram value

    Returns:
        Tuple of (number of data points, number of MC points)
    """
    n_data = np.sum(data_mask)
    n_mc = np.sum(mc_mask)

    temp_data[:n_data] = data[data_mask]
    temp_data[n_data] = hist_min
    temp_data[n_data + 1] = hist_max

    temp_mc[:n_mc] = mc[mc_mask]
    temp_mc[n_mc] = hist_min
    temp_mc[n_mc + 1] = hist_max

    temp_weights[:n_mc] = weights[mc_mask]
    temp_weights[n_mc : n_mc + 2] = 0

    return n_data + 2, n_mc + 2


@numba.njit
def xlogy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute x * log(y) with special handling for x == 0.

    Args:
        x: Input array
        y: Input array

    Returns:
        Array of x * log(y) values with zeros where x == 0
    """
    result = np.zeros_like(x, dtype=np.float32)
    mask = x != 0
    result[mask] = x[mask] * np.log(y[mask])
    return result


def apply_smearing(
    mc: np.ndarray, lead_smear: float, sublead_smear: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Apply smearing to Monte Carlo data.

    Args:
        mc: Input MC array
        lead_smear: Leading smearing factor
        sublead_smear: Subleading smearing factor
        rng: Random number generator instance

    Returns:
        Smeared MC array
    """
    return mc * np.sqrt(
        rng.normal(1, lead_smear, len(mc)) * rng.normal(1, sublead_smear, len(mc))
    ).astype(np.float32)


@numba.njit
def apply_scale(
    data: np.ndarray, lead_scale: float, sublead_scale: float
) -> np.ndarray:
    """
    Apply scaling factors to data.

    Args:
        data: Input data array
        lead_scale: Leading scale factor
        sublead_scale: Subleading scale factor

    Returns:
        Scaled data array
    """
    return data * np.sqrt(lead_scale * sublead_scale)


@numba.njit
def compute_loss(
    binned_data: np.ndarray, binned_mc: np.ndarray, epsilon: float = EPSILON
) -> float:
    """
    Compute Earth Mover's Distance (EMD) between binned distributions.

    Args:
        binned_data: Binned data distribution
        binned_mc: Binned MC distribution
        epsilon: Small value to prevent division by zero

    Returns:
        EMD loss value or infinity if invalid
    """
    sum_data = np.sum(binned_data)
    sum_mc = np.sum(binned_mc)

    if sum_data < epsilon or sum_mc < epsilon:
        return np.inf

    norm_data = binned_data / (sum_data + epsilon)
    norm_mc = binned_mc / (sum_mc + epsilon)

    return np.sum(np.abs(np.cumsum(norm_data) - np.cumsum(norm_mc)))


class ZCategory:
    """
    Z-boson category analysis class with optimized performance.
    """

    def __init__(
        self,
        lead_index: int,
        sublead_index: int,
        data: np.ndarray,
        mc: np.ndarray,
        weights: np.ndarray,
        **options,
    ):
        """
        Initialize ZCategory instance.

        Args:
            lead_index: Index for leading particle
            sublead_index: Index for subleading particle
            data: Input data array
            mc: Monte Carlo simulation array
            weights: Weights for MC events
            **options: Additional options including:
                - smear_i: Leading smear index
                - smear_j: Subleading smear index
                - hist_min: Minimum histogram value
                - hist_max: Maximum histogram value
                - bin_size: Histogram bin size
                - auto_bin: Whether to automatically determine bin size
        """
        self.lead_index = lead_index
        self.sublead_index = sublead_index
        self.lead_smear_index = options.get("smear_i", -1)
        self.sublead_smear_index = options.get("smear_j", -1)

        # Initialize histogram parameters
        self.hist_params = HistogramParams(
            hist_min=options.get("hist_min", DEFAULT_HIST_MIN),
            hist_max=options.get("hist_max", DEFAULT_HIST_MAX),
            bin_size=options.get("bin_size", DEFAULT_BIN_SIZE),
            auto_bin=options.get("_kAutoBin", True),
        )

        # Convert inputs to float32 for better performance
        self.data = np.asarray(data, dtype=np.float32)
        self.mc = np.asarray(mc, dtype=np.float32)
        self.weights = np.asarray(weights, dtype=np.float32)

        # Initialize random number generator
        self.rng = np.random.Generator(np.random.PCG64(3543136929))

        # Pre-compute masks for valid ranges
        self._compute_masks()

        # Pre-allocate buffers
        self._allocate_buffers()

        # Initialize state variables
        self.NLL = 0.0
        self.weight = 1.0 if lead_index == sublead_index else 0.1
        self.valid = True
        self.history: List[Tuple[float, ...]] = []
        self.lead_smear = 0.0
        self.sublead_smear = 0.0
        self.lead_scale = 1.0
        self.sublead_scale = 1.0

        # Set automatic bin size if needed
        if self.hist_params.auto_bin and self.hist_params.bin_size == DEFAULT_BIN_SIZE:
            self._set_bin_size()

    def _compute_masks(self) -> None:
        """Compute masks for valid data ranges."""
        self.data_mask = (self.hist_params.hist_min <= self.data) & (
            self.data <= self.hist_params.hist_max
        )
        self.mc_mask = (self.hist_params.hist_min <= self.mc) & (
            self.mc <= self.hist_params.hist_max
        )

    def _allocate_buffers(self) -> None:
        """Allocate temporary buffers for computations."""
        self.num_bins = int(
            round(
                (self.hist_params.hist_max - self.hist_params.hist_min)
                / self.hist_params.bin_size,
                0,
            )
        )
        self.temp_data = np.empty(len(self.data) + 2, dtype=np.float32)
        self.temp_mc = np.empty(len(self.mc) + 2, dtype=np.float32)
        self.temp_weights = np.empty(len(self.weights) + 2, dtype=np.float32)

    def update(
        self,
        lead_scale: float,
        sublead_scale: float,
        lead_smear: float = 0,
        sublead_smear: float = 0,
    ) -> None:
        """
        Update category with new scale and smear parameters.

        Args:
            lead_scale: Scale factor for leading particle
            sublead_scale: Scale factor for subleading particle
            lead_smear: Smearing for leading particle
            sublead_smear: Smearing for subleading particle
        """
        if not self.valid:
            return

        try:
            # Apply scales with validation
            lead_scale = max(EPSILON, lead_scale if lead_scale != 0 else 1.0)
            sublead_scale = max(EPSILON, sublead_scale if sublead_scale != 0 else 1.0)

            # Apply transformations
            temp_data = (
                self.temp_data
                if self.lead_scale == lead_scale and self.sublead_scale == sublead_scale
                else apply_scale(self.data, lead_scale, sublead_scale)
            )

            temp_mc = (
                self.temp_mc
                if lead_smear == self.lead_smear and sublead_smear == self.sublead_smear
                else apply_smearing(self.mc, lead_smear, sublead_smear, self.rng)
            )

            # Update state
            self.lead_scale = lead_scale
            self.sublead_scale = sublead_scale
            self.lead_smear = lead_smear
            self.sublead_smear = sublead_smear

            # Update masks and arrays
            data_mask = (self.hist_params.hist_min <= temp_data) & (
                temp_data <= self.hist_params.hist_max
            )
            mc_mask = (self.hist_params.hist_min <= temp_mc) & (
                temp_mc <= self.hist_params.hist_max
            )

            n_data, n_mc = _update_arrays_numba(
                temp_data,
                temp_mc,
                self.weights,
                self.temp_data,
                self.temp_mc,
                self.temp_weights,
                data_mask,
                mc_mask,
                self.hist_params.hist_min,
                self.hist_params.hist_max,
            )

            if self._check_invalid(n_data - 2, n_mc - 2):
                self.cleanup()
                return

            # Compute histograms
            binned_data, _ = numba_hist.numba_histogram(
                self.temp_data[:n_data], self.num_bins
            )
            binned_mc, _ = numba_hist.numba_weighted_histogram(
                self.temp_mc[:n_mc], self.temp_weights[:n_mc], self.num_bins
            )

            # Apply minimum value to MC bins
            binned_mc = np.maximum(binned_mc, EPSILON)

            # Compute loss
            self.NLL = compute_loss(binned_data, binned_mc)

            # Update history with size limit
            if len(self.history) >= MAX_HISTORY_SIZE:
                self.history.pop(0)

            self.history.append(
                (
                    lead_scale,
                    sublead_scale,
                    lead_smear,
                    sublead_smear,
                    self.NLL,
                    self.hist_params.bin_size,
                )
            )

            if np.isnan(self.NLL):
                self.cleanup()

        except Exception as e:
            warnings.warn(f"Error in update: {str(e)}")
            self.cleanup()

    def _set_bin_size(self) -> None:
        """Automatically determine optimal bin size using Freedman-Diaconis rule."""
        try:
            # Filter data within valid range
            temp_data = self.data[self.data_mask]
            temp_mc = self.mc[self.mc_mask]

            if self._check_invalid(len(temp_data), len(temp_mc)):
                warnings.warn(
                    f"Category ({self.lead_index},{self.sublead_index}) "
                    "deactivated due to insufficient statistics"
                )
                self.cleanup()
                return

            # Calculate optimal bin width using Freedman-Diaconis rule
            data_width = (
                2
                * stats.iqr(temp_data, rng=(25, 75), nan_policy="omit")
                / np.power(len(temp_data), 1.0 / 3.0)
            )
            mc_width = (
                2
                * stats.iqr(temp_mc, rng=(25, 75), nan_policy="omit")
                / np.power(len(temp_mc), 1.0 / 3.0)
            )

            self.hist_params.bin_size = max(data_width, mc_width)

        except Exception as e:
            warnings.warn(f"Error in bin size calculation: {str(e)}")
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources and invalidate category."""
        try:
            del self.temp_data
            del self.temp_mc
            del self.temp_weights
            del self.data
            del self.mc
            del self.weights
            self.valid = False
        except Exception as e:
            warnings.warn(f"Error in cleanup: {str(e)}")

    def _check_invalid(self, data_count: int = 0, mc_count: int = 0) -> bool:
        """
        Check if the category has sufficient statistics.

        Args:
            data_count: Number of data events
            mc_count: Number of MC events

        Returns:
            True if invalid, False if valid
        """
        return (
            data_count < cc.MIN_EVENTS_DATA
            or mc_count < cc.MIN_EVENTS_MC_DIAG
            or (
                mc_count < cc.MIN_EVENTS_MC_OFFDIAG
                and self.lead_index != self.sublead_index
            )
        )

    def display_info(self) -> None:
        """Display information about the category state."""
        info = {
            "Lead index": self.lead_index,
            "Sublead index": self.sublead_index,
            "Lead smearing index": self.lead_smear_index,
            "Sublead smearing index": self.sublead_smear_index,
            "Data events": len(self.data) if self.data is not None else 0,
            "MC events": len(self.mc) if self.mc is not None else 0,
            "NLL": f"{self.NLL:.4f}",
            "Bin size": f"{self.hist_params.bin_size:.4f}",
            "Weight": self.weight,
            "Valid": self.valid,
        }

        for key, value in info.items():
            print(f"{key}: {value}")

    def inject(
        self,
        lead_scale: float,
        sublead_scale: float,
        lead_smear: float,
        sublead_smear: float,
    ) -> None:
        """
        Inject scales and smearings into the toy MC (labeled as data).

        Args:
            lead_scale: Scale factor for leading electron
            sublead_scale: Scale factor for subleading electron
            lead_smear: Smearing for leading electron
            sublead_smear: Smearing for subleading electron
        """
        if not self.valid or self.data is None:
            warnings.warn("Cannot inject into invalid or empty category")
            return

        try:
            # Apply scaling
            scale_factor = np.sqrt(lead_scale * sublead_scale)
            self.data = (self.data * scale_factor).astype(np.float32)

            # Apply smearing if non-zero
            if abs(lead_smear) > EPSILON and abs(sublead_smear) > EPSILON:
                lead_smear_values = self.rng.normal(
                    1, abs(lead_smear), len(self.data)
                ).astype(np.float32)
                sublead_smear_values = self.rng.normal(
                    1, abs(sublead_smear), len(self.data)
                ).astype(np.float32)

                smear_factor = np.sqrt(
                    np.multiply(
                        lead_smear_values, sublead_smear_values, dtype=np.float32
                    )
                )
                self.data = (self.data * smear_factor).astype(np.float32)

        except Exception as e:
            warnings.warn(f"Error in inject: {str(e)}")
            self.cleanup()

    def get_history(
        self, n_entries: Optional[int] = None
    ) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Get optimization history with optional limit on entries.

        Args:
            n_entries: Number of most recent entries to return
                      If None, returns full history

        Returns:
            List of tuples containing (lead_scale, sublead_scale,
            lead_smear, sublead_smear, NLL, bin_size)
        """
        if n_entries is None or n_entries >= len(self.history):
            return self.history
        return self.history[-n_entries:]
