import numba
import numpy as np
from scipy import stats

import python.tools.numba_hist as numba_hist
from python.classes.constant_classes import CategoryConstants as cc


@numba.njit
def xlogy(x, y):
    """Compute x * log(y) with special handling for x == 0."""
    result = np.zeros_like(x)
    mask = x != 0
    result[mask] = x[mask] * np.log(y[mask])
    return result


@numba.njit
def apply_smearing(mc, lead_smear, sublead_smear, seed):
    """Pre-allocated version of smearing application"""
    np.random.seed(seed)
    # Avoid temporary arrays by computing directly
    return mc * np.sqrt(
        np.random.normal(1, lead_smear, len(mc))
        * np.random.normal(1, sublead_smear, len(mc))
    )


@numba.njit
def apply_scale(data, lead_scale, sublead_scale):
    """Apply scaling factors to data"""
    return data * np.sqrt(lead_scale * sublead_scale)


@numba.njit
def compute_loss(binned_data, binned_mc):
    """Optimized EMD computation"""
    # Pre-normalize to avoid division
    sum_data = np.sum(binned_data)
    sum_mc = np.sum(binned_mc)
    if sum_data == 0 or sum_mc == 0:
        return np.inf

    hist1_normalized = binned_data / sum_data
    hist2_normalized = binned_mc / sum_mc

    # Use in-place operations where possible
    np.cumsum(hist1_normalized, out=hist1_normalized)
    np.cumsum(hist2_normalized, out=hist2_normalized)

    return np.sum(np.abs(hist1_normalized - hist2_normalized))


class zcat:
    def __init__(self, i, j, data, mc, weights, **options):
        self.lead_index = i
        self.sublead_index = j
        self.lead_smear_index = options.get("smear_i", -1)
        self.sublead_smear_index = options.get("smear_j", -1)

        # Pre-allocate arrays and convert to float32 for better performance
        self.data = np.asarray(data, dtype=np.float32)
        self.mc = np.asarray(mc, dtype=np.float32)
        self.weights = np.asarray(weights, dtype=np.float32)

        # Store histogram parameters
        self.hist_min = options.get("hist_min", 80.0)
        self.hist_max = options.get("hist_max", 100.0)
        self.auto_bin = options.get("_kAutoBin", True)
        self.bin_size = options.get("bin_size", 0.25)

        # Pre-compute masks for valid ranges
        self.data_mask = (self.hist_min <= self.data) & (self.data <= self.hist_max)
        self.mc_mask = (self.hist_min <= self.mc) & (self.mc <= self.hist_max)

        # Pre-allocate buffers for frequent operations
        self.num_bins = int(round((self.hist_max - self.hist_min) / self.bin_size, 0))
        self.temp_data = np.empty(len(self.data) + 2, dtype=np.float32)
        self.temp_mc = np.empty(len(self.mc) + 2, dtype=np.float32)
        self.temp_weights = np.empty(len(self.weights) + 2, dtype=np.float32)

        # Initialize other attributes
        self.updated = False
        self.NLL = 0
        self.weight = 1 if i == j else 0.1
        self.seed = 3543136929
        self.valid = True
        self.history = []

        if self.auto_bin and self.bin_size == 0.25:
            self.set_bin_size()

    @numba.njit
    def _update_arrays(self, temp_data, temp_mc, temp_weights, data_mask, mc_mask):
        """Optimized array update function"""
        # Use pre-allocated arrays and avoid concatenation
        n_data = np.sum(data_mask)
        n_mc = np.sum(mc_mask)

        temp_data[:n_data] = self.data[data_mask]
        temp_data[n_data] = self.hist_min
        temp_data[n_data + 1] = self.hist_max

        temp_mc[:n_mc] = self.mc[mc_mask]
        temp_mc[n_mc] = self.hist_min
        temp_mc[n_mc + 1] = self.hist_max

        temp_weights[:n_mc] = self.weights[mc_mask]
        temp_weights[n_mc:] = 0

        return n_data + 2, n_mc + 2

    def update(self, lead_scale, sublead_scale, lead_smear=0, sublead_smear=0):
        """Optimized update function using pre-allocated arrays"""
        if not self.valid:
            return

        self.updated = True

        # Apply scales
        lead_scale = 1.0 if lead_scale == 0 else lead_scale
        sublead_scale = 1.0 if sublead_scale == 0 else sublead_scale

        # Use pre-allocated arrays
        temp_data = apply_scale(self.data, lead_scale, sublead_scale)
        temp_mc = (
            self.mc
            if lead_smear == 0 and sublead_smear == 0
            else apply_smearing(self.mc, lead_smear, sublead_smear, self.seed)
        )

        # Update masks
        data_mask = (self.hist_min <= temp_data) & (temp_data <= self.hist_max)
        mc_mask = (self.hist_min <= temp_mc) & (temp_mc <= self.hist_max)

        # Use pre-allocated arrays for histograms
        n_data, n_mc = self._update_arrays(
            self.temp_data, self.temp_mc, self.temp_weights, data_mask, mc_mask
        )

        if self.check_invalid(n_data - 2, n_mc - 2):
            self.clean_up()
            return

        # Compute histograms using pre-allocated arrays
        binned_data, _ = numba_hist.numba_histogram(
            self.temp_data[:n_data], self.num_bins
        )
        binned_mc, _ = numba_hist.numba_weighted_histogram(
            self.temp_mc[:n_mc], self.temp_weights[:n_mc], self.num_bins
        )

        # Clean binned data
        binned_mc[binned_mc == 0] = 1e-15

        # Compute loss
        self.NLL = compute_loss(binned_data, binned_mc)

        # Update history
        self.history.append(
            (
                lead_scale,
                sublead_scale,
                lead_smear,
                sublead_smear,
                self.NLL,
                self.bin_size,
            )
        )

        if np.isnan(self.NLL):
            self.clean_up()

    def set_bin_size(self):
        if self.auto_bin and self.bin_size == 0.25:
            # prune and check data and mc for validity
            temp_data = self.data[
                np.logical_and(self.hist_min <= self.data, self.data <= self.hist_max)
            ]
            mask_mc = np.logical_and(self.mc >= self.hist_min, self.mc <= self.hist_max)
            temp_mc = self.mc[mask_mc]
            if self.check_invalid(temp_data, temp_mc):
                print(
                    "[INFO][zcat][init] category ({},{}) was deactivated due to insufficient statistics in data".format(
                        self.lead_index, self.sublead_index
                    )
                )
                self.clean_up()
                return

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
            self.bin_size = max(
                data_width, mc_width
            )  # always choose the larger binning scheme

    def clean_up(self):
        """
        Set all variables to None to free up memory.
        """
        self.data = None
        self.mc = None
        self.weights = None
        self.bins = None
        self.valid = False

    def check_invalid(self, data=None, mc=None):
        """
        Check if the z category is valid.

        Args:
            None
        Returns:
            bool: True if the z category is invalid, False otherwise
        """
        if data is None:
            data = self.data
        if mc is None:
            mc = self.mc
        return (
            len(data) < cc.MIN_EVENTS_DATA
            or len(mc) < cc.MIN_EVENTS_MC_DIAG
            or (
                len(mc) < cc.MIN_EVENTS_MC_OFFDIAG
                and self.lead_index != self.sublead_index
            )
        )

    def print(self):
        """Print the z category object."""
        print("lead index:", self.lead_index)
        print("sublead index:", self.sublead_index)
        print("lead smearing index:", self.lead_smear_index)
        print("sublead smearing index:", self.sublead_smear_index)
        print("nevents, data:", len(self.data))
        print("nevents, mc: ", len(self.mc))
        print("NLL:", self.NLL, " || w/bin size:", self.bin_size)
        print("weight:", self.weight)
        print("valid:", self.valid)

    def inject(self, lead_scale, sublead_scale, lead_smear, sublead_smear):
        """
        Artificially inject scales and smearings in to the "toy mc" labelled here as data.

        Args:
            lead_scale (float): scale for the leading electron
            sublead_scale (float): scale for the subleading electron
            lead_smear (float): smearing for the leading electron
            sublead_smear (float): smearing for the subleading electron
        Returns:
            None
        """
        self.data = self.data * np.sqrt(lead_scale * sublead_scale, dtype=np.float32)
        if lead_smear != 0 and sublead_smear != 0:
            lead_smear_list = np.random.normal(
                1, np.abs(lead_smear), len(self.data), dtype=np.float32
            )
            sublead_smear_list = np.random.normal(
                1, np.abs(sublead_smear), len(self.data), dtype=np.float32
            )
            self.data = self.data * np.sqrt(
                np.multiply(lead_smear_list, sublead_smear_list, dtype=np.float32),
                dtype=np.float32,
            )
        return
