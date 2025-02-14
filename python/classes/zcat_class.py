from dataclasses import dataclass, field
from typing import List, Tuple

import numba
import numpy as np
from scipy import stats

import python.tools.numba_hist as numba_hist
from python.classes.constant_classes import CategoryConstants as cc

EPSILON = 1e-15


def apply_smearing(mc, lead_smear, sublead_smear, seed):
    """Pre-allocated version of smearing application"""
    rand = np.random.Generator(np.random.PCG64(seed))
    # Avoid temporary arrays by computing directly
    return mc * np.sqrt(
        rand.normal(1, lead_smear, len(mc)) * rand.normal(1, sublead_smear, len(mc))
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

    return np.sum(
        np.abs(
            np.cumsum(binned_data / (sum_data + EPSILON))
            - np.cumsum(binned_mc / (sum_mc + EPSILON))
        )
    )


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
        self.temp_data = np.copy(self.data[self.data_mask])
        self.temp_mc = np.copy(self.mc[self.mc_mask])

        # Initialize other attributes
        self.updated = False
        self.NLL = 0
        self.weight = 1 if i == j else 0.1
        self.seed = 3543136929
        self.valid = True
        self.history = []
        self.lead_smear = 0
        self.sublead_smear = 0
        self.lead_scale = 1
        self.sublead_scale = 1
        self.top_and_bottom = np.array([self.hist_min, self.hist_max])

        if self.auto_bin and self.bin_size == 0.25:
            self.set_bin_size()

    def update(self, lead_scale, sublead_scale, lead_smear=0, sublead_smear=0):
        """Optimized update function using pre-allocated arrays"""
        if not self.valid:
            return

        self.updated = True

        # Apply scales, only update if necessary
        lead_scale = 1.0 if lead_scale == 0 else lead_scale
        sublead_scale = 1.0 if sublead_scale == 0 else sublead_scale

        if self.lead_scale != lead_scale or self.sublead_scale != sublead_scale:
            self.temp_data = apply_scale(self.data, lead_scale, sublead_scale)
            self.lead_scale = lead_scale
            self.sublead_scale = sublead_scale
            self.data_mask = (self.hist_min <= self.temp_data) & (
                self.temp_data <= self.hist_max
            )
            self.temp_data = self.temp_data[self.data_mask]

        # Apply smearing, only update if necessary
        if self.lead_smear != lead_smear or self.sublead_smear != sublead_smear:
            self.temp_mc = apply_smearing(self.mc, lead_smear, sublead_smear, self.seed)
            self.lead_smear = lead_smear
            self.sublead_smear = sublead_smear
            self.mc_mask = (self.hist_min <= self.temp_mc) & (
                self.temp_mc <= self.hist_max
            )
            self.temp_mc = self.temp_mc[self.mc_mask]

        if self.check_invalid(len(self.temp_mc), len(self.temp_data)):
            self.clean_up()
            return

        # Compute histograms using pre-allocated arrays
        # print type
        binned_data, _ = numba_hist.numba_histogram(
            np.concatenate([self.temp_data, self.top_and_bottom]),
            self.num_bins,
        )
        binned_mc, _ = numba_hist.numba_weighted_histogram(
            np.concatenate([self.temp_mc, self.top_and_bottom]),
            np.concatenate([self.weights, [0, 0]]),
            self.num_bins,
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
            if self.check_invalid(len(temp_data), len(temp_mc)):
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

    def check_invalid(self, data: int = 0, mc: int = 0):
        """
        Check if the z category is valid.

        Args:
            None
        Returns:
            bool: True if the z category is invalid, False otherwise
        """
        return (
            data < cc.MIN_EVENTS_DATA
            or mc < cc.MIN_EVENTS_MC_DIAG
            or (mc < cc.MIN_EVENTS_MC_OFFDIAG and self.lead_index != self.sublead_index)
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
                1, np.abs(lead_smear), len(self.data)
            ).astype(np.float32)
            sublead_smear_list = np.random.normal(
                1, np.abs(sublead_smear), len(self.data)
            ).astype(np.float32)
            self.data = self.data * np.sqrt(
                np.multiply(lead_smear_list, sublead_smear_list, dtype=np.float32),
                dtype=np.float32,
            )
        return
