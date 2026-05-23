import numpy as np
import pandas as pd
from scipy import stats

import python.utilities.numba_hist as numba_hist
from python.classes.constant_classes import CategoryConstants as cc

import numba


@numba.njit
def xlogy(x, y):
    """Compute x * log(y) with special handling for x == 0."""
    result = np.zeros_like(x)
    mask = x != 0
    result[mask] = x[mask] * np.log(y[mask])
    return result


@numba.njit
def apply_smearing(mc, lead_smear, sublead_smear, seed):
    np.random.seed(seed)
    lead_rand = np.random.normal(0, lead_smear, len(mc))
    sublead_rand = np.random.normal(0, sublead_smear, len(mc))
    x = np.sqrt((1 + lead_rand) * (1 + sublead_rand))
    return mc * x


@numba.njit
def _generate_smearing_randn(n, seed):
    """Generate and return two standard normal vectors using numba's RNG."""
    np.random.seed(seed)
    lead = np.random.normal(0.0, 1.0, n)
    sublead = np.random.normal(0.0, 1.0, n)
    return lead, sublead


@numba.njit
def apply_smearing_cached(mc, lead_smear, sublead_smear, randn_lead, randn_sublead):
    """Apply smearing using precomputed standard normal draws (avoids re-seeding RNG)."""
    lead_rand = lead_smear * randn_lead
    sublead_rand = sublead_smear * randn_sublead
    x = np.sqrt((1 + lead_rand) * (1 + sublead_rand))
    return mc * x


@numba.njit
def compute_nll_chisqr(binned_data, norm_binned_mc, num_bins=80):
    # Implement NLL and Chi-squared computation here
    # This is a placeholder, replace with actual implementation
    scaled_mc = norm_binned_mc * np.sum(binned_data)
    err_mc = np.sqrt(scaled_mc).astype(np.float32)
    err_data = np.sqrt(binned_data).astype(np.float32)
    err = np.sqrt(
        np.add(
            np.multiply(err_mc, err_mc).astype(np.float32),
            np.multiply(err_data, err_data).astype(np.float32),
        ).astype(np.float32)
    ).astype(np.float32)
    chi_sqr = (
        np.sum(
            np.divide(
                np.multiply(binned_data - scaled_mc, binned_data - scaled_mc).astype(
                    np.float32
                ),
                err,
            ).astype(np.float32)
        )
        / num_bins
    )

    nll = xlogy(binned_data, norm_binned_mc)
    nll[nll == -np.inf] = 0
    nll = np.sum(nll) / len(nll)
    # evaluate penalty
    penalty = xlogy(np.sum(binned_data) - binned_data, 1 - norm_binned_mc)
    penalty[penalty == -np.inf] = 0
    penalty = np.sum(penalty) / len(penalty)
    return -2 * (nll + penalty) * chi_sqr


@numba.njit
def compute_earthmovers_distance(binned_data, binned_mc, emd_weights=None):
    """
    Compute the Earth Mover's Distance (EMD) between binned data and binned MC distributions.
    MC is rescaled to match the data total so CDFs are comparable.
    Bins are weighted so that middle bins contribute more and edge bins contribute less.
    If emd_weights is provided, skip recomputing the triangular kernel.
    """
    n = len(binned_data)
    if emd_weights is not None:
        weights = emd_weights
    else:
        # build a triangular weight peaking at the center
        weights = np.empty(n, dtype=np.float64)
        for i in range(n):
            weights[i] = 1.0 - abs(2.0 * i / (n - 1) - 1.0) if n > 1 else 1.0
        # ensure minimum weight so edge bins are not completely ignored
        for i in range(n):
            weights[i] = max(weights[i], 0.1)
    cdf_data = np.cumsum(binned_data)
    mc_sum = np.sum(binned_mc)
    if mc_sum > 0:
        cdf_mc = np.cumsum(np.sum(binned_data) * binned_mc / mc_sum)
    else:
        cdf_mc = np.cumsum(binned_mc)
    emd = 0.0
    for i in range(n):
        emd += weights[i] * abs(cdf_data[i] - cdf_mc[i])
    return emd


@numba.njit
def build_emd_weights(n):
    """Build triangular EMD weight vector peaking at center, minimum 0.1."""
    weights = np.empty(n, dtype=np.float64)
    for i in range(n):
        weights[i] = 1.0 - abs(2.0 * i / (n - 1) - 1.0) if n > 1 else 1.0
    for i in range(n):
        weights[i] = max(weights[i], 0.1)
    return weights


@numba.njit
def build_uniform_weights(n):
    """Flat weight vector of ones — corresponds to a hard fit window defined by hist_min/hist_max."""
    w = np.empty(n, dtype=np.float64)
    for i in range(n):
        w[i] = 1.0
    return w
    return weights


def _derive_cat_seed(base_seed, lead_index, sublead_index):
    """Deterministic per-category seed (uint32) derived from base seed and indices.

    Uses Knuth-style multipliers to mix the base seed with category indices,
    ensuring different (lead_index, sublead_index) pairs produce independent seeds
    while remaining bit-reproducible for a fixed base seed.

    Args:
        base_seed (int): Base RNG seed (e.g., 3543136929)
        lead_index (int): Index of the leading electron
        sublead_index (int): Index of the subleading electron

    Returns:
        int: Derived seed masked to uint32 range [0, 0xFFFFFFFF]
    """
    # Two large odd Knuth-style multipliers — independent across (i, j)
    mixed = (
        int(base_seed)
        ^ (int(lead_index) * 0x9E3779B1)
        ^ (int(sublead_index) * 0x85EBCA77)
    )
    return mixed & 0xFFFFFFFF


class zcat:
    """
    Produces a 'z category' object to be used in the scales and smearing derivation.
    """

    def __init__(self, i, j, data, mc, weights, **options):
        """
        Initialize a z category object.

        Args:
            i (int): index of the first electron in the category
            j (int): index of the second electron in the category
            data (np.array): invariant mass of the data
            mc (np.array): invariant mass of the mc
            weights (np.array): weights of the mc
            **options (dict): optional arguments
        Returns:
            None
        """
        self.lead_index = i
        self.sublead_index = j
        self.lead_smear_index = (
            options["smear_i"] if "smear_i" in options.keys() else -1
        )
        self.sublead_smear_index = (
            options["smear_j"] if "smear_j" in options.keys() else -1
        )
        self.data = np.array(data, dtype=np.float32)
        self.mc = np.array(mc, dtype=np.float32)
        print(
            "[INFO][zcat][init] category ({},{}, data = {}, mc = {})".format(
                self.lead_index, self.sublead_index, len(self.data), len(self.mc)
            )
        )
        self.weights = np.array(weights, dtype=np.float32)
        self.hist_min = options["hist_min"] if "hist_min" in options.keys() else 80.0
        self.hist_max = options["hist_max"] if "hist_max" in options.keys() else 100.0
        self.auto_bin = options["_kAutoBin"] if "_kAutoBin" in options.keys() else True
        self.bin_size = options["bin_size"] if "bin_size" in options.keys() else 0.25
        self.updated = False
        self.NLL = 0
        # Diagonal cats get weight 1; off-diag get the configured constant.
        # For 'sum-normalized' scheme the off-diag weight is renormalized
        # later by normalize_off_diagonal_weights() once all cats exist.
        off_diag_w = options.get("off_diag_weight", 0.1)
        self.weight = 1.0 if i == j else float(off_diag_w)
        base_seed = options.get("base_seed", 3543136929)
        self.seed = _derive_cat_seed(base_seed, self.lead_index, self.sublead_index)
        self.valid = True
        self.bins = np.array([])

        # set the bin size if auto binning is enabled
        if self.auto_bin and self.bin_size == 0.25:
            # prune and check data and mc for validity
            temp_data = self.data[
                np.logical_and(self.hist_min <= self.data, self.data <= self.hist_max)
            ]
            mask_mc = np.logical_and(self.mc >= self.hist_min, self.mc <= self.hist_max)
            temp_weights = self.weights[mask_mc]
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

        # Precompute hot-path data that doesn't change during optimization
        if self.valid:
            if self.bin_size <= 0:
                self.bin_size = 0.25  # fallback for degenerate IQR
            self._num_bins = int(
                round((self.hist_max - self.hist_min) / self.bin_size, 0)
            )
            self._bin_edges = numba_hist.make_bin_edges(
                self.hist_min, self.hist_max, self._num_bins
            )
            scheme = options.get("loss_weighting", "uniform")
            if scheme == "triangular":
                self._emd_weights = build_emd_weights(self._num_bins)
            else:
                # "uniform" — hard window. Defaults to uniform if option missing.
                self._emd_weights = build_uniform_weights(self._num_bins)
            self._randn_lead, self._randn_sublead = _generate_smearing_randn(
                len(self.mc), self.seed
            )
            self._data_sentinels = np.array(
                [self.hist_min, self.hist_max], dtype=np.float32
            )
            self._weight_sentinels = np.array([0, 0], dtype=np.float32)

    def clean_up(self):
        """
        Set all variables to None to free up memory.
        """
        self.data = None
        self.mc = None
        self.weights = None
        self.bins = None
        self._randn_lead = None
        self._randn_sublead = None
        self._bin_edges = None
        self._emd_weights = None
        self._data_sentinels = None
        self._weight_sentinels = None
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
            lead_smear_list = np.array(
                np.random.normal(1, np.abs(lead_smear), len(self.data)),
                dtype=np.float32,
            )
            sublead_smear_list = np.array(
                np.random.normal(1, np.abs(sublead_smear), len(self.data)),
                dtype=np.float32,
            )
            self.data = self.data * np.sqrt(
                np.multiply(lead_smear_list, sublead_smear_list, dtype=np.float32),
                dtype=np.float32,
            )
        return

    def get_smeared_mc(self, mc, lead_smear, sublead_smear, seed) -> np.array:
        """
        Returns the smeared mc.

        Args:
            mc (np.array): invariant mass of the mc
            lead_smear (float): smearing for the leading electron
            sublead_smear (float): smearing for the subleading electron
            seed (int): seed for the random number generator
        Returns:
            np.array: smeared mc
        """
        np.random.seed(seed)
        lead_smear_list = (
            np.array(np.random.normal(1, np.abs(lead_smear), len(mc)), dtype=np.float32)
            if lead_smear != 0
            else np.ones(len(mc), dtype=np.float32)
        )
        sublead_smear_list = (
            np.array(
                np.random.normal(1, np.abs(sublead_smear), len(mc)), dtype=np.float32
            )
            if sublead_smear != 0
            else np.ones(len(mc), dtype=np.float32)
        )
        return np.multiply(
            mc,
            np.sqrt(
                np.multiply(lead_smear_list, sublead_smear_list, dtype=np.float32),
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    def update(self, lead_scale, sublead_scale, lead_smear=0, sublead_smear=0):
        """
        Update the z category with new scales and smearings.

        Args:
            lead_scale (float): scale for the leading electron
            sublead_scale (float): scale for the subleading electron
            lead_smear (float): smearing for the leading electron
            sublead_smear (float): smearing for the subleading electron
        Returns:
            None
        """
        self.updated = True

        # apply the scales first
        lead_scale = 1.0 if lead_scale == 0 else lead_scale
        sublead_scale = 1.0 if sublead_scale == 0 else sublead_scale

        scale_factor = np.sqrt(lead_scale * sublead_scale, dtype=np.float32)
        temp_data = self.data * scale_factor

        # apply the smearings second (use cached random vectors)
        temp_mc = (
            self.mc
            if lead_smear == 0 and sublead_smear == 0
            else apply_smearing_cached(
                self.mc,
                lead_smear,
                sublead_smear,
                self._randn_lead,
                self._randn_sublead,
            )
        )

        # prune the data and add sentinel entries at either end of the histogram range
        # these end entries ensure the same number of bins in data and mc
        mask_data = (self.hist_min <= temp_data) & (temp_data <= self.hist_max)
        mask_mc = (self.hist_min <= temp_mc) & (temp_mc <= self.hist_max)

        temp_data = np.concatenate([temp_data[mask_data], self._data_sentinels])
        temp_mc = np.concatenate([temp_mc[mask_mc], self._data_sentinels])
        temp_weights = np.concatenate([self.weights[mask_mc], self._weight_sentinels])

        # use precomputed bin edges instead of recomputing from data
        binned_data = numba_hist.numba_histogram_with_edges(temp_data, self._bin_edges)
        binned_mc = numba_hist.numba_weighted_histogram_with_edges(
            temp_mc, temp_weights, self._bin_edges
        )

        if self.check_invalid(temp_data, temp_mc):
            print(
                "[INFO][zcat][update] category ({},{}) was deactivated due to insufficient statistics in data".format(
                    self.lead_index, self.sublead_index
                )
            )
            self.clean_up()
            return

        # clean binned data and mc
        binned_mc[binned_mc == 0] = 1e-15

        # normalize mc to use as a pdf
        norm_binned_mc = binned_mc / np.sum(binned_mc)

        # compute the EMD with precomputed weights
        self.NLL = compute_earthmovers_distance(
            binned_data, norm_binned_mc, self._emd_weights
        )

        if np.isnan(self.NLL):
            # if the NLL is nan, set the category to invalid
            self.clean_up()
