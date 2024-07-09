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
    lead_rand = np.random.normal(1, lead_smear, len(mc))
    sublead_rand = np.random.normal(1, sublead_smear, len(mc))
    x = np.sqrt((1 + lead_rand) * (1 + sublead_rand))
    return mc * x

@numba.njit
def apply_scale(data, lead_scale, sublead_scale):
    return data * np.sqrt(lead_scale * sublead_scale)

@numba.njit
def compute_nll_chisqr(binned_data, norm_binned_mc, num_bins=80):
    # Implement NLL and Chi-squared computation here
    # This is a placeholder, replace with actual implementation
    scaled_mc = norm_binned_mc*np.sum(binned_data)
    err_mc = np.sqrt(scaled_mc).astype(np.float32)
    err_data = np.sqrt(binned_data).astype(np.float32)
    err = np.sqrt(np.add(np.multiply(err_mc,err_mc).astype(np.float32), np.multiply(err_data,err_data).astype(np.float32)).astype(np.float32)).astype(np.float32)
    chi_sqr = np.sum( np.divide(np.multiply(binned_data-scaled_mc,binned_data-scaled_mc).astype(np.float32),err).astype(np.float32))/num_bins    

    nll = xlogy(binned_data, norm_binned_mc)
    nll[nll==-np.inf] = 0
    nll = np.sum(nll)/len(nll)
    #evaluate penalty
    penalty = xlogy(np.sum(binned_data)-binned_data, 1 - norm_binned_mc)
    penalty[penalty==-np.inf] = 0
    penalty = np.sum(penalty)/len(penalty)
    return -2*(nll + penalty)*chi_sqr


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
        self.lead_index=i
        self.sublead_index=j
        self.lead_smear_index=options['smear_i'] if 'smear_i' in options.keys() else -1
        self.sublead_smear_index=options['smear_j'] if 'smear_j' in options.keys() else -1
        self.data = np.array(data, dtype=np.float32)
        self.mc = np.array(mc, dtype=np.float32)
        print("[INFO][zcat][init] category ({},{}, data = {}, mc = {})".format(self.lead_index, self.sublead_index,len(self.data),len(self.mc)))
        self.weights = np.array(weights, dtype=np.float32)
        self.hist_min = options['hist_min'] if 'hist_min' in options.keys() else 80.
        self.hist_max = options['hist_max'] if 'hist_max' in options.keys() else 100.
        self.auto_bin = options['_kAutoBin'] if '_kAutoBin' in options.keys() else True
        self.bin_size = options['bin_size'] if 'bin_size' in options.keys() else 0.25
        self.updated = False
        self.NLL = 0
        self.weight = 1 if i == j else 0.1 #  penalize off diagonal fits
        self.seed = 3543136929 #  use a fixed random integer for your seed to avoid fluctuations in nll value from smearings
        self.valid=True
        self.bins=np.array([])

        # set the bin size if auto binning is enabled
        if self.auto_bin and self.bin_size == 0.25:
            # prune and check data and mc for validity
            temp_data = self.data[np.logical_and(self.hist_min <= self.data, self.data <= self.hist_max)]
            mask_mc = np.logical_and(self.mc >= self.hist_min,self.mc <= self.hist_max)
            temp_weights = self.weights[mask_mc]
            temp_mc = self.mc[mask_mc]
            if self.check_invalid(temp_data, temp_mc):
                print("[INFO][zcat][init] category ({},{}) was deactivated due to insufficient statistics in data".format(self.lead_index, self.sublead_index))
                self.clean_up()
                return
            
            data_width = 2*stats.iqr(temp_data, rng=(25,75), nan_policy="omit")/np.power(len(temp_data), 1./3.)
            mc_width = 2*stats.iqr(temp_mc, rng=(25,75), nan_policy="omit")/np.power(len(temp_mc), 1./3.)
            self.bin_size = max( data_width, mc_width) # always choose the larger binning scheme

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
        return (len(data) < cc.MIN_EVENTS_DATA or len(mc) < cc.MIN_EVENTS_MC_DIAG or (len(mc) < cc.MIN_EVENTS_MC_OFFDIAG and self.lead_index != self.sublead_index))

    def print(self):
        """Print the z category object."""
        print("lead index:", self.lead_index)
        print("sublead index:",self.sublead_index)
        print("lead smearing index:", self.lead_smear_index)
        print("sublead smearing index:",self.sublead_smear_index)
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
        self.data = self.data*np.sqrt(lead_scale*sublead_scale, dtype=np.float32)
        if lead_smear != 0 and sublead_smear != 0:
            lead_smear_list = np.random.normal(1, np.abs(lead_smear), len(self.data), dtype=np.float32)
            sublead_smear_list = np.random.normal(1, np.abs(sublead_smear), len(self.data), dtype=np.float32)
            self.data = self.data*np.sqrt(np.multiply(lead_smear_list,sublead_smear_list, dtype=np.float32), dtype=np.float32)
        return

    def update(self, lead_scale, sublead_scale, lead_smear=0, sublead_smear=0) -> None:
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
        self.updated=True

        # apply the scales first 
        lead_scale = 1.0 if lead_scale == 0 else lead_scale
        sublead_scale = 1.0 if sublead_scale == 0 else sublead_scale

        temp_data = apply_scale(self.data, lead_scale, sublead_scale)

        temp_mc = self.mc
        temp_weights = self.weights
        
        # apply the smearings second
        temp_mc = self.mc if lead_smear == 0 and sublead_smear == 0 else apply_smearing(self.mc, lead_smear, sublead_smear, self.seed)


        # prune the data and add a single entry at either end of the histogram range
        # these end entries ensure the same number of bins in data and mc returned by np.bincount
        mask_data = (self.hist_min <= temp_data) & (temp_data <= self.hist_max)
        mask_mc = (self.hist_min <= temp_mc) & (temp_mc <= self.hist_max)

        temp_data = np.concatenate([temp_data[mask_data], [self.hist_min, self.hist_max]])
        temp_mc = np.concatenate([temp_mc[mask_mc], [self.hist_min, self.hist_max]])
        temp_weights = np.concatenate([self.weights[mask_mc], [0, 0]])


        num_bins = int(round((self.hist_max-self.hist_min)/self.bin_size,0))
        binned_data,edges = numba_hist.numba_histogram(temp_data,num_bins)
        binned_mc,edges = numba_hist.numba_weighted_histogram(temp_mc,temp_weights,num_bins)

        if self.check_invalid(temp_data, temp_mc):
            print("[INFO][zcat][update] category ({},{}) was deactivated due to insufficient statistics in data".format(self.lead_index, self.sublead_index))
            self.clean_up()
            return

        # clean binned data and mc, log of 0 is a problem
        binned_mc[binned_mc == 0] = 1e-15

        # normalize mc to use as a pdf
        norm_binned_mc = binned_mc/np.sum(binned_mc)

        # compute the NLL
        self.NLL = compute_nll_chisqr(binned_data, norm_binned_mc, num_bins)

        if np.isnan(self.NLL):
            # if the NLL is nan, set the category to invalid
            self.clean_up()


