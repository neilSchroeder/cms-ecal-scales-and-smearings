import numpy as np
import pandas as pd
from scipy import stats 
from scipy.special import xlogy

import python.utilities.numba_hist as numba_hist

class zcat:
    """ produces a 'z category' object to be used in the scales and smearing derivation """

    def __init__(self, i, j, data, mc, weights, **options):
        self.lead_index=i
        self.sublead_index=j
        self.lead_smear_index=options['smear_i'] if 'smear_i' in options.keys() else -1
        self.sublead_smear_index=options['smear_j'] if 'smear_j' in options.keys() else -1
        self.data = np.array(data, dtype=np.float32)
        self.mc = np.array(mc, dtype=np.float32)
        self.weights = np.array(weights, dtype=np.float32)
        self.smearings = np.array()
        self.last_lead_smear = 0
        self.last_sub_smear = 0
        self.hist_min = options['hist_min'] if 'hist_min' in options.keys() else 80.
        self.hist_max = options['hist_max'] if 'hist_max' in options.keys() else 100.
        self.auto_bin = options['_kAutoBin'] if '_kAutoBin' in options.keys() else True
        self.bin_size = options['bin_size'] if 'bin_size' in options.keys() else 0.25
        self.updated = False
        self.NLL = 0
        self.weight = 1
        self.seed = 3543136929 #use a fixed random integer for your seed to avoid fluctuations in nll value from smearings
        self.valid=True
        self.bins=np.array([])

    def __delete__(self):
        del self.lead_index
        del self.sublead_index
        del self.data
        del self.mc
        del self.updated
        del self.NLL
        del self.hist_min
        del self.hist_max
        del self.auto_bin
        del self.bin_size
        del self.seed
        del self.weight
        del self.valid

    def print(self):
        print("lead index:", self.lead_index)
        print("sublead index:",self.sublead_index)
        print("lead smearing index:", self.lead_smear_index)
        print("sublead smearing index:",self.sublead_smear_index)
        print("nevents, data:", len(self.data))
        print("nevents, mc: ", len(self.mc))
        print("NLL:", self.NLL, " || w/bin size:", self.bin_size)
        print("weight:", self.weight)
        print("valid:", self.valid)


    def reset(self): 
        self.updated=False
        return

    def inject(self, lead_scale, sublead_scale, lead_smear, sublead_smear):
        #this function artificially injects scales and smearings in to the "toy mc" labelled here as data
        self.data = self.data*np.sqrt(lead_scale*sublead_scale, dtype=np.float32)
        if lead_smear != 0 and sublead_smear != 0:
            lead_smear_list = np.random.normal(1, np.abs(lead_smear), len(self.data), dtype=np.float32)
            sublead_smear_list = np.random.normal(1, np.abs(sublead_smear), len(self.data), dtype=np.float32)
            self.data = self.data*np.sqrt(np.multiply(lead_smear_list,sublead_smear_list, dtype=np.float32), dtype=np.float32)
        return

    def smear(self, mc, lead_smear, sublead_smear, seed):
        np.random.seed(seed)
        lead_smear_list = np.array(np.random.normal(1, np.abs(lead_smear), len(mc)), dtype=np.float32) if lead_smear != 0 else np.ones(len(mc), dtype=np.float32)
        sublead_smear_list = np.array(np.random.normal(1, np.abs(sublead_smear), len(mc)), dtype=np.float32) if sublead_smear != 0 else np.ones(len(mc), dtype=np.float32)
        return np.multiply(mc, np.sqrt(np.multiply(lead_smear_list,sublead_smear_list, dtype=np.float32), dtype=np.float32), dtype=np.float32)

    def get_nllChiSqr(self, binned_data, norm_binned_mc):
        #eval chi squared
        scaled_mc = norm_binned_mc*np.sum(binned_data)
        err_mc = np.sqrt(scaled_mc, dtype=np.float32)
        err_data = np.sqrt(binned_data, dtype=np.float32)
        err = np.sqrt(np.add(np.multiply(err_mc,err_mc, dtype=np.float32), np.multiply(err_data,err_data, dtype=np.float32), dtype=np.float32), dtype=np.float32)
        num_bins = int(round((self.hist_max-self.hist_min)/self.bin_size,0))
        chi_sqr = np.sum( np.divide(np.multiply(binned_data-scaled_mc,binned_data-scaled_mc, dtype=np.float32),err, dtype=np.float32))/num_bins

        #evalute nll
        nll = xlogy(binned_data, norm_binned_mc)
        nll[nll==-np.inf] = 0
        nll = np.sum(nll)/len(nll)
        #evaluate penalty
        penalty = xlogy(np.sum(binned_data)-binned_data, 1 - norm_binned_mc)
        penalty[penalty==-np.inf] = 0
        penalty = np.sum(penalty)/len(penalty)
        return -2*(nll + penalty)*chi_sqr

    def update(self, lead_scale, sublead_scale, lead_smear=0, sublead_smear=0):
        #updates the value of nll for the class using the appropriate scales and smearings
        self.updated=True

        #apply the scales first 
        temp_data = np.copy(self.data) * np.sqrt(lead_scale*sublead_scale, dtype=np.float32)
        temp_weights = np.copy(self.weights)
        
        temp_mc = np.copy(self.mc)
        #apply the smearings second
        if lead_smear!=0 or sublead_smear!=0:
            temp_mc = self.smear(temp_mc, lead_smear, sublead_smear, self.seed) 

        #determinite binning using the Freedman-Diaconis rule
        #data_width, mc_width = get_binning()
        if self.auto_bin and self.bin_size == 0.25:
            #prune and check data and mc for validity
            temp_data = temp_data[np.logical_and(self.hist_min <= temp_data, self.hist_max <= self.hist_max)]
            mask_mc = np.logical_and(temp_mc >= self.hist_min,temp_mc <= self.hist_max)
            temp_weights = temp_weights[mask_mc]
            temp_mc = temp_mc[mask_mc]
            if len(temp_data) < 10 or len(temp_mc) < 1000: 
                print("[INFO][zcat] category ({},{}, data = {}, mc = {}) was deactivated due to insufficient statistics".format(self.lead_index, self.sublead_index,len(temp_data),len(temp_mc)))
                self.NLL = 0
                self.valid=False
                del self.data
                del self.mc
                del self.weights
                del temp_data
                del temp_mc
                del temp_weights
                return
            if len(temp_mc) < 2000 and self.lead_index != self.sublead_index: 
                print("[INFO][zcat] category ({},{}, data = {}, mc = {}) was deactivated due to insufficient statistics in MC".format(self.lead_index, self.sublead_index,len(temp_data),len(temp_mc)))
                self.NLL = 0
                self.valid=False
                del self.data
                del self.mc
                del self.weights
                del temp_data
                del temp_mc
                del temp_weights
                return
            #since the data and mc are now pruned go ahead and find the bin size
            data_width = 2*stats.iqr(temp_data, rng=(25,75), scale="raw", nan_policy="omit")/np.power(len(temp_data), 1./3.)
            mc_width = 2*stats.iqr(temp_mc, rng=(25,75), scale="raw", nan_policy="omit")/np.power(len(temp_mc), 1./3.)
            self.bin_size = max( data_width, mc_width) #always choose the larger binning scheme

        #prune the data and add a single entry at either end of the histogram range
        #these end entries ensure the same number of bins in data and mc returned by np.bincount
        temp_data = temp_data[np.logical_and(self.hist_min <= temp_data, temp_data <= self.hist_max)]
        temp_data = np.append(temp_data,np.array([self.hist_min,self.hist_max], dtype=np.float32))
        mask_mc = np.logical_and(self.hist_min <= temp_mc, temp_mc <= self.hist_max)
        temp_weights = temp_weights[mask_mc]
        temp_mc = temp_mc[mask_mc]
        temp_weights = np.append(temp_weights,np.array([0,0], dtype=np.float32))
        temp_mc = np.append(temp_mc,np.array([self.hist_min,self.hist_max]))

        num_bins = int(round((self.hist_max-self.hist_min)/self.bin_size,0))
        binned_data,edges = numba_hist.numba_histogram(temp_data,num_bins)
        binned_mc,edges = numba_hist.numba_weighted_histogram(temp_mc,temp_weights,num_bins)

        if np.sum(binned_data) < 10:
            print("[INFO][zcat] category ({},{}) was deactivated due to insufficient statistics in data".format(self.lead_index, self.sublead_index))
            self.NLL = 0
            self.valid=False
            del self.data
            del self.mc
            del self.weights
            del temp_mc
            del temp_weights
            return
        if len(temp_mc) < 1000 and self.lead_index==self.sublead_index:
            print("[INFO][zcat] category ({},{}, data = {}, mc = {}) was deactivated due to insufficient statistics in MC".format(self.lead_index, self.sublead_index, len(temp_data),len(temp_mc)))
            self.NLL = 0
            self.valid=False
            del self.data
            del self.mc
            del self.weights
            del temp_mc
            del temp_weights
            return
        if len(temp_mc) < 2000 and self.lead_index!=self.sublead_index:
            print("[INFO][zcat] category ({},{}) was deactivated due to insufficient statistics in MC".format(self.lead_index, self.sublead_index))
            self.NLL = 0
            self.valid=False
            del self.data
            del self.mc
            del self.weights
            del temp_mc
            del temp_weights
            return

        #clean binned data and mc, log of 0 is a problem
        binned_mc[binned_mc == 0] = 1e-15

        #normalize mc to use as a pdf
        norm_binned_mc = binned_mc/np.sum(binned_mc)

        self.NLL = self.get_nllChiSqr(binned_data, norm_binned_mc)
        #penalize off-diagonal categories in the fit
        #self.weight = np.sum(binned_data) if self.lead_index == self.sublead_index else 0.01*np.sum(binned_data)
        if np.isnan(self.NLL):
            self.valid = False
            self.NLL = 0
            del self.data
            del self.mc
            del self.weights

        return

