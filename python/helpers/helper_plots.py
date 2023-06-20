import numpy as np
import pandas as pd

from python.classes.constant_classes import DataConstants as dc

def get_bin_uncertainties(bins, values, weights):
    """ 
    Calculates the uncertainty of weighted bins 
    ----------
    Args:
        bins: bin edges
        values: values to bin
        weights: weights of the values
    ----------
    Returns:
        ret: array of uncertainties
    ----------    
    """

    ret = []
    for i in range(len(bins)-1):
        val_mask = np.logical_and(bins[i] <= values, values < bins[i+1])
        ret.append(np.sqrt(np.sum(np.power(weights[val_mask], 2))))

    return np.array(ret)

def get_systematic_uncertainty(bins, data, data_up, data_down, mc, mc_weights):
    """
    Calculates the systematic uncertainties for data and MC
    ----------
    Args:
        bins: bin edges
        data: data values
        data_up: data values with systematic uncertainty up
        data_down: data values with systematic uncertainty down
        mc: MC values
        mc_up: MC values with systematic uncertainty up
        mc_down: MC values with systematic uncertainty down
        mc_weights: MC weights
    ----------
    Returns:
        ret: array of uncertainties
    ----------
    """

    d, d_bins = np.histogram(data, bins=bins, range=[dc.MIN_INVMASS,dc.MAX_INVMASS])
    d_up, _ = np.histogram(data_up, bins=d_bins)
    d_up = d_up*sum(d)/sum(d_up)
    d_down, _ = np.histogram(data_down, bins=d_bins)
    d_down = d_down*sum(d)/sum(d_down)
    m, m_bins = np.histogram(mc, bins=bins, range=[dc.MIN_INVMASS,dc.MAX_INVMASS], weights=mc_weights)

    diff_up_data = np.abs(np.subtract(d, d_up))
    diff_down_data = np.abs(np.subtract(d, d_down))

    max_data = np.maximum(diff_up_data, diff_down_data)

    return max_data
