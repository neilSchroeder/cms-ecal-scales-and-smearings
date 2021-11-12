import numpy as np
import pandas as pd

def get_bin_uncertainties(bins, values, weights):
    """ calculates the uncertainty of weighted bins """

    ret = []
    for i in range(len(bins)-1):
        val_mask = np.logical_and(bins[i] <= values, values < bins[i+1])
        ret.append(np.sqrt(np.sum(np.power(weights[val_mask], 2))))

    return np.array(ret)

def get_systematic_uncertainty(bins, data, data_up, data_down, mc, mc_up, mc_down, mc_weights):

    d, d_bins = np.histogram(data, bins=bins, range=[80,100])
    d_up, _ = np.histogram(data_up, bins=d_bins)
    d_up = d_up*sum(d)/sum(d_up)
    d_down, _ = np.histogram(data_down, bins=d_bins)
    d_down = d_down*sum(d)/sum(d_down)
    m, m_bins = np.histogram(mc, bins=bins, range=[80,100], weights=mc_weights)
    m_up, _ = np.histogram(mc_up, bins=m_bins, weights=mc_weights)
    m_down, _ = np.histogram(mc_down, bins=m_bins, weights=mc_weights)

    diff_up_data = np.abs(np.subtract(d, d_up))
    diff_down_data = np.abs(np.subtract(d, d_down))
    diff_up_mc = np.abs(np.subtract(m, m_up))
    diff_down_mc = np.abs(np.subtract(m, m_down))

    max_data = np.maximum(diff_up_data, diff_down_data)
    max_mc = np.maximum(diff_up_mc, diff_down_mc)

    return np.sqrt(np.add(np.power(max_data,2),np.power(max_mc,2)))
