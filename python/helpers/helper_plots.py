import numpy as np
import pandas as pd

def get_systematic_uncertainty(bins, data, data_up, data_down, mc, mc_up, mc_down, mc_weights):

    d, _ = np.histogram(data, bins=bins, range=[80,100])
    d_up, _ = np.histogram(data_up, bins=bins, range=[80.,100.])
    d_down, _ = np.histogram(data_down, bins=bins, range=[80.,100.])
    m, _ = np.histogram(mc, bins=bins, range=[80,100], weights=mc_weights)
    m = m*sum(d)/sum(m)
    m_up, _ = np.histogram(mc_up, bins=bins, range=[80.,100.], weights=mc_weights)
    m_up = m_up*sum(m)/sum(m_up)
    m_down, _ = np.histogram(mc_down, bins=bins, range=[80.,100.], weights=mc_weights)
    m_down = m_down*sum(m)/sum(m_down)

    diff_up_data = np.abs(np.subtract(d, d_up))
    diff_down_data = np.abs(np.subtract(d, d_down))
    diff_up_mc = np.abs(np.subtract(m, m_up))
    diff_down_mc = np.abs(np.subtract(m, m_down))

    max_data = np.maximum(diff_up_data, diff_down_data)
    max_mc = np.maximum(diff_up_mc, diff_down_mc)

    return np.sqrt(np.add(np.power(max_data,2),np.power(max_mc,2)))
