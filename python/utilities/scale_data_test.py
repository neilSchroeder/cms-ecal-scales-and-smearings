import numpy as np
import pandas as pd

from python.classes.constant_classes import DataConstants as dc
from python.classes.constant_classes import PyValConstants as pvc
import python.utilities.data_loader as data_loader

def prepare_scales_lookup(scales_df):
    """
    Prepares a lookup table for scale corrections.

    Parameters
    ----------
    corrections_df : pd.DataFrame
        DataFrame with columns 'min_run', 'max_run', 'min_eta', 'max_eta', 'min_r9', 'max_r9', 'min_et', 'max_et', 'gain', 'scale', 'err'

    Returns
    -------
    """
    # Assume corrections_df has columns 'min_run', 'max_run', 'min_eta', 'max_eta', 'min_r9', 'max_r9', 'min_et', 'max_et', 'scale'
    scales_df = scales_df.sort_values([dc.i_run_min, dc.i_eta_min, dc.i_r9_min, dc.i_et_min])
    
    # Create bin edges
    run_edges = np.unique(np.concatenate([scales_df[dc.i_run_min], scales_df[dc.i_run_max]]))
    eta_edges = np.unique(np.concatenate([scales_df[dc.i_eta_min], scales_df[dc.i_eta_max]]))
    r9_edges = np.unique(np.concatenate([scales_df[dc.i_r9_min], scales_df[dc.i_r9_max]]))
    et_edges = np.unique(np.concatenate([scales_df[dc.i_et_min], scales_df[dc.i_et_max]]))
    gain_edges = np.unique(scales_df[dc.i_gain])

    #TODO implement gain functionality
    
    # Create lookup array
    lookup = np.full((len(run_edges)-1, len(eta_edges)-1, len(r9_edges)-1, len(et_edges)-1), len(gain_edges)-1, np.nan)
    for _, row in scales_df.iterrows():
        run_idx = np.searchsorted(run_edges, row['min_run'])
        eta_idx = np.searchsorted(eta_edges, row['min_eta'])
        r9_idx = np.searchsorted(r9_edges, row['min_r9'])
        et_idx = np.searchsorted(et_edges, row['min_et'])
        gain_idx = np.searchsorted(gain_edges, row['gain'])
        lookup[run_idx, eta_idx, r9_idx, et_idx, gain_idx] = (row['scale'], row['err'])
    
    return run_edges, eta_edges, r9_edges, et_edges, gain_edges, lookup

def apply_corrections(data, run_edges, eta_edges, r9_edges, et_edges, gain_edges, lookup):
    # Assume events_df has columns 'x' and 'y'
    run_indices = np.digitize(data['run'], run_edges) - 1
    eta_indices = np.digitize(data['eta'], eta_edges) - 1
    r9_indices = np.digitize(data['r9'], r9_edges) - 1
    et_indices = np.digitize(data['et'], et_edges) - 1
    gain_indices = np.digitize(data['gain'], gain_edges) - 1

    # Clip indices to valid range
    run_indices = np.clip(run_indices, 0, len(run_edges)-2)
    eta_indices = np.clip(eta_indices, 0, len(eta_edges)-2)
    r9_indices = np.clip(r9_indices, 0, len(r9_edges)-2)
    et_indices = np.clip(et_indices, 0, len(et_edges)-2)
    gain_indices = np.clip(gain_indices, 0, len(gain_edges)-2)

    # Apply scales
    scales = lookup[run_indices, eta_indices, r9_indices, et_indices, gain_indices]
    
    # Handle any events that fall outside the correction bins
    mask = np.isnan(scales)
    scales[mask] = 1.0  # or any other default value
    
    return scales


def scale(data, scales):
    """
    This function applies the scales in a multi-threaded way.

    Args:
        data (pd.DataFrame): dataframe to apply scales to
        scales (str): path to scales file
    Returns:
        data (pd.DataFrame): dataframe with scales applied
    """
    info = "[INFO][scale_data.py]"
    # newformat of scales files is 
    # runMin runMax etaMin etaMax r9Min r9Max etMin etMax gain val err

    # read in scales to df
    scales_df = pd.read_csv(scales, sep="\t", comment="#", header=None)

    # add 'lead_et' and 'sublead_et' columns
    data['lead_et'] = data[dc.E_LEAD] / np.cosh(data[dc.ETA_LEAD])
    data['sublead_et'] = data[dc.E_SUB] / np.cosh(data[dc.ETA_SUB])

    # split data into lead_data and sublead_data
    lead_data = data[[dc.RUN, dc.ETA_LEAD, dc.R9_LEAD, 'lead_et', dc.GAIN_LEAD]]
    sublead_data = data[[dc.RUN, dc.ETA_SUB, dc.R9_SUB, 'sublead_et', dc.GAIN_SUB]]

    # replace column names with 'run', 'eta', 'r9', 'et', 'gain'
    lead_data.columns = ['run', 'eta', 'r9', 'et', 'gain']
    sublead_data.columns = ['run', 'eta', 'r9', 'et', 'gain']

    # replace gain values: 0 -> 12, 1 -> 6, else -> 1
    gain12 = lead_data['gain'] == 0
    gain6 = lead_data['gain'] == 1
    gain1 = ~(gain12 | gain6)
    lead_data.loc[gain12, 'gain'] = 12
    lead_data.loc[gain6, 'gain'] = 6
    lead_data.loc[gain1, 'gain'] = 1

    gain12 = sublead_data['gain'] == 0
    gain6 = sublead_data['gain'] == 1
    gain1 = ~(gain12 | gain6)
    sublead_data.loc[gain12, 'gain'] = 12
    sublead_data.loc[gain6, 'gain'] = 6
    sublead_data.loc[gain1, 'gain'] = 1

    # apply corrections
    lead_data['scale'] = apply_corrections(lead_data, *prepare_scales_lookup(scales_df))
    sublead_data['scale'] = apply_corrections(sublead_data, *prepare_scales_lookup(scales_df))

    # calculate new energies, errors, and invmasses
    # lead_data['scale'] returns a tuple of (scale, err)
    data[dc.E_LEAD] = data[dc.E_LEAD] * lead_data['scale'][:,0]
    data[dc.E_SUB] = data[dc.E_SUB] * sublead_data['scale'][:,0]
    invmass = data[dc.INVMASS].values.copy()
    data[pvc.KEY_INVMASS_UP] = invmass * np.sqrt(
        np.multiply(
            np.add(lead_data['scale'][:,0], lead_data['scale'][:,1]),
            np.add(sublead_data['scale'][:,0], sublead_data['scale'][:,1])
        )
    )
    data[pvc.KEY_INVMASS_DOWN] = invmass * np.sqrt(
        np.multiply(
            np.subtract(lead_data['scale'][:,0], lead_data['scale'][:,1]),
            np.add(sublead_data['scale'][:,0], sublead_data['scale'][:,1])
        )
    )
    data[dc.INVMASS] = invmass * np.sqrt(
        np.multiply(lead_data['scale'][:,0], sublead_data['scale'][:,0])
    )

    return data_loader.custom_cuts(
                                    data,
                                    inv_mass_cuts=(80, 100),
                                    eta_cuts=(0, 1.4442, 1.566, 2.5),
                                    et_cuts=((32, 14000), (20, 14000)),
    ) 
    