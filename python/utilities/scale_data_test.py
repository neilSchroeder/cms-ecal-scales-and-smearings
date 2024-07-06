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
    run_edges = np.unique(np.concatenate([scales_df[dc.i_run_min].values - 0.1, np.array([999999])]))
    eta_edges = np.unique(scales_df[[dc.i_eta_min, dc.i_eta_max]].values)
    r9_edges = np.unique(scales_df[[dc.i_r9_min, dc.i_r9_max]].values)
    et_edges = np.unique(scales_df[[dc.i_et_min, dc.i_et_max]].values) if all(scales_df[dc.i_gain] == 0) else np.array([0.5, 5.5, 6.5, 12.5])

    # Create lookup array
    lookup_scales = np.full((len(run_edges)-1, len(eta_edges)-1, len(r9_edges)-1, len(et_edges)-1), np.nan)
    lookup_errs = np.full((len(run_edges)-1, len(eta_edges)-1, len(r9_edges)-1, len(et_edges)-1), np.nan)

    for _, row in scales_df.iterrows():
        run_idx = np.searchsorted(run_edges, row[dc.i_run_min], side='right') - 1
        eta_idx = np.searchsorted(eta_edges, row[dc.i_eta_min], side='right') - 1
        r9_idx = np.searchsorted(r9_edges, row[dc.i_r9_min], side='right') - 1
        et_idx = np.searchsorted(et_edges, row[dc.i_et_min], side='right') - 1

        if any([run_idx < 0, eta_idx < 0, r9_idx < 0, et_idx < 0]):
            print(f"[ERROR][scale_data.py] Negative index found: {run_idx}, {eta_idx}, {r9_idx}, {et_idx}")
            print(row)
            continue

        # Ensure indices are within bounds
        run_idx = min(run_idx, len(run_edges) - 2)
        eta_idx = min(eta_idx, len(eta_edges) - 2)
        r9_idx = min(r9_idx, len(r9_edges) - 2)
        et_idx = min(et_idx, len(et_edges) - 2)

        lookup_scales[run_idx, eta_idx, r9_idx, et_idx] = row[dc.i_scale]
        lookup_errs[run_idx, eta_idx, r9_idx, et_idx] = row[dc.i_err]

    # Print the number of NaN values
    nan_count = np.isnan(lookup_scales).sum()
    print(f"Number of NaN values in lookup_scales: {nan_count}")

    return run_edges, eta_edges, r9_edges, et_edges, lookup_scales, lookup_errs

def apply_corrections(data, run_edges, eta_edges, r9_edges, et_edges, lookup_scales, lookup_errs):
    # Assume events_df has columns 'x' and 'y'
    # print every variable:
    run_indices = np.digitize(data['run'], run_edges) - 2
    eta_indices = np.digitize(data['eta'], eta_edges) - 1
    r9_indices = np.digitize(data['r9'], r9_edges) - 1
    et_indices = np.digitize(data['et'], et_edges) - 1

    # Clip indices to valid range
    run_indices = np.clip(run_indices, 0, len(run_edges)-2)
    eta_indices = np.clip(eta_indices, 0, len(eta_edges)-2)
    r9_indices = np.clip(r9_indices, 0, len(r9_edges)-2)
    et_indices = np.clip(et_indices, 0, len(et_edges)-2)

    #iterate over data

    # Apply scales
    scales = lookup_scales[run_indices, eta_indices, r9_indices, et_indices]
    print(scales)
    errs = lookup_errs[run_indices, eta_indices, r9_indices, et_indices]
    
    # Handle any events that fall outside the correction bins
    mask = np.isnan(scales)
    i = 0
    data['run_index'] = run_indices
    data['eta_index'] = eta_indices
    data['r9_index'] = r9_indices
    data['et_index'] = et_indices

    for idx, row in data[mask].iterrows():
        print(f"[INFO][scale_data.py] Event {idx} falls outside the correction bins")
        print(row)
        print(row['run'], row['run_index'], run_edges[int(row['run_index'])], run_edges[int(row['run_index'])+1])
        print(row['eta'], row['eta_index'], eta_edges[int(row['eta_index'])], eta_edges[int(row['eta_index'])+1])
        print(row['r9'], row['r9_index'], r9_edges[int(row['r9_index'])], r9_edges[int(row['r9_index'])+1])
        print(row['et'], row['et_index'], et_edges[int(row['et_index'])], et_edges[int(row['et_index'])+1])
        print(lookup_scales[int(row['run_index']), int(row['eta_index']), int(row['r9_index']), int(row['et_index'])])
        print("=====================================")
        i += 1
    print(f"[INFO][scale_data.py] {mask.sum()} events fall outside the correction bins")
    print(data[mask].describe())
    scales[mask] = -1.0  # or any other default value
    errs[mask] = 0.0  # or any other default value
    
    return scales, errs


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
    lead_data['scale'], lead_data['err'] = apply_corrections(lead_data, *prepare_scales_lookup(scales_df))
    sublead_data['scale'], sublead_data['err'] = apply_corrections(sublead_data, *prepare_scales_lookup(scales_df))

    # calculate new energies, errors, and invmasses
    # lead_data['scale'] returns a tuple of (scale, err)
    data[dc.E_LEAD] = data[dc.E_LEAD] * lead_data['scale']
    data[dc.E_SUB] = data[dc.E_SUB] * sublead_data['scale']
    invmass = data[dc.INVMASS].values.copy()
    data[pvc.KEY_INVMASS_UP] = invmass * np.sqrt(
        np.multiply(
            np.add(lead_data['scale'], lead_data['err']),
            np.add(sublead_data['scale'], sublead_data['err'])
        )
    )
    data[pvc.KEY_INVMASS_DOWN] = invmass * np.sqrt(
        np.multiply(
            np.subtract(lead_data['scale'], lead_data['err']),
            np.subtract(sublead_data['scale'], sublead_data['err'])
        )
    )
    data[dc.INVMASS] = invmass * np.sqrt(
        np.multiply(lead_data['scale'], sublead_data['scale'])
    )

    return data_loader.custom_cuts(
                                    data,
                                    inv_mass_cuts=(80, 100),
                                    eta_cuts=(0, 1.4442, 1.566, 2.5),
                                    et_cuts=((32, 14000), (20, 14000)),
    ) 
    