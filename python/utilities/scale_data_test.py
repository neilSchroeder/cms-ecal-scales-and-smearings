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
    scales_df : pd.DataFrame
        DataFrame with columns 'min_run', 'max_run', 'min_eta', 'max_eta', 'min_r9', 'max_r9', 'min_et', 'max_et', 'gain', 'scale', 'err'

    Returns
    -------
    tuple : (run_edges, eta_edges, r9_edges, et_edges, lookup_scales, lookup_errs)
    """
    # Print the input DataFrame to verify its structure and contents
    print("Scales DataFrame:")
    print(scales_df.head())

    scales_df = scales_df.sort_values([dc.i_run_min, dc.i_eta_min, dc.i_r9_min, dc.i_et_min])

    # Create bin edges
    run_edges = np.unique(np.concatenate([scales_df[dc.i_run_min].values - 0.1, np.array([999999])]))
    eta_edges = np.unique(scales_df[[dc.i_eta_min, dc.i_eta_max]].values)
    r9_edges = np.unique(scales_df[[dc.i_r9_min, dc.i_r9_max]].values)
    et_edges = np.unique(scales_df[[dc.i_et_min, dc.i_et_max]].values) if all(scales_df[dc.i_gain] == 0) else np.array([0.5, 5.5, 6.5, 12.5])

    # Create lookup array
    lookup_scales = np.full((len(run_edges)-1, len(eta_edges)-1, len(r9_edges)-1, len(et_edges)-1), np.nan)
    lookup_errs = np.full((len(run_edges)-1, len(eta_edges)-1, len(r9_edges)-1, len(et_edges)-1), np.nan)

    for idx, row in scales_df.iterrows():
        run_id = np.digitize(row[dc.i_run_min], run_edges) - 1
        eta_id = np.digitize(row[dc.i_eta_min]+1e-6, eta_edges) - 1
        r9_id = [np.digitize(row[dc.i_r9_min]+1e-6, r9_edges) - 1]
        if np.digitize(row[dc.i_r9_max]-1e-6, r9_edges) - 1  != r9_id:
            r9_id = [x for x in range(r9_id[0], np.digitize(row[dc.i_r9_max]-1e-6, r9_edges), 1)]
        # TODO: implement gain handling
        et_id = [np.digitize(row[dc.i_et_min]+1e-6, et_edges) - 1]
        if np.digitize(row[dc.i_et_max], et_edges) - 1 != et_id:
            et_id = [x for x in range(et_id[0], np.digitize(row[dc.i_et_max]-1e-6, et_edges), 1)]

        for r9 in r9_id:
            for et in et_id:
                lookup_scales[run_id, eta_id, r9, et] = row[dc.i_scale]
                lookup_errs[run_id, eta_id, r9, et] = row[dc.i_err]

    return run_edges, eta_edges, r9_edges, et_edges, lookup_scales, lookup_errs


def apply_corrections(data, run_edges, eta_edges, r9_edges, et_edges, lookup_scales, lookup_errs):
    # Assume events_df has columns 'x' and 'y'
    # print every variable:
    run_indices = np.digitize(data['run'], run_edges) - 1
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
    errs = lookup_errs[run_indices, eta_indices, r9_indices, et_indices]
    
    # Handle any events that fall outside the correction bins
    mask = np.isnan(scales)
    i = 0
    data['run_index'] = run_indices
    data['eta_index'] = eta_indices
    data['r9_index'] = r9_indices
    data['et_index'] = et_indices

    print(f"[WARNING][scale_data.py] {mask.sum()} events fall outside the correction bins, please check scales file for completeness.")
    print(f"[WARNING][scale_data.py] Run `python pytyhon/tools/scales_validator.py -s <scales_file>` to check coverage.")
    scales[mask] = 1.0  # or any other default value
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
    print(f"{info} Applying corrections to lead_data and sublead_data")
    args = prepare_scales_lookup(scales_df)
    lead_data['scale'], lead_data['err'] = apply_corrections(lead_data, *args)
    sublead_data['scale'], sublead_data['err'] = apply_corrections(sublead_data, *args)

    # calculate new energies, errors, and invmasses
    # lead_data['scale'] returns a tuple of (scale, err)
    data['lead_scale'] = lead_data['scale']
    data['lead_err'] = lead_data['err']
    data['sublead_scale'] = sublead_data['scale']
    data['sublead_err'] = sublead_data['err']
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

    # grab one event of every scale value
    lead_data = lead_data.drop_duplicates(subset=['scale'])
    sublead_data = sublead_data.drop_duplicates(subset=['scale'])
    lead_data.to_csv("lead_data.csv")
    sublead_data.to_csv("sublead_data.csv")

    return data_loader.custom_cuts(
                                    data,
                                    inv_mass_cuts=(80, 100),
                                    eta_cuts=(0, 1.4442, 1.566, 2.5),
                                    et_cuts=((32, 14000), (20, 14000)),
    ) 
    