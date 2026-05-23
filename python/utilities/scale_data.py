import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
import multiprocessing as mp

# import concurrent futures
import concurrent.futures as cf
import gc
import time
import tqdm

from python.classes.constant_classes import DataConstants as dc, PyValConstants as pvc


def apply(arg):
    """
    Applies the scales to the dataframe.

    Args:
        arg (tuple(pd.DataFrame, pd.DataFrame)): tuple of data and scales
    Returns:
        data (pd.DataFrame): scaled data dataframe
    """
    data, scales = arg
    if len(data) == 0:
        return data
    if len(scales) == 0:
        return data

    n_events = len(data)
    n_cats = len(scales)

    # Extract data columns as 1-D arrays (shape: [N_events])
    run_vals = data[dc.RUN].values
    eta_lead = data[dc.ETA_LEAD].values
    eta_sub = data[dc.ETA_SUB].values
    r9_lead = data[dc.R9_LEAD].values
    r9_sub = data[dc.R9_SUB].values
    e_lead = data[dc.E_LEAD].values
    e_sub = data[dc.E_SUB].values
    gain_lead_raw = data[dc.GAIN_LEAD].values
    gain_sub_raw = data[dc.GAIN_SUB].values

    # Extract scales columns as 1-D arrays (shape: [N_cats])
    s_run_min = scales[:, dc.i_run_min]
    s_run_max = scales[:, dc.i_run_max]
    s_eta_min = scales[:, dc.i_eta_min]
    s_eta_max = scales[:, dc.i_eta_max]
    s_r9_min = scales[:, dc.i_r9_min]
    s_r9_max = scales[:, dc.i_r9_max]
    s_et_min = scales[:, dc.i_et_min]
    s_et_max = scales[:, dc.i_et_max]
    s_gain = scales[:, dc.i_gain]
    s_scale = scales[:, dc.i_scale]
    s_err = scales[:, dc.i_err]

    # Build [N_events, N_cats] boolean masks via broadcasting
    # run mask: s_run_min <= run <= s_run_max
    run_mask = (s_run_min[np.newaxis, :] <= run_vals[:, np.newaxis]) & (
        run_vals[:, np.newaxis] <= s_run_max[np.newaxis, :]
    )

    # eta/r9 masks for lead and sublead
    lead_mask = (
        run_mask
        & (s_eta_min[np.newaxis, :] <= eta_lead[:, np.newaxis])
        & (eta_lead[:, np.newaxis] < s_eta_max[np.newaxis, :])
        & (s_r9_min[np.newaxis, :] <= r9_lead[:, np.newaxis])
        & (r9_lead[:, np.newaxis] < s_r9_max[np.newaxis, :])
    )

    sub_mask = (
        run_mask
        & (s_eta_min[np.newaxis, :] <= eta_sub[:, np.newaxis])
        & (eta_sub[:, np.newaxis] < s_eta_max[np.newaxis, :])
        & (s_r9_min[np.newaxis, :] <= r9_sub[:, np.newaxis])
        & (r9_sub[:, np.newaxis] < s_r9_max[np.newaxis, :])
    )

    # Et-dependent scales
    if np.any(s_et_min != dc.MIN_ET):
        et_lead_vals = e_lead / np.cosh(eta_lead)
        et_sub_vals = e_sub / np.cosh(eta_sub)
        lead_mask &= (s_et_min[np.newaxis, :] <= et_lead_vals[:, np.newaxis]) & (
            et_lead_vals[:, np.newaxis] < s_et_max[np.newaxis, :]
        )
        sub_mask &= (s_et_min[np.newaxis, :] <= et_sub_vals[:, np.newaxis]) & (
            et_sub_vals[:, np.newaxis] < s_et_max[np.newaxis, :]
        )

    # Gain-dependent scales
    if np.any(s_gain != 0):
        # Convert gain encoding: 0->12, 1->6, >2->1
        lead_gain = np.where(gain_lead_raw == 1, 6, np.where(gain_lead_raw > 2, 1, 12))
        sub_gain = np.where(gain_sub_raw == 1, 6, np.where(gain_sub_raw > 2, 1, 12))
        lead_mask &= s_gain[np.newaxis, :] == lead_gain[:, np.newaxis]
        sub_mask &= s_gain[np.newaxis, :] == sub_gain[:, np.newaxis]

    # Count matches per event
    lead_match_count = lead_mask.sum(axis=1)
    sub_match_count = sub_mask.sum(axis=1)

    # Warn on multi-match events
    lead_multi = np.where(lead_match_count > 1)[0]
    if len(lead_multi) > 0:
        for idx in lead_multi[:5]:  # limit warnings
            row = data.iloc[idx]
            print(f"[WARNING][scale_data.py] more than one lead scale found")
            print(f"[WARNING][scale_data.py] lead eta: {row[dc.ETA_LEAD]}")
            print(f"[WARNING][scale_data.py] lead r9: {row[dc.R9_LEAD]}")
            print(
                f"[WARNING][scale_data.py] lead et: {row[dc.E_LEAD]/np.cosh(row[dc.ETA_LEAD])}"
            )
            print(f"[WARNING][scale_data.py] lead gain: {row[dc.GAIN_LEAD]}")
            print(f"[WARNING][scale_data.py] lead scales: {scales[lead_mask[idx]]}")
    sub_multi = np.where(sub_match_count > 1)[0]
    if len(sub_multi) > 0:
        for idx in sub_multi[:5]:
            row = data.iloc[idx]
            print(f"[WARNING][scale_data.py] more than one sublead scale found")
            print(f"[WARNING][scale_data.py] sublead eta: {row[dc.ETA_SUB]}")
            print(f"[WARNING][scale_data.py] sublead r9: {row[dc.R9_SUB]}")
            print(
                f"[WARNING][scale_data.py] sublead et: {row[dc.E_SUB]/np.cosh(row[dc.ETA_SUB])}"
            )
            print(f"[WARNING][scale_data.py] sublead gain: {row[dc.GAIN_SUB]}")
            print(f"[WARNING][scale_data.py] sublead scales: {scales[sub_mask[idx]]}")

    assert np.all(
        lead_match_count <= 1
    ), f"Multiple lead scale matches for {np.sum(lead_match_count > 1)} events"
    assert np.all(
        sub_match_count <= 1
    ), f"Multiple sublead scale matches for {np.sum(sub_match_count > 1)} events"

    # Extract scale and error values using the masks
    # For events with exactly one match, argmax gives the matching index;
    # for events with zero matches, we'll override with 0.0 below.
    lead_cat_idx = np.argmax(lead_mask, axis=1)
    sub_cat_idx = np.argmax(sub_mask, axis=1)

    lead_has_match = lead_match_count > 0
    sub_has_match = sub_match_count > 0

    lead_scales = np.where(lead_has_match, s_scale[lead_cat_idx], 0.0)
    lead_err = np.where(lead_has_match, s_err[lead_cat_idx], 0.0)
    sub_scales = np.where(sub_has_match, s_scale[sub_cat_idx], 0.0)
    sub_err = np.where(sub_has_match, s_err[sub_cat_idx], 0.0)
    et_lead = data[dc.E_LEAD] / np.cosh(data[dc.ETA_LEAD])
    et_lead_mask = et_lead > 80
    lead_non_lin_unc = np.multiply(
        np.add(et_lead_mask * 0.0001, (~et_lead_mask) * 0.0005), lead_scales
    )
    lead_err = np.sqrt(np.power(lead_non_lin_unc, 2) + np.power(lead_err, 2))
    lead_scales_up = np.add(lead_scales, lead_err)
    lead_scales_down = np.subtract(lead_scales, lead_err)

    et_sub = data[dc.E_SUB] / np.cosh(data[dc.ETA_SUB])
    et_sub_mask = et_sub > 80
    sub_non_lin_unc = np.multiply(
        np.add(et_sub_mask * 0.0001, (~et_sub_mask) * 0.0005), sub_scales
    )
    sub_err = np.sqrt(np.power(sub_non_lin_unc, 2) + np.power(sub_err, 2))
    sub_scales_up = np.add(sub_scales, sub_err)
    sub_scales_down = np.subtract(sub_scales, sub_err)

    if np.any(np.sqrt(np.multiply(lead_scales, sub_scales)) <= 0.9):
        print(f"[WARNING][scale_data.py] some scales are less than 0.9")
        print(f"[WARNING][scale_data.py] lead scales: {lead_scales}")
        print(f"[WARNING][scale_data.py] sub scales: {sub_scales}")
        print(data.head())
    data[dc.E_LEAD] = np.multiply(data[dc.E_LEAD].values, lead_scales, dtype=np.float32)
    data[dc.E_SUB] = np.multiply(data[dc.E_SUB].values, sub_scales, dtype=np.float32)
    invmass = data[dc.INVMASS].values.copy()
    data[pvc.KEY_INVMASS_UP] = np.multiply(
        invmass, np.sqrt(np.multiply(lead_scales_up, sub_scales_up)), dtype=np.float32
    )
    data[pvc.KEY_INVMASS_DOWN] = np.multiply(
        invmass,
        np.sqrt(np.multiply(lead_scales_down, sub_scales_down)),
        dtype=np.float32,
    )
    data[dc.INVMASS] = np.multiply(
        invmass, np.sqrt(np.multiply(lead_scales, sub_scales)), dtype=np.float32
    )

    return data


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
    run = dc.RUN
    i_run_min = 0
    i_run_max = 1

    # read in scales to df
    scales_df = pd.read_csv(scales, sep="\t", comment="#", header=None)

    # drop MC runs, they are not needed
    scales_df = scales_df[~scales_df[i_run_min].isin(dc.MC_RUNS)]
    scales_df = scales_df[~scales_df[i_run_max].isin(dc.MC_RUNS)]

    processors = mp.cpu_count() - 1

    # grab unique run values low and high from df
    unique_runnums_low = scales_df[:][i_run_min].unique().tolist()

    run_bins = unique_runnums_low[::processors]
    run_bins.append(999999)

    # divide data by run number
    print(f"{info} dividing data by run")
    divided_data = [
        data[
            np.logical_and(
                run_bins[i] <= data[run].values, data[run].values < run_bins[i + 1]
            )
        ]
        for i in range(len(run_bins) - 1)
    ]
    assert len(data) == sum([len(x) for x in divided_data])

    # divide scales by run and tuple with divided data
    print(f"{info} dividing scales by run and tuple")
    divided_scales = [
        (
            divided_data[i],  # divided data
            scales_df.loc[
                np.logical_and(
                    scales_df[:][i_run_min] >= run_bins[i],
                    scales_df[:][i_run_min] < run_bins[i + 1],
                )
            ].values,  # scales divided by run
        )
        for i in range(len(run_bins) - 1)
    ]
    assert len(scales_df) == sum([len(x[1]) for x in divided_scales])

    # initiate multiprocessing of scales application
    print(f"{info} distributing application of scales")
    print(f"{info} please be patient, there are {len(data)} rows to apply scales to")
    print(
        f"{info} it takes ~ 0.0003 seconds per row, and you've requested {processors} processors"
    )
    proc_futures = []
    executor = cf.ProcessPoolExecutor(max_workers=processors)
    for x in divided_scales:
        proc_futures.append(executor.submit(apply, x))

    # if any of the processes fail, raise an error
    exceptions = [x.exception() for x in proc_futures]
    if any(exceptions):
        print(f"[ERROR][scale_data.py] some processes failed")
        for exc in exceptions:
            if exc:
                print(exc)
        raise RuntimeError

    ret = pd.concat([x.result() for x in proc_futures])
    executor.shutdown()
    print(f"{info} done applying scales")

    return ret
