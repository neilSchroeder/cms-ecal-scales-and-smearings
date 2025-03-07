import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
# import concurrent futures
import concurrent.futures as cf
import gc
import multiprocessing as mp
import time

import tqdm

import src.core.data_loader as data_loader
from src.classes.constant_classes import DataConstants as dc
from src.classes.constant_classes import PyValConstants as pvc


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

    def find_scales(row):
        """
        Finds the scales for a given row in a dataframe

        Args:
            row (pandas series): a row in a dataframe
        Returns:
            tuple: a tuple of scales
        """
        # find the run bin (only needs to be computed once)
        run_mask = np.logical_and(
            (scales[:, dc.i_run_min] <= row[dc.RUN]),
            (row[dc.RUN] <= scales[:, dc.i_run_max]),
        )

        # find the eta and r9 bins
        lead_mask = np.logical_and.reduce(
            (
                run_mask,
                np.logical_and(
                    (scales[:, dc.i_eta_min] <= row[dc.ETA_LEAD]),
                    (row[dc.ETA_LEAD] < scales[:, dc.i_eta_max]),
                ),
                np.logical_and(
                    (scales[:, dc.i_r9_min] <= row[dc.R9_LEAD]),
                    (row[dc.R9_LEAD] < scales[:, dc.i_r9_max]),
                ),
            )
        )
        sublead_mask = np.logical_and.reduce(
            (
                run_mask,
                np.logical_and(
                    (scales[:, dc.i_eta_min] <= row[dc.ETA_SUB]),
                    (row[dc.ETA_SUB] < scales[:, dc.i_eta_max]),
                ),
                np.logical_and(
                    (scales[:, dc.i_r9_min] <= row[dc.R9_SUB]),
                    (row[dc.R9_SUB] < scales[:, dc.i_r9_max]),
                ),
            )
        )

        if any(scales[:, dc.i_et_min] != dc.MIN_ET):  # these scales are Et dependent
            lead_et = row[dc.E_LEAD] / np.cosh(row[dc.ETA_LEAD])
            sub_et = row[dc.E_SUB] / np.cosh(row[dc.ETA_SUB])
            lead_mask = np.logical_and(
                lead_mask,
                np.logical_and(
                    (scales[:, dc.i_et_min] <= lead_et),
                    (scales[:, dc.i_et_max] > lead_et),
                ),
            )
            sublead_mask = np.logical_and(
                sublead_mask,
                np.logical_and(
                    (scales[:, dc.i_et_min] <= sub_et),
                    (scales[:, dc.i_et_max] > sub_et),
                ),
            )

        if any(scales[:, dc.i_gain] != 0):  # these scales are gain dependent
            lead_gain = 12
            sub_gain = 12
            if row[dc.GAIN_LEAD] == 1:
                lead_gain = 6
            if row[dc.GAIN_LEAD] > 2:
                lead_gain = 1
            if row[dc.GAIN_SUB] == 1:
                sub_gain = 6
            if row[dc.GAIN_SUB] > 2:
                sub_gain = 1

            lead_mask = np.logical_and(lead_mask, scales[:, dc.i_gain] == lead_gain)
            sublead_mask = np.logical_and(
                sublead_mask, scales[:, dc.i_gain] == sub_gain
            )

        # one category should be found for each electron
        if len(scales[lead_mask]) > 1:
            print(f"[WARNING][scale_data.py] more than one lead scale found")
            print(f"[WARNING][scale_data.py] lead eta: {row[dc.ETA_LEAD]}")
            print(f"[WARNING][scale_data.py] lead r9: {row[dc.R9_LEAD]}")
            print(
                f"[WARNING][scale_data.py] lead et: {row[dc.E_LEAD]/np.cosh(row[dc.ETA_LEAD])}"
            )
            print(f"[WARNING][scale_data.py] lead gain: {row[dc.GAIN_LEAD]}")
            print(f"[WARNING][scale_data.py] lead scales: {scales[lead_mask]}")
        if len(scales[sublead_mask]) > 1:
            print(f"[WARNING][scale_data.py] more than one sublead scale found")
            print(f"[WARNING][scale_data.py] sublead eta: {row[dc.ETA_SUB]}")
            print(f"[WARNING][scale_data.py] sublead r9: {row[dc.R9_SUB]}")
            print(
                f"[WARNING][scale_data.py] sublead et: {row[dc.E_SUB]/np.cosh(row[dc.ETA_SUB])}"
            )
            print(f"[WARNING][scale_data.py] sublead gain: {row[dc.GAIN_SUB]}")
            print(f"[WARNING][scale_data.py] sublead scales: {scales[sublead_mask]}")

        assert len(scales[lead_mask]) <= 1
        assert len(scales[sublead_mask]) <= 1

        lead_scale = (
            np.ravel(scales[lead_mask])[dc.i_scale]
            if len(np.ravel(scales[lead_mask])) > 0
            else 0.0
        )
        lead_err = (
            np.ravel(scales[lead_mask])[dc.i_err]
            if len(np.ravel(scales[lead_mask])) > 0
            else 0.0
        )
        sublead_scale = (
            np.ravel(scales[sublead_mask])[dc.i_scale]
            if len(np.ravel(scales[sublead_mask])) > 0
            else 0.0
        )
        sublead_err = (
            np.ravel(scales[sublead_mask])[dc.i_err]
            if len(np.ravel(scales[sublead_mask])) > 0
            else 0.0
        )

        return (lead_scale, lead_err, sublead_scale, sublead_err)

    # put values in their own columns
    these_scales = data.apply(find_scales, axis=1)
    these_scales = pd.DataFrame(
        these_scales.values.tolist(),
        columns=["lead_scale", "lead_err", "sublead_scale", "sublead_err"],
    )

    lead_scales = these_scales["lead_scale"].values
    lead_err = these_scales["lead_err"].values
    et_lead = data[dc.E_LEAD] / np.cosh(data[dc.ETA_LEAD])
    et_lead_mask = et_lead > 80
    lead_non_lin_unc = np.multiply(
        np.add(et_lead_mask * 0.0001, (~et_lead_mask) * 0.0005), lead_scales
    )
    lead_err = np.sqrt(np.power(lead_non_lin_unc, 2) + np.power(lead_err, 2))
    lead_scales_up = np.add(lead_scales, lead_err)
    lead_scales_down = np.subtract(lead_scales, lead_err)

    sub_scales = these_scales["sublead_scale"].values
    sub_err = these_scales["sublead_err"].values
    et_sub = data[dc.E_SUB] / np.cosh(data[dc.ETA_SUB])
    et_sub_mask = et_sub > 80
    sub_non_lin_unc = np.multiply(
        np.add(et_sub_mask * 0.0001, (~et_sub_mask) * 0.0005), sub_scales
    )
    sub_err = np.sqrt(np.power(sub_non_lin_unc, 2) + np.power(sub_err, 2))
    sub_scales_up = np.add(sub_scales, sub_err)
    sub_scales_down = np.subtract(sub_scales, sub_err)

    if np.sqrt(np.multiply(lead_scales, sub_scales)).any() <= 0.9:
        print(f"[WARNING][scale_data.py] some scales are less than 0.9")
        print(f"[WARNING][scale_data.py] lead scales: {lead_scales}")
        print(f"[WARNING][scale_data.py] sub scales: {sub_scales}")
        print(data.head())
        print(these_scales)
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
    if any([x.exception() for x in proc_futures]):
        print(f"[ERROR][scale_data.py] some processes failed")
        for x in proc_futures:
            if x.exception():
                print(x.result())
        raise RuntimeError

    ret = pd.concat([x.result() for x in proc_futures])
    executor.shutdown()
    print(f"{info} done applying scales")

    return data_loader.apply_custom_event_selection(
        ret,
        # inv_mass_cuts=(80, 100),
        eta_cuts=(0, 1.4442, 1.566, 2.5),
        # et_cuts=((32, 14000), (20, 14000)),
    )
