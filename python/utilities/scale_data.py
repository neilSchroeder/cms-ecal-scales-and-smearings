import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import multiprocessing as mp
import gc
import time

from python.classes.constant_classes import (
    DataConstants as dc,
    PyValConstants as pvc
)


def apply(arg):
    #make a returnable df with all runs in this set of scales:
    data,scales = arg    

    def find_scales(row):
        """finds the scales"""
        # find the run bin
        lead_mask = np.logical_and(
            (scales[:,dc.i_run_min] <= row[dc.RUN]),(row[dc.RUN] <= scales[:,dc.i_run_max])
            )
        sublead_mask = np.logical_and(
            (scales[:,dc.i_run_min] <= row[dc.RUN]),(row[dc.RUN] <= scales[:,dc.i_run_max])
            )
        

        # find the eta and r9 bins
        lead_mask = np.logical_and( lead_mask, np.logical_and(
            (scales[:,dc.i_eta_min] <= row[dc.ETA_LEAD]),(row[dc.ETA_LEAD] < scales[:,dc.i_eta_max])
            ))
        lead_mask = np.logical_and(lead_mask,
            np.logical_and(
                (scales[:,dc.i_r9_min] <= row[dc.R9_LEAD]),(row[dc.R9_LEAD] < scales[:,dc.i_r9_max])
                )
            )

        sublead_mask = np.logical_and( sublead_mask, np.logical_and(
            (scales[:,dc.i_eta_min] <= row[dc.ETA_SUB]),(row[dc.ETA_SUB] < scales[:,dc.i_eta_max])
            ))
        sublead_mask = np.logical_and(sublead_mask,
            np.logical_and(
                (scales[:,dc.i_r9_min] <= row[dc.R9_SUB]),(row[dc.R9_SUB] < scales[:,dc.i_r9_max])
                )
            )

        if any(scales[:,dc.i_et_min] != dc.MIN_ET): #these scales are Et dependent
            lead_et = row[dc.E_LEAD]/np.cosh(row[dc.ETA_LEAD])
            sub_et = row[dc.E_SUB]/np.cosh(row[dc.ETA_SUB])
            lead_mask = np.logical_and( lead_mask, np.logical_and( 
                (scales[:,dc.i_et_min] <= lead_et), (scales[:,dc.i_et_max] > lead_et)
                ))
            sublead_mask = np.logical_and(sublead_mask, np.logical_and( 
                (scales[:,dc.i_et_min] <= sub_et), (scales[:,dc.i_et_max] > sub_et)\
                ))
            


        if any(scales[:,dc.i_gain] != 0): #these scales are gain dependent
            lead_gain = 12
            sub_gain = 12
            if row[dc.GAIN_LEAD] == 1: lead_gain = 6
            if row[dc.GAIN_LEAD] > 2: lead_gain = 1
            if row[dc.GAIN_SUB] == 1: sub_gain = 6
            if row[dc.GAIN_SUB] > 2: sub_gain = 1

            lead_mask = np.logical_and(lead_mask, scales[:,dc.i_gain] == lead_gain)
            sublead_mask = np.logical_and(sublead_mask, scales[:,dc.i_gain] == sub_gain)

        # one category should be found for each electron
        if len(scales[lead_mask]) > 1:
            print(f"[WARNING][scale_data.py] more than one lead scale found")
            print(f"[WARNING][scale_data.py] lead eta: {row[dc.ETA_LEAD]}")
            print(f"[WARNING][scale_data.py] lead r9: {row[dc.R9_LEAD]}")
            print(f"[WARNING][scale_data.py] lead et: {row[dc.E_LEAD]/np.cosh(row[dc.ETA_LEAD])}")
            print(f"[WARNING][scale_data.py] lead gain: {row[dc.GAIN_LEAD]}")
            print(f"[WARNING][scale_data.py] lead scales: {scales[lead_mask]}")
        if len(scales[sublead_mask]) > 1:
            print(f"[WARNING][scale_data.py] more than one sublead scale found")
            print(f"[WARNING][scale_data.py] sublead eta: {row[dc.ETA_SUB]}")
            print(f"[WARNING][scale_data.py] sublead r9: {row[dc.R9_SUB]}")
            print(f"[WARNING][scale_data.py] sublead et: {row[dc.E_SUB]/np.cosh(row[dc.ETA_SUB])}")
            print(f"[WARNING][scale_data.py] sublead gain: {row[dc.GAIN_SUB]}")
            print(f"[WARNING][scale_data.py] sublead scales: {scales[sublead_mask]}")

        assert(len(scales[lead_mask]) <= 1)
        assert(len(scales[sublead_mask]) <= 1)



        lead_scale = np.ravel(scales[lead_mask])[dc.i_scale] \
            if len(np.ravel(scales[lead_mask])) > 0 else 0.
        lead_err = np.ravel(scales[lead_mask])[dc.i_err] \
            if len(np.ravel(scales[lead_mask])) > 0 else 0.
        sublead_scale = np.ravel(scales[sublead_mask])[dc.i_scale] \
            if len(np.ravel(scales[sublead_mask])) > 0 else 0.
        sublead_err = np.ravel(scales[sublead_mask])[dc.i_err] \
            if len(np.ravel(scales[sublead_mask])) > 0 else 0.
         
        return (lead_scale, lead_err, sublead_scale, sublead_err)

    # time this function
    these_scales = data.apply(find_scales, axis=1)

    lead_scales = np.array([x[0] if len(x) > 0 else 0. for x in these_scales])
    lead_err = np.array([x[1] if len(x) > 0 else 0. for x in these_scales])
    lead_scales_up = np.add(lead_scales,lead_err)
    lead_scales_down = np.subtract(lead_scales, lead_err)

    sub_scales = np.array([x[2] if len(x) > 0 else 0. for x in these_scales])
    sub_err = np.array([x[3] if len(x) > 0 else 0. for x in these_scales])
    sub_scales_up = np.add(sub_scales,sub_err)
    sub_scales_down = np.subtract(sub_scales, sub_err)

    data[dc.E_LEAD] = np.multiply(data[dc.E_LEAD].values,lead_scales, dtype=np.float32)
    data[dc.E_SUB] = np.multiply(data[dc.E_SUB].values,sub_scales, dtype=np.float32)
    data[pvc.KEY_INVMASS_UP] = np.multiply(data[dc.INVMASS].values, 
                                     np.sqrt(np.multiply(lead_scales_up,sub_scales_up)), dtype=np.float32)
    data[pvc.KEY_INVMASS_DOWN] = np.multiply(data[dc.INVMASS].values, 
                                       np.sqrt(np.multiply(lead_scales_down,sub_scales_down)), dtype=np.float32)
    data[dc.INVMASS] = np.multiply(data[dc.INVMASS].values, np.sqrt(np.multiply(lead_scales,sub_scales)), dtype=np.float32)

    return data


def scale(data, scales):
    """
    This function applies the scales in a multi-threaded way
    """
    info = "[INFO][scale_data.py]"
    #newformat of scales files is 
    #runMin runMax etaMin etaMax r9Min r9Max etMin etMax gain val err
    run = dc.RUN
    i_run_min = 0
    i_run_max = 1

    #read in scales to df
    scales_df = pd.read_csv(scales, sep="\t", comment="#", header=None)
    
    processors = mp.cpu_count() - 1

    #grab unique run values low and high from df
    unique_runnums_low = scales_df[:][i_run_min].unique().tolist()

    run_bins = unique_runnums_low[::processors]
    run_bins.append(999999)

    #divide data by run number
    print(f"{info} dividing data by run")
    divided_data = [ data[np.logical_and(run_bins[i] <= data[run].values, data[run].values < run_bins[i+1])] for i in range(len(run_bins)-1)]
    assert(len(data) == sum([len(x) for x in divided_data]))

    #divide scales by run and tuple with divided data
    print(f"{info} dividing scales by run and tuple")
    divided_scales = [(divided_data[i],
        scales_df.loc[
        np.logical_and(scales_df[:][i_run_min] >= run_bins[i],
            scales_df[:][i_run_min] < run_bins[i+1])
        ].values
        ) for i in range(len(run_bins)-1)]
    assert(len(scales_df) == sum([len(x[1]) for x in divided_scales]))

    #initiate multiprocessing of scales application
    print(f"{info} distributing application of scales")
    print(f"{info} please be patient, there are {len(data)} rows to apply scales to")
    print(f"{info} it takes ~ 0.0003 seconds per row, and you've requested {processors} processors")
    pool = mp.Pool(processes=processors)
    scaled_data=pool.map(apply, divided_scales)
    pool.close()
    pool.join()
    return pd.concat(scaled_data)

