import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import multiprocessing as mp
import gc

import python.classes.const_class as constants

def apply(arg):
    #make a returnable df with all runs in this set of scales:
    data,scales = arg

    #constants
    c = constants.const()

    def find_scales(row):
        """finds the scales"""
        #lead scale
        lead_eta_mask = np.logical_and((scales[:,c.i_eta_min] <= row[c.ETA_LEAD]),(row[c.ETA_LEAD] < scales[:,c.i_eta_max]))
        lead_r9_mask = np.logical_and((scales[:,c.i_r9_min] <= row[c.R9_LEAD]),(row[c.R9_LEAD] < scales[:,c.i_r9_max]))
        lead_gainEt_mask = np.ones(len(lead_eta_mask),dtype=bool)

        sub_eta_mask = np.logical_and((scales[:,c.i_eta_min] <= row[c.ETA_SUB]),(row[c.ETA_SUB] < scales[:,c.i_eta_max]))
        sub_r9_mask = np.logical_and((scales[:,c.i_r9_min] <= row[c.R9_SUB]),(row[c.R9_SUB] < scales[:,c.i_r9_max]))
        sub_gainEt_mask = np.ones(len(sub_eta_mask),dtype=bool)

        if any(scales[:,c.i_et_min] != c.MIN_ET): #these scales are Et dependent
            lead_et = row[c.E_LEAD]/np.cosh(row[c.ETA_LEAD])
            sub_et = row[c.E_SUB]/np.cosh(row[c.ETA_SUB])
            lead_gainEt_mask = np.logical_and((scales[:,c.i_et_min] <= lead_et), (scales[:,c.i_et_max] > lead_et))
            sub_gainEt_mask = np.logical_and((scales[:,c.i_et_min] <= sub_et), (scales[:,c.i_et_max] > sub_et))

        if any(scales[:,c.i_gain] != 0): #these scales are gain dependent
            lead_gain = 12
            sub_gain = 12
            if row[c.GAIN_LEAD] == 1: lead_gain = 6
            if row[c.GAIN_LEAD] > 2: lead_gain = 1
            if row[c.GAIN_SUB] == 1: sub_gain = 6
            if row[c.GAIN_SUB] > 2: sub_gain = 1

            lead_gainEt_mask = scales[:,c.i_gain] == lead_gain
            sub_gainEt_mask = scales[:,c.i_gain] == sub_gain
        
        lead_mask = np.logical_and(lead_eta_mask,np.logical_and(lead_r9_mask,lead_gainEt_mask))
        sublead_mask = np.logical_and(sub_eta_mask,np.logical_and(sub_r9_mask,sub_gainEt_mask))

        if len(np.ravel(scales[sublead_mask])) < 10:
            print(row)
            print(scales[sublead_mask])
        return (np.ravel(scales[lead_mask])[c.i_scale], np.ravel(scales[lead_mask])[c.i_err],
                np.ravel(scales[sublead_mask])[c.i_scale], np.ravel(scales[sublead_mask])[c.i_err])


    scales = data.apply(find_scales, axis=1)
    lead_scales = np.array([x[0] if len(x) > 0 else 0. for x in scales])
    lead_err = np.array([x[1] if len(x) > 0 else 0. for x in scales])
    lead_scales_up = np.add(lead_scales,lead_err)
    lead_scales_down = np.subtract(lead_scales, lead_err)
    sub_scales = np.array([x[2] if len(x) > 0 else 0. for x in scales])
    sub_err = np.array([x[3] if len(x) > 0 else 0. for x in scales])
    sub_scales_up = np.add(sub_scales,sub_err)
    sub_scales_down = np.subtract(sub_scales, sub_err)

    data[c.E_LEAD] = np.multiply(data[c.E_LEAD].values,lead_scales, dtype=np.float32)
    data[c.E_SUB] = np.multiply(data[c.E_SUB].values,sub_scales, dtype=np.float32)
    data["invmass_up"] = np.multiply(data[c.INVMASS].values, np.sqrt(np.multiply(lead_scales_up,sub_scales_up)), dtype=np.float32)
    data["invmass_down"] = np.multiply(data[c.INVMASS].values, np.sqrt(np.multiply(lead_scales_down,sub_scales_down)), dtype=np.float32)
    data[c.INVMASS] = np.multiply(data[c.INVMASS].values, np.sqrt(np.multiply(lead_scales,sub_scales)), dtype=np.float32)

    return data


def scale(data, scales):
    """
    This function applies the 
    """
    #newformat of scales files is 
    #runMin runMax etaMin etaMax r9Min r9Max etMin etMax gain val err
    run = 'runNumber'
    i_run_min = 0
    i_run_max = 1

    #read in scales to df
    scales_df = pd.read_csv(scales, sep="\t", comment="#", header=None)
    
    processors = mp.cpu_count() - 1

    #grab unique run values low and high from df
    unique_runnums_low = scales_df[:][i_run_min].unique().tolist()
    unique_runnums_high = scales_df[:][i_run_max].unique().tolist()

    run_bins = unique_runnums_low[::processors]
    run_bins.append(999999)

    #divide data by run number
    divided_data = [
            data[np.logical_and(
                run_bins[i] <= data[run].values,  
                data[run].values < run_bins[i+1])
                ] 
            for i in range(len(run_bins)-1)
            ]

    #divide scales by run and tuple with divided data
    divided_scales = [(divided_data[i],
        scales_df.loc[
        np.logical_and(scales_df[:][i_run_min] >= run_bins[i],
            scales_df[:][i_run_min] < run_bins[i+1])
        ].values
        ) for i in range(len(run_bins)-1)]

    #initiate multiprocessing of scales application
    pool = mp.Pool(processes=processors)
    scaled_data=pool.map(apply, divided_scales)
    pool.close()
    pool.join()
    return pd.concat(scaled_data)

