import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import multiprocessing as mp
import gc

import python.classes.constants as constants

def apply(arg):
    #make a returnable df with all runs in this set of scales:
    data,scales = arg

    #constants
    c = constants.const()

    def find_scales(row):
        """finds the scales"""
        #lead scale
        LEAD_eta_mask = np.logical_and((scales[:,c.i_eta_min] <= row[c.ETA_LEAD]),(row[c.ETA_LEAD] < scales[:,c.i_eta_max]))
        LEAD_r9_mask = np.logical_and((scales[:,c.i_r9_min] <= row[c.R9_LEAD]),(row[c.R9_LEAD] < scales[:,c.i_r9_max]))
        LEAD_gainEt_mask = np.ones(len(LEAD_eta_mask),dtype=bool)

        SUB_eta_mask = np.logical_and((scales[:,c.i_eta_min] <= row[c.ETA_SUB]),(row[c.ETA_SUB] < scales[:,c.i_eta_max]))
        SUB_r9_mask = np.logical_and((scales[:,c.i_r9_min] <= row[c.R9_SUB]),(row[c.R9_SUB] < scales[:,c.i_r9_max]))
        SUB_gainEt_mask = np.ones(len(SUB_eta_mask),dtype=bool)

        if any(scales[:,c.i_et_min] != c.MIN_ET): #these scales are Et dependent
            LEAD_et = row[c.E_LEAD]/np.cosh(row[c.ETA_LEAD])
            SUB_et = row[c.E_SUB]/np.cosh(row[c.ETA_SUB])
            LEAD_gainEt_mask = np.logical_and((scales[:,c.i_et_min] <= LEAD_et), (scales[:,c.i_et_max] > LEAD_et))
            SUB_gainEt_mask = np.logical_and((scales[:,c.i_et_min] <= SUB_et), (scales[:,c.i_et_max] > SUB_et))

        if any(scales[:,c.i_gain] != 0): #these scales are gain dependent
            LEAD_gain = 12
            SUB_gain = 12
            if row[c.GAIN_LEAD] == 1: LEAD_gain = 6
            if row[c.GAIN_LEAD] >= 2: LEAD_gain = 1
            if row[c.GAIN_SUB] == 1: SUB_gain = 6
            if row[c.GAIN_SUB] >= 2: SUB_gain = 1

            LEAD_gainEt_mask = scales[:,c.i_gain] == LEAD_gain
            SUB_gainEt_mask = scales[:,c.i_gain] == SUB_gain
        
        LEAD_mask = np.logical_and(LEAD_eta_mask,np.logical_and(LEAD_r9_mask,LEAD_gainEt_mask))
        SUBLEAD_mask = np.logical_and(SUB_eta_mask,np.logical_and(SUB_r9_mask,SUB_gainEt_mask))
        return (np.ravel(scales[LEAD_mask])[c.i_scale], np.ravel(scales[LEAD_mask])[c.i_err],
                np.ravel(scales[SUBLEAD_mask])[c.i_scale], np.ravel(scales[SUBLEAD_mask])[c.i_err])


    scales = data.apply(find_scales, axis=1)
    LEAD_scales = np.array([x[0] if len(x) > 0 else 0. for x in scales])
    LEAD_err = np.array([x[1] if len(x) > 0 else 0. for x in scales])
    LEAD_scales_up = np.add(LEAD_scales,LEAD_err)
    LEAD_scales_down = np.subtract(LEAD_scales, LEAD_err)
    SUB_scales = np.array([x[2] if len(x) > 0 else 0. for x in scales])
    SUB_err = np.array([x[3] if len(x) > 0 else 0. for x in scales])
    SUB_scales_up = np.add(SUB_scales,SUB_err)
    SUB_scales_down = np.subtract(SUB_scales, SUB_err)

    data[c.E_LEAD] = np.multiply(data[c.E_LEAD].values,LEAD_scales, dtype=np.float32)
    data[c.E_SUB] = np.multiply(data[c.E_SUB].values,SUB_scales, dtype=np.float32)
    data["invmass_up"] = np.multiply(data[c.INVMASS].values, np.sqrt(np.multiply(LEAD_scales_up,SUB_scales_up)), dtype=np.float32)
    data["invmass_down"] = np.multiply(data[c.INVMASS].values, np.sqrt(np.multiply(LEAD_scales_down,SUB_scales_down)), dtype=np.float32)
    data[c.INVMASS] = np.multiply(data[c.INVMASS].values, np.sqrt(np.multiply(LEAD_scales,SUB_scales)), dtype=np.float32)
    data["lead_scales"] = LEAD_scales
    data["lead_scales_err"] = LEAD_scales_err
    data["sub_scales"] = SUB_scales
    data["sub_scales_err"] = SUB_scales_err

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

