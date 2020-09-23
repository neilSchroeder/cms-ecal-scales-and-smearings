import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import multiprocessing as mp
import gc

def apply(arg):
    #make a returnable df with all runs in this set of scales:
    data,scales = arg
    #print("[INFO][python/scale_data_fast][apply] Applying scales for runs between {} and {}".format(scales[0][0], scales[0][1]))

    for index,row in enumerate(scales):
        ele0_eta = data['etaEle[0]'].between(row[2], row[3])
        ele1_eta = data['etaEle[1]'].between(row[2], row[3])
        ele0_r9 = data['R9Ele[0]'].between(row[4], row[5])
        ele1_r9 = data['R9Ele[1]'].between(row[4], row[5])
        if row[6] != 0 or row[7] != 14000:
            ele0_et = np.divide(data['energy_ECAL_ele[0]'].values, np.cosh(data['etaEle[0]'].values))
            ele1_et = np.divide(data['energy_ECAL_ele[1]'].values, np.cosh(data['etaEle[1]'].values))
            ele0_et = ele0_et.tolist()
            ele1_et = ele1_et.tolist()
            ele0_et = (ele0_et > row[6]) & (ele0_et < row[7])
            ele1_et = (ele1_et > row[6]) & (ele1_et < row[7])
            data.loc[ele0_eta&ele0_r9&ele0_et,'energy_ECAL_ele[0]'] *= row[9]
            data.loc[ele0_eta&ele0_r9&ele0_et,'invMass_ECAL_ele'] *= np.sqrt(row[9])
            data.loc[ele1_eta&ele1_r9&ele1_et,'energy_ECAL_ele[1]'] *= row[9]
            data.loc[ele1_eta&ele1_r9&ele1_et,'invMass_ECAL_ele'] *= np.sqrt(row[9])
        else:
            data.loc[ele0_eta&ele0_r9,'energy_ECAL_ele[0]'] *= row[9]
            data.loc[ele0_eta&ele0_r9,'invMass_ECAL_ele'] *= np.sqrt(row[9])
            data.loc[ele1_eta&ele1_r9,'energy_ECAL_ele[1]'] *= row[9]
            data.loc[ele1_eta&ele1_r9,'invMass_ECAL_ele'] *= np.sqrt(row[9])

    return data

def scale(data, scales):
#newformat of scales files is 
#runMin runMax etaMin etaMax r9Min r9Max etMin etMax gain val err
    scales_df = pd.read_csv(scales, sep="\t", comment="#", header=None)
    processors = mp.cpu_count() - 1
    unique_runnums_low = scales_df[:][0].unique().tolist()
    unique_runnums_high = scales_df[:][1].unique().tolist()
    divided_data = [data.loc[data['runNumber'].between(unique_runnums_low[i], unique_runnums_high[i])] for i in range(len(unique_runnums_low))]
    del data
    gc.collect()
    divided_scales = [(divided_data[i],scales_df.loc[scales_df[:][0] == unique_runnums_low[i]].values) for i in range(len(unique_runnums_low))]
    print("[INFO][python/scale_data_fast][scale] dividing scales and parallelizing")
    pool = mp.Pool(processes=processors)
    scaled_data=pool.map(apply, divided_scales)
    pool.close()
    pool.join()
    return pd.concat(scaled_data)

