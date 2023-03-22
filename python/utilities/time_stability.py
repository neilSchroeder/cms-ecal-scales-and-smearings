from collections import OrderedDict
import datetime
import numpy as np
import pandas as pd
import statistics as stat

from python.classes.constant_classes import DataConstants as dc
from python.classes.config_class import SSConfig
import python.utilities.write_files as write_files
ss_config = SSConfig()


"""
Author: 
    Neil Schroeder, schr1077@umn.edu, neil.raymond.schroeder@cern.ch

About:
    This function derives the scales known as 'step1' scales. These scales 
    correct data to the pdg Z mass in bins of Run and Eta to stabilize the
    scale as a function of time and location in the detector.
"""

def derive(data, runs, output, _kWriteData=True):
    """
    Derives the scale for each run and eta bin
    ----------
    Args:
        data: dataframe of the data
        runs: path to the file containing the run bins
        output: output file tag
        _kWriteData: boolean to write the data to a file
    ---------- 
    Returns:
        None
    ----------
    """
    print(f"[INFO][python/time_stability][derive] Deriving scale for runs in {runs}")

    eta_min = [0, 1., 1.2, 1.566, 2.]
    eta_max = [1., 1.2, 1.4442, 2., 2.5]
    ret = [[] for i in range(len(eta_min))]

    run_bins = pd.read_csv(runs, delimiter='\t', header=None)

    headers = dc.TIME_STABILITY_HEADERS
    dictForDf = OrderedDict.fromkeys(headers)
    for col in headers:
        dictForDf[col] = []

    for i,pair in run_bins.iterrows():
        mask_run = np.logical_and(pair[0] <= data[dc.RUN].values, data[dc.RUN].values <= pair[1])
        mask_mass = np.logical_and(dc.MIN_INVMASS <= data[dc.INVMASS].values, data[dc.INVMASS].values <= dc.MAX_INVMASS)
        mask = np.logical_and(mask_run, mask_mass)
        for j,eta_bin in enumerate(ret):
            bin_mask = np.logical_and(mask, np.logical_and( eta_min[j] <= data[dc.ETA_LEAD].values, data[dc.ETA_LEAD].values < eta_max[j]))
            bin_mask = np.logical_and(bin_mask, np.logical_and( eta_min[j] <= data[dc.ETA_SUB].values, data[dc.ETA_SUB].values < eta_max[j]))
            eta_bin.append(91.188/stat.median(data[bin_mask][dc.INVMASS].values))
            if _kWriteData:
                dictForDf['run_min'].append(pair[0])
                dictForDf['run_max'].append(pair[1])
                dictForDf['eta_min'].append(eta_min[j])
                dictForDf['eta_max'].append(eta_max[j])
                dictForDf['median'].append(stat.median(data[bin_mask][dc.INVMASS].values))
                dictForDf['mean'].append(stat.mean(data[bin_mask][dc.INVMASS].values))
                dictForDf['sigma'].append(stat.stdev(np.array(data[bin_mask][dc.INVMASS].values,dtype=np.float64)))
                dictForDf['events'].append(len(data[bin_mask][dc.INVMASS].values))
                dictForDf['scale'].append(eta_bin[-1])
                invmass = data[bin_mask][dc.INVMASS].values * eta_bin[-1]
                invmass = invmass[ np.logical_and(dc.MIN_INVMASS <= invmass, invmass <= dc.MAX_INVMASS) ]
                dictForDf['median_corr'].append(stat.median(invmass))
                dictForDf['mean_corr'].append(stat.mean(invmass))
                dictForDf['sigma_corr'].append(stat.stdev(np.array(invmass,dtype=np.float64)))

    if _kWriteData: 
        dfOut = pd.DataFrame(dictForDf)
        dfOut.to_csv(f"{ss_config.DEFAULT_WRITE_FILES_PATH}time_stability_outputData_{output}.dat",
                sep='\t', header=True,index=False)

    return ret, f"{ss_config.DEFAULT_WRITE_FILES_PATH}time_stability_outputData_{output}.dat"
    
