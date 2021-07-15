from collections import OrderedDict
import datetime
import numpy as np
import pandas as pd
import statistics as stat

import python.classes.const_class as constants
import python.utilities.write_files as write_files

c = constants.const()

"""
Author: 
    Neil Schroeder, schr1077@umn.edu, neil.raymond.schroeder@cern.ch

About:
    This function derives the scales known as 'step1' scales. These scales 
    correct data to the pdg Z mass in bins of Run and Eta to stabilize the
    scale as a function of time and location in the detector.
"""

def derive(data, runs, _kWriteData=True):
    print("[INFO][python/time_stability][derive] Deriving scale for runs in {}".format(runs))

    eta_min = [0, 1., 1.566, 2.]
    eta_max = [1., 1.4442, 2., 2.5]
    ret = [[] for i in range(4)] #there are 4 eta bins

    run_bins = pd.read_csv(runs, delimiter='\t', header=None)

    headers = ['run_min', 'run_max', 'eta_min', 'eta_max', 'median', 'mean', 'sigma', 'scale', 'median_corr', 'mean_corr', 'sigma_corr', 'events']
    dictForDf = OrderedDict.fromkeys(headers)
    for col in headers:
        dictForDf[col] = []

    for i,pair in run_pairs.iterrows():
        mask_run = np.logical_and(pair[0] <= data[c.RUN].values, data[c.RUN].values <= pair[1])
        mask_mass = np.logical_and(c.MIN_INVMASS <= data[c.INVMASS].values, data[c.INVMASS].values <= c.MAX_INVMASS)
        mask = np.logical_and(mask_run, mask_mass)
        for j,eta_bin in enumerate(ret):
            bin_mask = np.logical_and(mask, np.logical_and( eta_min[j] <= data[c.ETA_LEAD].values, data[c.ETA_LEAD].values < eta_max[j]))
            bin_mask = np.logical_and(bin_mask, np.logical_and( eta_min[j] <= data[c.ETA_SUB].values, data[c.ETA_SUB].values < eta_max[j]))
            eta_bin.append(91.188/stat.median(data[bin_mask][c.INVMASS].values))
            if _kWriteData:
                dictForDf['run_min'].append(pair[0])
                dictForDf['run_max'].append(pair[1])
                dictForDf['eta_min'].append(eta_min[j])
                dictForDf['eta_max'].append(eta_max[j])
                dictForDf['median'].append(stat.median(data[bin_mask][c.INVMASS].values))
                dictForDf['mean'].append(stat.mean(data[bin_mask][c.INVMASS].values))
                dictForDf['sigma'].append(stat.stdev(data[bin_mask][c.INVMASS].values))
                dictForDf['events'].append(len(data[bin_mask][c.INVMASS].values))
                dictForDf['scale'].append(eta_bin[-1])
                invmass = data[bin_mask][c.INVMASS].values * eta_bin[-1]
                invmass = invmass[ np.logical_and(c.MIN_INVMASS <= invmass, invmass <= c.MAX_INVMASS) ]
                dictForDf['median_corr'].append(stat.median(invmass))
                dictForDf['mean_corr'].append(stat.mean(invmass))
                dictForDf['sigma_corr'].append(stat.stdev(invmass))

    if _kWriteData: 
        dfOut = pd.DataFrame(dictForDf)
        dfOut.to_csv("time_stability_outputData_{}-{}-{}.dat".format(datetime.datetime.today().year, datetime.datetime.today().month, datetime.datetime.today().day),
                sep='\t', header=True,index=False)

    return ret
    
