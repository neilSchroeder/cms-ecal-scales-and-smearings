from collections import OrderedDict
import datetime
import numpy as np
import pandas as pd
import statistics as stat

import python.utilities.write_files as write_files

def derive(data, runs, _kWriteData=True):
    print("[INFO][python/time_stability][derive] Deriving scale for runs in {}".format(runs))
    ret = []
    ret_eta0 = [] #eta 0 - 1
    ret_eta1 = [] #eta 1 - 1.4442
    ret_eta2 = [] #eta 1.566 - 2
    ret_eta3 = [] #eta 2 - 2.5
    run_pairs = pd.read_csv(runs, delimiter='\t', header=None)
    headers = ['run_min', 'run_max', 'eta_min', 'eta_max', 'median', 'mean', 'sigma', 'scale', 'median_corr', 'mean_corr', 'sigma_corr', 'events']
    dictForDf = OrderedDict.fromkeys(headers)
    for col in headers:
        dictForDf[col] = []

    for i,pair in run_pairs.iterrows():
        mask_run = data['runNumber'].between(pair[0], pair[1])
        mask_eta0 = data['etaEle[0]'].between(0, 1) & data['etaEle[1]'].between(0, 1)
        mask_eta1 = data['etaEle[0]'].between(1, 1.4442) & data['etaEle[1]'].between(1, 1.4442)
        mask_eta2 = data['etaEle[0]'].between(1.566, 2) & data['etaEle[1]'].between(1.566, 2)
        mask_eta3 = data['etaEle[0]'].between(2, 2.5) & data['etaEle[1]'].between(2, 2.5)
        mask_invMass = data['invMass_ECAL_ele'].between(80, 100)
        df0 = data.loc[mask_run&mask_eta0&mask_invMass]
        df1 = data.loc[mask_run&mask_eta1&mask_invMass]
        df2 = data.loc[mask_run&mask_eta2&mask_invMass]
        df3 = data.loc[mask_run&mask_eta3&mask_invMass]
        invMass0 = df0['invMass_ECAL_ele'].values
        invMass1 = df1['invMass_ECAL_ele'].values
        invMass2 = df2['invMass_ECAL_ele'].values
        invMass3 = df3['invMass_ECAL_ele'].values
        #scales correct median of run distributions to pdg Z mass
        ret_eta0.append(91.188/stat.median(invMass0))
        ret_eta1.append(91.188/stat.median(invMass1))
        ret_eta2.append(91.188/stat.median(invMass2))
        ret_eta3.append(91.188/stat.median(invMass3))

        if _kWriteData:
        #eta0
           dictForDf['run_min'].append(pair[0])
           dictForDf['run_max'].append(pair[1])
           dictForDf['eta_min'].append(0)
           dictForDf['eta_max'].append(1)
           dictForDf['median'].append(stat.median(invMass0))
           dictForDf['mean'].append(stat.mean(invMass0))
           dictForDf['sigma'].append(stat.stdev(invMass0))
           dictForDf['events'].append(len(invMass0))
           dictForDf['scale'].append(ret_eta0[-1])
           df0 = data.loc[mask_run&mask_eta0]
           invMass0 = df0['invMass_ECAL_ele'].values
           invMass0 *= ret_eta0[-1]
           invMass0 = invMass0[invMass0>=80]
           invMass0 = invMass0[invMass0<=100]
           dictForDf['median_corr'].append(stat.median(invMass0))
           dictForDf['mean_corr'].append(stat.mean(invMass0))
           dictForDf['sigma_corr'].append(stat.stdev(invMass0))
    
           dictForDf['run_min'].append(pair[0])
           dictForDf['run_max'].append(pair[1])
           dictForDf['eta_min'].append(1)
           dictForDf['eta_max'].append(1.4442)
           dictForDf['median'].append(stat.median(invMass1))
           dictForDf['mean'].append(stat.mean(invMass1))
           dictForDf['sigma'].append(stat.stdev(invMass1))
           dictForDf['events'].append(len(invMass1))
           dictForDf['scale'].append(ret_eta1[-1])
           df1 = data.loc[mask_run&mask_eta1]
           invMass1 = df1['invMass_ECAL_ele'].values
           invMass1 *= ret_eta1[-1]
           invMass1 = invMass1[invMass1>=80]
           invMass1 = invMass1[invMass1<=100]
           dictForDf['median_corr'].append(stat.median(invMass1))
           dictForDf['mean_corr'].append(stat.mean(invMass1))
           dictForDf['sigma_corr'].append(stat.stdev(invMass1))
    
           dictForDf['run_min'].append(pair[0])
           dictForDf['run_max'].append(pair[1])
           dictForDf['eta_min'].append(1.566)
           dictForDf['eta_max'].append(2)
           dictForDf['median'].append(stat.median(invMass2))
           dictForDf['mean'].append(stat.mean(invMass2))
           dictForDf['sigma'].append(stat.stdev(invMass2))
           dictForDf['events'].append(len(invMass2))
           dictForDf['scale'].append(ret_eta2[-1])
           df2 = data.loc[mask_run&mask_eta2]
           invMass2 = df2['invMass_ECAL_ele'].values
           invMass2 *= ret_eta2[-1]
           invMass2 = invMass2[invMass2>=80]
           invMass2 = invMass2[invMass2<=100]
           dictForDf['median_corr'].append(stat.median(invMass2))
           dictForDf['mean_corr'].append(stat.mean(invMass2))
           dictForDf['sigma_corr'].append(stat.stdev(invMass2))
    
           dictForDf['run_min'].append(pair[0])
           dictForDf['run_max'].append(pair[1])
           dictForDf['eta_min'].append(2)
           dictForDf['eta_max'].append(2.5)
           dictForDf['median'].append(stat.median(invMass3))
           dictForDf['mean'].append(stat.mean(invMass3))
           dictForDf['sigma'].append(stat.stdev(invMass3))
           dictForDf['events'].append(len(invMass3))
           dictForDf['scale'].append(ret_eta3[-1])
           df3 = data.loc[mask_run&mask_eta3]
           invMass3 = df3['invMass_ECAL_ele'].values
           invMass3 *= ret_eta3[-1]
           invMass3 = invMass3[invMass3>=80]
           invMass3 = invMass3[invMass3<=100]
           dictForDf['median_corr'].append(stat.median(invMass3))
           dictForDf['mean_corr'].append(stat.mean(invMass3))
           dictForDf['sigma_corr'].append(stat.stdev(invMass3))



    ret.append(ret_eta0)
    ret.append(ret_eta1)
    ret.append(ret_eta2)
    ret.append(ret_eta3)
    if _kWriteData: 
        dfOut = pd.DataFrame(dictForDf)
        dfOut.to_csv("time_stability_outputData_{}_{}_{}.dat".format(datetime.datetime.today().year, datetime.datetime.today().month, datetime.datetime.today().day),
                sep='\t', header=True,index=False)
    return ret
