import numpy as np
import pandas as pd
import statistics as stat
import write_files

def derive(data, runs):
    print("[INFO][python/time_stability][derive] Deriving scale for runs in {}".format(runs))
    ret = []
    ret_eta0 = [] #eta 0 - 1
    ret_eta1 = [] #eta 1 - 1.4442
    ret_eta2 = [] #eta 1.566 - 2
    ret_eta3 = [] #eta 2 - 2.5
    run_pairs = pd.read_csv(runs, delimiter='\t', header=None)
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

    ret.append(ret_eta0)
    ret.append(ret_eta1)
    ret.append(ret_eta2)
    ret.append(ret_eta3)
    return ret
