import numpy as np
import pandas as pd

def scale(data, scales):
#newformat of scales files is 
#runMin runMax etaMin etaMax r9Min r9Max etMin etMax gain val err
    scales_df = pd.read_csv(scales, sep="\t", comment="#", header=None)
    for index,row in scales_df.iterrows():
        run_vals = data['runNumber'].between(row[0],row[1])
        ele0_eta = data['etaEle[0]'].between(row[2], row[3])
        ele1_eta = data['etaEle[1]'].between(row[2], row[3])
        ele0_r9 = data['R9Ele[0]'].between(row[4], row[5])
        ele1_r9 = data['R9Ele[1]'].between(row[4], row[5])
        data.loc[run_vals&ele0_eta&ele0_r9,'energy_ECAL_ele[0]'] *= row[9]
        data.loc[run_vals&ele0_eta&ele0_r9,'invMass_ECAL_ele'] *= np.sqrt(row[9])
        data.loc[run_vals&ele1_eta&ele1_r9,'energy_ECAL_ele[1]'] *= row[9]
        data.loc[run_vals&ele1_eta&ele1_r9,'invMass_ECAL_ele'] *= np.sqrt(row[9])

