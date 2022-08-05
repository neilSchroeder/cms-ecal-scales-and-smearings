import pandas as pd
import numpy as np
import time

import python.classes.const_class as constants

def smear(mc,smearings):
    #applies gaussian smearings to the MC

    def weighted_std(x_vals, y_vals):
        avg = np.average(x_vals, weights=y_vals)
        var = np.average(np.power(x_vals - avg,2), weights=y_vals)
        return np.sqrt(var)
        
    #constants
    c = constants.const()

    np.random.seed(c.SEED)

    i_cat = 0
    delim_cat = "-"
    delim_var = "_"

    mc["invmass_up"] = np.copy(mc[c.INVMASS].values)
    mc["invmass_down"] = np.copy(mc[c.INVMASS].values)
                                   
    smear_df = pd.read_csv(smearings, delimiter='\t', header=None, comment='#')
    tot = len(mc)
    mc["et_lead"] = np.divide(mc[c.E_LEAD].values, np.cosh(mc[c.ETA_LEAD].values))
    mc["et_sub"] = np.divide(mc[c.E_SUB].values, np.cosh(mc[c.ETA_SUB].values))
    #format is category, emain, err_mean, rho, err_rho, phi, err_phi
    count_lead = np.zeros(len(mc[c.ETA_LEAD].values),dtype=bool)
    count_sub = np.zeros(len(mc[c.ETA_LEAD].values),dtype=bool)
    for i,row in smear_df.iterrows():
    
        #split cat into parts
        cat = row[i_cat]
        cat_list = cat.split(delim_cat) 
        #cat_list[0] is eta, cat_list[1] is r9
        eta_list = cat_list[0].split(delim_var)
        r9_list = cat_list[1].split(delim_var)
        et_list = cat_list[2].split(delim_var) if len(cat_list) > 2 else ['Et', c.MIN_ET, c.MAX_ET]
    
        #format of lists is [VarName, VarMin, VarMax]
        eta_min, eta_max = float(eta_list[1]), float(eta_list[2])
        r9_min, r9_max = float(r9_list[1]), float(r9_list[2])
        et_min, et_max = float(et_list[1]), float(et_list[2])

        #build masks
        mask_eta_lead = np.logical_and( eta_min <= mc[c.ETA_LEAD].values, mc[c.ETA_LEAD].values < eta_max)
        mask_eta_sub = np.logical_and( eta_min <= mc[c.ETA_SUB].values, mc[c.ETA_SUB].values < eta_max)
        mask_r9_lead = np.logical_and( r9_min <= mc[c.R9_LEAD].values, mc[c.R9_LEAD].values < r9_max)
        mask_r9_sub = np.logical_and( r9_min <= mc[c.R9_SUB].values, mc[c.R9_SUB].values < r9_max)
        mask_et_lead = np.ones(len(mask_eta_lead), dtype=bool)
        mask_et_sub = np.ones(len(mask_eta_sub), dtype=bool)
        if not (et_min == c.MIN_ET and et_max == c.MAX_ET):
            mask_et_lead = np.logical_and(et_min <= mc["et_lead"].values, mc["et_lead"].values < et_max)
            mask_et_sub = np.logical_and(et_min <= mc["et_sub"].values, mc["et_sub"].values < et_max)
    
        mask_lead = np.logical_and(mask_eta_lead,np.logical_and(mask_r9_lead,mask_et_lead))
        mask_sub = np.logical_and(mask_eta_sub,np.logical_and(mask_r9_sub,mask_et_sub))

        #smear the mc
        smears_lead = np.multiply(mask_lead, np.random.normal(1, row[3], len(mask_lead)),dtype=np.float32)
        smears_lead_up = np.multiply(mask_lead, np.random.normal(1, row[3] + row[4], len(mask_lead)),dtype=np.float32)
        smears_lead_down = np.multiply(mask_lead, np.random.normal(1, np.abs(row[3] - row[4]), len(mask_lead)),dtype=np.float32)
        smears_sub = np.multiply(mask_sub, np.random.normal(1, row[3], len(mask_sub)),dtype=np.float32)
        smears_sub_up = np.multiply(mask_sub, np.random.normal(1, row[3] + row[4], len(mask_sub)),dtype=np.float32)
        smears_sub_down = np.multiply(mask_sub, np.random.normal(1, np.abs(row[3] - row[4]), len(mask_sub)),dtype=np.float32)
        smears_lead[smears_lead==0] = 1.
        smears_lead_up[smears_lead_up==0] = 1.
        smears_lead_down[smears_lead_down==0] = 1.
        smears_sub[smears_sub==0] = 1.
        smears_sub_up[smears_sub_up==0] = 1.
        smears_sub_down[smears_sub_down==0] = 1.

        #get energies and smear them
        mc[c.E_LEAD] = np.multiply(mc[c.E_LEAD].values, smears_lead,dtype=np.float32)
        mc[c.E_SUB] = np.multiply(mc[c.E_SUB].values, smears_sub,dtype=np.float32)
        mc["invmass_up"] = np.multiply(mc["invmass_up"].values, np.sqrt(np.multiply(smears_lead_up,smears_sub_up)),dtype=np.float32)
        mc["invmass_down"] = np.multiply(mc["invmass_down"].values, np.sqrt(np.multiply(smears_lead_down,smears_sub_down)),dtype=np.float32)
        mc[c.INVMASS] = np.multiply(mc[c.INVMASS].values, np.sqrt(np.multiply(smears_lead,smears_sub)),dtype=np.float32)

    mc.drop(["et_lead", "et_sub"], axis=1, inplace=True)

    h_down, _ = np.histogram(mc["invmass_down"].values, bins=80, range=[80.,100.])
    h, _ = np.histogram(mc[c.INVMASS].values, bins=80, range=[80.,100.])
    h_up, _ = np.histogram(mc["invmass_up"].values, bins=80, range=[80.,100.])


    bins = np.arange(80.,100.25,0.25)
    mids = np.array([(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)])
    print("h_down", weighted_std(mids, h_down))
    print("h", weighted_std(mids, h))
    print("h_up", weighted_std(mids, h_up))


    return mc
