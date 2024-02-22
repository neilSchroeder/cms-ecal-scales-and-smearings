import pandas as pd
import numpy as np
import time

from python.classes.constant_classes import DataConstants as dc
from python.classes.constant_classes import CategoryConstants as cc

def smear(mc,smearings):
    """
    Applies gaussian smearings to the MC
    ----------
    Args:
        mc (pd.DataFrame): dataframe of mc
        smearings (str): path to the smearings file
    ----------
    Returns:
        mc (pd.DataFrame): dataframe of mc with smeared variables
    ----------
    """ 
    np.random.seed(dc.SEED) # otherwise there won't be any consistency

    i_cat = 0
    delim_cat = "-"
    delim_var = "_"

    smear_df = pd.read_csv(smearings, delimiter='\t', header=None, comment='#')
    tot = len(mc)
    mc["et_lead"] = np.divide(mc[dc.E_LEAD].values, np.cosh(mc[dc.ETA_LEAD].values))
    mc["et_sub"] = np.divide(mc[dc.E_SUB].values, np.cosh(mc[dc.ETA_SUB].values))
    #format is category, emain, err_mean, rho, err_rho, phi, err_phi
    for i, row in smear_df.iterrows():
    
        # split cat into parts
        cat = row[i_cat]
        cat_list = cat.split(delim_cat) 
        # cat_list[0] is eta, cat_list[1] is r9
        eta_list = cat_list[0].split(delim_var)
        r9_list = cat_list[1].split(delim_var)
        et_list = cat_list[2].split(delim_var) if len(cat_list) > 2 else ['Et', dc.MIN_ET, dc.MAX_ET]
    
        # format of lists is [VarName, VarMin, VarMax]
        eta_min, eta_max = float(eta_list[1]), float(eta_list[2])
        r9_min, r9_max = float(r9_list[1]), float(r9_list[2])
        et_min, et_max = float(et_list[1]), float(et_list[2])

        # build masks
        mask_eta_lead = np.logical_and( eta_min <= mc[dc.ETA_LEAD].values, mc[dc.ETA_LEAD].values < eta_max)
        mask_eta_sub = np.logical_and( eta_min <= mc[dc.ETA_SUB].values, mc[dc.ETA_SUB].values < eta_max)
        mask_r9_lead = np.logical_and( r9_min <= mc[dc.R9_LEAD].values, mc[dc.R9_LEAD].values < r9_max)
        mask_r9_sub = np.logical_and( r9_min <= mc[dc.R9_SUB].values, mc[dc.R9_SUB].values < r9_max)
        mask_et_lead = np.ones(len(mask_eta_lead), dtype=bool)
        mask_et_sub = np.ones(len(mask_eta_sub), dtype=bool)
        if not (et_min == dc.MIN_ET and et_max == dc.MAX_ET):
            mask_et_lead = np.logical_and(et_min <= mc["et_lead"].values, mc["et_lead"].values < et_max)
            mask_et_sub = np.logical_and(et_min <= mc["et_sub"].values, mc["et_sub"].values < et_max)
    
        mask_lead = np.logical_and(mask_eta_lead,np.logical_and(mask_r9_lead,mask_et_lead))
        tot -= np.sum(mask_lead)
        assert tot >= 0 #will catch you if you're double counting
        mask_sub = np.logical_and(mask_eta_sub,np.logical_and(mask_r9_sub,mask_et_sub))

        # smear the mc
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
        mc[dc.E_LEAD] = np.multiply(mc[dc.E_LEAD].values, smears_lead,dtype=np.float32)
        mc[dc.E_SUB] = np.multiply(mc[dc.E_SUB].values, smears_sub,dtype=np.float32)
        mc[dc.INVMASS] = np.multiply(mc[dc.INVMASS].values, np.sqrt(smears_lead),dtype=np.float32)
        mc[dc.INVMASS] = np.multiply(mc[dc.INVMASS].values, np.sqrt(smears_sub),dtype=np.float32)

    mc.drop(["et_lead", "et_sub"], axis=1, inplace=True)

    return mc
