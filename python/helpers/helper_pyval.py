import pandas as pd
import numpy as np
import uproot as up
import os

import python.classes.pyval_constants_class as constants

def extract_files(filename):
    """
    takes in the config dataframe and returns 5 lists separating out 
    data, mc, scales, smearings, and cats files
    """

    df = pd.read_csv(filename, sep='\t', header=None, comment="#")

    ret_dict = {}
    ret_dict["DATA"] = []
    ret_dict["MC"] = []
    ret_dict["SCALES"] = []
    ret_dict["SMEARINGS"] = []
    ret_dict["WEIGHTS"] = []
    ret_dict["CATS"] = []

    for i,row in df.iterrows():
        if os.path.exists(row[1]): 
            ret_dict[row[0]].append(row[1])
        else:
            print(f'[ERROR] file does not exist {row[1]}')
            raise KeyboardInterrupt

    return ret_dict

def get_dataframe(files):
    """
    takes in a list of root files for data and mc.
    opens them with uproot into dataframes
    """

    #load constants
    c = constants.const()

    #branches we'll need
    TREE_NAME = 'selected'
    keep_cols = [
            'R9Ele',
            'energy_ECAL_ele',
            'etaEle',
            'phiEle',
            'gainSeedSC',
            'invMass_ECAL_ele',
            'runNumber',
            'mcGenWeight',
            'eleID',
            ]

    df = pd.DataFrame()

    if ".root" in files[0]:
        #this takes a long time, so avoid it if possible
        df = pd.concat([up.open(f)[TREE_NAME].pandas.df(keep_cols) for f in files])
        #drop unnecessary columns
        drop_list = ['R9Ele[2]', 'energy_ECAL_ele[2]', 'etaEle[2]', 'gainSeedSC[2]', 'phiEle[2]', 'eleID[2]']
        df.drop(drop_list, axis=1, inplace=True)
    elif ".csv" in files[0] or ".tsv" in files[0]:
        data_types = {
                c.R9_LEAD: np.float32,
                c.R9_SUB: np.float32,
                c.ETA_LEAD: np.float32,
                c.ETA_SUB: np.float32,
                c.E_LEAD: np.float32,
                c.E_SUB: np.float32,
                c.PHI_LEAD: np.float32,
                c.PHI_SUB: np.float32,
                c.INVMASS: np.float32,
                c.RUN: np.int32,
                c.GAIN_LEAD: np.int16,
                c.GAIN_SUB: np.int16,
                }
        df = pd.concat([pd.read_csv(f, sep='\t',dtype=data_types) for f in files])
    else:
        print("[python][helpers][helper_main] get goofed")


    #clean the data a bit before sending back

    df[c.ETA_LEAD] = np.abs(df[c.ETA_LEAD])
    df[c.ETA_SUB] = np.abs(df[c.ETA_SUB])
    
    transition_mask_lead = ~df[c.ETA_LEAD].between(c.MAX_EB,c.MIN_EE)
    transition_mask_sub = ~df[c.ETA_SUB].between(c.MAX_EB,c.MIN_EE)
    tracker_mask_lead = ~df[c.ETA_LEAD].between(c.MAX_EE, c.TRACK_MAX)
    tracker_mask_sub = ~df[c.ETA_SUB].between(c.MAX_EE, c.TRACK_MAX)
    invmass_mask = df[c.INVMASS].between(c.invmass_min, c.invmass_max)
    mask = transition_mask_lead&transition_mask_sub&tracker_mask_lead&tracker_mask_sub&invmass_mask
    df = df.loc[mask]

    return df

def standard_cuts(df):
    """
    Takes in a dataframe and applies the following cuts:
    pt_lead > 32 GeV
    pt_sublead > 20 GeV
    80 GeV < invMass < 100 GeV
    |eta| < 2.5 and !(1.4442 < |eta| < 1.566)
    """

    #constants
    c = constants.const()

    #masks
    mask_lead = (np.divide(df[c.E_LEAD].values, np.cosh(df[c.ETA_LEAD].values))) >= c.MIN_PT_LEAD
    mask_sub = (np.divide(df[c.E_SUB].values, np.cosh(df[c.ETA_SUB].values))) >= c.MIN_PT_SUB

    mask_lead = np.logical_and(mask_lead,np.logical_or(df[c.ETA_LEAD].values < c.MAX_EB, c.MIN_EE < df[c.ETA_LEAD].values))
    mask_lead = np.logical_and(mask_lead, df[c.ETA_LEAD].values < c.MAX_EE)

    mask_sub = np.logical_and(mask_sub,np.logical_or(df[c.ETA_SUB].values < c.MAX_EB, c.MIN_EE < df[c.ETA_SUB].values))
    mask_sub = np.logical_and(mask_sub, df[c.ETA_SUB].values < c.MAX_EE)

    mask_invmass = np.logical_and(c.MIN_INVMASS <= df[c.INVMASS].values, df[c.INVMASS].values <= c.MAX_INVMASS)

    mask = np.logical_and(mask_lead,mask_sub)
    mask = np.logical_and(mask, mask_invmass)

    return df[mask]

def custom_cuts(df, custom_cuts):

    pass #TODO write out custom cuts function

    return df
