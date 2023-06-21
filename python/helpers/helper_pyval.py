import pandas as pd
import numpy as np
import uproot as up
import os

from python.classes.constant_classes import PyValConstants as pvc
from python.classes.constant_classes import DataConstants as dc
import python.classes.config_class as config_class
ss_config = config_class.SSConfig()

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
            raise RuntimeError

    return ret_dict

def get_dataframe(files, debug=False):
    """
    takes in a list of root files for data and mc.
    opens them with uproot into dataframes
    """

    df = pd.DataFrame()

    if ".root" in files[0]:
        #this takes a long time, so avoid it if possible
        df = pd.concat([up.open(f)[pvc.TREE_NAME].pandas.df(pvc.KEEP_COLS) for f in files])
        #drop unnecessary columns
        drop_list = ['R9Ele[2]', 'energy_ECAL_ele[2]', 'etaEle[2]', 'gainSeedSC[2]', 'phiEle[2]', 'eleID[2]']
        df.drop(drop_list, axis=1, inplace=True)
    elif ".csv" in files[0] or ".tsv" in files[0]:
        df = pd.concat([pd.read_csv(f, sep='\t',dtype=dc.DATA_TYPES) for f in files])
    else:
        print("[python][helpers][helper_main] ERROR: file type not recognized")
        raise ValueError("file type not recognized: must be .root, .csv, or .tsv")
    
    if debug:
        # use a smaller dataset for debugging
        df = df.head(1000000)


    #clean the data a bit before sending back

    df[dc.ETA_LEAD] = np.abs(df[dc.ETA_LEAD])
    df[dc.ETA_SUB] = np.abs(df[dc.ETA_SUB])
    
    transition_mask_lead = ~df[dc.ETA_LEAD].between(dc.MAX_EB,dc.MIN_EE)
    transition_mask_sub = ~df[dc.ETA_SUB].between(dc.MAX_EB,dc.MIN_EE)
    tracker_mask_lead = ~df[dc.ETA_LEAD].between(dc.MAX_EE, dc.TRACK_MAX)
    tracker_mask_sub = ~df[dc.ETA_SUB].between(dc.MAX_EE, dc.TRACK_MAX)
    invmass_mask = df[dc.INVMASS].between(dc.invmass_min, dc.invmass_max)
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

    #masks
    mask_lead = (np.divide(df[dc.E_LEAD].values, np.cosh(df[dc.ETA_LEAD].values))) >= dc.MIN_PT_LEAD
    mask_sub = (np.divide(df[dc.E_SUB].values, np.cosh(df[dc.ETA_SUB].values))) >= dc.MIN_PT_SUB

    mask_lead = np.logical_and(mask_lead,np.logical_or(df[dc.ETA_LEAD].values < dc.MAX_EB, dc.MIN_EE < df[dc.ETA_LEAD].values))
    mask_lead = np.logical_and(mask_lead, df[dc.ETA_LEAD].values < dc.MAX_EE)

    mask_sub = np.logical_and(mask_sub,np.logical_or(df[dc.ETA_SUB].values < dc.MAX_EB, dc.MIN_EE < df[dc.ETA_SUB].values))
    mask_sub = np.logical_and(mask_sub, df[dc.ETA_SUB].values < dc.MAX_EE)

    mask_invmass = np.logical_and(dc.MIN_INVMASS <= df[dc.INVMASS].values, df[dc.INVMASS].values <= dc.MAX_INVMASS)

    mask = np.logical_and(mask_lead,mask_sub)
    mask = np.logical_and(mask, mask_invmass)

    return df[mask]


def custom_cuts(df, custom_cuts):

    pass #TODO write out custom cuts function