import pandas as pd
import numpy as np
import uproot3 as up
import os

from python.classes.constant_classes import PyValConstants as pvc
from python.classes.constant_classes import DataConstants as dc
import python.classes.config_class as config_class
ss_config = config_class.SSConfig()

def extract_files(filename):
    """
    Extract files to use from a config file.

    Args:
        filename (str): the name of the config file
    Returns:
        ret_dict (dict): a dictionary of lists of files to use
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
    Loads root files into a pandas dataframe.

    Args:
        files (list): a list of files to load
        debug (bool): whether to use a smaller dataset for debugging
    Returns:
        df (pandas dataframe): the dataframe containing the data
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
        df = df.head(100000)


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