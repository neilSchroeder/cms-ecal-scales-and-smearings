"""Utilities for loading root files into dataframes."""

import numpy as np
import pandas as pd
import uproot3 as up

from python.classes.constant_classes import (
    DataConstants as dc,
)

def get_dataframe(files, 
                  apply_cuts='standard', 
                  eta_cuts=None,
                  inv_mass_cuts=None,
                  et_cuts=None, 
                  r9_cut=None, 
                  working_point=None, 
                  debug=False):
    """
    Loads root files into a pandas dataframe.

    Args:
        files (list): a list of files to load
        apply_cuts (str): whether to apply standard or custom cuts
        eta_cuts (tuple(float, float, float, float)): a tuple of floats to use as eta cuts for leading and SUBing electrons
        inv_mass_cuts (tuple(float, float)): a tuple of floats to use as inv mass cuts
        et_cuts (tuple(float, float)): a tuple of floats to use as et cuts for leading and SUBing electrons
        r9_cut (tuple(float, float) | tuple(tuple(float, float),tuple(float,float))): a float to use as an r9 cut (default is 0.96)
        working_point (str): a string to use as a working point for the electron ID
            default is "loose", other options are "medium", "tight"
        debug (bool): whether to use a smaller dataset for debugging
    Returns:
        df (pandas dataframe): the dataframe containing the data
    """

    df = pd.DataFrame()

    if ".root" in files[0]:
        #this takes a long time, so avoid it if possible
        df = pd.concat([up.open(f)[dc.TREE_NAME].pandas.df(dc.KEEP_COLS) for f in files])
        #drop unnecessary columns
        df.drop(dc.DROP_LIST, axis=1, inplace=True)
    elif ".csv" in files[0] or ".tsv" in files[0]:
        df = pd.concat([pd.read_csv(f, sep='\t',dtype=dc.DATA_TYPES) for f in files])
    else:
        print("[python][helpers][helper_main] ERROR: file type not recognized")
        raise ValueError("file type not recognized: must be .root, .csv, or .tsv")
    
    if debug:
        # use a smaller dataset for debugging
        df = df.head(100000)

    if apply_cuts == 'standard':
        df = standard_cuts(df)
    else:
        df = custom_cuts(df,
                        eta_cuts,
                        inv_mass_cuts,
                        et_cuts,
                        r9_cut,
                        working_point)
        
    return df



def standard_cuts(df):
    """
    Takes in a dataframe and applies the following cuts:
    pt_lead > 32 GeV
    pt_SUB > 20 GeV
    80 GeV < invMass < 100 GeV
    |eta| < 2.5 and !(1.4442 < |eta| < 1.566)
    """

    #masks
    return custom_cuts(df,
                        eta_cuts=(0, dc.MAX_EB, dc.MIN_EE, dc.MAX_EE),
                        inv_mass_cuts=(dc.MIN_INVMASS, dc.MAX_INVMASS),
                        et_cuts=(dc.MIN_ET_LEAD, dc.MIN_ET_SUB),
    )


def custom_cuts(df,
                eta_cuts=None,
                inv_mass_cuts=None,
                et_cuts=None,
                r9_cuts=None,
                working_point=None,
                **kwargs):
    """
    Takes in a dataframe and applies the cuts specified in custom_cuts.

    Args:
        df (pandas dataframe): the dataframe to cut
        eta_cuts (tuple(float, float, float, float) | tuple(tuple(float,float),tuple(float,float))): a tuple of floats to use as eta cuts for leading and SUBing electrons
        inv_mass_cuts (tuple(float, float)): a tuple of floats to use as inv mass cuts
        et_cuts (tuple(float, float) | tuple(tuple(float,float),tuple(float,float))): a tuple of floats to use as et cuts for leading and SUBing electrons
        r9_cut (tuple(float, float) | tuple(tuple(float,float),tuple(float,float))): a float to use as an r9 cut (default is 0.96)
        working_point (str): a string to use as a working point for the electron ID
    Returns:
        df (pandas dataframe): the dataframe with the cuts applied
    """

    mask = np.ones(len(df), dtype=bool)
    if eta_cuts:
        if isinstance(eta_cuts[0], tuple):
            # this means cuts on both leading and SUBing electrons
            if eta_cuts[0][0] != -1:
                mask &= (df[dc.ETA_LEAD] > eta_cuts[0][0])
            if eta_cuts[0][1] != -1:
                mask &= (df[dc.ETA_LEAD] < eta_cuts[0][1])
            if eta_cuts[1][0] != -1:
                mask &= (df[dc.ETA_SUB] > eta_cuts[1][0])
            if eta_cuts[1][1] != -1:
                mask &= (df[dc.ETA_SUB] < eta_cuts[1][1])
        else:
            # this means one set of cuts for both leading and SUBing electrons
            mask = mask & ((df[dc.ETA_LEAD] > eta_cuts[0]) & (df[dc.ETA_LEAD] < eta_cuts[1]) | (df[dc.ETA_LEAD] > eta_cuts[2]) & (df[dc.ETA_LEAD] < eta_cuts[3]))
            mask = mask & ((df[dc.ETA_SUB] > eta_cuts[0]) & (df[dc.ETA_SUB] < eta_cuts[1]) | (df[dc.ETA_SUB] > eta_cuts[2]) & (df[dc.ETA_SUB] < eta_cuts[3]))

    if inv_mass_cuts:
        mask = mask & (df[dc.INVMASS] > inv_mass_cuts[0]) & (df[dc.INVMASS] < inv_mass_cuts[1])

    if et_cuts:
        if isinstance(et_cuts[0], tuple):
            # this means cuts on both leading and SUBing electrons
            et_lead = np.divide(df[dc.E_LEAD].values, np.cosh(df[dc.ETA_LEAD].values))
            et_sub = np.divide(df[dc.E_SUB].values, np.cosh(df[dc.ETA_SUB].values))
            if et_cuts[0][0] != -1:
                mask &= (et_lead > et_cuts[0][0])
            if et_cuts[0][1] != -1:
                mask &= (et_lead < et_cuts[0][1])
            if et_cuts[1][0] != -1:
                mask &= (et_sub > et_cuts[1][0])
            if et_cuts[1][1] != -1:
                mask &= (et_sub < et_cuts[1][1])
        else:
            # otherwise you're just providing minimum values for both electrons
            mask = mask & (et_lead > et_cuts[0]) & (et_sub > et_cuts[1])

    if r9_cuts:
        if isinstance(r9_cuts[0], tuple):
            # this means cuts on both leading and SUBing electrons
            if r9_cuts[0][0] != -1:
                mask &= (df[dc.R9_LEAD] > r9_cuts[0][0])
            if r9_cuts[0][1] != -1:
                mask &= (df[dc.R9_LEAD] < r9_cuts[0][1])
            if r9_cuts[1][0] != -1:
                mask &= (df[dc.R9_SUB] > r9_cuts[1][0])
            if r9_cuts[1][1] != -1:
                mask &= (df[dc.R9_SUB] < r9_cuts[1][1])

        else:
            # otherwise you're just providing a minimum value for both electrons
            mask = mask & (df[dc.R9_LEAD] > r9_cuts[0]) & (df[dc.R9_SUB] > r9_cuts[1])

    if working_point:
        wp_id = dc.TIGHT_ID if working_point == "tight" else dc.MEDIUM_ID
        mask = mask & np.array([x&wp_id for x in df[dc.ID_LEAD].values]) & np.array([x&wp_id for x in df[dc.ID_SUB].values])

    return df[mask]