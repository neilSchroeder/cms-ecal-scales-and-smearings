import pandas as pd
import numpy as np

from python.classes.constant_classes import DataConstants as dc

def standard(df):
    """
    Takes in a dataframe and applies the following cuts:
    pt_lead > 32 GeV
    pt_sublead > 20 GeV
    80 GeV < invMass < 100 GeV
    |eta| < 2.5 and !(1.4442 < |eta| < 1.566)
    ----------
    Args:
        df: pandas dataframe
    ----------
    Returns:
        df: pandas dataframe with cuts applied
    ----------
    """

    #masks
    mask_pt_lead = (np.divide(df[dc.E_LEAD].values,
                            np.cosh(df[dc.ETA_LEAD].values))) > dc.MIN_PT_LEAD
    mask_pt_sub = (np.divide(df[dc.E_SUB].values,
                            np.cosh(df[dc.ETA_SUB].values))) > dc.MIN_PT_SUB
    mask_eta_lead = ~(df.loc[:,dc.ETA_LEAD].between(dc.MAX_EB, dc.MIN_EE))
    mask_eta_lead = mask_eta_lead&(df[dc.ETA_LEAD] < dc.MAX_EE)
    mask_eta_sub = ~(df.loc[:,dc.ETA_SUB].between(dc.MAX_EB, dc.MIN_EE))
    mask_eta_sub = mask_eta_sub&(df[dc.ETA_SUB] < dc.MAX_EE)
    mask_invmass = df.loc[:,dc.INVMASS].between(dc.MIN_INVMASS, dc.MAX_INVMASS)

    return df[mask_pt_lead&mask_pt_sub&mask_eta_lead&mask_eta_sub&mask_invmass]

def custom(df, custom_cuts):
    """
    Takes in a dataframe and applies custom cuts
    ----------
    Args:
        df: pandas dataframe
        custom_cuts: ???
    ----------
    Returns:
        df: pandas dataframe with cuts applied
    ----------
    """

    pass #TODO write out custom cuts function

    # return df
