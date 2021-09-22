import pandas as pd
import numpy as np

import python.classes.constants as constants

def standard(df):
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
    mask_pt_lead = (np.divide(df[c.E_LEAD].values,
                            np.cosh(df[c.ETA_LEAD].values))) > c.MIN_PT_LEAD
    mask_pt_sub = (np.divide(df[c.E_SUB].values,
                            np.cosh(df[c.ETA_SUB].values))) > c.MIN_PT_SUB
    mask_eta_lead = ~(df.loc[:,c.ETA_LEAD].between(c.MAX_EB, c.MIN_EE))
    mask_eta_lead = mask_eta_lead&(df[c.ETA_LEAD] < c.MAX_EE)
    mask_eta_sub = ~(df.loc[:,c.ETA_SUB].between(c.MAX_EB, c.MIN_EE))
    mask_eta_sub = mask_eta_sub&(df[c.ETA_SUB] < c.MAX_EE)
    mask_invmass = df.loc[:,c.INVMASS].between(c.MIN_INVMASS, c.MAX_INVMASS)

    return df[mask_pt_lead&mask_pt_sub&mask_eta_lead&mask_eta_sub&mask_invmass]

def custom(df, custom_cuts):

    pass #TODO write out custom cuts function

    return df
