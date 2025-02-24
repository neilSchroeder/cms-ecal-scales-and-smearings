import time

import numba
import numpy as np
import pandas as pd

from python.classes.constant_classes import CategoryConstants as cc
from python.classes.constant_classes import DataConstants as dc
from python.tools.data_loader import custom_cuts


@numba.njit
def transform_smearings(smearings, new_sigma, old_sigma):
    return (smearings - 1) * (new_sigma / old_sigma) + 1


def smear(mc, smearings):
    """
    Applies gaussian smearings to the MC in a double loop approach

    This may end up being a little slower, but the logic flows better
    """
    delim_cat = "-"
    delim_var = "_"
    # read in the smearings
    smear_df = pd.read_csv(smearings, delimiter="\t", header=None, comment="#")

    rand = np.random.Generator(np.random.PCG64(dc.SEED))
    lead_smearings = rand.normal(1, 0.001, len(mc))
    current_smearing_lead = 0.001
    sublead_smearings = rand.normal(1, 0.001, len(mc))
    current_smearing_sublead = 0.001

    lead_eta = mc[dc.ETA_LEAD].values
    lead_r9 = mc[dc.R9_LEAD].values
    lead_e = mc[dc.E_LEAD].values
    lead_et = np.divide(lead_e, np.cosh(lead_eta))
    sublead_eta = mc[dc.ETA_SUB].values
    sublead_r9 = mc[dc.R9_SUB].values
    sublead_e = mc[dc.E_SUB].values
    sublead_et = np.divide(sublead_e, np.cosh(sublead_eta))

    for i, row_i in smear_df.iterrows():
        # transform the lead smearings
        lead_smearings = transform_smearings(
            lead_smearings, row_i[3], current_smearing_lead
        )
        current_smearing_lead = row_i[3]

        # split cat into parts
        cat = row_i[0]
        cat_list = cat.split(delim_cat)
        # cat_list[0] is eta, cat_list[1] is r9
        eta_list = cat_list[0].split(delim_var)
        r9_list = cat_list[1].split(delim_var)
        et_list = (
            cat_list[2].split(delim_var)
            if len(cat_list) > 2
            else ["Et", dc.MIN_ET, dc.MAX_ET]
        )

        lead_mask = np.logical_and.reduce(
            (
                lead_eta >= float(eta_list[1]),
                lead_eta < float(eta_list[2]),
                lead_r9 >= float(r9_list[1]),
                lead_r9 < float(r9_list[2]),
                lead_et >= float(et_list[1]),
                lead_et < float(et_list[2]),
            )
        )
        for j, row_j in smear_df.iterrows():
            # transform the sublead smearings
            sublead_smearings = transform_smearings(
                sublead_smearings, row_j[3], current_smearing_sublead
            )
            current_smearing_sublead = row_j[3]

            # split cat into parts
            cat = row_j[0]
            cat_list = cat.split(delim_cat)
            # cat_list[0] is eta, cat_list[1] is r9
            eta_list = cat_list[0].split(delim_var)
            r9_list = cat_list[1].split(delim_var)
            et_list = (
                cat_list[2].split(delim_var)
                if len(cat_list) > 2
                else ["Et", dc.MIN_ET, dc.MAX_ET]
            )

            sublead_mask = np.logical_and.reduce(
                (
                    sublead_eta >= float(eta_list[1]),
                    sublead_eta < float(eta_list[2]),
                    sublead_r9 >= float(r9_list[1]),
                    sublead_r9 < float(r9_list[2]),
                    sublead_et >= float(et_list[1]),
                    sublead_et < float(et_list[2]),
                )
            )

            mask = np.logical_and(lead_mask, sublead_mask)
            mc.loc[mask, dc.E_LEAD] *= lead_smearings[mask]
            mc.loc[mask, dc.E_SUB] *= sublead_smearings[mask]
            mc.loc[mask, dc.INVMASS] *= np.sqrt(
                lead_smearings[mask] * sublead_smearings[mask]
            ) / (1 - current_smearing_lead * current_smearing_sublead / 8)

    return custom_cuts(
        mc,
        inv_mass_cuts=(80, 100),
        eta_cuts=(0, 1.4442, 1.566, 2.5),
        et_cuts=((32, 14000), (20, 14000)),
    )
