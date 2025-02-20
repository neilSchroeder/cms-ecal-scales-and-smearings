import time

import numba
import numpy as np
import pandas as pd

from python.classes.constant_classes import CategoryConstants as cc
from python.classes.constant_classes import DataConstants as dc
from python.tools.data_loader import custom_cuts


@numba.njit
def multiply(arr1, arr2):
    """
    Multiplies two arrays element-wise.
    ----------
    Args:
        arr1 (np.array): array 1
        arr2 (np.array): array 2
    ----------
    Returns:
        np.array: element-wise product of arr1 and arr2
    ----------
    """
    return np.multiply(arr1, arr2).astype(np.float32)


@numba.njit
def apply_smearing(mc, lead_smear, sublead_smear, seed):
    np.random.seed(seed)
    lead_rand = np.random.normal(1, lead_smear, len(mc))
    sublead_rand = np.random.normal(1, sublead_smear, len(mc))
    x = np.sqrt((lead_rand) * (sublead_rand))
    return mc * x


def smear_old(mc, smearings):
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
    np.random.seed(dc.SEED)  # otherwise there won't be any consistency
    i_cat = 0
    delim_cat = "-"
    delim_var = "_"

    mc_eta_lead = mc[dc.ETA_LEAD].values
    mc_eta_sub = mc[dc.ETA_SUB].values
    mc_r9_lead = mc[dc.R9_LEAD].values
    mc_r9_sub = mc[dc.R9_SUB].values
    mc_et_lead = np.divide(mc[dc.E_LEAD].values, np.cosh(mc_eta_lead))
    mc_et_sub = np.divide(mc[dc.E_SUB].values, np.cosh(mc_eta_sub))

    smear_df = pd.read_csv(smearings, delimiter="\t", header=None, comment="#")
    tot = len(mc)
    # format is category, emain, err_mean, rho, err_rho, phi, err_phi
    total_mask_lead = np.zeros(len(mc), dtype=bool)
    for i, row in smear_df.iterrows():

        # split cat into parts
        cat = row[i_cat]
        cat_list = cat.split(delim_cat)
        # cat_list[0] is eta, cat_list[1] is r9
        eta_list = cat_list[0].split(delim_var)
        r9_list = cat_list[1].split(delim_var)
        et_list = (
            cat_list[2].split(delim_var)
            if len(cat_list) > 2
            else ["Et", dc.MIN_ET, dc.MAX_ET]
        )

        # format of lists is [VarName, VarMin, VarMax]
        eta_min, eta_max = float(eta_list[1]), float(eta_list[2])
        r9_min, r9_max = float(r9_list[1]), float(r9_list[2])
        et_min, et_max = float(et_list[1]), float(et_list[2])

        # build masks
        mask_eta_lead = np.logical_and(eta_min <= mc_eta_lead, mc_eta_lead < eta_max)
        mask_eta_sub = np.logical_and(eta_min <= mc_eta_sub, mc_eta_sub < eta_max)
        mask_r9_lead = np.logical_and(r9_min <= mc_r9_lead, mc_r9_lead < r9_max)
        mask_r9_sub = np.logical_and(r9_min <= mc_r9_sub, mc_r9_sub < r9_max)
        mask_et_lead = np.ones(len(mask_eta_lead), dtype=bool)
        mask_et_sub = np.ones(len(mask_eta_sub), dtype=bool)
        if not (et_min == dc.MIN_ET and et_max == dc.MAX_ET):
            mask_et_lead = np.logical_and(et_min <= mc_et_lead, mc_et_lead < et_max)
            mask_et_sub = np.logical_and(et_min <= mc_et_sub, mc_et_sub < et_max)

        mask_lead = np.logical_and(
            mask_eta_lead, np.logical_and(mask_r9_lead, mask_et_lead)
        )
        total_mask_lead = np.logical_or(total_mask_lead, mask_lead)
        tot -= np.sum(mask_lead)
        assert tot >= 0  # will catch you if you're double counting
        mask_sub = np.logical_and(
            mask_eta_sub, np.logical_and(mask_r9_sub, mask_et_sub)
        )

        # smear the mc
        smears_lead = multiply(
            mask_lead, np.random.normal(1, row[3], len(mask_lead)).astype(np.float32)
        )
        # smears_lead_up = multiply(mask_lead, normal(1, row[3] + row[4], len(mask_lead)))
        # smears_lead_down = multiply(mask_lead, normal(1, np.abs(row[3] - row[4]), len(mask_lead)))
        smears_sub = multiply(
            mask_sub, np.random.normal(1, row[3], len(mask_sub)).astype(np.float32)
        )
        # smears_sub_up = multiply(mask_sub, normal(1, row[3] + row[4], len(mask_sub)))
        # smears_sub_down = multiply(mask_sub, normal(1, np.abs(row[3] - row[4]), len(mask_sub)))
        no_smear_mask_lead = smears_lead == 0
        no_smear_mask_sub = smears_sub == 0
        smears_lead[no_smear_mask_lead] = 1.0
        # smears_lead_up[smears_lead_up==0] = 1.
        # smears_lead_down[smears_lead_down==0] = 1.
        smears_sub[no_smear_mask_sub] = 1.0
        # smears_sub_up[smears_sub_up==0] = 1.
        # smears_sub_down[smears_sub_down==0] = 1.

        # get energies and smear them
        mc[dc.E_LEAD] = multiply(mc[dc.E_LEAD].values, smears_lead)
        mc[dc.E_SUB] = multiply(mc[dc.E_SUB].values, smears_sub)
        mc[dc.INVMASS] = multiply(
            mc[dc.INVMASS].values, np.sqrt(multiply(smears_lead, smears_sub))
        )

    if np.sum(total_mask_lead) != len(mc):
        print("[WARNING] Not all events were smeared")
        print(mc[~total_mask_lead].head())
        print(mc[~total_mask_lead].describe())

    return custom_cuts(
        mc,
        inv_mass_cuts=(80, 100),
        eta_cuts=(0, 1.4442, 1.566, 2.5),
        et_cuts=((32, 14000), (20, 14000)),
    )


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

    for i in range(len(smear_df)):
        # grab the row
        row_i = smear_df.iloc[i]

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
                mc[dc.ETA_LEAD].values >= float(eta_list[1]),
                mc[dc.ETA_LEAD].values < float(eta_list[2]),
                mc[dc.R9_LEAD].values >= float(r9_list[1]),
                mc[dc.R9_LEAD].values < float(r9_list[2]),
                mc[dc.E_LEAD].values >= float(et_list[1]),
                mc[dc.E_LEAD].values < float(et_list[2]),
            )
        )
        for j in range(len(smear_df)):
            row_j = smear_df.iloc[j]

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
                    mc[dc.ETA_SUB].values >= float(eta_list[1]),
                    mc[dc.ETA_SUB].values < float(eta_list[2]),
                    mc[dc.R9_SUB].values >= float(r9_list[1]),
                    mc[dc.R9_SUB].values < float(r9_list[2]),
                    mc[dc.E_SUB].values >= float(et_list[1]),
                    mc[dc.E_SUB].values < float(et_list[2]),
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
