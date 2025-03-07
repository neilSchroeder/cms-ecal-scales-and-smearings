import numba
import numpy as np
import pandas as pd

from src.classes.constant_classes import CategoryConstants as cc
from src.classes.constant_classes import DataConstants as dc
from src.core.data_loader import apply_custom_event_selection


@numba.njit
def transform_smearings(smearings, new_sigma, old_sigma):
    return (smearings - 1) * (new_sigma / old_sigma) + 1


def parse_category_values(cat_str, delim_cat="-", delim_var="_"):
    """Parse category string into numeric arrays"""
    cat_list = cat_str.split(delim_cat)
    eta_vals = np.array([float(x) for x in cat_list[0].split(delim_var)[1:]])
    r9_vals = np.array([float(x) for x in cat_list[1].split(delim_var)[1:]])
    if len(cat_list) > 2:
        et_vals = np.array(
            [
                float(x) if x != "-1" else dc.MAX_ET
                for x in cat_list[2].split(delim_var)[1:]
            ]
        )
    else:
        et_vals = np.array([dc.MIN_ET, dc.MAX_ET])
    return eta_vals, r9_vals, et_vals


def smear(mc, smearings_file) -> pd.DataFrame:
    """
    Applies gaussian smearings to the MC in a double loop approach

    Args:
        mc (pd.DataFrame): the MC dataframe
        smearings_file (str): the file containing the smearings
    Returns:
        pd.DataFrame: the MC dataframe with the smearings applied
    """
    delim_cat = "-"
    delim_var = "_"
    # read in the smearings
    smear_df = pd.read_csv(smearings_file, delimiter="\t", header=None, comment="#")

    rand = np.random.Generator(np.random.PCG64(dc.SEED))
    lead_smearings = rand.normal(1, 0.001, len(mc)).astype(np.float32)
    current_smearing_lead = 0.001
    sublead_smearings = rand.normal(1, 0.001, len(mc)).astype(np.float32)
    current_smearing_sublead = 0.001

    lead_eta = mc[dc.ETA_LEAD].values
    lead_r9 = mc[dc.R9_LEAD].values
    lead_e = mc[dc.E_LEAD].values
    lead_et = np.divide(lead_e, np.cosh(lead_eta))
    sublead_eta = mc[dc.ETA_SUB].values
    sublead_r9 = mc[dc.R9_SUB].values
    sublead_e = mc[dc.E_SUB].values
    sublead_et = np.divide(sublead_e, np.cosh(sublead_eta))

    categories = [parse_category_values(x, delim_cat, delim_var) for x in smear_df[0]]
    smearings = smear_df[3]
    num_rows = len(smear_df)
    for i in range(num_rows):
        # transform the lead smearings
        lead_smearings = transform_smearings(
            lead_smearings, smearings[i], current_smearing_lead
        )
        current_smearing_lead = smearings[i]

        # split cat into parts
        eta_list, r9_list, et_list = categories[i]

        lead_mask = (
            (lead_eta >= eta_list[0])
            & (lead_eta < eta_list[1])
            & (lead_r9 >= r9_list[0])
            & (lead_r9 < r9_list[1])
            & (lead_et >= et_list[0])
            & (lead_et < et_list[1])
        )

        for j in range(num_rows):
            # transform the sublead smearings
            sublead_smearings = transform_smearings(
                sublead_smearings, smearings[j], current_smearing_sublead
            )
            current_smearing_sublead = smearings[j]

            eta_list, r9_list, et_list = categories[j]

            sublead_mask = (
                (sublead_eta >= eta_list[0])
                & (sublead_eta < eta_list[1])
                & (sublead_r9 >= r9_list[0])
                & (sublead_r9 < r9_list[1])
                & (sublead_et >= et_list[0])
                & (sublead_et < et_list[1])
            )

            mask = np.logical_and(lead_mask, sublead_mask)
            mc.loc[mask, dc.E_LEAD] *= lead_smearings[mask]
            mc.loc[mask, dc.E_SUB] *= sublead_smearings[mask]
            mc.loc[mask, dc.INVMASS] *= np.sqrt(
                lead_smearings[mask] * sublead_smearings[mask]
            ) / (1 - current_smearing_lead * current_smearing_sublead / 8)

    return apply_custom_event_selection(
        mc,
        inv_mass_cuts=(80, 100),
        eta_cuts=(0, 1.4442, 1.566, 2.5),
        et_cuts=((32, 14000), (20, 14000)),
    )
