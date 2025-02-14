"""Utilities for loading root files into dataframes and manipulating dataframes."""

import numpy as np
import pandas as pd
import uproot3 as up

from python.classes.constant_classes import DataConstants as dc, CategoryConstants as cc
from python.classes.zcat_class import zcat


def get_dataframe(
    files: list[str],
    apply_cuts: str = "standard",
    eta_cuts: tuple = None,
    inv_mass_cuts: tuple = None,
    et_cuts: tuple = None,
    r9_cuts: tuple = None,
    working_point: tuple = None,
    debug: bool = False,
    nrows: int = 1_000_000,
):
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
        # read from root files
        # this takes a long time, so avoid it if possible
        df = pd.concat(
            [up.open(f)[dc.TREE_NAME].pandas.df(dc.KEEP_COLS) for f in files]
        )
        # drop unnecessary columns
        df.drop(dc.DROP_LIST, axis=1, inplace=True)
    elif ".csv" in files[0] or ".tsv" in files[0]:
        # read from csv or tsv files
        rows = nrows if debug else None
        df = pd.concat([pd.read_csv(f, sep="\t", dtype=dc.DATA_TYPES, nrows=rows) for f in files])
    else:
        print("[python][helpers][helper_main] ERROR: file type not recognized")
        raise ValueError("file type not recognized: must be .root, .csv, or .tsv")

    if apply_cuts == "standard":
        df = standard_cuts(df)
    else:
        df = custom_cuts(df, eta_cuts, inv_mass_cuts, et_cuts, r9_cuts, working_point)

    return df


def standard_cuts(df):
    """
    Takes in a dataframe and applies the following cuts:
    pt_lead > 32 GeV
    pt_SUB > 20 GeV
    80 GeV < invMass < 100 GeV
    |eta| < 2.5 and !(1.4442 < |eta| < 1.566)
    """
    # masks
    print(f"number of events before cuts: {len(df)}")
    mask = np.ones(len(df), dtype=bool)
    df[dc.ETA_LEAD] = np.abs(df[dc.ETA_LEAD])
    df[dc.ETA_SUB] = np.abs(df[dc.ETA_SUB])
    transition_mask_lead = ~df[dc.ETA_LEAD].between(dc.MAX_EB, dc.MIN_EE)
    transition_mask_sub = ~df[dc.ETA_SUB].between(dc.MAX_EB, dc.MIN_EE)
    tracker_mask_lead = ~df[dc.ETA_LEAD].between(dc.MAX_EE, dc.TRACK_MAX)
    tracker_mask_sub = ~df[dc.ETA_SUB].between(dc.MAX_EE, dc.TRACK_MAX)
    invmass_mask = df[dc.INVMASS].between(dc.invmass_min, dc.invmass_max)

    mask = (
        transition_mask_lead
        & transition_mask_sub
        & tracker_mask_lead
        & tracker_mask_sub
        & invmass_mask
    )
    print(f"number of events after cuts: {sum(mask)}")

    return df[mask]


def custom_cuts(
    df: pd.DataFrame,
    eta_cuts: tuple = None,
    inv_mass_cuts: tuple = None,
    et_cuts: tuple = None,
    r9_cuts: tuple = None,
    working_point: tuple = None,
    **kwargs,
):
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
        df[dc.ETA_LEAD] = np.abs(df[dc.ETA_LEAD])
        df[dc.ETA_SUB] = np.abs(df[dc.ETA_SUB])
        if isinstance(eta_cuts[0], tuple):
            # this means cuts on both leading and SUBing electrons
            if eta_cuts[0][0] != -1:
                mask &= df[dc.ETA_LEAD] >= eta_cuts[0][0]
            if eta_cuts[0][1] != -1:
                mask &= df[dc.ETA_LEAD] <= eta_cuts[0][1]
            if eta_cuts[1][0] != -1:
                mask &= df[dc.ETA_SUB] >= eta_cuts[1][0]
            if eta_cuts[1][1] != -1:
                mask &= df[dc.ETA_SUB] <= eta_cuts[1][1]
        else:
            # this means one set of cuts for both leading and SUBing electrons
            mask = mask & (
                (df[dc.ETA_LEAD] > eta_cuts[0]) & (df[dc.ETA_LEAD] < eta_cuts[1])
                | (df[dc.ETA_LEAD] > eta_cuts[2]) & (df[dc.ETA_LEAD] < eta_cuts[3])
            )
            mask = mask & (
                (df[dc.ETA_SUB] > eta_cuts[0]) & (df[dc.ETA_SUB] < eta_cuts[1])
                | (df[dc.ETA_SUB] > eta_cuts[2]) & (df[dc.ETA_SUB] < eta_cuts[3])
            )

    if inv_mass_cuts:
        mask = (
            mask
            & (df[dc.INVMASS] > inv_mass_cuts[0])
            & (df[dc.INVMASS] < inv_mass_cuts[1])
        )

    if et_cuts:
        et_lead = np.divide(df[dc.E_LEAD].values, np.cosh(df[dc.ETA_LEAD].values))
        et_sub = np.divide(df[dc.E_SUB].values, np.cosh(df[dc.ETA_SUB].values))
        df["et_lead"] = et_lead
        df["et_sub"] = et_sub
        if isinstance(et_cuts[0], tuple):
            # this means cuts on both leading and subleading electrons
            if et_cuts[0][0] != -1:
                mask &= et_lead > et_cuts[0][0]
            if et_cuts[0][1] != -1:
                mask &= et_lead < et_cuts[0][1]
            if et_cuts[1][0] != -1:
                mask &= et_sub > et_cuts[1][0]
            if et_cuts[1][1] != -1:
                mask &= et_sub < et_cuts[1][1]
        else:
            # otherwise you're just providing minimum values for both electrons
            n_events = sum(mask)
            print(n_events - sum(mask & (et_lead > et_cuts[0]) & (et_sub > et_cuts[1])))
            mask = mask & (et_lead > et_cuts[0]) & (et_sub > et_cuts[1])

    if r9_cuts:
        if isinstance(r9_cuts[0], tuple):
            # this means cuts on both leading and SUBing electrons
            if r9_cuts[0][0] != -1:
                mask &= df[dc.R9_LEAD] > r9_cuts[0][0]
            if r9_cuts[0][1] != -1:
                mask &= df[dc.R9_LEAD] < r9_cuts[0][1]
            if r9_cuts[1][0] != -1:
                mask &= df[dc.R9_SUB] > r9_cuts[1][0]
            if r9_cuts[1][1] != -1:
                mask &= df[dc.R9_SUB] < r9_cuts[1][1]

        else:
            # otherwise you're just providing a minimum value for both electrons
            mask = mask & (df[dc.R9_LEAD] > r9_cuts[0]) & (df[dc.R9_SUB] > r9_cuts[1])

    if working_point is not None:
        wp_id = dc.TIGHT_ID if working_point == "tight" else dc.MEDIUM_ID
        id_lead_mask = np.array([(x & wp_id) == wp_id for x in df[dc.ID_LEAD].values])
        id_sub_mask = np.array([(x & wp_id) == wp_id for x in df[dc.ID_SUB].values])
        mask = mask & id_lead_mask & id_sub_mask

    return df[mask]


def add_transverse_energy(
    data: pd.DataFrame, mc: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds a transverse energy column to the data and mc dataframes.

    Args:
        data (pandas.DataFrame): data dataframe
        mc (pandas.DataFrame): mc dataframe
    Returns:
        data (pandas.DataFrame): data dataframe with transverse energy column
        mc (pandas.DataFrame): mc dataframe with transverse energy column
    """
    energy_0 = np.array(data[dc.E_LEAD].values)
    energy_1 = np.array(data[dc.E_SUB].values)
    eta_0 = np.array(data[dc.ETA_LEAD].values)
    eta_1 = np.array(data[dc.ETA_SUB].values)
    data[dc.ET_LEAD] = np.divide(energy_0, np.cosh(eta_0))
    data[dc.ET_SUB] = np.divide(energy_1, np.cosh(eta_1))
    energy_0 = np.array(mc[dc.E_LEAD].values)
    energy_1 = np.array(mc[dc.E_SUB].values)
    eta_0 = np.array(mc[dc.ETA_LEAD].values)
    eta_1 = np.array(mc[dc.ETA_SUB].values)
    mc[dc.ET_LEAD] = np.divide(energy_0, np.cosh(eta_0))
    mc[dc.ET_SUB] = np.divide(energy_1, np.cosh(eta_1))

    #  impose an et cut of 32 on leading and 20 on subleading
    mask_lead = data[dc.ET_LEAD].between(dc.MIN_ET_LEAD, dc.MAX_ET_LEAD) & data[
        dc.ET_SUB
    ].between(dc.MIN_ET_SUB, dc.MAX_ET_SUB)
    data = data[mask_lead]
    mask_lead = mc[dc.ET_LEAD].between(dc.MIN_ET_LEAD, dc.MAX_ET_LEAD) & mc[
        dc.ET_SUB
    ].between(dc.MIN_ET_SUB, dc.MAX_ET_SUB)
    mc = mc[mask_lead]
    return data, mc


def extract_cats(
    data: pd.DataFrame, mc: pd.DataFrame, cats_df: pd.DataFrame, **options
) -> list[zcat]:
    """
    Extract the dielectron categories from the data and mc dataframes.

    This is done by querying the dataframe from events satisfying the category
    cuts.

    Args:
        data (pandas.DataFrame): data dataframe
        mc (pandas.DataFrame): mc dataframe
        cats_df (pandas.DataFrame): dataframe containing the categories
        **options: keyword arguments, which contain the following:
            num_scales (int): number of scales to be derived
            num_smears (int): number of smearings to be derived

    Returns:
        categories (list): list of zcat objects, each representing a dielectron category
    """
    # check for empty data
    if len(data) == 0:
        print("[WARNING][python/utilities/data_loader] no data, returning")
        return []
    if len(mc) == 0:
        print("[WARNING][python/utilities/data_loader] no mc, returning")
        return []

    categories = []
    for index1 in range(options["num_scales"]):
        for index2 in range(index1 + 1):
            cat1 = cats_df.iloc[index1]
            cat2 = cats_df.iloc[index2]
            # thisCat should have the form: type etaMin etaMax r9Min r9Max gain etMin etMax

            eta_mask = np.ones(len(data), dtype=bool)
            if cat1[cc.i_eta_min] != cc.empty:
                eta_mask = data[dc.ETA_LEAD].between(
                    cat1[cc.i_eta_min], cat1[cc.i_eta_max]
                ) & data[dc.ETA_SUB].between(cat2[cc.i_eta_min], cat2[cc.i_eta_max])
                eta_mask = eta_mask | (
                    data[dc.ETA_SUB].between(cat1[cc.i_eta_min], cat1[cc.i_eta_max])
                    & data[dc.ETA_LEAD].between(cat2[cc.i_eta_min], cat2[cc.i_eta_max])
                )

            r9_mask = np.ones(len(data), dtype=bool)
            if cat1[cc.i_r9_min] != cc.empty:
                r9_mask = data[dc.R9_LEAD].between(
                    cat1[cc.i_r9_min], cat1[cc.i_r9_max]
                ) & data[dc.R9_SUB].between(cat2[cc.i_r9_min], cat2[cc.i_r9_max])
                r9_mask = r9_mask | (
                    data[dc.R9_SUB].between(cat1[cc.i_r9_min], cat1[cc.i_r9_max])
                    & data[dc.R9_LEAD].between(cat2[cc.i_r9_min], cat2[cc.i_r9_max])
                )

            et_mask = np.ones(len(data), dtype=bool)
            if cat1[cc.i_et_min] != cc.empty:
                et_mask = data[dc.ET_LEAD].between(
                    cat1[cc.i_et_min], cat1[cc.i_et_max]
                ) & data[dc.ET_SUB].between(cat2[cc.i_et_min], cat2[cc.i_et_max])
                et_mask = et_mask | (
                    data[dc.ET_SUB].between(cat1[cc.i_et_min], cat1[cc.i_et_max])
                    & data[dc.ET_LEAD].between(cat2[cc.i_et_min], cat2[cc.i_et_max])
                )

            # gain mask should be all true if gain is not specified
            gain_mask = np.ones(len(data), dtype=bool)
            if cat1[cc.i_gain] != cc.empty:
                # possible gain values are 12, 6, 1
                # if gain is 12, then gainSeedSC is 0
                # if gain is 6, then gainSeedSC is 1
                # if gain is 1, then gainSeedSC is greater than 1
                gainlow1, gainhigh1, gainlow2, gainhigh2 = 0, 0, 0, 0
                if cat1[cc.i_gain] == 6:
                    gainlow1, gainhigh1 = 1, 1
                if cat1[cc.i_gain] == 1:
                    gainlow1, gainhigh1 = 2, 99999
                if cat2[cc.i_gain] == 6:
                    gainlow2, gainhigh2 = 1, 1
                if cat2[cc.i_gain] == 1:
                    gainlow2, gainhigh2 = 2, 99999
                gain_mask = data[dc.GAIN_LEAD].between(gainlow1, gainhigh1) & data[
                    dc.GAIN_SUB
                ].between(gainlow2, gainhigh2)
                gain_mask = gain_mask | (
                    data[dc.GAIN_SUB].between(gainlow1, gainhigh1)
                    & data[dc.GAIN_LEAD].between(gainlow2, gainhigh2)
                )

            df = data[eta_mask & r9_mask & et_mask & gain_mask]
            mass_list_data = np.array(df[dc.INVMASS])

            eta_mask = np.ones(len(mc), dtype=bool)
            if cat1[cc.i_eta_min] != cc.empty:
                eta_mask = mc[dc.ETA_LEAD].between(
                    cat1[cc.i_eta_min], cat1[cc.i_eta_max]
                ) & mc[dc.ETA_SUB].between(cat2[cc.i_eta_min], cat2[cc.i_eta_max])
                eta_mask = eta_mask | (
                    mc[dc.ETA_SUB].between(cat1[cc.i_eta_min], cat1[cc.i_eta_max])
                    & mc[dc.ETA_LEAD].between(cat2[cc.i_eta_min], cat2[cc.i_eta_max])
                )

            r9_mask = np.ones(len(mc), dtype=bool)
            if cat1[cc.i_r9_min] != cc.empty:
                r9_mask = mc[dc.R9_LEAD].between(
                    cat1[cc.i_r9_min], cat1[cc.i_r9_max]
                ) & mc[dc.R9_SUB].between(cat2[cc.i_r9_min], cat2[cc.i_r9_max])
                r9_mask = r9_mask | (
                    mc[dc.R9_SUB].between(cat1[cc.i_r9_min], cat1[cc.i_r9_max])
                    & mc[dc.R9_LEAD].between(cat2[cc.i_r9_min], cat2[cc.i_r9_max])
                )

            et_mask = np.ones(len(mc), dtype=bool)
            if cat1[cc.i_et_min] != cc.empty:
                et_mask = mc[dc.ET_LEAD].between(
                    cat1[cc.i_et_min], cat1[cc.i_et_max]
                ) & mc[dc.ET_SUB].between(cat2[cc.i_et_min], cat2[cc.i_et_max])
                et_mask = et_mask | (
                    mc[dc.ET_SUB].between(cat1[cc.i_et_min], cat1[cc.i_et_max])
                    & mc[dc.ET_LEAD].between(cat2[cc.i_et_min], cat2[cc.i_et_max])
                )

            gain_mask = np.ones(len(mc), dtype=bool)
            if cat1[cc.i_gain] != cc.empty:
                gainlow1, gainhigh1, gainlow2, gainhigh2 = 0, 0, 0, 0
                if cat1[cc.i_gain] == 6:
                    gainlow1, gainhigh1 = 1, 1
                if cat1[cc.i_gain] == 1:
                    gainlow1, gainhigh1 = 2, 99999
                if cat2[cc.i_gain] == 6:
                    gainlow2, gainhigh2 = 1, 1
                if cat2[cc.i_gain] == 1:
                    gainlow2, gainhigh2 = 2, 99999
                gain_mask = mc[dc.GAIN_LEAD].between(gainlow1, gainhigh1) & mc[
                    dc.GAIN_SUB
                ].between(gainlow2, gainhigh2)
                gain_mask = gain_mask | (
                    mc[dc.GAIN_SUB].between(gainlow1, gainhigh1)
                    & mc[dc.GAIN_LEAD].between(gainlow2, gainhigh2)
                )

            df = mc[eta_mask & r9_mask & et_mask & gain_mask]
            mass_list_mc = np.array(df[dc.INVMASS].values, dtype=np.float32)
            weight_list_mc = (
                np.array(df["pty_weight"].values, dtype=np.float32)
                if "pty_weight" in df.columns
                else np.ones(len(mass_list_mc))
            )
            # MC needs to be over smeared in order to have good "resolution" on the scales and smearings
            while (
                len(mass_list_mc) < max(50 * len(mass_list_data), 50000)
                and len(mass_list_mc) > 100
                and len(mass_list_data) > 10
                and len(mass_list_mc) < 1000000
            ):
                mass_list_mc = np.append(mass_list_mc, mass_list_mc)
                weight_list_mc = np.append(weight_list_mc, weight_list_mc)

            # drop any "bad" entries
            mass_list_data = mass_list_data[~np.isnan(mass_list_data)]
            weight_list_mc = weight_list_mc[~np.isnan(mass_list_mc)]
            mass_list_mc = mass_list_mc[~np.isnan(mass_list_mc)]

            if options["num_smears"] > 0:
                categories.append(
                    zcat(
                        index1,
                        index2,
                        mass_list_data.copy(),
                        mass_list_mc.copy(),
                        weight_list_mc.copy(),
                        smear_i=get_smearing_index(cats_df, index1),
                        smear_j=get_smearing_index(cats_df, index2),
                        **options,
                    )
                )
            else:
                categories.append(
                    zcat(
                        index1,
                        index2,  # no smearing categories, so no smearing indices
                        mass_list_data.copy(),
                        mass_list_mc.copy(),
                        weight_list_mc.copy(),
                        **options,
                    )
                )

    return categories


def get_smearing_index(cats: pd.DataFrame, cat_index: int) -> int:
    """
    Return the index of the smearing category that corresponds to the given category index

    Args:
        cats (pandas.DataFrame): dataframe containing the categories
        cat_index (int): index of the category
    Returns:
        (int): index of the smearing category that corresponds to the given category index
    """

    eta_min = cats.iloc[int(cat_index), cc.i_eta_min]
    eta_max = cats.iloc[int(cat_index), cc.i_eta_max]
    r9_min = cats.iloc[int(cat_index), cc.i_r9_min]
    r9_max = cats.iloc[int(cat_index), cc.i_r9_max]
    et_min = cats.iloc[int(cat_index), cc.i_et_min]
    et_max = cats.iloc[int(cat_index), cc.i_et_max]

    truth_type = cats.loc[:, cc.i_type] == "smear"
    truth_eta_min = np.array([True for x in truth_type])
    truth_eta_max = np.array([True for x in truth_type])
    truth_r9_min = np.array([True for x in truth_type])
    truth_r9_max = np.array([True for x in truth_type])
    truth_et_min = np.array([True for x in truth_type])
    truth_et_max = np.array([True for x in truth_type])
    if eta_min != cc.empty and eta_max != cc.empty:
        truth_eta_min = cats.loc[:, cc.i_eta_min] <= eta_min
        truth_eta_max = cats.loc[:, cc.i_eta_max] >= eta_max
    if r9_min != cc.empty and r9_max != -1:
        truth_r9_min = cats.loc[:, cc.i_r9_min] <= r9_min
        truth_r9_max = cats.loc[:, cc.i_r9_max] >= r9_max
    if et_min != cc.empty and et_max != cc.empty:
        truth_et_min = cats.loc[:, cc.i_et_min] <= et_min
        truth_et_max = cats.loc[:, cc.i_et_max] >= et_max

    truth = (
        truth_type
        & truth_eta_min
        & truth_eta_max
        & truth_r9_min
        & truth_r9_max
        & truth_et_min
        & truth_et_max
    )
    return cats.loc[truth].index[0]


def clean_up(data, mc, cats):
    """
    Clean up dataframes, add transverse energy if necessary, and drop unnecessary columns.

    Args:
        data (pandas.DataFrame): data dataframe
        mc (pandas.DataFrame): mc dataframe
        cats (pandas.DataFrame): dataframe containing the categories
    Returns:
        data (pandas.DataFrame): cleaned data dataframe
        mc (pandas.DataFrame): cleaned mc dataframe
    """
    if cats.iloc[0, cc.i_et_min] != cc.empty:
        data, mc = add_transverse_energy(data, mc)

    drop_list = [dc.E_LEAD, dc.E_SUB, dc.RUN]
    if cats.iloc[0, cc.i_gain] == cc.empty:
        drop_list += [dc.GAIN_LEAD, dc.GAIN_SUB]

    print("[INFO][python/nll] dropping {}".format(drop_list))
    data.drop(drop_list, axis=1, inplace=True)
    mc.drop(drop_list, axis=1, inplace=True)

    return data, mc
