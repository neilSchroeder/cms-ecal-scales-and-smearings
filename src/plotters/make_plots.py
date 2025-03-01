import numpy as np
import pandas as pd
import seaborn as sns

from src.classes.config_class import SSConfig
from src.classes.constant_classes import DataConstants as dc
from src.classes.constant_classes import PlottingConstants as pc
from src.classes.constant_classes import PyValConstants as pvc
from src.plotters.plots import (
    plot_style_bw_cb_fit,
    plot_style_paper,
    plot_style_validation_mc,
)
from src.tools.data_loader import apply_custom_event_selection

config = SSConfig()
pc.plotting_functions = {
    "crossCheckMC": plot_style_validation_mc,
    "fit": plot_style_bw_cb_fit,
    "paper": plot_style_paper,
}


def convert_str_to_tuple(this_string):
    """
    converts a string of the form "(a,b)" to a tuple (a,b)
    ----------
    Args:
        this_string: string to convert
    ----------
    Returns:
        tuple: tuple of the form (a,b)
    ----------
    """
    this_string = this_string.replace("(", "")
    this_string = this_string.replace(")", "")
    this_string = this_string.split(",")
    return (float(this_string[0]), float(this_string[1]))


def get_array_with_cuts(df, info, _isData=True, title=None):
    """
    Gets the variable to plot from the dataframe
    ----------
    Args:
        df: dataframe to get the variable from
        info: dictionary of the form {"variable":var, "eta0":eta0, "r90":r90, "et0":et0, "eta1":eta1, "r91":r91, "et1":et1}
        _isData: boolean to determine if we're plotting data or MC
    ----------
    Returns:
        var: variable to plot
    ----------
    """

    # unpack the info
    bounds_eta_lead = convert_str_to_tuple(info[pvc.ETA_LEAD])
    bounds_r9_lead = convert_str_to_tuple(info[pvc.R9_LEAD])
    bounds_et_lead = convert_str_to_tuple(info[pvc.ET_LEAD])
    bounds_eta_sub = convert_str_to_tuple(info[pvc.ETA_SUB])
    bounds_r9_sub = convert_str_to_tuple(info[pvc.R9_SUB])
    bounds_et_sub = convert_str_to_tuple(info[pvc.ET_SUB])

    # determine what variable we're plotting
    var_key = None
    if info["variable"] not in df.columns:
        # gonna have to do something fancier here
        raise ValueError("variable not in dataframe")
    else:
        var_key = info["variable"]

    # apply the cuts
    print(
        f"Lead eta: {bounds_eta_lead}, Lead r9: {bounds_r9_lead}, Lead et: {bounds_et_lead}"
    )
    print(
        f"Sub eta: {bounds_eta_sub}, Sub r9: {bounds_r9_sub}, Sub et: {bounds_et_sub}"
    )
    df_with_cuts = apply_custom_event_selection(
        df,
        inv_mass_cuts=(80, 100),
        eta_cuts=(bounds_eta_lead, bounds_eta_sub),
        et_cuts=(bounds_et_lead, bounds_et_sub),
        r9_cuts=(bounds_r9_lead, bounds_r9_sub),
    )

    if var_key == dc.INVMASS:
        # check if systematics available

        if _isData:
            df_title = "_" + title if title else ""
            if (
                pvc.KEY_INVMASS_UP not in df.columns
                or pvc.KEY_INVMASS_DOWN not in df.columns
            ):
                # there's no systematics, so just return the data
                return np.array(df[var_key].values)
            # return the data
            return [
                np.array(df_with_cuts[var_key].values),
                np.array(df_with_cuts[pvc.KEY_INVMASS_UP].values),
                np.array(df_with_cuts[pvc.KEY_INVMASS_DOWN].values),
            ]
        else:
            df_title = "_" + title if title else ""
        # otherwise return mc
        return [
            np.array(df_with_cuts[var_key].values),
            np.array(df_with_cuts[pvc.KEY_PTY].values),
        ]

    return np.array(df_with_cuts[var_key].values)


def plot(data, mc, cats, **options):
    """
    Plots the data and mc in the categories specified in cats.

    Args:
        data (pd.DataFrame): dataframe containing the data
        mc (pd.DataFrame): dataframe containing the mc
        cats (str): path to the file containing the categories
        **options: options to pass to the plotting function
    Returns:
        None
    """

    # get constants
    df_cats = pd.read_csv(cats, sep="\t", comment="#")
    plot_consts = pc()
    # copy categories into a new dataframe
    df_results = df_cats.copy()
    # add a column for the results
    df_results[pvc.i_plot_results] = None

    # loop over cats
    results = []
    for i, row in df_cats.iterrows():
        print(row[pvc.i_plot_name])
        # determine what plot style and plotting function to use
        plotting_class = plot_consts.get_plotting_function(row[pvc.i_plot_style])
        # run the plotting function
        val = plotting_class(
            get_array_with_cuts(
                data, row[pvc.i_plot_var : :], title=row[pvc.i_plot_name]
            ),  # gets events from data to plot
            get_array_with_cuts(
                mc, row[pvc.i_plot_var : :], _isData=False, title=row[pvc.i_plot_name]
            ),  # gets events from mc to plot
            row[pvc.i_plot_name],  # plot title
            **options,
            syst=(
                row[pvc.i_plot_var] == dc.INVMASS
                and "invmass_up" in data.columns
                and "invmass_down" in data.columns
            ),  # if we're plotting the invariant mass, we need to plot the systematic uncertainty,
            # no_ratio=True
        )

        df_results.loc[i, pvc.i_plot_results] = val

    # save the results
    print(
        f"Saving plot results to {config.DEFAULT_WRITE_FILES_PATH}/{options['tag']}_plot_results.csv"
    )
    df_results.to_csv(
        f"{config.DEFAULT_WRITE_FILES_PATH}/{options['tag']}_plot_results.csv",
        sep="\t",
        index=False,
    )

    return
