import pandas as pd
import numpy as np

from python.classes.constant_classes import PyValConstants as pvc
from python.classes.constant_classes import DataConstants as dc
import python.plotters.plots as plots
pvc.plotting_functions = {
        'paper': plots.plot_style_paper,
        'crossCheckMC': plots.plot_style_validation_mc
    }

def get_tuple(this_string):
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
    this_string = this_string.replace("(","")
    this_string = this_string.replace(")","")
    this_string = this_string.split(",")
    return (float(this_string[0]),float(this_string[1]))

def get_var(df, info, _isData=True):
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
    var = info["variable"]
    bounds_eta_lead = get_tuple(info[pvc.ETA_LEAD])
    bounds_r9_lead = get_tuple(info[pvc.R9_LEAD])
    bounds_et_lead = get_tuple(info[pvc.ET_LEAD])
    bounds_eta_sub = get_tuple(info[pvc.ETA_SUB])
    bounds_r9_sub = get_tuple(info[pvc.R9_SUB])
    bounds_et_sub = get_tuple(info[pvc.ET_SUB])

     # determine what variable we're plotting
    var_key = None
    if var not in df.columns:
         # gonna have to do something fancier here
        pass
    else:
        var_key = var

    mask_lead = np.ones(len(df), dtype=bool)
    mask_sub = np.ones(len(df), dtype=bool)
    if bounds_eta_lead[0] != -1 and bounds_eta_lead[1] != -1:
        mask_lead = np.logical_and(mask_lead,
                np.logical_and(bounds_eta_lead[0] <= df[dc.ETA_LEAD].values, 
                    df[dc.ETA_LEAD].values < bounds_eta_lead[1]
                    )
                )
    if bounds_eta_sub[0] != -1 and bounds_eta_sub[1] != -1:
        mask_sub = np.logical_and(mask_sub,
                np.logical_and(bounds_eta_sub[0] <= df[dc.ETA_SUB].values, 
                    df[dc.ETA_SUB].values < bounds_eta_sub[1]
                    )
                )
    if bounds_r9_lead[0] != -1 and bounds_r9_lead[1] != -1:
        mask_lead = np.logical_and(mask_lead, 
                np.logical_and(bounds_r9_lead[0] <= df[dc.R9_LEAD].values, 
                    df[dc.R9_LEAD].values < bounds_r9_lead[1]
                    )
                )
    if bounds_r9_sub[0] != -1 and bounds_r9_sub[1] != -1:
        mask_sub = np.logical_and(mask_sub, 
                np.logical_and(bounds_r9_sub[0] <= df[dc.R9_SUB].values, 
                    df[dc.R9_SUB].values < bounds_r9_sub[1]
                    )
                )
    if bounds_et_lead[0] != -1 and bounds_et_lead[1] != -1:
       mask_lead = np.logical_and(mask_lead, 
               np.logical_and(
                   bounds_et_lead[0] <= np.divide(df[dc.E_LEAD].values, np.cosh(df[dc.ETA_LEAD].values)), 
                   np.divide(df[dc.E_LEAD].values, np.cosh(df[dc.ETA_LEAD].values)) < bounds_et_lead[1]
                   )
               )
    if bounds_et_sub[0] != -1 and bounds_et_sub[1] != -1:
       mask_sub = np.logical_and(mask_sub, 
               np.logical_and(
                   bounds_et_sub[0] <= np.divide(df[dc.E_SUB].values, np.cosh(df[dc.ETA_SUB].values)), 
                   np.divide(df[dc.E_SUB].values, np.cosh(df[dc.ETA_SUB].values)) < bounds_et_sub[1]
                   )
               )

    mask = np.logical_and(mask_lead,mask_sub)
    df_new = df[mask]

    if var_key == dc.INVMASS:
        if not _isData:
            return [np.array(df[mask][var_key].values),
                    np.array(df[mask][pvc.KEY_INVMASS_UP].values),
                    np.array(df[mask][pvc.KEY_INVMASS_DOWN].values),
                    np.array(df[mask][pvc.KEY_PTY].values)]
        return [np.array(df[mask][var_key].values),
            np.array(df[mask][pvc.KEY_INVMASS_UP].values),
            np.array(df[mask][pvc.KEY_INVMASS_DOWN].values)]
    return np.array(df[mask][var_key].values)

def plot(data, mc, cats, **options):
    """
    Plots the data and mc in the categories specified in cats
    ----------
    Args:
        data: dataframe of the data
        mc: dataframe of the mc
        cats: dataframe of the categories
        **options: options to pass to the plotting functions
    ----------
    Returns:
        None
    ----------
    """

    # get constants
    df_cats = pd.read_csv(cats, sep='\t', comment='#')

    # loop over cats
    for i,row in df_cats.iterrows():
        print(row[pvc.i_plot_name])
        pvc.get_plotting_function(row[pvc.i_plot_style])(  # gets the plotting function
            get_var(data, row[pvc.i_plot_var::]),  # gets events from data to plot
            get_var(mc, row[pvc.i_plot_var::], False),  # gets events from mc to plot
            row[pvc.i_plot_name],  # plot title
            **options,
            syst=True if row[pvc.i_plot_var] == dc.INVMASS else False,
            )

    return
