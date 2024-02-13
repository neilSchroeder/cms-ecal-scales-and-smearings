import pandas as pd
import numpy as np

from python.classes.constant_classes import PyValConstants as pvc
from python.classes.constant_classes import DataConstants as dc
import python.plotters.plots as plots
from python.utilities.data_loader import custom_cuts

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
    bounds_eta_lead = get_tuple(info[pvc.ETA_LEAD])
    bounds_r9_lead = get_tuple(info[pvc.R9_LEAD])
    bounds_et_lead = get_tuple(info[pvc.ET_LEAD])
    bounds_eta_sub = get_tuple(info[pvc.ETA_SUB])
    bounds_r9_sub = get_tuple(info[pvc.R9_SUB])
    bounds_et_sub = get_tuple(info[pvc.ET_SUB])

     # determine what variable we're plotting
    var_key = None
    if info["variable"] not in df.columns:
         # gonna have to do something fancier here
        raise ValueError("variable not in dataframe")
    else:
        var_key = info["variable"]

    df_with_cuts = custom_cuts(df,
                     eta_cuts=(bounds_eta_lead, bounds_eta_sub),
                     et_cuts=(bounds_et_lead, bounds_et_sub),
                     r9_cuts=(bounds_r9_lead, bounds_r9_sub))

    if var_key == dc.INVMASS:
        # check if systematics available
        
        if _isData:
            if pvc.KEY_INVMASS_UP not in df.columns or pvc.KEY_INVMASS_DOWN not in df.columns:
                # there's no systematics, so just return the data
                return np.array(df[var_key].values)
            # return the data
            return [np.array(df_with_cuts[var_key].values),
                    np.array(df_with_cuts[pvc.KEY_INVMASS_UP].values),
                    np.array(df_with_cuts[pvc.KEY_INVMASS_DOWN].values)]

        # otherwise return mc
        return [np.array(df_with_cuts[var_key].values),
                np.array(df_with_cuts[pvc.KEY_PTY].values)]
        
    
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
    df_cats = pd.read_csv(cats, sep='\t', comment='#')

    # loop over cats
    plotting_class = pvc()
    for i,row in df_cats.iterrows():
        print(row[pvc.i_plot_name])
        plotting_class.get_plotting_function(row[pvc.i_plot_style])(  # gets the plotting function
            get_var(data, row[pvc.i_plot_var::]),  # gets events from data to plot
            get_var(mc, row[pvc.i_plot_var::], _isData=False),  # gets events from mc to plot
            row[pvc.i_plot_name],  # plot title
            **options,
            syst= (row[pvc.i_plot_var] == dc.INVMASS and 'invmass_up' in data.columns and 'invmass_down' in data.columns),  # if we're plotting the invariant mass, we need to plot the systematic uncertainty,
            #no_ratio=True
            )

    return
