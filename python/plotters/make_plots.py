import pandas as pd
import numpy as np

import python.classes.const_class_pyval as constants
import python.plotters.plots as plots

def get_tuple(this_string):
    this_string = this_string.replace("(","")
    this_string = this_string.replace(")","")
    this_string = this_string.split(",")
    return (float(this_string[0]),float(this_string[1]))

def get_var(df, info, _isData=True):

    #constants
    c = constants.const()

    #unpack the info
    var = info["variable"]
    bounds_eta_lead = get_tuple(info["eta0"])
    bounds_r9_lead = get_tuple(info["r90"])
    bounds_et_lead = get_tuple(info["et0"])
    bounds_eta_sub = get_tuple(info["eta1"])
    bounds_r9_sub = get_tuple(info["r91"])
    bounds_et_sub = get_tuple(info["et1"])

    #determine what variable we're plotting
    var_key = None
    if var not in df.columns:
        #gonna have to do something fancier here
        pass
    else:
        var_key = var

    mask_lead = np.ones(len(df), dtype=bool)
    mask_sub = np.ones(len(df), dtype=bool)
    if bounds_eta_lead[0] != -1 and bounds_eta_lead[1] != -1:
        mask_lead = np.logical_and(mask_lead,
                np.logical_and(bounds_eta_lead[0] <= df[c.ETA_LEAD].values, 
                    df[c.ETA_LEAD].values < bounds_eta_lead[1]
                    )
                )
    if bounds_eta_sub[0] != -1 and bounds_eta_sub[1] != -1:
        mask_sub = np.logical_and(mask_sub,
                np.logical_and(bounds_eta_sub[0] <= df[c.ETA_SUB].values, 
                    df[c.ETA_SUB].values < bounds_eta_sub[1]
                    )
                )
    if bounds_r9_lead[0] != -1 and bounds_r9_lead[1] != -1:
        mask_lead = np.logical_and(mask_lead, 
                np.logical_and(bounds_r9_lead[0] <= df[c.R9_LEAD].values, 
                    df[c.R9_LEAD].values < bounds_r9_lead[1]
                    )
                )
    if bounds_r9_sub[0] != -1 and bounds_r9_sub[1] != -1:
        mask_sub = np.logical_and(mask_sub, 
                np.logical_and(bounds_r9_sub[0] <= df[c.R9_SUB].values, 
                    df[c.R9_SUB].values < bounds_r9_sub[1]
                    )
                )
    if bounds_et_lead[0] != -1 and bounds_et_lead[1] != -1:
       mask_lead = np.logical_and(mask_lead, 
               np.logical_and(
                   bounds_et_lead[0] <= np.divide(df[c.E_LEAD].values, np.cosh(df[c.ETA_LEAD].values)), 
                   np.divide(df[c.E_LEAD].values, np.cosh(df[c.ETA_LEAD].values)) < bounds_et_lead[1]
                   )
               )
    if bounds_et_sub[0] != -1 and bounds_et_sub[1] != -1:
       mask_sub = np.logical_and(mask_sub, 
               np.logical_and(
                   bounds_et_sub[0] <= np.divide(df[c.E_SUB].values, np.cosh(df[c.ETA_SUB].values)), 
                   np.divide(df[c.E_SUB].values, np.cosh(df[c.ETA_SUB].values)) < bounds_et_sub[1]
                   )
               )

    mask = np.logical_and(mask_lead,mask_sub)
    if var_key == c.INVMASS:
        if not _isData:
            print(np.std(df[mask]["invmass_down"].values), np.std(df[mask][var_key].values), np.std(df[mask]["invmass_up"].values))
            h_up, _ = np.histogram(df[mask]["invmass_up"].values, bins=80, range=[80.,100.])
            h, _ = np.histogram(df[mask][var_key].values, bins=80, range=[80.,100.])
            h_down, _ = np.histogram(df[mask]["invmass_down"].values, bins=80, range=[80.,100.])
            print(h_up[-1], h[-1], h_down[-1])
            return [np.array(df[mask][var_key].values),
                    np.array(df[mask]["invmass_up"].values),
                    np.array(df[mask]["invmass_down"].values),
                    np.array(df[mask]['pty_weight'].values)]
        return [np.array(df[mask][var_key].values),
            np.array(df[mask]["invmass_up"].values),
            np.array(df[mask]["invmass_down"].values)]
    return np.array(df[mask][var_key].values)

def plot(data, mc, cats, **options):

    #get constants
    c = constants.const()
    #open cats
    df_cats = pd.read_csv(cats, sep='\t', comment='#')

    #loop over cats
    for i,row in df_cats.iterrows():
        c.get_plotting_function(row[c.i_plot_style])( #gets the plotting function
            get_var(data, row[c.i_plot_var::]), #gets events from data to plot
            get_var(mc, row[c.i_plot_var::], False), #gets events from mc to plot
            row[c.i_plot_name], #plot title
            **options,
            syst=True if row[c.i_plot_var] == c.INVMASS else False,
            )

    return
