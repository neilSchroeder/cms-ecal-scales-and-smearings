import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
"""
This function will plot histograms of data and mc in the categories provided
"""

def plot_cats(zcats, cats):
    
    bins = np.arange(60, 120, 0.5)

    for cat in zcats:
        if cat.valid:
            print("[INFO][python/plot_masses][plot_cats] Plotting category with indices ({}, {})".format(cat.lead_index,cat.sublead_index))
            fig, axs = plt.subplots(nrows = 1, ncols = 1)
            axs.set_xlim(80,100)
            eta_label = "{} $< |\eta| <$ {} $\oplus$ {} $< |\eta| <$ {}".format(round(cats.iloc[cat.lead_index,1],4), round(cats.iloc[cat.lead_index,2],4), round(cats.iloc[cat.sublead_index,1],4), round(cats.iloc[cat.sublead_index,2],4))
            r9_label = "{} $< R_9 <$ {} $\oplus$ {} $< R_9 <$ {}".format(cats.iloc[cat.lead_index, 3], cats.iloc[cat.lead_index,4], cats.iloc[cat.sublead_index,3], cats.iloc[cat.sublead_index,4]) if cats.iloc[cat.lead_index,3] != -1 else ""
            et_label = "{} $< E_T <$ {} $\oplus$ {} $< E_T <$ {}".format(cats.iloc[cat.lead_index,6], cats.iloc[cat.lead_index,7], cats.iloc[cat.sublead_index,6], cats.iloc[cat.sublead_index,7]) if cats.iloc[cat.lead_index,6] != -1 else ""
            dy, dx, _ = axs.hist(data[cat.lead_index][cat.sublead_index], bins, alpha=0.5, label="Data")
            my, mx = np.histogram(mc[cat.lead_index][cat.sublead_index], bins=bins)
            weight = np.sum(dy)*my/np.sum(my)
            weight[weight == np.inf] = 0
            weight[weight ==-np.inf] = 0
            my, mx, _ = axs.hist(mx[:-1], bins, weights=weight, alpha=0.5, label="MC")
            axs.set_ylim(0, max(dy.max(),my.max())*1.75)
            axs.plot([],[], '', label=eta_label)
            axs.plot([],[], '', label=str(r9_label+et_label))
            axs.set_ylabel("Events / 0.5 GeV")
            axs.set_xlabel("M$_{ee}$ [GeV]")
            fig.savefig(plot_dir+"invMass_{}_{}.png".format(cat.lead_index, cat.sublead_index))
            plt.close(fig)

