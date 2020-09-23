import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
"""
This function will plot histograms of data and mc in the categories provided
"""

def plot_cats(data, mc, cats):
    
    bins = np.arange(60, 120, 0.5)
        
    for index1 in range(len(data)):
        for index2 in range(index1+1):
            test_events_mc, test_x_mc = np.histogram(mc[index1][index2], bins=np.arange(80,120,0.25))
            valid_cat_mc = np.sum(test_events_mc) >= 1000 and index1 == index2
            valid_cat_mc = valid_cat_mc or ( np.sum(test_events_mc) >= 2000 and index1 != index2 )
            if valid_cat_mc:
                print("[INFO][python/plot_masses][plot_cats] Plotting category with indices ({}, {})".format(index1,index2))
                fig, axs = plt.subplots(nrows = 1, ncols = 1)
                axs.set_xlim(60,120)
                eta_label = "{} $< |\eta| <$ {} $\oplus$ {} $< |\eta| <$ {}".format(round(cats.iloc[index1,1],4), round(cats.iloc[index1,2],4), round(cats.iloc[index2,1],4), round(cats.iloc[index2,2],4))
                r9_label = "{} $< R_9 <$ {} $\oplus$ {} $< R_9 <$ {}".format(cats.iloc[index1, 3], cats.iloc[index1,4], cats.iloc[index2,3], cats.iloc[index2,4]) if cats.iloc[index1,3] != -1 else ""
                et_label = "{} $< E_T <$ {} $\oplus$ {} $< E_T <$ {}".format(cats.iloc[index1,6], cats.iloc[index1,7], cats.iloc[index2,6], cats.iloc[index2,7]) if cats.iloc[index1,6] != -1 else ""
                dy, dx, _ = axs.hist(data[index1][index2], bins, alpha=0.5, label="Data")
                my, mx = np.histogram(mc[index1][index2], bins=bins)
                weight = np.sum(dy)*my/np.sum(my)
                weight[weight == np.inf] = 0
                weight[weight ==-np.inf] = 0
                my, mx, _ = axs.hist(mx[:-1], bins, weights=weight, alpha=0.5, label="MC")
                axs.set_ylim(0, max(dy.max(),my.max())*1.75)
                axs.plot([],[], '', label=eta_label)
                axs.plot([],[], '', label=str(r9_label+et_label))
                axs.set_ylabel("Events / 0.5 GeV")
                axs.set_xlabel("M$_{ee}$ [GeV]")
                axs.legend(loc='best')
                fig.savefig("/eos/home-n/nschroed/ECALELF/ul17/step5/invMass_{}_{}.png".format(index1, index2))
                plt.close(fig)

