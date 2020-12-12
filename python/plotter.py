import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from zcat_class import zcat
"""
This function will plot histograms of data and mc in the categories provided
"""

def plot_cats(plot_dir, zcats, cats):
    """
    Plots the invariant mass distributions
    """
    
    for cat in zcats:
        if cat.valid:
            bins = np.arange(cat.hist_min, cat.hist_max, cat.bin_size)
            print("[INFO][python/plotter][plot_cats] Plotting category with indices ({}, {})".format(cat.lead_index,cat.sublead_index))
            fig, axs = plt.subplots(nrows = 1, ncols = 1)
            axs.set_xlim(80,100)
            eta_label = "{} $< |\eta| <$ {} $\oplus$ {} $< |\eta| <$ {}".format(round(cats.iloc[cat.lead_index,1],4), round(cats.iloc[cat.lead_index,2],4), round(cats.iloc[cat.sublead_index,1],4), round(cats.iloc[cat.sublead_index,2],4))
            r9_label = "{} $< R_9 <$ {} $\oplus$ {} $< R_9 <$ {}".format(cats.iloc[cat.lead_index, 3], cats.iloc[cat.lead_index,4], cats.iloc[cat.sublead_index,3], cats.iloc[cat.sublead_index,4]) if cats.iloc[cat.lead_index,3] != -1 else ""
            et_label = "{} $< E_T <$ {} $\oplus$ {} $< E_T <$ {}".format(cats.iloc[cat.lead_index,6], cats.iloc[cat.lead_index,7], cats.iloc[cat.sublead_index,6], cats.iloc[cat.sublead_index,7]) if cats.iloc[cat.lead_index,6] != -1 else ""
            dy, dx, _ = axs.hist(cat.data, bins, alpha=0.5, label="Data")
            my, mx = np.histogram(cat.mc, bins=bins)
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
            fig.savefig(plot_dir+"invMass_{}_{}.png".format(cat.lead_index if cat.lead_index > 9 else str('0'+str(cat.lead_index)), cat.sublead_index if cat.sublead_index > 9 else str('0'+str(cat.sublead_index))))
            plt.close(fig)

def plot_1Dscan(plot_dir, scan_file, zcats):
    """
    Plots the 1D scans in the diagonal dielectron categories
    """

    scales_df = pd.read_csv(scan_file,sep='\t',header=None)
    print(scan_file)
    for i,row in scales_df.iterrows():
        print("[INFO][python/plotter][plot_1Dscan] Plotting single category with index {}".format(i))
        scan_vals = np.arange(row[8]*0.95, row[8]*0.975, (row[8]*0.975-row[8]*0.95)/5)
        scan_vals = np.append(scan_vals, np.arange(row[8]*0.975, row[8]*0.99, (row[8]*0.99-row[8]*0.975)/10))
        scan_vals = np.append(scan_vals, np.arange(row[8]*0.99, row[8]*1.01, (row[8]*1.01-row[8]*0.99)/25))
        scan_vals = np.append(scan_vals, np.arange(row[8]*1.01, row[8]*1.025, (row[8]*1.025-row[8]*1.01)/10))
        scan_vals = np.append(scan_vals, np.arange(row[8]*1.025, row[8]*1.05, (row[8]*1.05-row[8]*1.025)/5))
        scan_vals = np.unique(scan_vals)
        nll = []
        for val in scan_vals:
            nll.append(0)
            for cat in zcats: 
                if cat.valid:
                    if row[0]=='scale':
                        if cat.lead_index == i:
                            cat.update(val, 1)
                            nll[-1] += cat.NLL
                        if cat.sublead_index == i:
                            cat.update(1, val)
                            nll[-1] += cat.NLL
                    if row[0]=='smear':
                        if cat.lead_smear_index == i:
                            cat.update(1,1,val,0)
                            nll[-1] += cat.NLL
                        if cat.sublead_smear_index == i:
                            cat.update(1,1,0,val)
                            nll[-1] += cat.NLL
        nll = [x - min(nll) for x in nll]

        fig, axs = plt.subplots(nrows = 2, ncols = 1)
        eta_label = "{} $< |\eta| <$ {}".format(round(row[1],4), round(row[2],4))
        r9_label = "{} $< R_9 <$ {}".format(row[3], row[4]) if row[3] != -1 else ""
        et_label = "{} $< E_T <$ {}".format(row[6], row[7]) if row[6] != -1 else ""
        axs[1].plot(scan_vals, nll, '')
        axs[1].plot([],[],'',label=eta_label)
        mindex = nll.index(min(nll))
        axs[0].plot(scan_vals[mindex-5:mindex+6],nll[mindex-5:mindex+6], '')
        axs[0].set_ylabel('-2$\Delta$NLL * $\Chi^2$')
        axs[1].set_ylabel('-2$\Delta$NLL * $\Chi^2$')
        axs[0].set_xlabel('1+$\Delta$P' if row[0]=='scale' else '$\Delta\sigma$')
        axs[1].set_xlabel('1+$\Delta$P' if row[0]=='scale' else '$\Delta\sigma$')
        axs[1].legend(loc='best')
        fig.savefig(plot_dir+"scan1D_{}.png".format(i if i >= 10 else str('0'+str(i))))
        plt.close(fig)
            
    return
