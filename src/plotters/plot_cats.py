import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.classes.config_class import SSConfig
from src.classes.constant_classes import CategoryConstants as cc
from src.classes.constant_classes import DataConstants as dc
from src.classes.zcat_class import zcat

ss_config = SSConfig()


def plot_cats(zcats, cats, plot_dir=ss_config.DEFAULT_PLOT_PATH):
    """
    Plots the invariant mass distributions
    ----------
    Args:
        zcats: list of zcat objects
        cats: dataframe of the categories
        plot_dir: directory to save the plots
    ----------
    Returns:
        None
    ----------
    """

    for cat in zcats:
        if cat.valid:
            bins = np.arange(cat.hist_min, cat.hist_max, cat.bin_size)
            print(
                f"[INFO][python/plotters][plot_cats] Plotting category with indices ({cat.lead_index}, {cat.sublead_index})"
            )
            fig, axs = plt.subplots(nrows=1, ncols=1)
            axs.set_xlim(dc.MIN_INVMASS, dc.MAX_INVMASS)

            eta_label = (
                f"{round(cats.iloc[cat.lead_index, cc.i_eta_min],4)} $< |\eta| <$ {round(cats.iloc[cat.lead_index,cc.i_eta_max],4)} $\oplus$ {round(cats.iloc[cat.sublead_index,cc.i_eta_min],4)} $< |\eta| <$ {round(cats.iloc[cat.sublead_index,cc.i_eta_max],4)}"
                if cats.iloc[cat.lead_index, cc.i_eta_min] != -1
                else ""
            )

            r9_label = (
                f"{cats.iloc[cat.lead_index, cc.i_r9_min]} $< R_9 <$ {cats.iloc[cat.lead_index, cc.i_r9_max]} $\oplus$ {cats.iloc[cat.sublead_index, cc.i_r9_min]} $< R_9 <$ {cats.iloc[cat.sublead_index, cc.i_r9_max]}"
                if cats.iloc[cat.lead_index, 3] != -1
                else ""
            )

            et_label = (
                f"{cats.iloc[cat.lead_index,cc.i_et_min]} $< E_T <$ {cats.iloc[cat.lead_index,cc.i_et_max]} $\oplus$ {cats.iloc[cat.sublead_index, cc.i_et_min]} $< E_T <$ {cats.iloc[cat.sublead_index, cc.i_et_max]}"
                if cats.iloc[cat.lead_index, 6] != -1
                else ""
            )

            dy, dx, _ = axs.hist(cat.data, bins, alpha=0.5, label="Data")
            my, mx = np.histogram(cat.mc, bins=bins)
            weight = np.sum(dy) * my / np.sum(my)
            weight[weight == np.inf] = 0
            weight[weight == -np.inf] = 0
            my, mx, _ = axs.hist(mx[:-1], bins, weights=weight, alpha=0.5, label="MC")
            axs.set_ylim(0, max(dy.max(), my.max()) * 1.75)
            axs.plot([], [], "", label=eta_label)
            axs.plot([], [], "", label=str(r9_label + et_label))
            axs.set_ylabel("Events / 0.5 GeV")
            axs.set_xlabel("M$_{ee}$ [GeV]")
            axs.legend(loc="best")
            fig_path = f"{plot_dir}/invMass_{cat.lead_index if cat.lead_index > 9 else str('0'+str(cat.lead_index))}_{cat.sublead_index if cat.sublead_index > 9 else str('0'+str(cat.sublead_index))}.png"
            fig.savefig(fig_path)
            print(f"[INFO][python/plotters][plot_cats] Saved plot to {fig_path}")
            plt.close(fig)


# this is probably broken, so use it with caution
def plot_1Dscan(scan_file, zcats, plot_dir=ss_config.DEFAULT_PLOT_PATH):
    """
    Plots the 1D scans in the diagonal dielectron categories
    ----------
    Args:
        scan_file: file containing the 1D scan results
        zcats: list of zcat objects
        plot_dir: directory to save the plots
    ----------
    Returns:
        None
    ----------
    """

    num_scales = 0
    scales_df = pd.read_csv(scan_file, sep="\t", header=None)
    for i, row in scales_df.iterrows():
        if row[0] == "scale":
            num_scales += 1
            continue
        if i > 49:
            print(
                "[INFO][python/plotter][plot_1Dscan] Plotting single category with index {}".format(
                    i
                )
            )
            scan_vals = np.arange(
                row[8] * 0.8, row[8] * 0.95, (row[8] * 0.95 - row[8] * 0.8) / 15
            )
            scan_vals = np.append(
                scan_vals,
                np.arange(
                    row[8] * 0.95, row[8] * 0.975, (row[8] * 0.975 - row[8] * 0.95) / 5
                ),
            )
            scan_vals = np.append(
                scan_vals,
                np.arange(
                    row[8] * 0.975, row[8] * 0.99, (row[8] * 0.99 - row[8] * 0.975) / 10
                ),
            )
            scan_vals = np.append(
                scan_vals,
                np.arange(
                    row[8] * 0.99, row[8] * 1.01, (row[8] * 1.01 - row[8] * 0.99) / 25
                ),
            )
            scan_vals = np.append(
                scan_vals,
                np.arange(
                    row[8] * 1.01, row[8] * 1.025, (row[8] * 1.025 - row[8] * 1.01) / 10
                ),
            )
            scan_vals = np.append(
                scan_vals,
                np.arange(
                    row[8] * 1.025, row[8] * 1.05, (row[8] * 1.05 - row[8] * 1.025) / 5
                ),
            )
            scan_vals = np.append(
                scan_vals,
                np.arange(
                    row[8] * 1.05, row[8] * 1.5, (row[8] * 1.5 - row[8] * 1.05) / 45
                ),
            )
            scan_vals = np.unique(scan_vals)
            nll = []
            for val in scan_vals:
                nll.append(0)
                for cat in zcats:
                    if cat.valid:
                        """
                        if row[0]=='scale':
                            if cat.lead_index == i:
                                cat.update(val, 1)
                                nll[-1] += cat.NLL
                            if cat.sublead_index == i:
                                cat.update(1, val)
                                nll[-1] += cat.NLL
                        """
                        if row[0] == "smear":
                            if cat.lead_smear_index == i:
                                cat.update(1, 1, val, 0)
                                nll[-1] += cat.NLL
                            if cat.sublead_smear_index == i:
                                cat.update(1, 1, 0, val)
                                nll[-1] += cat.NLL
            nll = [x - min(nll) for x in nll]
            fig, axs = plt.subplots(nrows=2, ncols=1)
            eta_label = "{} $< |\eta| <$ {}".format(round(row[1], 4), round(row[2], 4))
            r9_label = "{} $< R_9 <$ {}".format(row[3], row[4]) if row[3] != -1 else ""
            et_label = "{} $< E_T <$ {}".format(row[6], row[7]) if row[6] != -1 else ""
            axs[1].plot(scan_vals, nll, "")
            axs[1].plot([], [], "", label=eta_label)
            mindex = nll.index(min(nll))
            axs[0].plot(
                scan_vals[mindex - 5 : mindex + 6], nll[mindex - 5 : mindex + 6], ""
            )
            axs[0].set_ylabel("-2$\Delta$NLL * $\chi^2$")
            axs[1].set_ylabel("-2$\Delta$NLL * $\chi^2$")
            axs[0].set_xlabel("1+$\Delta$P" if row[0] == "scale" else "$\Delta\sigma$")
            axs[1].set_xlabel("1+$\Delta$P" if row[0] == "scale" else "$\Delta\sigma$")
            axs[1].legend(loc="best")
            fig.savefig(
                plot_dir + "scan1D_{}.png".format(i if i >= 10 else str("0" + str(i)))
            )
            plt.close(fig)

    return
