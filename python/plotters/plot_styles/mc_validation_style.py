import pandas as pd
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
mpl.rc("text", usetex=True)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.font_manager as font_manager
import time

import python.helpers.helper_plots as helper_plots
from python.classes.config_class import SSConfig

ss_config = SSConfig()
from python.classes.constant_classes import PlottingConstants as pc
from python.plotters.fit_bw_cb import fit_bw_cb

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
    }
)

plot_dir = ss_config.DEFAULT_PLOT_PATH


def plot_style_validation_mc(data, mc, plot_title, **options):
    """
    Plots the invariant mass distributions with the MC validation style
    ----------
    Args:
        data: data histogram
        mc: mc histogram
        plot_title: title of the plot
        options: dictionary of options
            - syst: boolean to plot the systematics
            - bins: number of bins
            - no_ratio: boolean to not plot the ratio
    ----------
    Returns:
        None
    ----------
    """

    style = "crossCheckMC_"
    print("plotting {}".format(plot_title))

    if options["syst"]:
        data, data_up, data_down = data
        mc, mc_up, mc_down, mc_weights = mc

    # binning scheme
    binning = "auto"
    hist_min = 80.0
    hist_max = 100.0
    if options["bins"] != None or options["bins"] != "auto":
        binning = [
            hist_min + float(i) * (hist_max - hist_min) / float(options["bins"])
            for i in range(int(options["bins"]) + 1)
        ]

    # histogram data and mc
    h_mc, h_bins = np.histogram(
        mc, bins=binning, range=[hist_min, hist_max], weights=mc_weights
    )
    h_mc_up, h_bins = np.histogram(
        mc_up, bins=binning, range=[hist_min, hist_max], weights=mc_weights
    )
    h_mc_down, h_bins = np.histogram(
        mc_down, bins=binning, range=[hist_min, hist_max], weights=mc_weights
    )

    h_mc_up = h_mc_up * sum(h_mc) / sum(h_mc_up)
    h_mc_down = h_mc_down * sum(h_mc) / sum(h_mc_down)
    bin_width = round(h_bins[1] - h_bins[0], 4)
    marker_size = 20 * bin_width

    mids = [(h_bins[i] + h_bins[i + 1]) / 2 for i in range(len(h_bins) - 1)]

    rows = 2
    if "no_ratio" in options.keys():
        rows = 1
    fig, axs = plt.subplots(
        nrows=rows, ncols=1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(6, 8)
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.99,
        top=0.95,
        bottom=0.1,
        hspace=None if "no_ratio" in options.keys() else 0.06,
    )

    x_err = (h_bins[1] - h_bins[0]) / 2
    axs[0].errorbar(
        mids,
        h_mc,  # plot mc
        xerr=x_err,
        label="mc",
        color="cornflowerblue",
        drawstyle="steps-mid",
        capsize=0.0,
    )
    axs[0].errorbar(
        mids,
        h_mc_up,  # plot data
        xerr=x_err,
        yerr=np.sqrt(h_mc_up),
        label="data",
        linestyle="None",
        color="blue",
        marker="o",
        markersize=marker_size,
        capsize=0.0,
        capthick=0.0,
    )
    axs[0].errorbar(
        mids,
        h_mc_down,  # plot data
        xerr=x_err,
        yerr=np.sqrt(h_mc_down),
        label="data",
        linestyle="None",
        color="green",
        marker="o",
        markersize=marker_size,
        capsize=0.0,
        capthick=0.0,
    )

    axs[0].set_ylim(bottom=0)
    axs[0].set_xlim(hist_min - 1, hist_max + 1)
    axs[0].set_ylabel(
        "Events/{x:.3f} GeV".format(x=bin_width),
        horizontalalignment="right",
        y=1.0,
        labelpad=5,
    )

    # invert legend order because python is a hassle
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles[::-1], labels[::-1], loc="best")
    y_err_mc = np.sqrt(h_mc)

    if "no_ratio" in options.keys():
        axs[0].set_xlabel("M$_{ee}$ [GeV]", horizontalalignment="right", x=1.0)
    else:
        ratio = np.maximum(
            np.abs(np.subtract(h_mc, h_mc_up)), np.abs(np.subtract(h_mc, h_mc_down))
        )
        axs[1].plot(
            mids, [1.0 for x in mids], linestyle="dashed", color="black", alpha=0.5
        )
        # add syst+unc band to unity line

        axs[1].errorbar(
            mids,
            ratio,
            xerr=x_err,
            yerr=[0 for x in mids],
            label="data / mc",
            linestyle="None",
            color="black",
            marker="o",
            markersize=marker_size,
            capsize=0.0,
        )
        # invert legend order because python is a hassle
        handles, labels = axs[1].get_legend_handles_labels()
        axs[1].legend(handles[::-1], labels[::-1], loc="best")
        axs[1].set_ylabel("data/mc", horizontalalignment="right", y=1.0)
        axs[1].set_xlabel("M$_{ee}$ [GeV]", horizontalalignment="right", x=1.0)
        axs[1].set_xlim(hist_min - 1, hist_max + 1)

    # save fig
    fig.savefig(f"{plot_dir}{style}{options['tag']}_{plot_title}.png")
    fig.savefig(f"{plot_dir}{style}{options['tag']}_{plot_title}.pdf")

    plt.close(fig)
