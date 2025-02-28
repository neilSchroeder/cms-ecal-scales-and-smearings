from src.tools.data_loader import custom_cuts
from src.tools.data_loader import get_dataframe
import numpy as np
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd

from src.classes.constant_classes import (
    DataConstants as dc,
)

from src.plotters.plots import (
    plot_style_bw_cb_fit,
    plot_style_paper,
    plot_style_validation_mc,
)


def main():

    # import the data
    data_path = "examples/data/pruned_ul18_data.csv"
    mc_path = "examples/data/pruned_ul18_mc.csv"

    df_data = get_dataframe(
        [data_path],
        apply_cuts="custom",
        eta_cuts=(0, 1.4442, 1.566, 2.5),
        et_cuts=((32, 14000), (20, 14000)),
    )
    # print data columns
    print(df_data.columns)

    df_mc = get_dataframe(
        [mc_path],
        apply_cuts="custom",
        eta_cuts=(0, 1.4442, 1.566, 2.5),
        et_cuts=((32, 14000), (20, 14000)),
    )

    # derive pt y weights
    from src.tools.reweight_pt_y import (
        derive_pt_y_weights,
        add_pt_y_weights,
        get_zpt,
    )

    weights_file = derive_pt_y_weights(df_data, df_mc, "ul18")
    df_mc = add_pt_y_weights(df_data, weights_file)

    data_ptz = get_zpt(df_data)
    mc_ptz = get_zpt(df_mc)

    # plot the ptz
    ptz_hist_data, bins = np.histogram(data_ptz, bins=dc.PTZ_BINS)
    ptz_hist_mc, bins = np.histogram(
        mc_ptz, bins=dc.PTZ_BINS, weights=df_mc[dc.PTY_WEIGHT].values
    )

    ptz_hist_mc = np.sum(ptz_hist_data) * ptz_hist_mc / np.sum(ptz_hist_mc)
    mids = 0.5 * (bins[1:] + bins[:-1])

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    x_err = 0.5 * (bins[1:] - bins[:-1])
    ax[0].errorbar(
        mids,
        ptz_hist_data,
        xerr=x_err,
        yerr=np.sqrt(ptz_hist_data),
        fmt="o",
        label="Data",
        alpha=0.5,
    )
    ax[0].errorbar(
        mids,
        ptz_hist_mc,
        xerr=x_err,
        yerr=np.sqrt(ptz_hist_mc),
        fmt="o",
        label="MC",
        alpha=0.5,
    )
    ax[0].set_ylabel("Events")
    ax[0].legend()
    ax[1].errorbar(
        mids,
        ptz_hist_data / ptz_hist_mc,
        xerr=x_err,
        yerr=np.sqrt(ptz_hist_data) / ptz_hist_mc,
        fmt="o",
    )
    ax[1].set_ylabel("Data/MC")
    ax[1].set_xlabel("Pt(Z) [GeV]")
    ax[1].set_xscale("log")
    ax[0].set_xscale("log")

    plt.savefig("ptz.png")


if __name__ == "__main__":
    main()
