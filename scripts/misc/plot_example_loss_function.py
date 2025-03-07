"""Plot the example loss function."""

import matplotlib
import numpy as np
import seaborn as sns
from scipy.special import xlogy

matplotlib.use("Agg")
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import rel_entr

from src.classes.constant_classes import CategoryConstants as cc
from src.classes.constant_classes import DataConstants as dc
from src.classes.zcat_class import zcat
from src.core.data_loader import get_dataframe


def jensen_shannon_divergence(p, q):
    """
    Calculate the Jensen-Shannon divergence between two probability distributions.

    Parameters:
    p, q: numpy arrays of equal length representing probability distributions
         (they should sum to 1)

    Returns:
    float: The Jensen-Shannon divergence between the distributions
    """
    # First, let's ensure we're working with probability distributions
    # by normalizing the inputs
    p = np.array(p) / np.sum(p)
    q = np.array(q) / np.sum(q)

    # Calculate the middle point between the two distributions
    m = 0.5 * (p + q)

    # The JS divergence is the average of KL(P||M) and KL(Q||M)
    # We use scipy's rel_entr which computes x * log(x/y) element-wise
    # and handles edge cases properly

    # Calculate KL(P||M)
    kl_p_m = np.sum(rel_entr(p, m))

    # Calculate KL(Q||M)
    kl_q_m = np.sum(rel_entr(q, m))

    # Take the average
    js_divergence = 0.5 * (kl_p_m + kl_q_m)

    return js_divergence


def earth_movers_distance(hist1, hist2):
    """
    Calculate the Earth Mover's Distance between two histograms.

    Parameters:
    hist1, hist2: numpy arrays of equal length containing bin contents

    Returns:
    float: The Earth Mover's Distance between the histograms
    """
    # Calculate the cumulative distribution functions (CDFs)
    # The CDF at each point represents the total probability mass up to that point
    cdf1 = np.cumsum(hist1 / hist1.sum())
    cdf2 = np.cumsum(hist2 / hist2.sum())

    # The EMD in 1D is actually just the area between the CDFs
    # We can calculate this using the absolute difference between CDFs
    emd = np.sum(np.abs(cdf1 - cdf2) ** 0.5 * hist1**2) / (hist1**2).sum()

    return emd, np.abs(cdf1 - cdf2) ** 0.5 * (hist1**2)


# define the target function
def nll_chi_sqr(data, mc):
    # evaluate the nll and chi squared between data and mc
    err_mc = np.sqrt(mc)
    err_data = np.sqrt(data)

    chi_sqr = (data - mc) ** 2 / np.sqrt(err_data**2 + err_mc**2)
    mc = mc / np.sum(mc)
    data = data / np.sum(data)

    temperature = 0.5
    nll = xlogy(data, mc**temperature)
    penalty = xlogy(1 - data, (1 - mc) ** temperature)
    cross_entropy = -2 * (nll + penalty)
    return -np.sum(cross_entropy * chi_sqr) / np.sum(cross_entropy)


def main():
    # import the data
    data_path = "examples/data/pruned_ul18_data.csv"
    mc_path = "examples/data/pruned_ul18_mc.csv"

    df_data = get_dataframe(
        [data_path],
        apply_cuts="custom",
        eta_cuts=((0, 1), (0, 1)),
        et_cuts=((32, 14000), (20, 14000)),
        r9_cuts=((0, 0.96), (0, 0.96)),
        debug=True,
        nrows=200000,
    )
    # print data columns
    print(df_data.columns)

    df_mc = get_dataframe(
        [mc_path],
        apply_cuts="custom",
        eta_cuts=((0, 1), (0, 1)),
        et_cuts=((32, 14000), (20, 14000)),
        r9_cuts=((0, 0.96), (0, 0.96)),
        debug=True,
        nrows=200000,
    )

    # create zcat class
    zcat_obj = zcat(
        0,
        0,
        df_data[dc.INVMASS],
        df_mc[dc.INVMASS],
        [1] * len(df_mc[dc.INVMASS]),
    )

    loss_fun = earth_movers_distance
    df_data[dc.INVMASS] = (
        df_data[dc.INVMASS].values
        * 1.004040404040404
        * 0.9996464646464646
        * 0.9987373737373737
    )
    losses = []
    arrs = []
    for scale in np.linspace(0.98, 1.02, 100):
        # scale the data
        invmass = df_data[dc.INVMASS] * scale

        data_hist, bins = np.histogram(invmass, bins=80, range=(80, 100))
        mc_hist, bins = np.histogram(df_mc[dc.INVMASS], bins=80, range=(80, 100))
        loss, arr = loss_fun(data_hist, mc_hist)
        losses.append((scale, loss))
        arrs.append(arr)

    # # plot losses
    fig, ax = plt.subplots()
    ax.plot([x[0] for x in losses], [x[1] for x in losses])
    # y scale log
    # ax.set_yscale("log")
    ax.set_xlabel("Scale factor")
    ax.set_ylabel("Loss")
    plt.savefig("loss_function_scale.png")
    # close all
    plt.close("all")

    # plot arrs as gif
    fig, ax = plt.subplots()
    ax.set_xlabel("Bin")
    ax.set_ylabel("Cumulative difference")
    for i, arr in enumerate(arrs):
        ax.plot(arr)
        plt.savefig(f"plots/loss_function_arr_scale_{i}.png")
    plt.close("all")

    # collect arrs as gif
    import imageio

    images = []
    for i in range(len(arrs)):
        images.append(imageio.imread(f"plots/loss_function_arr_scale_{i}.png"))
    imageio.mimsave("loss_function_arr_scale.gif", images)

    losses = []
    arrs = []
    seed = 123456789
    datas = []
    mcs = []
    rand = np.random.Generator(np.random.PCG64(seed))
    s1 = rand.normal(loc=1, scale=0.001, size=10 * len(df_mc[dc.INVMASS]))
    s2 = rand.normal(loc=1, scale=0.001, size=10 * len(df_mc[dc.INVMASS]))
    old_smearing = 0.001
    new_smearing = 0.001
    for smearing in np.linspace(0.001, 0.02, 1000):
        # smear the mc
        # concatenate the mass 10 x
        s1 = (s1 - 1) * (new_smearing / old_smearing) + 1
        s2 = (s2 - 1) * (new_smearing / old_smearing) + 1
        old_smearing, new_smearing = new_smearing, smearing
        invmass = np.concatenate([df_mc[dc.INVMASS]] * 10)
        invmass = invmass * np.sqrt(s1 * s2)

        data_hist, bins = np.histogram(df_data[dc.INVMASS], bins=80, range=(80, 100))
        mc_hist, bins = np.histogram(invmass, bins=80, range=(80, 100))
        datas.append(data_hist)
        mcs.append(np.sum(data_hist) * mc_hist / np.sum(mc_hist))
        loss, arr = loss_fun(data_hist, mc_hist)
        losses.append((smearing, loss))
        arrs.append(arr)

        # # plot losses
    fig, ax = plt.subplots()
    ax.plot([x[0] for x in losses], [x[1] for x in losses])
    # y scale log
    # ax.set_yscale("log")
    ax.set_xlabel("Smearing factor")
    ax.set_yscale("log")
    ax.set_ylabel("Loss")
    plt.savefig("loss_function_smearing.png")
    plt.close("all")

    # plot arrs as gif
    for i, arr in enumerate(arrs):
        fig, ax = plt.subplots()
        ax.set_xlabel("Bin")
        ax.set_ylabel("Cumulative difference")
        ax.plot(arr, label=f"Cumulative difference: Smearing = {losses[i][0]:.4f}")
        ax.legend(loc="upper right")
        plt.savefig(f"plots/loss_function_arr_smearing_{i}.png")
        plt.close("all")

    # collect arrs as gif
    images = []
    for i in range(len(arrs)):
        images.append(imageio.imread(f"plots/loss_function_arr_smearing_{i}.png"))
    imageio.mimsave("loss_function_arr_smearing.gif", images)

    for i, (data, mc) in enumerate(zip(datas, mcs)):
        # plot data and mc
        fig, ax = plt.subplots()
        ax.plot(data, label="Data")
        ax.plot(mc, label=f"MC: smearing = {losses[i][0]:.3f}")
        ax.set_xlabel("Bin")
        ax.set_ylabel("Events")
        ax.legend()
        plt.savefig(f"plots/data_mc_{i}.png")
        plt.close("all")

    # collect as gif
    images = []
    for i in range(len(datas)):
        images.append(imageio.imread(f"plots/data_mc_{i}.png"))
    imageio.mimsave("data_mc.gif", images)

    losses = []
    rand = np.random.Generator(np.random.PCG64(seed))
    s1 = rand.normal(loc=1, scale=0.001, size=10 * len(df_mc[dc.INVMASS]))
    s2 = rand.normal(loc=1, scale=0.001, size=10 * len(df_mc[dc.INVMASS]))
    new_smear = 0.001
    old_smear = 0.001
    for smearing in np.linspace(0.01, 0.015, 100):
        # concatenate the mass 10 x
        invmass_mc = np.concatenate([df_mc[dc.INVMASS]] * 10)
        new_smear = smearing
        s1 = (s1 - 1) * (new_smear / old_smear) + 1
        s2 = (s2 - 1) * (new_smear / old_smear) + 1
        old_smear = new_smear
        invmass_mc = invmass_mc * np.sqrt(s1 * s2) / (1 - smearing**2 / 8)
        for scale in np.linspace(0.995, 1.005, 100):
            invmass = df_data[dc.INVMASS] * scale
            # scale the data
            # smear the mc

            data_hist, bins = np.histogram(invmass, bins=80, range=(80, 100))
            mc_hist, bins = np.histogram(invmass_mc, bins=80, range=(80, 100))
            loss, arr = loss_fun(data_hist, mc_hist)
            losses.append((scale, smearing, 1 / loss))

    # plot 2d heatmap losses
    fig, ax = plt.subplots()
    scales = [x[0] for x in losses]
    smears = [x[1] for x in losses]
    ax.set_ylabel("Scale factor")
    ax.set_xlabel("smearing factor")
    # set heatmap to log scale
    norm = matplotlib.colors.LogNorm(
        vmin=np.min([x[2] for x in losses]), vmax=np.max([x[2] for x in losses])
    )
    ax = sns.heatmap(
        np.array([x[2] for x in losses]).reshape(100, 100),
        xticklabels=10,
        yticklabels=10,
        cmap="viridis",
        norm=norm,
    )
    # set ticks
    ax.set_title("Loss function")
    plt.savefig("loss_function_2d.png")

    min_loss = np.max([x[2] for x in losses])
    idx = np.amax(np.where([x[2] for x in losses] == min_loss))
    print(
        f"Minimum loss: {min_loss} at scale: {losses[idx][0]} and smearing: {losses[idx][1]}"
    )


if __name__ == "__main__":
    main()
