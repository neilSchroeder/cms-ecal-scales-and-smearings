import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
mpl.rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.font_manager as font_manager
import time

import python.helpers.helper_plots as helper_plots
from python.classes.config_class import SSConfig
ss_config = SSConfig()
from python.classes.constant_classes import PlottingConstants as pc
from python.plotters.fit_bw_cb import fit_bw_cb

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
})

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

    if options['syst']:
        data, data_up, data_down = data
        mc, mc_up, mc_down, mc_weights = mc

    #binning scheme
    binning = 'auto'
    hist_min = 80.
    hist_max = 100.
    if options['bins'] != None or options['bins'] != 'auto':
        binning = [hist_min + float(i)*(hist_max-hist_min)/float(options['bins']) for i in range(int(options['bins'])+1)]

    #histogram data and mc
    h_mc, h_bins = np.histogram(mc, bins=binning, range=[hist_min,hist_max], weights=mc_weights)
    h_mc_up, h_bins = np.histogram(mc_up, bins=binning, range=[hist_min,hist_max], weights=mc_weights)
    h_mc_down, h_bins = np.histogram(mc_down, bins=binning, range=[hist_min,hist_max], weights=mc_weights)

    h_mc_up = h_mc_up * sum(h_mc)/sum(h_mc_up)
    h_mc_down = h_mc_down * sum(h_mc)/sum(h_mc_down)
    bin_width = round(h_bins[1] - h_bins[0], 4)
    marker_size = 20*bin_width

    mids = [(h_bins[i]+h_bins[i+1])/2 for i in range(len(h_bins)-1)]

    rows = 2
    if 'no_ratio' in options.keys(): rows = 1
    fig,axs = plt.subplots(nrows=rows, ncols=1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(6,8))
    fig.subplots_adjust(left=0.12, right=0.99, top=0.95, bottom=0.1,
            hspace=None if 'no_ratio' in options.keys() else 0.06)

    x_err = (h_bins[1]-h_bins[0])/2
    axs[0].errorbar(mids, h_mc, # plot mc
            xerr=x_err,
            label = 'mc', 
            color='cornflowerblue',
            drawstyle='steps-mid',
            capsize=0.,
            )
    axs[0].errorbar(mids, h_mc_up, # plot data
            xerr=x_err, yerr=np.sqrt(h_mc_up), 
            label = 'data', 
            linestyle='None', 
            color='blue',
            marker='o',
            markersize=marker_size, 
            capsize=0., 
            capthick=0.)
    axs[0].errorbar(mids, h_mc_down, # plot data
            xerr=x_err, yerr=np.sqrt(h_mc_down), 
            label = 'data', 
            linestyle='None', 
            color='green',
            marker='o',
            markersize=marker_size, 
            capsize=0., 
            capthick=0.)

    axs[0].set_ylim(bottom=0)
    axs[0].set_xlim(hist_min-1, hist_max+1)
    axs[0].set_ylabel("Events/{x:.3f} GeV".format(x=bin_width), horizontalalignment='right',y=1., labelpad=5)

    # invert legend order because python is a hassle
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles[::-1], labels[::-1],loc='best')
    y_err_mc = np.sqrt(h_mc)

    if 'no_ratio' in options.keys():
        axs[0].set_xlabel('M$_{ee}$ [GeV]',horizontalalignment='right',x=1.)
    else:
        ratio = np.maximum(np.abs(np.subtract(h_mc, h_mc_up)), np.abs(np.subtract(h_mc, h_mc_down)))
        axs[1].plot(mids, [1. for x in mids], linestyle='dashed', color='black', alpha=0.5)
        # add syst+unc band to unity line
            
        axs[1].errorbar(mids, ratio, 
                xerr=x_err, yerr=[0 for x in mids],
                label='data / mc', 
                linestyle='None',
                color='black',
                marker='o',
                markersize=marker_size,
                capsize=0.,)
        #invert legend order because python is a hassle
        handles, labels = axs[1].get_legend_handles_labels()
        axs[1].legend(handles[::-1], labels[::-1], loc='best')
        axs[1].set_ylabel('data/mc',horizontalalignment='right', y=1.)
        axs[1].set_xlabel('M$_{ee}$ [GeV]',horizontalalignment='right', x=1.)
        axs[1].set_xlim(hist_min-1, hist_max+1)

    #save fig
    fig.savefig(f"{plot_dir}{style}{options['tag']}_{plot_title}.png")
    fig.savefig(f"{plot_dir}{style}{options['tag']}_{plot_title}.pdf")

    plt.close(fig)


def plot_style_paper(data, mc, plot_title, **options):
    """
    Plotting function for paper style plots
    ---------------------------------------
    Args:
        data: data histogram
        mc: mc histogram
        plot_title: title of plot
        options: dictionary of options
            - syst: bool, whether to plot systematics
            - bins: int, number of bins
            - tag: str, tag for plot name
            - no_ratio: bool, whether to plot ratio
    ---------------------------------------
    Returns:
        None
    ---------------------------------------
    """
    style = pc.paper_style
    print("plotting {}".format(plot_title))

    # systematics
    if options['syst']:
        data, data_up, data_down = data
        mc, mc_weights = mc

    # binning scheme
    binning = style.binning
    if options['bins'] and options['bins'] != 'auto':
        binning = [pc.HIST_MIN + float(i)*(pc.HIST_MAX-pc.HIST_MIN)/float(options['bins']) for i in range(int(options['bins'])+1)]

    # histogram data and mc
    h_data, h_bins = np.histogram(data, bins=binning, range=[pc.HIST_MIN,pc.HIST_MAX])
    try:
        h_mc, h_bins = np.histogram(mc, bins=h_bins, weights=mc_weights)
    except Exception as e:
        # probably failed because no weights (unusual)
        h_mc, h_bins = np.histogram(mc, bins=h_bins) 
    bin_width = round(h_bins[1] - h_bins[0], 4)
    marker_size = 20*bin_width

    h_mc = h_mc*np.sum(h_data)/np.sum(h_mc)

    ratio = np.divide(h_data,h_mc)
    ratio[ratio==np.inf] = np.nan
    
    mids = [(h_bins[i]+h_bins[i+1])/2 for i in range(len(h_bins)-1)]
    mids_full = mids.copy()
    mids_full[0], mids_full[-1] = pc.HIST_MIN, pc.HIST_MAX
    x_err = (h_bins[1]-h_bins[0])/2

    # fit the data and mc
    fit_params_data = None
    fit_params_mc = None
    if options['_kFit']:
        # fit the data
        fit_params_data = fit_bw_cb(mids, h_data, [1.424, 1.86, np.average(mids, weights=h_data)-91.188, 1.])
        fit_params_mc = fit_bw_cb(mids, h_mc, [1.424, 1.86, np.average(mids, weights=h_mc)-91.188, 1.])

    # calculate errors
    y_err_data = np.sqrt(h_data)
    try:
        y_err_mc = helper_plots.get_bin_uncertainties(h_bins, mc, mc_weights)
    except Exception as e:
        # probably failed because no weights (unusual)
        y_err_mc = np.sqrt(h_mc)
    y_err_ratio = np.array([])

    mc_err_max = np.add(h_mc,y_err_mc)
    mc_err_min = np.subtract(h_mc,y_err_mc)

    # include syst uncertainties if they exist
    syst_unc = []
    if options['syst']:
        syst_unc = helper_plots.get_systematic_uncertainty(h_bins, data, data_up, data_down, mc, mc_weights)
        err = np.sqrt(np.add(np.power(y_err_mc, 2), np.power(syst_unc, 2)))
        mc_err_max, mc_err_min = np.add(h_mc, err), np.subtract(h_mc, err)
        y_err_ratio = np.divide(y_err_data, h_data)

    # define figure
    rows = 2
    if 'no_ratio' in options: rows = 1
    fig,axs = plt.subplots(
        nrows=rows, 
        ncols=1, 
        figsize=style.fig['size'],
        sharex=style.fig['sharex'],
        gridspec_kw={'height_ratios': style.fig['subplot_ratio']}
        )
    
    if 'no_ratio' in options:
        axs = [axs]

    fig.subplots_adjust(
        left=style.subplot['left'], 
        right=style.subplot['right'], 
        top=style.subplot['top'], 
        bottom=style.subplot['bottom'],
        hspace=None if 'no_ratio' in options.keys() else style.subplot['hspace'],
    )

    # top plot
    axs[0].fill_between(
        mids_full,
        h_mc,
        y2=0,
        step='mid',
        alpha=0.2,
        color=style.colors['mc'],
    ) #fill mc area
    axs[0].fill_between(
        mids_full,
        mc_err_max,
        y2=mc_err_min,
        step='mid', 
        alpha=0.3, 
        color=style.colors['syst'], 
        label=style.labels['syst']) #fill mc error
    axs[0].errorbar(mids, h_mc, # plot mc
            xerr=x_err,
            label = options['mc_title'], 
            color=style.colors['mc'],
            drawstyle=style.error_bar_style,
            capsize=0.,
            )
    axs[0].errorbar(mids, h_data, # plot data
            xerr=x_err, yerr=y_err_data, 
            label = options['data_title'], 
            color=style.colors['data'],
            linestyle='None', 
            marker='o',
            markersize=marker_size, 
            capsize=0., 
            capthick=0.)
    
    axs[0].set_ylim(bottom=0, top=np.max(h_data)*style.y_scale)

    axs[0].set_ylabel("Events/{x:.3f} GeV".format(x=bin_width), horizontalalignment='right',y=1., labelpad=5, fontsize=14)
    # set grid
    axs[0].grid(which='major',axis='both')
    # set scientific notation
    axs[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axs[0].yaxis.offsetText.set_position(style.sci_notation_offset)

    # invert legend order because python is a hassle
    handles, labels = axs[0].get_legend_handles_labels()
    handles, labels = handles[::-1], labels[::-1]
    if options['_kFit'] and fit_params_data and fit_params_mc:
        labels[0] = f"{labels[0]}: $\mu$={fit_params_data['mu']:.3f} $\sigma$={fit_params_data['sigma']:.3f}"
        labels[1] = f"{labels[1]}: $\mu$={fit_params_mc['mu']:.3f} $\sigma$={fit_params_mc['sigma']:.3f}"

    axs[0].legend( 
        handles[::-1], 
        labels[::-1], 
        loc=style.legend['loc'], 
        fontsize=style.legend['fontsize'],
        )   

    # labels
    for text in style.annotations.keys():
        if text == 'plot_title': 
            print('plot_title', plot_title)
            if plot_title in style.annotations[text]['annot'].keys():
                print("plot_title found in annotations", style.annotations[text]['annot'][plot_title])
                axs[0].annotate(
                    style.annotations[text]['annot'][plot_title],
                    xy=style.annotations[text]['xy'],
                    xycoords=style.annotations[text]['xycoords'],
                    ha=style.annotations[text]['ha'],
                    va=style.annotations[text]['va'],
                    weight=style.annotations[text]['weight'],
                    fontsize=style.annotations[text]['fontsize'],
                )
            else:
                print(f"WARNING: {plot_title} not found in annotations, no plot title will be added")
        else:
            lumi = None
            if text == 'lumi':
                lumi = options['lumi'] if 'lumi' in options.keys() else None
            axs[0].annotate(
                lumi if lumi else r'{}'.format(style.annotations[text]['annot']),
                xy=style.annotations[text]['xy'],
                xycoords=style.annotations[text]['xycoords'],
                ha=style.annotations[text]['ha'],
                va=style.annotations[text]['va'],
                fontsize=style.annotations[text]['fontsize'],
            )
    
    # ratio pad
    if 'no_ratio' in options.keys():
        axs[0].set_xlabel(style.labels['x_axis']['label'],
                          ha=style.labels['x_axis']['ha'],
                        #   fontsize=style.labels['x_axis']['fontsize'],
                          x=1.)
    else:
        axs[1].plot(mids, 
                    [1. for x in mids], 
                    linestyle='dashed', 
                    color=style.colors['data'], 
                    alpha=0.5)

        # add syst+unc band to unity line
        if 'syst' in options and options['syst']:
            err = np.sqrt(np.add(np.power(y_err_mc,2),np.power(syst_unc,2)))
            syst_err = np.divide(err,h_mc)
            axs[1].fill_between(mids_full, 
                                syst_err+1, 
                                1-syst_err, 
                                step='mid', 
                                alpha=0.3, 
                                color=style.colors['syst'], 
                                label=style.labels['syst'])
            
        axs[1].errorbar(mids, ratio, 
                xerr=x_err, yerr=y_err_ratio, 
                label=style.labels['ratio'], 
                linestyle='None',
                color=style.colors['data'],
                marker='o',
                markersize=marker_size,
                capsize=0.,)

        # invert legend order because python is a hassle
        handles, labels = axs[1].get_legend_handles_labels()
        axs[1].legend(
            handles[::-1], 
            labels[::-1], 
            loc=style.legend['loc'],
            fontsize=style.legend['fontsize'],
        )

        axs[1].set_ylabel(
            style.labels['ratio'],
            horizontalalignment='right', 
            y=1.,
            fontsize=14,
        )
        axs[1].set_xlabel(
            style.labels['x_axis']['label'],
            horizontalalignment=style.labels['x_axis']['ha'],
            x=1.,
            fontsize=style.labels['x_axis']['fontsize'],
        )
        axs[1].set_ylim(pc.RATIO_MIN, pc.RATIO_MAX)
        axs[1].set_xlim(pc.HIST_MIN, pc.HIST_MAX)
        axs[1].grid(which='major',axis='both')
        # set tick size to zero on top plot
        axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        # set x label size


    # save fig
    fig.savefig(f"{plot_dir}{style.style}{options['tag']}_{plot_title}.png")
    fig.savefig(f"{plot_dir}{style.style}{options['tag']}_{plot_title}.pdf")

    plt.close(fig)

