import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib as mpl
import time

import python.helpers.helper_plots as helper_plots

mpl.rc('font',family="Helvetica")

plot_dir = '/afs/cern.ch/work/n/nschroed/SS_PyMin/plots/'

def plot_style_validation_mc(data, mc, plot_title, **options):

    style = "crossCheckMC_"
    print("plotting {}".format(plot_title))

    if options['syst']:
        data, data_up, data_down = data
        mc, mc_up, mc_down, mc_weights = mc

    #binning scheme
    binning = 'auto'
    hist_min = 80.
    hist_max = 100.
    if options['bins'] is not None or options['bins'] is not 'auto':
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
    fig,axs = plt.subplots(nrows=rows, ncols=1)
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

    plt.grid(which='major',axis='both')
    #save fig
    fig.savefig(plot_dir+style+plot_title+".png")
    fig.savefig(plot_dir+style+plot_title+".pdf")

    plt.close(fig)


def plot_style_paper(data, mc, plot_title, **options):
    style = "paperStyle_"
    print("plotting {}".format(plot_title))

    # systematics
    if options['syst']:
        data, data_up, data_down = data
        mc, mc_up, mc_down, mc_weights = mc

    # binning scheme
    binning = 'auto'
    hist_min = 80.
    hist_max = 100.
    if options['bins'] is not None or options['bins'] is not 'auto':
        binning = [hist_min + float(i)*(hist_max-hist_min)/float(options['bins']) for i in range(int(options['bins'])+1)]

    # histogram data and mc
    h_data, h_bins = np.histogram(data, bins=binning, range=[hist_min,hist_max])
    h_mc, h_bins = np.histogram(mc, bins=h_bins, weights=mc_weights)
    bin_width = round(h_bins[1] - h_bins[0], 4)
    marker_size = 20*bin_width

    h_mc = h_mc*np.sum(h_data)/np.sum(h_mc)

    ratio = np.divide(h_data,h_mc)
    ratio[ratio==np.inf] = np.nan
    
    mids = [(h_bins[i]+h_bins[i+1])/2 for i in range(len(h_bins)-1)]
    mids_full = mids.copy()
    mids_full[0], mids_full[-1] = hist_min, hist_max
    x_err = (h_bins[1]-h_bins[0])/2

    # calculate errors
    y_err_data = np.sqrt(h_data)
    y_err_mc = helper_plots.get_bin_uncertainties(h_bins, mc, mc_weights)
    y_err_ratio = np.array([])

    mc_err_max = np.add(h_mc,y_err_mc)
    mc_err_min = np.subtract(h_mc,y_err_mc)

    # include syst uncertainties if they exist
    syst_unc = []
    if options['syst']:
        syst_unc = helper_plots.get_systematic_uncertainty(h_bins, data, data_up, data_down, mc, mc_up, mc_down, mc_weights)
        err = np.sqrt(np.add(np.power(y_err_mc, 2), np.power(syst_unc, 2)))
        mc_err_max, mc_err_min = np.add(h_mc, err), np.subtract(h_mc, err)
        y_err_ratio = np.divide(y_err_data, h_data)

    # define figure
    rows = 2
    if 'no_ratio' in options.keys(): rows = 1
    fig,axs = plt.subplots(nrows=rows, ncols=1)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.95, bottom=0.1,
            hspace=None if 'no_ratio' in options.keys() else 0.06)

    # top plot
    axs[0].set_xticklabels([])
    axs[0].fill_between(mids_full,h_mc,y2=0,step='mid',alpha=0.2,color='cornflowerblue') #fill mc area
    axs[0].fill_between(mids_full,mc_err_max,y2=mc_err_min,step='mid', alpha=0.3, color='red', label='mc stat. $\oplus$ syst. unc.') #fill mc error
    axs[0].errorbar(mids, h_mc, # plot mc
            xerr=x_err,
            label = 'mc', 
            color='cornflowerblue',
            drawstyle='steps-mid',
            capsize=0.,
            )
    axs[0].errorbar(mids, h_data, # plot data
            xerr=x_err, yerr=y_err_data, 
            label = 'data', 
            linestyle='None', 
            color='black',
            marker='o',
            markersize=marker_size, 
            capsize=0., 
            capthick=0.)
    axs[0].set_ylim(bottom=0)
    axs[0].set_xlim(hist_min-1, hist_max+1)
    axs[0].set_ylabel("Events/{x:.3f} GeV".format(x=bin_width), horizontalalignment='right',y=1., labelpad=5)

    # invert legend order because python is a hassle
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend( handles[::-1], labels[::-1], loc='best')

    # labels
    axs[0].annotate("$\\bf{CMS} \ \\it{Preliminary}$", 
            xy=(0,1.), xycoords='axes fraction', 
            ha='left', va='bottom')
    lumi = options['lumi'] if 'lumi' in options.keys() else 'XX.X fb$^{-1} (13 TeV) 20XX'
    axs[0].annotate(lumi, 
            xy=(1,1.), xycoords='axes fraction', 
            ha='right', va='bottom')
    
    # ratio pad
    if 'no_ratio' in options.keys():
        axs[0].set_xlabel('M$_{ee}$ [GeV]',horizontalalignment='right',x=1.)
    else:
        axs[1].plot(mids, [1. for x in mids], linestyle='dashed', color='black', alpha=0.5)

        # add syst+unc band to unity line
        if 'syst' in options.keys():
            err = np.sqrt(np.add(np.power(y_err_mc,2),np.power(syst_unc,2)))
            syst_err = np.divide(err,h_mc)
            for i in range(len(err)):
                print(err[i], h_mc[i], syst_err[i])
            axs[1].fill_between(mids_full, syst_err+1, 1-syst_err, step='mid', alpha=0.3, color='red', label='mc stat. $\oplus$ syst. unc.')
            
        axs[1].errorbar(mids, ratio, 
                xerr=x_err, yerr=y_err_ratio, 
                label='data / mc', 
                linestyle='None',
                color='black',
                marker='o',
                markersize=marker_size,
                capsize=0.,)

        # invert legend order because python is a hassle
        handles, labels = axs[1].get_legend_handles_labels()
        axs[1].legend(handles[::-1], labels[::-1], loc='best')
        axs[1].set_ylabel('data/mc',horizontalalignment='right', y=1.)
        axs[1].set_xlabel('M$_{ee}$ [GeV]',horizontalalignment='right', x=1.)
        axs[1].set_ylim(0.75, 1.25)
        axs[1].set_xlim(hist_min-1, hist_max+1)

    plt.grid(which='major',axis='both')
    # save fig
    fig.savefig(plot_dir+style+plot_title+".png")
    fig.savefig(plot_dir+style+plot_title+".pdf")

    plt.close(fig)

