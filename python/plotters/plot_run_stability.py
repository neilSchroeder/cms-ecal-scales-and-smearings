"""
Plots the run stability for the dielectron mass
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from python.classes.config_class import SSConfig
ss_config = SSConfig()

def main():
    parser = argparse.ArgumentParser(description='Plot the RunFineEtaR9 scales')
    parser.add_argument('-i', '--inputFile',
                        help='input dat file')
    parser.add_argument('-o', '--outputFile',
                        help='output pdf file')
    parser.add_argument('--lumi-label', default='59.7 fb^{-1} (13 TeV) 2018', dest='lumi_label',
                        help='Luminosity label: format is "XX.X fb^{-1} (13 TeV) YEAR"')
    parser.add_argument('--corrected', action='store_true', default=False,
                        help='Use corrected values')

    args = parser.parse_args()

    data = np.genfromtxt(args.inputFile, dtype=float, delimiter='\t', comments='#')

    eta_lowEdges = np.unique(np.array([data[i][2] for i in range(data.shape[0])]))
    eta_highEdges = np.unique(np.array([data[i][3] for i in range(data.shape[0])]))
    eta_bins = eta_lowEdges.tolist()
    eta_bins.append(eta_highEdges[-1])

    scales = []
    unc = []
    r9vals = []
    for i in range(len(eta_bins)-1):
        temp_scales = []
        temp_unc = []
        temp_r9 = []
        for row in data:
            if row[2] == eta_bins[i]:
                if args.corrected:
                    temp_scales.append(row[8])
                    temp_unc.append(row[10]/np.sqrt(row[-1]))
                else:
                    temp_scales.append(row[4])
                    temp_unc.append(row[6]/np.sqrt(row[-1]))
                temp_r9.append(row[0:2])

        scales.append(temp_scales)
        unc.append(temp_unc)
        r9vals.append(temp_r9)


    for i in range(len(eta_bins)-1):
        mids = np.array([(x[0]+x[1])/2 for x in r9vals[i]])
        xloerr = [mids[j] - r9vals[i][j][0] for j in range(len(mids))]
        xhierr = [r9vals[i][j][1] - mids[j] for j in range(len(mids))]
        myarray = np.array([xloerr,xhierr])
        plt.errorbar(mids,
                     scales[i],
                     yerr=unc[i],
                     xerr=myarray,
                     fmt='--o',
                     label=f"|$\eta$| between {round(eta_bins[i],4)} and {round(eta_bins[i+1],4)}",
                     linestyle='',
                     mew=0,
                     capsize=0,
                     elinewidth=0)

    plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.09)
    plt.ylim(80, 100)
    plt.text(plt.xlim()[0], plt.ylim()[1]+0.1, "$\\bf{CMS} \ \\it{Preliminary}$", va='bottom')
    plt.text(plt.xlim()[1], plt.ylim()[1]+0.1, args.lumi_label , ha='right', va='bottom')
    plt.xlabel('Run Number', horizontalalignment='right',x=1.0)
    plt.ylabel('Median Dielectron Mass [GeV]', horizontalalignment='right',y=1.0)
    plt.legend(loc='best')
    corrected_tag = "corrected" if args.corrected else "uncorrected"
    file_name = f"{ss_config.DEFAULT_PLOT_PATH}/run_stability_{args.outputFile}_{corrected_tag}.pdf"
    plt.savefig(file_name)
    print(f"Saved plot to {file_name}")
    file_name = f"{ss_config.DEFAULT_PLOT_PATH}/run_stability_{args.outputFile}.png"
    plt.savefig(file_name)
    print(f"Saved plot to {file_name}")

main()

