import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import rc
import matplotlib

def main():
    parser = argparse.ArgumentParser(description='Plot the RunFineEtaR9 scales')
    parser.add_argument('inputFile',
                        help='input dat file')

    args = parser.parse_args()

    data = np.genfromtxt(args.inputFile, dtype=float, delimiter='\t', comments='#')

    eta_lowEdges = np.unique(np.array([data[i][2] for i in range(data.shape[0])]))
    eta_highEdges = np.unique(np.array([data[i][3] for i in range(data.shape[0])]))
    r9_lowEdges = np.unique(np.array([data[i][0] for i in range(data.shape[0])]))
    r9_highEdges = np.unique(np.array([data[i][1] for i in range(data.shape[0])]))


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
                temp_scales.append(row[8])
                temp_unc.append(row[10]/np.sqrt(row[-1]))
                #temp_scales.append(row[4])
                #temp_unc.append(row[6]/np.sqrt(row[-1]))
                temp_r9.append(row[0:2])

        scales.append(temp_scales)
        unc.append(temp_unc)
        r9vals.append(temp_r9)


    for i,row in enumerate(scales):
        if i < 4:
            mids = np.array([(x[0]+x[1])/2 for x in r9vals[i]])
            xloerr = [mids[j] - r9vals[i][j][0] for j in range(len(mids))]
            xhierr = [r9vals[i][j][1] - mids[j] for j in range(len(mids))]
            myarray = np.array([xloerr,xhierr])
            plt.errorbar(mids,
                         row,
                         yerr=unc[i],
                         xerr=myarray,
                         fmt='--o',
                         label="|$\eta$| between {} and {}".format(round(eta_bins[i],4), round(eta_bins[i+1],4)),
                         linestyle='',
                         mew=0,
                         capsize=0,
                         elinewidth=0)

    plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.09)
    plt.ylim(86, 93)
    plt.text(plt.xlim()[0], 93.1, "$\\bf{CMS} \ \\it{Preliminary}$")
    plt.text(plt.xlim()[1], 93.1, "59.7 fb$^{-1}$ (13 TeV) 2018",horizontalalignment='right')
    plt.xlabel('Run Number', horizontalalignment='right',x=1.0)
    plt.ylabel('Median Dielectron Mass [GeV]', horizontalalignment='right',y=1.0)
    plt.legend(loc='best')
    plt.show()

main()

