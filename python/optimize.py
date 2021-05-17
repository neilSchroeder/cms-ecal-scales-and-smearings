#!/usr/bin/env python3
"""
This function will derive the residual scales and additional smearings.

Data and MC should be taken from the ECALELF step-0 outputs.

Usage/Order of Operations:
    prune: start by supplying a list of data and mc files you'll be using for your derivation.
        the prune step will read in these files, join them, remove unnecessary columns,
        and write them to a csv in a directory you specify (/eos/home-<initial>/<username> is recommended)
    run_divide: this step will produce the "time bins" or bins of runs containing a minimum number of events.
        this step is vital as it will be used in the next step, and all future steps, to stabilize the data
        over time.
    time_stability: this step will produce the scales required per run bin to align the data with the pdg Z mass.
        this stabilizes the data as a function of time and eta.
    standard use: once the previous steps have been completed you can now provide the data and mc on which you will 
        derive the scales and smearings, the scales which will be applied to the data prior to derivation, and the 
        categories in which you wish to derive the scales and smearings. The result will automatically be written to
        a table in the datFiles/ directory with the appropriate name.

"""

import argparse as ap
import gc
import numpy as np
import os
import pandas as pd
import sys
import uproot as up

import divide_by_run 
import nll_wClass
import scale_data
import scale_data_fast
import pruner
import time_stability
import write_files
import condor_handler
import reweight_pt_y

def main():
###############################################################################
    #setup options
    parser = ap.ArgumentParser(description="Derivation of Scales and Smearings")

    parser.add_argument("-i","--inputFile", required=True, 
                    	help="input file containing paths to data and mc")
    parser.add_argument("-s","--scales",
                    	help="path to scales file to apply to data")
    parser.add_argument("--smearings", default=None,
                        help="path to smearings file to apply to MC")
    parser.add_argument("-c","--cats", 
                    	help="path to file describing categories to use in minimization")
    parser.add_argument("-w", "--weights", default='',
                        help="tsv containing rapidity x ptz weights, if empty, they will be derived. It is recommended that these be derived just after deriving time stability (step1) corrections.")
    parser.add_argument("-o","--output", default='',
                    	help="output tag to add to file names")
    parser.add_argument("--ingore", default='',
                    	help="list of categories to ignore for the current derivation")
    parser.add_argument("--hist_min", default=80, type=float,
                    	help="Min of histogram for binned NLL evaluation")
    parser.add_argument("--hist_max", default=100, type=float,
                    	help="Max of histogram for binned NLL evaluation")
    parser.add_argument("--start_style", default='scan', type=str,
                    	help="Determines how the minimizer chooses its starting location. Allowed options are 'scan', 'random', 'specify'. If using specify, used the scan_scales argument to provide the starting location")
    parser.add_argument("--scan_min", default=0.98, type=float,
                    	help="Min value for scan start")
    parser.add_argument("--scan_max", default=1.02, type=float,
                    	help="Max value for scan start")
    parser.add_argument("--scan_step", default=0.001, type=float,
                    	help="Step size for scan start")
    parser.add_argument("--min_step_size", default=None, 
                        help="Min step size for scipy.optimize.minimize function. This is an advanced option, please use with care.")
    parser.add_argument("--prune", default=False, action='store_true',
                    	help="prune down the data and mc to keep memory usage low")
    parser.add_argument("--pruned_file_name", 
                    	help="basic string for naming the pruned files")
    parser.add_argument("--pruned_file_dest",
                    	help="destination for the pruned files, a directory in /eos/home-<initial>/<username>/ is recommended")
    parser.add_argument("--run_divide", default=False, action='store_true',
                    	help="option to make the run division file for time_stability")
    parser.add_argument("--minEvents", default=10000,
                    	help="minimum number of events allowed in a given run bin")
    parser.add_argument("--time_stability", default=False, action='store_true',
                    	help="scale data to PDG Z mass, pass the run divide file as your categories")
    parser.add_argument("--closure", default=False, action='store_true',
                    	help="derive the closure of the scales for a given step")
    parser.add_argument("--rewrite", default=False, action='store_true',
                    	help="only writes the scales file associated with this step")
    parser.add_argument("--plot", default=False, action='store_true',
                    	help="Plot the invariant mass distributions for data and mc in the provided categories")
    parser.add_argument("--plot_dir", default='./',
                    	help="directory to write the plots")
    parser.add_argument("--test_method_accuracy", default=False, action='store_true',
                    	help="Treat MC as data, inject known scales into mc cats, see how well method recovers known scale injection")
    parser.add_argument("--scan_nll", default=False, action='store_true',
                    	help="Scan the NLL phase space for a given set of categories. A set of scales in the 'onlystepX' format should be provided as the scan center")
    parser.add_argument("--scan_scales", default='', 
                    	help="Scales defining the center of the scan [this is highly recommended when scanning the NLL phase space]")
    parser.add_argument("--no_auto_bin", default=False, action='store_true',
                    	help="Turns off the auto binning feature (using Freedman-Diaconis method)")
    parser.add_argument("--bin_size", default=0.25, type=float,
                    	help="Size of bins for binned NLL evaluation")
    parser.add_argument("--condor", default=False, action='store_true',
                        help="Submit this job to condor")
    parser.add_argument("--queue", default='tomorrow',
                        help="flavour of submitted job")
    parser.add_argument("--from_condor", default=False, action='store_true',
                        help="[NOT FOR USER] flag added by condor_handler to indicate this has been submitted from condor")
    parser.add_argument("--fix_scales", default=False, action='store_true',
                        help="[ADVANCED] flag to keep scales fixed at 1. and only derive smearings")
    parser.add_argument("--combine_files", default=False, action='store_true',
                    	help="[ADVANCED] combines two specified files, using the --only_step and --step options")
    parser.add_argument("--only_step", default='', type=str,
                    	help="[ADVANCED] only step file, to be used with the --combine_files and --step options")

    args = parser.parse_args()
    print("[INFO] welcome to SS_PyMin")
    print("[INFO] you have run the following command:")

    step = -1
    if args.cats is not None: 
        step = int(args.cats[args.cats.find("step")+4])

###############################################################################

###############################################################################
    #submit this job to condor
    cmd = ''
    for arg in sys.argv:
        if ' ' in arg:
            cmd += '"{}"  '.format(arg)
        else:
            cmd +="{}  ".format(arg)
    print(cmd) 
    if args.condor and not args.from_condor:
        #remove the condor/queue information
        if cmd.find('--condor') != -1:
            cmd = cmd.replace("--condor ","")

        condor_handler.manage(cmd, args.output, args.queue) 
        return

###############################################################################

###############################################################################
    #if you need to rewrite a scales file after making changes by hand
    # you can use this option to do so.
    if args.rewrite:
        scales_out = os.getcwd()+"/datFiles/step"+str(step)+"_"+args.output+"_scales.dat"
        if args.scales != '':
            scales_out = os.path.dirname(args.scales)+"/step"+str(step)+"_"+args.output+"_scales.dat"
        if args.from_condor:
            scales_out = os.getcwd()+"/condor/"+args.output+"/step"+str(step)+"_"+args.output+"_scales.dat"
        if args.closure: scales_out = scales_out.replace("step"+str(step), "step"+str(step)+"closure",1)
        else: scales_out = scales_out.replace("step"+str(step-1),"step"+str(step),1)
        new_scales = scales_out.replace("step", "onlystep")
        new_smears = os.path.dirname(scales_out)+"/"+os.path.basename(scales_out).replace("scales", "smearings")
        write_files.rewrite_smearings(args.cats, new_smears)
        write_files.combine( new_scales, args.scales, scales_out )
        return

    if args.combine_files:
        scales_out = os.getcwd()+"/datFiles/step"+str(step)+"_"+args.output+"_scales.dat"
        if args.scales != '':
            scales_out = os.path.dirname(args.scales)+"/step"+str(step)+"_"+args.output+"_scales.dat"
        if args.from_condor:
            scales_out = os.getcwd()+"/condor/"+args.output+"/step"+str(step)+"_"+args.output+"_scales.dat"
        if args.closure: scales_out = scales_out.replace("step"+str(step), "step"+str(step)+"closure",1)
        else: scales_out = scales_out.replace("step"+str(step-1),"step"+str(step),1)
        write_files.combine( args.only_step, args.scales, scales_out )
        return

###############################################################################

###############################################################################
    #prune files to manage running memory usage
    if args.prune:
        pruner.prune(args.inputFile, args.pruned_file_name, args.pruned_file_dest)
        return

###############################################################################

###############################################################################

    #load data/mc
    data = pd.DataFrame()
    mc = pd.DataFrame()

    root_files = open(args.inputFile, 'r').readlines()
    root_files = [x.strip() for x in root_files]

    #import data and mc to dataframes
    print("[INFO] importing data and mc to dataframes (this might take a bit) ...")
    if root_files[0].find("data") != -1:
        data = pd.read_csv(root_files[0], sep='\t')
        mc = pd.read_csv(root_files[1], sep='\t')
    elif root_files[1].find("data") != -1:
        data = pd.read_csv(root_files[1], sep='\t')
        mc = pd.read_csv(root_files[0], sep='\t')
    else:
        print("[ERROR] could not find a data file to open")
        return
    if args.test_method_accuracy:
        data = mc.copy()

    data['etaEle[0]'] = np.absolute(data['etaEle[0]'].values)
    data['etaEle[1]'] = np.absolute(data['etaEle[1]'].values)
    mc['etaEle[0]'] = np.absolute(mc['etaEle[0]'].values)
    mc['etaEle[1]'] = np.absolute(mc['etaEle[1]'].values)
    

###############################################################################
    #derive time bins (by run) in which to stabilize data
    if args.run_divide:
        outFile = "datFiles/run_divide_"+args.output+".dat"
        write_files.write_runs(divide_by_run.divide(data, args.minEvents), outFile)
        return

###############################################################################
    #derive the scales for the time stability
    if args.time_stability:
        outFile = "datFiles/step1_"+args.output+"_scales.dat"
        write_files.write_time_stability(time_stability.derive(data, args.cats), args.cats, outFile)
        return

###############################################################################
    #scale the data
    if args.scales:
        print("[INFO] applying {} to the data".format(args.scales))
        data = scale_data_fast.scale(data, args.scales)
        gc.collect()

    weight_file = args.weights
    if args.weights == '':
        print("[INFO] deriving Y(Z), Pt(Z) weights")
        weight_file = reweight_pt_y.derive_pt_y_weights(data, mc, args.output)
    mc = reweight_pt_y.add_pt_y_weights(mc, weight_file)


###############################################################################
    #load categories for the derivation
    print("[INFO] importing categories from {}".format(args.cats))
    cats = pd.read_csv(args.cats, sep="\t", comment="#", header=None)
    num_scales = sum([val.find('scale') != -1 for val in cats[0]])

###############################################################################
    #derive scales and smearings
    print("[INFO] initiating minimization using scipy.optimize.minimize")
    scales_smears = nll_wClass.minimize(data, mc, cats, args.ingore, args.smearings,
                                 round(float(args.hist_min),2), round(float(args.hist_max),2), round(float(args.bin_size),2),
                                 args.start_style,
                                 args.scan_min, args.scan_max, args.scan_step,
                                 args.min_step_size,
                                 args.closure, args.scales, 
                                 args.plot, args.plot_dir,
                                 args.test_method_accuracy,
                                 args.scan_nll, args.scan_scales,
                                 args.fix_scales,
                                 not args.no_auto_bin)
    if args.plot:
        print("[INFO] plotting is done, please review")
        return
    if scales_smears == []:
        print("[ERROR] Review code, minimization did not succeed") 
        return

###############################################################################
    #write the onlystepX file and smearings file

    scales_out = os.path.dirname(args.scales)+"/step"+str(step)+"_"+args.output+"_scales.dat"
    if args.scales != '':
        scales_out = os.path.dirname(args.scales)+"/step"+str(step)+"_"+args.output+"_scales.dat"
    if args.from_condor:
        scales_out = os.getcwd()+"/condor/"+args.output+"/step"+str(step)+"_"+args.output+"_scales.dat"
    if args.closure: scales_out = scales_out.replace("step"+str(step), "step"+str(step)+"closure",1)
    else: scales_out = scales_out.replace("step"+str(step-1),"step"+str(step),1)

    new_scales = os.path.dirname(scales_out)+"/"+os.path.basename(scales_out).replace("step", "onlystep")
    new_smears = os.path.dirname(scales_out)+"/"+os.path.basename(scales_out).replace("scales", "smearings")

    write_files.write_scales(scales_smears[:num_scales], cats, new_scales)
    if not args.closure: write_files.write_smearings(scales_smears, cats, new_smears)
###############################################################################

###############################################################################
    #make scales file here
    print("[INFO] creating new scales file: {}".format(scales_out))
    write_files.combine( new_scales, args.scales, scales_out )
###############################################################################

main()
