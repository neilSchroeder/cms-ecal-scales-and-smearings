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
import pandas as pd
import sys

import python.helpers.helper_pymin as helper_pymin

import python.utilities.divide_by_run as divide_by_run
import python.utilities.minimizer as minimizer
import python.utilities.scale_data as scale_data
import python.utilities.smear_mc as smear_mc
import python.utilities.pruner as pruner
import python.utilities.time_stability as time_stability
import python.utilities.write_files as write_files
import python.utilities.condor_handler as condor_handler
import python.utilities.reweight_pt_y as reweight_pt_y

def main():
###############################################################################
    #setup options
    parser = ap.ArgumentParser(description="Derivation of Scales and Smearings")

    parser.add_argument("-i","--inputFile", default=None,
                    	help="input file containing paths to data and mc")

    parser.add_argument("--prune", default=False, dest='_kPrune', action='store_true',
                    	help="prune down the data and mc to keep memory usage low")
    parser.add_argument("--pruned-file-name", default=None, dest='pruned_file_name',
                    	help="basic string for naming the pruned files")
    parser.add_argument("--pruned-file-dest", default=None, dest='pruned_file_dest',
                    	help="destination for the pruned files, a directory in /eos/home-<initial>/<username>/ is recommended")

    parser.add_argument("--run-divide", default=False, dest='_kRunDivide', action='store_true',
                    	help="option to make the run division file for time_stability")
    parser.add_argument("--min-events", default=10000, dest='min_events',
                    	help="minimum number of events allowed in a given run bin")

    parser.add_argument("-c","--cats", default=None,
                    	help="path to file describing categories to use in minimization")
    parser.add_argument("--time-stability", default=False, dest="_kTimeStability", action='store_true',
                    	help="scale data to PDG Z mass, pass the run divide file as your categories")

    #options provided to the minimizer
    parser.add_argument("-s","--scales", default=None,
                    	help="path to scales file to apply to data")
    parser.add_argument("--smearings", default=None,
                        help="path to smearings file to apply to MC")
    parser.add_argument("-w", "--weights", default=None,
                        help="tsv containing rapidity x ptz weights, if empty, they will be derived. It is recommended that these be derived just after deriving time stability (step1) corrections.")
    parser.add_argument("-o","--output", default=None,
                    	help="output tag to add to file names")
    parser.add_argument("--closure", default=False, action='store_true', dest='_kClosure',
                    	help="derive the closure of the scales for a given step")
    parser.add_argument("--ignore", default=None,
                    	help="list of categories to ignore for the current derivation")
    parser.add_argument("--hist-min", default=80, type=float, dest='hist_min',
                    	help="Min of histogram for binned NLL evaluation")
    parser.add_argument("--hist-max", default=100, type=float, dest='hist_max',
                    	help="Max of histogram for binned NLL evaluation")
    parser.add_argument("--no-auto-bin", default=False, action='store_true', dest='_kNoAutoBin',
                    	help="Turns off the auto binning feature (using Freedman-Diaconis method)")
    parser.add_argument("--bin-size", default=0.25, type=float, dest='bin_size',
                    	help="Size of bins for binned NLL evaluation")
    parser.add_argument("--start-style", default='scan', type=str, dest='start_style',
                    	help="Determines how the minimizer chooses its starting location. Allowed options are 'scan', 'random', 'specify'. If using specify, used the scan_scales argument to provide the starting location")
    parser.add_argument("--scan-min", default=0.98, type=float, dest='scan_min',
                    	help="Min value for scan start")
    parser.add_argument("--scan-max", default=1.02, type=float, dest='scan_max',
                    	help="Max value for scan start")
    parser.add_argument("--scan-step", default=0.001, type=float, dest='scan_step',
                    	help="Step size for scan start")
    parser.add_argument("--min-step-size", default=None, dest='min_step_size',
                        help="Min step size for scipy.optimize.minimize function. This is an advanced option, please use with care.")
    parser.add_argument("--fix-scales", default=False, action='store_true', dest='_kFixScales',
                        help="[ADVANCED] flag to keep scales fixed at 1. and only derive smearings")
    parser.add_argument("--no-reweight", default=False, action='store_true', dest='_kNoReweight',
                        help="[ADVANCED] flag to turn off deriving pt(Z),y(Z) weights")
    parser.add_argument("--condor", default=False, action='store_true', dest='_kCondor',
                        help="Submit this job to condor")
    parser.add_argument("--queue", default='tomorrow',
                        help="flavour of submitted job")
    parser.add_argument("--from-condor", default=False, action='store_true', dest='_kFromCondor',
                        help="[NOT FOR USER] flag added by condor_handler to indicate this has been submitted from condor")

    #additional use options
    parser.add_argument("--rewrite", default=False, action='store_true', dest='_kRewrite',
                    	help="only writes the scales file associated with this step")
    parser.add_argument("--combine-files", default=False, action='store_true', dest='_kCombine',
                    	help="[ADVANCED] combines two specified files, using the --only_step and --step options")
    parser.add_argument("--only-step", default='', type=str, dest='only_step',
                    	help="[ADVANCED] only step file, to be used with the --combine_files and --step options")

    #plotting options
    parser.add_argument("--plot", default=False, action='store_true', dest='_kPlot',
                    	help="Plot the invariant mass distributions for data and mc in the provided categories")
    parser.add_argument("--plot-dir", default='./', dest='plot_dir',
                    	help="directory to write the plots")

    #advanced diagnostic options
    parser.add_argument("--test-method-accuracy", default=False, action='store_true', dest='_kTestMethodAccuracy',
                    	help="Treat MC as data, inject known scales into mc cats, see how well method recovers known scale injection")
    parser.add_argument("--scan-nll", default=False, action='store_true', dest='_kScanNLL',
                    	help="Scan the NLL phase space for a given set of categories. A set of scales in the 'onlystepX' format should be provided as the scan center")
    parser.add_argument("--scan-scales", default=None, dest='scan_scales',
                    	help="Scales defining the center of the scan [this is highly recommended when scanning the NLL phase space]")

    args = parser.parse_args()
    print("[INFO] welcome to SS_PyMin")
    print("[INFO] you have run the following command:")

    cmd = helper_pymin.get_cmd(sys.argv)
    print(cmd)

    if args._kClosure and args.smearings is None:
        if args._kTestMethodAccuracy:
            pass
        elif args._kRewrite:
            pass
        else:
            print("[ERROR] you have submitted a closure test without a smearings file.")
            print("[ERROR] please resubmit this job with a smearings file.")
            return

    step = helper_pymin.get_step(args)

    #submit this job to condor
    if args._kCondor and not args._kFromCondor:
        #remove the condor/queue information
        if cmd.find('--condor') != -1:
            cmd = cmd.replace("--condor ","")

        condor_handler.manage(cmd, args.output, args.queue)
        return

    #if you need to rewrite a scales file after making changes by hand
    # you can use these options to do so.
    if args._kRewrite:
        helper_pymin.rewrite(args)
        return

    if args._kCombine:
        helper_pymin.combine_files(args)
        return

    #prune files to manage running memory usage
    if args._kPrune:
        pruner.prune(args.inputFile, args.pruned_file_name, args.pruned_file_dest)
        return

    #load data/mc
    data, mc = helper_pymin.load_dataframes(args.inputFile, args)

    if data is None or mc is None:
        print("[ERROR] data or MC is empty")
        return

    #derive time bins (by run) in which to stabilize data
    if args._kRunDivide:
        outFile = "datFiles/run_divide_"+args.output+".dat"
        write_files.write_runs(divide_by_run.divide(data, args.min_events), outFile)
        return

    #derive the scales for the time stability
    if args._kTimeStability:
        outFile = "datFiles/step1_"+args.output+"_scales.dat"
        write_files.write_time_stability(time_stability.derive(data, args.cats), args.cats, outFile)
        return

    #scale the data
    if args.scales is not None:
        print("[INFO] applying {} to the data".format(args.scales))
        data = scale_data.scale(data, args.scales)

    #reweight MC or deriving the weights
    if args._kNoReweight:
        pass #don't reweight
    else:
        weight_file = args.weights
        if args.weights is None:
            print("[INFO] deriving Y(Z), Pt(Z) weights")
            weight_file = reweight_pt_y.derive_pt_y_weights(data, mc, args.output)
        mc = reweight_pt_y.add_pt_y_weights(mc, weight_file)


    #load categories for the derivation
    print("[INFO] importing categories from {}".format(args.cats))
    cats = pd.read_csv(args.cats, sep="\t", comment="#", header=None)
    num_scales = sum([val.find('scale') != -1 for val in cats[0]])

    if args._kClosure:
        if args._kTestMethodAccuracy:
            pass
        else:
            mc = smear_mc.smear(mc, args.smearings)

    #derive scales and smearings
    print("[INFO] initiating minimization using scipy.optimize.minimize")
    scales_smears = minimizer.minimize(data, mc, cats, args)


    #if we're plotting there's nothing to write, so just print a done message and exit
    if args._kPlot:
        print("[INFO] plotting is done, please review")
        return

    #minimization may not succeed, in this case the program will just end
    if scales_smears == []:
        print("[ERROR] Review code, minimization did not succeed")
        return

    #write the results
    print(scales_smears)
    helper_pymin.write_results(args, scales_smears)

    return

if __name__ == "__main__":
    main()
