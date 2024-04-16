import argparse as ap
import os
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
from python.plotters.plot_run_stability import plot_run_stability

from python.classes.config_class import SSConfig

def main():
    """
    Main function for the pymin.py script. This function will parse the command line arguments and call the appropriate
    functions to derive the scales and smearings. 
    --------------------------------
    Args:
        -i, --inputFile: path to file containing the data and mc paths
        --prune: option to prune down the data and mc to keep memory usage low
        --pruned-file-dest: destination for the pruned files
        --run-divide: option to make the run division file for time_stability
        --min-events: minimum number of events allowed in a given run bin
        -c, --cats: path to file describing categories to use in minimization
        --time-stability: scale data to PDG Z mass, pass the run divide file as your categories
        -s, --scales: path to scales file to apply to data
        --smearings: path to smearings file to apply to MC
        -w, --weights: tsv containing rapidity x ptz weights, if empty, they will be derived. It is recommended that these be derived just after deriving time stability (step1) corrections.
        -o, --output: output tag to add to file names
        --closure: derive the closure of the scales for a given step
        --ignore: list of categories to ignore for the current derivation
        --hist-min: Min of histogram for binned NLL evaluation
        --hist-max: Max of histogram for binned NLL evaluation
        --no-auto-bin: Turns off the auto binning feature (using Freedman-Diaconis method)
        --bin-size: Size of bins for binned NLL evaluation
        --start-style: Style of starting values for minimization
        --condor: submit the script to run on condor
        --queue: (only use with --condor) tell condor which queue to submit to
    --------------------------------
    Returns:
        None
    --------------------------------
    """

    ss_config = SSConfig() # initialize the config class
    # setup options
    parser = ap.ArgumentParser(description="Derivation of Scales and Smearings")

    parser.add_argument("-i","--inputFile", default=None,
                    	help="input file containing paths to data and mc")
    parser.add_argument("--prune", default=False, dest='_kPrune', action='store_true',
                    	help="prune down the data and mc to keep memory usage low")
    parser.add_argument("--pruned-file-dest", default=ss_config.DEFAULT_DATA_PATH, dest='pruned_file_dest',
                    	help="destination for the pruned files")

    parser.add_argument("--run-divide", default=False, dest='_kRunDivide', action='store_true',
                    	help="option to make the run division file for time_stability")
    parser.add_argument("--min-events", default=10000, dest='min_events',
                    	help="minimum number of events allowed in a given run bin")

    parser.add_argument("-c","--catsFile", default=None,
                    	help="path to file describing categories to use in minimization")
    parser.add_argument("--time-stability", default=False, dest="_kTimeStability", action='store_true',
                    	help="scale data to PDG Z mass, pass the run divide file as your categories")

    # options provided to the minimizer
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

    # additional use options
    parser.add_argument("--rewrite", default=False, action='store_true', dest='_kRewrite',
                    	help="only writes the scales file associated with this step")
    parser.add_argument("--combine-files", default=False, action='store_true', dest='_kCombine',
                    	help="[ADVANCED] combines two specified files, using the --only_step and --step options")
    parser.add_argument("--only-step", default='', type=str, dest='only_step',
                    	help="[ADVANCED] only step file, to be used with the --combine_files and --step options")

    # plotting options
    parser.add_argument("--plot", default=False, action='store_true', dest='_kPlot',
                    	help="Plot the invariant mass distributions for data and mc in the provided categories")
    parser.add_argument("--plot-dir", default='./', dest='plot_dir',
                    	help="directory to write the plots")
    parser.add_argument("--lumi-label", default=None, dest='lumi_label',
                        help="lumi label to add to plots, example: '35.9 fb^{-1} (13 TeV) 2016'")

    # advanced diagnostic options
    parser.add_argument("--test-method-accuracy", default=False, action='store_true', dest='_kTestMethodAccuracy',
                    	help="Treat MC as data, inject known scales into mc cats, see how well method recovers known scale injection")
    parser.add_argument("--scan-nll", default=False, action='store_true', dest='_kScanNLL',
                    	help="Scan the NLL phase space for a given set of categories. A set of scales in the 'onlystepX' format should be provided as the scan center")
    parser.add_argument("--scan-scales", default=None, dest='scan_scales',
                    	help="Scales defining the center of the scan [this is highly recommended when scanning the NLL phase space]")
    parser.add_argument("--debug", default=False, action='store_true', dest='_kDebug',
                        help="Turn on debug mode, which prints out additional information and uses a smaller dataset")

    args = parser.parse_args()
    print("[INFO] welcome to SS_PyMin")
    print("[INFO] you have run the following command:")

    cmd = helper_pymin.get_cmd()
    print(cmd)
    if "-" not in cmd:
        print("[ERROR] you have not provided any arguments to this script.")
        print("[ERROR] please resubmit this job with the correct arguments.")
        return

    if args._kClosure and args.smearings is None:
        if not args._kTestMethodAccuracy and not args._kRewrite:
            print("[ERROR] you have submitted a closure test without a smearings file.")
            print("[ERROR] please resubmit this job with a valid --smearings file.")
            return

    # submit this job to condor
    if args._kCondor and not args._kFromCondor:
        # remove the condor/queue information
        if cmd.find('--condor') != -1:
            cmd = cmd.replace("--condor ","")

        condor_handler.manage(cmd, args.output, args.queue)
        return

    # if you need to rewrite a scales file after making changes by hand
    #  you can use these options to do so.
    if args._kRewrite:
        helper_pymin.rewrite(args)
        return

    if args._kCombine:
        #TODO deprecated, fix this
        helper_pymin.combine_files(args)
        return

    # prune files to manage running memory usage
    if args._kPrune:
        if args.pruned_file_dest is None:
            print("[ERROR] you have requested a pruned file, but have not specified a destination")
            print("[ERROR] please resubmit with a --pruned_file_dest argument")
            return
        if args.output is None:
            print("[ERROR] you have requested a pruned file, but have not specified an output name")
            print("[ERROR] please resubmit with a --output argument")
            return
        if args.pruned_file_dest is None:
            print("[ERROR] you have requested a pruned file, but have not specified a destination")
            print("[ERROR] please resubmit with a --pruned_file_dest argument")
            return
        pruner.prune(args.inputFile, args.output, args.pruned_file_dest)
        return

    # load data/mc
    data, mc = helper_pymin.load_dataframes(args.inputFile, args)

    if data is None:
        print("[ERROR] data is None")
        return
    if mc is None:
        print("[ERROR] mc is None")
        return

    # derive time bins (by run) in which to stabilize data
    if args._kRunDivide:
        if args.output is None:
            print("[ERROR] you have requested a run divide, but have not specified an output name")
            print("[ERROR] please resubmit with a --output argument")
            return
        try:
            int(args.min_events)
        except:
            print("[ERROR] you have requested a run divide, but have not provided a valid minimum number of events")
            print("[ERROR] please resubmit with a valid --min_events argument")
            return
        outFile = "datFiles/run_divide_"+args.output+".dat"
        write_files.write_runs(divide_by_run.divide(data, args.min_events), outFile)
        return

    # derive the scales for the time stability
    if args._kTimeStability:
        if args.output is None:
            print("[ERROR] you have requested a time stability, but have not specified an output name")
            print("[ERROR] please resubmit with a --output argument")
            return
        if args.catsFile is None:
            print("[ERROR] you have requested a time stability, but have not specified a category")
            print("[ERROR] please resubmit with a --cats argument")
            return
        outFile = "datFiles/step1_"+args.output+"_scales.dat"
        ts_data, data_path = time_stability.derive(data, args.catsFile, args.output)
        write_files.write_time_stability(ts_data, args.catsFile, outFile)
        if args._kPlot:
            plot_run_stability(data_path, args.output, args.lumi_label, corrected=False)
            plot_run_stability(data_path, args.output, args.lumi_label, corrected=True)

        return

    # scale the data
    if args.scales and os.path.isfile(args.scales):
        print(f"[INFO] applying {args.scales} to the data")
        data = scale_data.scale(data, args.scales)
    else:
        print(f"[INFO] no scales file provided, skipping data scaling")

    # reweight MC or deriving the weights
    if not args._kNoReweight:
        weight_file = args.weights
        if args.weights is None:
            if args.output is None:
                print("[ERROR] you have requested a weight derivation, but have not specified an output name")
                print("[ERROR] please resubmit with a --output argument")
                return
            print("[INFO] deriving Y(Z), Pt(Z) weights")
            print(data.head())
            weight_file = reweight_pt_y.derive_pt_y_weights(data, mc, args.output)
            print(data.head())
        mc = reweight_pt_y.add_pt_y_weights(mc, weight_file)


    # load categories for the derivation
    if args.catsFile is None:
        print("[ERROR] you have not provided a category file")
        print("[ERROR] please resubmit with a --cats argument")
        return
    if not os.path.isfile(args.catsFile):
        print("[ERROR] you have provided a category file that does not exist")
        print("[ERROR] please resubmit with a valid --cats argument")
        return
    print(f"[INFO] importing categories from {args.catsFile}")
    cats_df = pd.read_csv(args.catsFile, sep="\t", comment="#", header=None)

    if args._kClosure:
        if not args._kTestMethodAccuracy:
            if args.smearings is None:
                print("[ERROR] you have requested a closure test, but have not provided a smearings file")
                print("[ERROR] please resubmit with a --smearings argument")
                return
            mc = smear_mc.smear(mc, args.smearings)

    # derive scales and smearings
    print("[INFO] initiating minimization using scipy.optimize.minimize")
    options = helper_pymin.get_options(args)
    scales_smears = minimizer.minimize(data, mc, cats_df, options)


    # if we're plotting there's nothing to write, so just print a done message and exit
    if args._kPlot:
        print("[INFO] plotting is done, please review")
        return

    # minimization may not succeed, in this case the program will just end
    if len(scales_smears) == 0:
        print("[ERROR] Review logs, minimization did not succeed")
        return

    # write the results
    file_written = helper_pymin.write_results(args, scales_smears)
    if not file_written:
        print("[ERROR] file not written, something went wrong")

    return


if __name__ == "__main__":
    main()
