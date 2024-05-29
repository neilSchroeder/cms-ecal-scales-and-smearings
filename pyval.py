#!/usr/bin/env python3

#built-in
import argparse as ap
import os
import sys
import time

#3rd party
import uproot3 as up
import numpy as np
import pandas as pd

#project functions
from python.helpers.helper_pyval import (
    extract_files,
    get_dataframe,
)
# from python.utilities.data_loader import get_dataframe
import python.utilities.reweight_pt_y as reweight_pt_y
import python.utilities.scale_data as scale_data
import python.utilities.smear_mc as smear_mc
import python.plotters.make_plots as make_plots
from python.utilities.evaluate_systematics import evaluate_systematics

from python.classes.constant_classes import(
    PyValConstants as pvc,
    DataConstants as dc
)
import python.classes.config_class as config_class
ss_config = config_class.SSConfig()


def main():
    """
    Main function for the scales and smearings validation and plotting.
    --------------------------------
    Args:
        -i, --in-file (str): path to input config file
        -o, --output-file (str): string used to create output files
        --data-title (str): title used in plots for the data
        --mc-title (str): title used in plots for the mc
        --lumi-label (str): luminosity label: i.e. 35 fb^{-1} (13 TeV) 2016
        --binning (int | str): Either number of bins, or 'auto' for automatic binning
        --systematics-study (flag): flag to dump the systematic uncertainty due to variations in R9, Et, and working point ID 
            [WARNING]: this is currently not implemented, it is a high priority TODO
        --fit (flag): flag to fit invariant mass distributions with BW conv. CB 
        --debug: flat to turn on debug mode.
    --------------------------------
    Returns:
        None
    --------------------------------       
    """

    #setup options
    parser = ap.ArgumentParser(description="Validation of Scales and Smearings")

    parser.add_argument(
            "-i","--in-file",
            help="input file",
            dest="input_file",
            default=None,
            )
    parser.add_argument(
            "-o","--output-file",
            help="string used to create output files",
            dest="output_file",
            default=None,
            )
    parser.add_argument(
            "--data-title",
            help="title used in plots for the data",
            dest="data_title",
            default=None,
            )
    parser.add_argument(
            "--mc-title",
            help="title used in plots for the mc",
            dest="mc_title",
            default=None,
            )
    parser.add_argument(
            "--lumi-label",
            dest="lumi_label",
            help="luminosity label: i.e. 35 fb^{-1} (13 TeV) 2016",
            default=None,
            )
    parser.add_argument(
            "--binning",
            dest="bins",
            help="Either number of bins, or 'auto' for automatic binning",
            default=80,
            )
    parser.add_argument(
            "--systematics-study",
            help="flag to dump the systematic uncertainty due to variations in R9, Et, and working point ID",
            default=False,
            dest="_kSystStudy",
            action="store_true",
            )
    parser.add_argument(
            "--no-reweight",
            help="turn off pt,Y reweighting",
            dest="no_reweight",
            default=False,
            action="store_true",
            )
    parser.add_argument(
            "--write",
            help="write the scaled data into csvs",
            dest="write_location",
            default=None,
            )
    parser.add_argument(
            "--fit",
            help="flag to turn on BW conv. CB fitting for all distributions",
            dest="_kFit",
            default=False,
            action="store_true",
    )
    parser.add_argument(
            "--debug",
            default=False,
            action="store_true",
            dest="_kDebug",
            help="turn on debug mode",
        )

    args = parser.parse_args()

    #greeting
    print(40*"#")
    print(40*"#")
    print("[INFO] Welcome to SS_PyVal")
    print("[INFO] You've provided the following options")
    for arg in vars(args):
        print(f'[INFO] {arg}: {getattr(args,arg)}')
    print(40*"#")
    print(40*"#")

    #open input file and prep our variables and such
    dict_config = extract_files(args.input_file) 

    #load and handle data first
    if len(dict_config[pvc.KEY_DAT]) > 0:
        print("[INFO] loading data")
        df_data = get_dataframe(dict_config[pvc.KEY_DAT],)
                                # apply_cuts='standard' if not args._kSystStudy else 'custom',
                                # eta_cuts=(0, dc.MAX_EB, dc.MIN_EE, dc.MAX_EE),
                                # debug = args._kDebug)
        if len(dict_config[pvc.KEY_SC]) > 0:
            print("[INFO] scaling data")
            df_data = scale_data.scale(df_data, dict_config[pvc.KEY_SC][0])

    #load and handle mc next
    if len(dict_config[pvc.KEY_MC]) > 0:
        print("[INFO] loading mc")
        df_mc = get_dataframe(dict_config[pvc.KEY_MC],)
                                # apply_cuts='standard' if not args._kSystStudy else 'custom',
                                # eta_cuts=(0, dc.MAX_EB, dc.MIN_EE, dc.MAX_EE),
                                # debug = args._kDebug)
        if len(dict_config[pvc.KEY_SM]) > 0:
            print("[INFO] smearing mc")
            df_mc = smear_mc.smear(df_mc, dict_config[pvc.KEY_SM][0])
        if len(dict_config[pvc.KEY_WT]) != 0:
            print("[INFO] reweighting mc")
            df_mc = reweight_pt_y.add_pt_y_weights(df_mc, dict_config[pvc.KEY_WT][0])
        else:
            if not args.no_reweight:
                #avoid this if you can, just to save time
                print("[INFO] deriving  pt,Y reweighting for mc")
                weight_file = reweight_pt_y.derive_pt_y_weights(df_data, df_mc, args.output_file)
                df_mc = reweight_pt_y.add_pt_y_weights(df_mc, weight_file)


    if args.write_location is not None:
        output_name = "/".join((args.write_location, args.output_file))
        if len(dict_config[pvc.KEY_DAT]) != 0: df_data.to_csv(output_name+"_data.tsv", sep='\t')
        if len(dict_config[pvc.KEY_MC]) != 0: df_mc.to_csv(output_name+"_mc.tsv", sep='\t')

    #plot the plots
    if not args._kSystStudy:
        make_plots.plot(df_data, df_mc, dict_config[pvc.KEY_CAT][0],
                data_title=args.data_title,
                mc_title=args.mc_title,
                lumi=args.lumi_label,
                bins=args.bins,
                tag=args.output_file,
                _kFit=args._kFit,
                )
    else:
        evaluate_systematics(df_data, df_mc, args.output_file)

    return

if __name__ == "__main__":
    main()
