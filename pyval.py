#!/usr/bin/env python3

#built-in
import argparse as ap
import os
import sys
import time

#3rd party
import uproot as up
import numpy as np
import pandas as pd

#project functions
import python.helpers.helper_pyval as helper_pyval
import python.utilities.reweight_mc as reweight_mc
import python.utilities.scale_data as scale_data
import python.utilities.smear_mc_pyval as smear_mc
import python.plotters.make_plots as make_plots

"""
This code will validate the scales and smearings obtained through the "cms-ecal-scales-and-smearings" 

Usage:

"""


def main():

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
            default=None,
            )
    parser.add_argument(
            "--mc-title",
            help="title used in plots for the mc",
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
            default=None,
            )
    parser.add_argument(
            "--systematics-study",
            help="flag to dump the systematic uncertainty due to variations in R9, Et, and working point ID",
            default=False,
            )
    parser.add_argument(
            "--no-reweight",
            help="turn off pt,Y reweighting",
            dest="reweight",
            default=False,
            action="store_true",
            )
    parser.add_argument(
            "--write",
            help="write the scaled data into csvs",
            dest="write_location",
            default=None,
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
    KEY_DAT = "DATA"
    KEY_MC = "MC"
    KEY_SC = "SCALES"
    KEY_SM = "SMEARINGS"
    KEY_WT = "WEIGHTS"
    KEY_CAT = "CATS"
    dict_config = helper_pyval.extract_files(args.input_file) 

    #load and handle data first
    if len(dict_config[KEY_DAT]) > 0:
        print("[INFO] loading data")
        df_data = helper_pyval.get_dataframe(dict_config[KEY_DAT])
        if len(dict_config[KEY_SC]) > 0:
            print("[INFO] scaling data")
            df_data = scale_data.scale(df_data, dict_config[KEY_SC][0])
        df_data = helper_pyval.standard_cuts(df_data)

    #load and handle mc next
    if len(dict_config[KEY_MC]) > 0:
        print("[INFO] loading mc")
        df_mc = helper_pyval.get_dataframe(dict_config[KEY_MC])
        print(max(df_mc['invMass_ECAL_ele'].values))
        if len(dict_config[KEY_SM]) > 0:
            print("[INFO] smearing mc")
            df_mc = smear_mc.smear(df_mc, dict_config[KEY_SM][0])
        df_mc = helper_pyval.standard_cuts(df_mc)
        if len(dict_config[KEY_WT]) != 0:
            print("[INFO] reweighting mc")
            df_mc = reweight_mc.add_pt_y_weights(df_mc, dict_config[KEY_WT][0])
        else:
            if args.reweight:
                pass
            else:
                #avoid this if you can, just to save time
                print("[INFO] deriving  pt,Y reweighting for mc")
                weight_file = reweight_mc.derive_pt_y_weights(df_data, df_mc, args.output_file)
                df_mc = reweight_mc.add_pt_y_weights(df_mc, weight_file)


    if args.write_location is not None:
        output_name = "/".join((args.write_location, args.output_file))
        if len(dict_config[KEY_DAT]) != 0: df_data.to_csv(output_name+"_data.tsv", sep='\t')
        if len(dict_config[KEY_MC]) != 0: df_mc.to_csv(output_name+"_mc.tsv", sep='\t')

    #plot the plots
    make_plots.plot(df_data, df_mc, dict_config[KEY_CAT][0],
            lumi=args.lumi_label,
            bins=args.bins,
            tag=args.output_file,
            )

    return

if __name__ == "__main__":
    main()
