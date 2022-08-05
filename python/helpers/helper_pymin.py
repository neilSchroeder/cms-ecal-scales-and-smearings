"""
helper functions for optimize.py
"""

import os
import sys

import pandas as pd
import numpy as np
import uproot as up

import python.classes.const_class as constants
import python.utilities.write_files as write_files

def get_cmd(args):
    #reconstructs and returns the command line string used to run the program

    cmd = ''
    for arg in sys.argv:
        if ' ' in arg:
            cmd += '"{}"  '.format(arg)
        else:
            cmd +="{}  ".format(arg)

    return cmd

def get_step(args):
    #gets the step number from the category file

    if args.cats is not None and not args._kTimeStability:
        if "." in args.cats.split("_")[1]:
            return int(args.cats.split("_")[1].split(".")[0][-1])
        return int(args.cats.split("_")[1][-1])

    return -1

def rewrite(args):
    # rewrites a set of scales/smearings files
    step = get_step(args)
    file_name_only_step = args.only_step
    only_step_path = os.path.dirname(file_name_only_step)

    file_name_scales = f"{only_step_path}/step{step}closure_{args.output}_scales.dat" if args._kClosure else f"{only_step_path}/step{step}_{args.output}_scales.dat"

    #if not args._kClosure: write_files.rewrite_smearings(args.cats, new_smears)
    write_files.combine( file_name_only_step, args.scales, file_name_scales)


def load_dataframes(files, args):

    data = None
    mc = None

    root_files = open(files, 'r').readlines()
    root_files = [x.strip() for x in root_files]

    #import data and mc to dataframes
    print("[INFO] importing data and mc to dataframes (this might take a bit) ...")

    c = constants.const()

    data_types = {
            c.R9_LEAD: np.float32,
            c.R9_SUB: np.float32,
            c.ETA_LEAD: np.float32,
            c.ETA_SUB: np.float32,
            c.E_LEAD: np.float32,
            c.E_SUB: np.float32,
            c.PHI_LEAD: np.float32,
            c.PHI_SUB: np.float32,
            c.INVMASS: np.float32,
            c.RUN: np.int32,
            c.GAIN_LEAD: np.int16,
            c.GAIN_SUB: np.int16,
            }

    if root_files[0].find("data") != -1:
        data = pd.read_csv(root_files[0], sep='\t',dtype=data_types)
        mc = pd.read_csv(root_files[1], sep='\t',dtype=data_types)
    elif root_files[1].find("data") != -1:
        data = pd.read_csv(root_files[1], sep='\t',dtype=data_types)
        mc = pd.read_csv(root_files[0], sep='\t',dtype=data_types)
    else:
        print("[ERROR] could not find a data file to open")
        return
    if args._kTestMethodAccuracy:
        data = mc.copy()

    #clean the data a bit before sending back

    data[c.ETA_LEAD] = np.abs(data[c.ETA_LEAD])
    data[c.ETA_SUB] = np.abs(data[c.ETA_SUB])

    transition_mask_lead = ~data[c.ETA_LEAD].between(c.MAX_EB,c.MIN_EE)
    transition_mask_sub = ~data[c.ETA_SUB].between(c.MAX_EB,c.MIN_EE)
    tracker_mask_lead = ~data[c.ETA_LEAD].between(c.MAX_EE, c.TRACK_MAX)
    tracker_mask_sub = ~data[c.ETA_SUB].between(c.MAX_EE, c.TRACK_MAX)
    invmass_mask = data[c.INVMASS].between(c.invmass_min, c.invmass_max)
    mask = transition_mask_lead&transition_mask_sub&tracker_mask_lead&tracker_mask_sub&invmass_mask
    data = data.loc[mask]

    mc[c.ETA_LEAD] = np.abs(mc[c.ETA_LEAD])
    mc[c.ETA_SUB] = np.abs(mc[c.ETA_SUB])

    transition_mask_lead = ~mc[c.ETA_LEAD].between(c.MAX_EB,c.MIN_EE)
    transition_mask_sub = ~mc[c.ETA_SUB].between(c.MAX_EB,c.MIN_EE)
    tracker_mask_lead = ~mc[c.ETA_LEAD].between(c.MAX_EE, c.TRACK_MAX)
    tracker_mask_sub = ~mc[c.ETA_SUB].between(c.MAX_EE, c.TRACK_MAX)
    invmass_mask = mc[c.INVMASS].between(c.invmass_min, c.invmass_max)
    mask = transition_mask_lead&transition_mask_sub&tracker_mask_lead&tracker_mask_sub&invmass_mask
    mc = mc.loc[mask]

    return data, mc

def write_results(args, scales_smears):
    step = get_step(args)
    cats = pd.read_csv(args.cats, sep="\t", comment="#", header=None)

    tag = args.output                                   
    this_dir = f"{os.getcwd()}/condor/{tag}/" if args._kFromCondor else os.getcwd()     
    file_name =  f"{this_dir}/step{step}_{tag}" if not args._kClosure else f"{this_dir}/step{step}closure_{tag}"
    only_step_file_name =  f"{this_dir}/onlystep{step}_{tag}" if not args._kClosure else f"{this_dir}/onlystep{step}closure_{tag}"

    new_scales = f"{only_step_file_name}_scales.dat"
    new_smears = f"{file_name}_smearings.dat"
    scales_out = f"{file_name}_scales.dat"

    write_files.write_scales(scales_smears, cats, new_scales)
    if not args._kClosure: write_files.write_smearings(scales_smears, cats, new_smears)
###############################################################################

###############################################################################
    #make scales file here
    print("[INFO] creating new scales file: {}".format(scales_out))
    write_files.combine( new_scales, args.scales, scales_out )
###############################################################################
