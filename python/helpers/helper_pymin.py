"""
helper functions for optimize.py
"""

import os
import sys

import pandas as pd
import numpy as np
import uproot as up

from python.classes.constant_classes import DataConstants as dc
from python.classes.config_class import SSConfig
import python.utilities.write_files as write_files

def get_options(args):
    """ deletes the options that are not needed for the current step """
    ret = vars(args)
    del ret['weights']
    return ret


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

    if args.catsFile is not None and not args._kTimeStability:
        if "." in args.catsFile.split("_")[1]:
            return int(args.catsFile.split("_")[1].split(".")[0][-1])
        return int(args.catsFile.split("_")[1][-1])

    return -1

def rewrite(args):
    # rewrites a set of scales/smearings files
    step = get_step(args)
    file_name_only_step = args.only_step
    only_step_path = os.path.dirname(file_name_only_step)

    file_name_scales = f"{only_step_path}/step{step}closure_{args.output}_scales.dat" if args._kClosure else f"{only_step_path}/step{step}_{args.output}_scales.dat"

    #if not args._kClosure: write_files.rewrite_smearings(args.catsFile, new_smears)
    write_files.combine( file_name_only_step, args.scales, file_name_scales)


def load_dataframes(files, args):

    data = None
    mc = None

    root_files = open(files, 'r').readlines()
    root_files = [x.strip() for x in root_files]

    #import data and mc to dataframes
    print("[INFO] importing data and mc to dataframes (this might take a bit) ...")

    data_types = {
            dc.R9_LEAD: np.float32,
            dc.R9_SUB: np.float32,
            dc.ETA_LEAD: np.float32,
            dc.ETA_SUB: np.float32,
            dc.E_LEAD: np.float32,
            dc.E_SUB: np.float32,
            dc.PHI_LEAD: np.float32,
            dc.PHI_SUB: np.float32,
            dc.INVMASS: np.float32,
            dc.RUN: np.int32,
            dc.GAIN_LEAD: np.int16,
            dc.GAIN_SUB: np.int16,
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

    data[dc.ETA_LEAD] = np.abs(data[dc.ETA_LEAD])
    data[dc.ETA_SUB] = np.abs(data[dc.ETA_SUB])

    transition_mask_lead = ~data[dc.ETA_LEAD].between(dc.MAX_EB, dc.MIN_EE)
    transition_mask_sub = ~data[dc.ETA_SUB].between(dc.MAX_EB, dc.MIN_EE)
    tracker_mask_lead = ~data[dc.ETA_LEAD].between(dc.MAX_EE, dc.TRACK_MAX)
    tracker_mask_sub = ~data[dc.ETA_SUB].between(dc.MAX_EE, dc.TRACK_MAX)
    invmass_mask = data[dc.INVMASS].between(dc.invmass_min, dc.invmass_max)
    mask = transition_mask_lead&transition_mask_sub&tracker_mask_lead&tracker_mask_sub&invmass_mask
    data = data.loc[mask]

    mc[dc.ETA_LEAD] = np.abs(mc[dc.ETA_LEAD])
    mc[dc.ETA_SUB] = np.abs(mc[dc.ETA_SUB])

    transition_mask_lead = ~mc[dc.ETA_LEAD].between(dc.MAX_EB, dc.MIN_EE)
    transition_mask_sub = ~mc[dc.ETA_SUB].between(dc.MAX_EB, dc.MIN_EE)
    tracker_mask_lead = ~mc[dc.ETA_LEAD].between(dc.MAX_EE, dc.TRACK_MAX)
    tracker_mask_sub = ~mc[dc.ETA_SUB].between(dc.MAX_EE, dc.TRACK_MAX)
    invmass_mask = mc[dc.INVMASS].between(dc.invmass_min, dc.invmass_max)
    mask = transition_mask_lead&transition_mask_sub&tracker_mask_lead&tracker_mask_sub&invmass_mask
    mc = mc.loc[mask]

    return data, mc

def write_results(args, scales_smears):
    try:
        step = get_step(args)
        cats = pd.read_csv(args.catsFile, sep="\t", comment="#", header=None)

        tag = args.output                                   
        this_dir = f"{os.getcwd()}/condor/{tag}/" if args._kFromCondor else SSConfig.DEFAULT_WRITE_FILES_PATH
        file_name =  f"{this_dir}/step{step}_{tag}" if not args._kClosure else f"{this_dir}/step{step}closure_{tag}"
        only_step_file_name =  f"{this_dir}/onlystep{step}_{tag}" if not args._kClosure else f"{this_dir}/onlystep{step}closure_{tag}"

        new_scales = f"{only_step_file_name}_scales.dat"
        new_smears = f"{file_name}_smearings.dat"
        scales_out = f"{file_name}_scales.dat"

        write_files.write_scales(scales_smears, cats, new_scales)
        if not args._kClosure: write_files.write_smearings(scales_smears, cats, new_smears)

        #make scales file here
        print("[INFO] creating new scales file: {}".format(scales_out))
        write_files.combine( new_scales, args.scales, scales_out )
        return True
    except:
        return False
