"""
helper functions for optimize.py
"""

import os
import sys

import pandas as pd
import numpy as np
import uproot as up

import python.classes.const_class as constants

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
        return int(args.cats[args.cats.find("step")+4])

    return -1

def rewrite(args):
    #rewrites a set of scales/smearings files

    scales_out = os.getcwd()+"/datFiles/step"+str(step)+"_"+args.output+"_scales.dat"
    if args.scales != '':
        scales_out = os.path.dirname(args.scales)+"/step"+str(step)+"_"+args.output+"_scales.dat"
    if args.from_condor:
        scales_out = os.getcwd()+"/condor/"+args.output+"/step"+str(step)+"_"+args.output+"_scales.dat"
    if args.closure: scales_out = scales_out.replace("step"+str(step), "step"+str(step)+"closure",1)
    else: scales_out = scales_out.replace("step"+str(step-1),"step"+str(step),1)
    new_scales = scales_out.replace("step", "onlystep")
    new_smears = os.path.dirname(scales_out)+"/"+os.path.basename(scales_out).replace("scales", "smearings")
    if not args.closure: write_files.rewrite_smearings(args.cats, new_smears)
    write_files.combine( new_scales, args.scales, scales_out )

def combine_files(args):
    #combines an onlystep file and a step file

    scales_out = os.getcwd()+"/datFiles/step"+str(step)+"_"+args.output+"_scales.dat"
    if args.scales != '':
        scales_out = os.path.dirname(args.scales)+"/step"+str(step)+"_"+args.output+"_scales.dat"
    if args.from_condor:
        scales_out = os.getcwd()+"/condor/"+args.output+"/step"+str(step)+"_"+args.output+"_scales.dat"
    if args.closure: scales_out = scales_out.replace("step"+str(step), "step"+str(step)+"closure",1)
    else: scales_out = scales_out.replace("step"+str(step-1),"step"+str(step),1)
    write_files.combine( args.only_step, args.scales, scales_out )

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

def write_results(args, scales_smears, unc):
    step = get_step(args)
    scale_path = args.scales if args.scales is not None else os.getcwd()+"/blah.dat"
    scales_out = os.path.dirname(scales_path)+"/step"+str(step)+"_"+args.output+"_scales.dat"
    if args.scales != '':
        scales_out = os.path.dirname(scales_path)+"/step"+str(step)+"_"+args.output+"_scales.dat"
    if args.from_condor:
        scales_out = os.getcwd()+"/condor/"+args.output+"/step"+str(step)+"_"+args.output+"_scales.dat"
    if args.closure: scales_out = scales_out.replace("step"+str(step), "step"+str(step)+"closure",1)
    else: scales_out = scales_out.replace("step"+str(step-1),"step"+str(step),1)

    new_scales = os.path.dirname(scales_out)+"/"+os.path.basename(scales_out).replace("step", "onlystep")
    new_smears = os.path.dirname(scales_out)+"/"+os.path.basename(scales_out).replace("scales", "smearings")

    scales_smears = [(scales_smears[i], unc[i]) for i,x in enumerate(unc)]
    write_files.write_scales(scales_smears[:num_scales], cats, new_scales)
    if not args.closure: write_files.write_smearings(scales_smears, cats, new_smears)
###############################################################################

###############################################################################
    #make scales file here
    print("[INFO] creating new scales file: {}".format(scales_out))
    write_files.combine( new_scales, args.scales, scales_out )
###############################################################################
