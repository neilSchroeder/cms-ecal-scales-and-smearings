import pandas as pd
import numpy as np
import uproot3 as up
import os

from python.classes.constant_classes import PyValConstants as pvc
from python.classes.constant_classes import DataConstants as dc
import python.classes.config_class as config_class
ss_config = config_class.SSConfig()

def check_args(args):
    """
    Check args for consistency.
    
    Args:
        args: parsed cmd line args from pyval run command
    Returns:
        None
    """

    if args.input_file is None:
        print("[ERROR] input file not specified")
        raise ValueError("input file not specified")
    if os.path.exists(args.input_file) == False:
        print(f"[ERROR] input file {args.input_file} does not exist")
        raise FileNotFoundError(f"input file {args.input_file} does not exist")
    
    if args.output_file is None:
        print("[ERROR] output file not specified")
        raise ValueError("output file not specified")
    if not isinstance(args.output_file, str):
        print("[ERROR] output file must be a string")
        raise ValueError("output file must be a string")
    
    if args.data_title is None:
        print("[ERROR] data title not specified")
        raise ValueError("data title not specified")        
    if not isinstance(args.data_title, str):
        print("[ERROR] data title must be a string")
        raise ValueError("data title must be a string")
    
    if args.mc_title is None:
        print("[ERROR] mc title not specified")
        raise ValueError("mc title not specified")
    if not isinstance(args.mc_title, str):
        print("[ERROR] mc title must be a string")
        raise ValueError("mc title must be a string")
    
    if args.lumi_label is None:
        print("[WARNING] lumi label not specified")
    if args.lumi_label and not isinstance(args.lumi_label, str):
        print("[ERROR] lumi label must be a string")
        raise ValueError("lumi label must be a string")
    
    if args.bins is None:
        print("[ERROR] binning not specified")
        raise ValueError("binning not specified")
    if args.bins and not isinstance(args.bins, int) and args.bins != "auto":
        print("[ERROR] binning must be an integer or 'auto'")
        raise ValueError("binning must be an integer or 'auto'")
    
    if args.write_location is None:
        print("[WARNING] write location not specified, scaled and smeared csvs will not be saved")
    if args.write_location and not isinstance(args.write_location, str):
        print("[ERROR] write location must be a string")
        raise ValueError("write location must be a string")
    
    if args._kPlotFit and not args._kFit:
        print("[ERROR] cannot plot fit without fitting")
        raise ValueError("cannot plot fit without fitting")
    
    if args.no_reweight:
        print("[WARNING] no reweighting flag set")


def extract_files(filename):
    """
    Extract files to use from a config file.

    Args:
        filename (str): the name of the config file
    Returns:
        ret_dict (dict): a dictionary of lists of files to use
    """

    df = pd.read_csv(filename, sep='\t', header=None, comment="#")

    ret_dict = {}
    ret_dict["DATA"] = []
    ret_dict["MC"] = []
    ret_dict["SCALES"] = []
    ret_dict["SMEARINGS"] = []
    ret_dict["WEIGHTS"] = []
    ret_dict["CATS"] = []

    for i,row in df.iterrows():
        if os.path.exists(row[1]): 
            ret_dict[row[0]].append(row[1])
        else:
            print(f'[ERROR] file does not exist {row[1]}')
            raise RuntimeError

    return ret_dict

def get_dataframe(files, debug=False):
    """
    Loads root files into a pandas dataframe.

    Args:
        files (list): a list of files to load
        debug (bool): whether to use a smaller dataset for debugging
    Returns:
        df (pandas dataframe): the dataframe containing the data
    """

    df = pd.DataFrame()

    if ".root" in files[0]:
        #this takes a long time, so avoid it if possible
        df = pd.concat([up.open(f)[pvc.TREE_NAME].pandas.df(pvc.KEEP_COLS) for f in files])
        #drop unnecessary columns
        drop_list = ['R9Ele[2]', 'energy_ECAL_ele[2]', 'etaEle[2]', 'gainSeedSC[2]', 'phiEle[2]', 'eleID[2]']
        df.drop(drop_list, axis=1, inplace=True)
    elif ".csv" in files[0] or ".tsv" in files[0]:
        df = pd.concat([pd.read_csv(f, sep='\t',dtype=dc.DATA_TYPES) for f in files])
    else:
        print("[python][helpers][helper_main] ERROR: file type not recognized")
        raise ValueError("file type not recognized: must be .root, .csv, or .tsv")
    
    if debug:
        # use a smaller dataset for debugging
        df = df.head(100000)


    #clean the data a bit before sending back

    df[dc.ETA_LEAD] = np.abs(df[dc.ETA_LEAD])
    df[dc.ETA_SUB] = np.abs(df[dc.ETA_SUB])
    
    transition_mask_lead = ~df[dc.ETA_LEAD].between(dc.MAX_EB,dc.MIN_EE)
    transition_mask_sub = ~df[dc.ETA_SUB].between(dc.MAX_EB,dc.MIN_EE)
    tracker_mask_lead = ~df[dc.ETA_LEAD].between(dc.MAX_EE, dc.TRACK_MAX)
    tracker_mask_sub = ~df[dc.ETA_SUB].between(dc.MAX_EE, dc.TRACK_MAX)
    invmass_mask = df[dc.INVMASS].between(dc.invmass_min, dc.invmass_max)
    mask = transition_mask_lead&transition_mask_sub&tracker_mask_lead&tracker_mask_sub&invmass_mask
    df = df.loc[mask]

    return df

def standard_cuts(df):
    """
    Takes in a dataframe and applies the following cuts:
    pt_lead > 32 GeV
    pt_sublead > 20 GeV
    80 GeV < invMass < 100 GeV
    |eta| < 2.5 and !(1.4442 < |eta| < 1.566)
    """

    #masks
    mask_lead = (np.divide(df[dc.E_LEAD].values, np.cosh(df[dc.ETA_LEAD].values))) >= dc.MIN_PT_LEAD
    mask_sub = (np.divide(df[dc.E_SUB].values, np.cosh(df[dc.ETA_SUB].values))) >= dc.MIN_PT_SUB

    mask_lead = np.logical_and(mask_lead,np.logical_or(df[dc.ETA_LEAD].values < dc.MAX_EB, dc.MIN_EE < df[dc.ETA_LEAD].values))
    mask_lead = np.logical_and(mask_lead, df[dc.ETA_LEAD].values < dc.MAX_EE)

    mask_sub = np.logical_and(mask_sub,np.logical_or(df[dc.ETA_SUB].values < dc.MAX_EB, dc.MIN_EE < df[dc.ETA_SUB].values))
    mask_sub = np.logical_and(mask_sub, df[dc.ETA_SUB].values < dc.MAX_EE)

    mask_invmass = np.logical_and(dc.MIN_INVMASS <= df[dc.INVMASS].values, df[dc.INVMASS].values <= dc.MAX_INVMASS)

    mask = np.logical_and(mask_lead,mask_sub)
    mask = np.logical_and(mask, mask_invmass)

    return df[mask]


def custom_cuts(df, custom_cuts):
    """
    Takes in a dataframe and applies the cuts specified in custom_cuts.

    Args:
        df (pandas dataframe): the dataframe to cut
        custom_cuts (list): a list of cuts to apply
    Returns:
        df (pandas dataframe): the dataframe with the cuts applied
    """

    for cut in custom_cuts:
        df = df.query(cut)

    return df
