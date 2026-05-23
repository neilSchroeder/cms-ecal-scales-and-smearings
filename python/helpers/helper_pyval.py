import pandas as pd
import numpy as np
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
        print(
            "[WARNING] write location not specified, scaled and smeared csvs will not be saved"
        )
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

    df = pd.read_csv(filename, sep="\t", header=None, comment="#")

    ret_dict = {}
    ret_dict["DATA"] = []
    ret_dict["MC"] = []
    ret_dict["SCALES"] = []
    ret_dict["SMEARINGS"] = []
    ret_dict["WEIGHTS"] = []
    ret_dict["CATS"] = []

    for i, row in df.iterrows():
        if os.path.exists(row[1]):
            ret_dict[row[0]].append(row[1])
        else:
            print(f"[ERROR] file does not exist {row[1]}")
            raise RuntimeError

    return ret_dict
