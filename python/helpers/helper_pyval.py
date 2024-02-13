import pandas as pd
import numpy as np
import uproot3 as up
import os

from python.classes.constant_classes import PyValConstants as pvc
from python.classes.constant_classes import DataConstants as dc
import python.classes.config_class as config_class
ss_config = config_class.SSConfig()

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
