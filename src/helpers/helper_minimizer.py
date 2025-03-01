import gc
import numpy as np
import pandas as pd
import time

from src.classes.constant_classes import DataConstants as dc
from src.classes.constant_classes import CategoryConstants as cc
from src.classes.zcat_class import zcat
from src.tools.data_loader import add_transverse_energy


def deactivate_cats(__ZCATS__, ignore_cats):
    """
    Deactivate categories that are in the ignore_cats file. This is rarely necessary.

    Args:
        __ZCATS__ (list): list of zcat objects, each representing a dielectron category
        ignore_cats (str): path to the ignore_cats file
    Returns:
        None
    """
    if ignore_cats is not None:
        df_ignore = pd.read_csv(ignore_cats, sep="\t", header=None)
        for cat in __ZCATS__:
            for row in df_ignore.iterrows():
                if (
                    row[cc.i_type] == cat.lead_index
                    and row[cc.i_eta_min] == cat.sublead_index
                ):
                    cat.valid = False
