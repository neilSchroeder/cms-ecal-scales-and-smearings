"""Script to evaluate the systematics of the analysis."""

import numpy as np

from python.classes.constant_classes import PyValConstants as pvc
from python.classes.constant_classes import DataConstants as dc
from python.classes.config_class import SSConfig
global_config = SSConfig()

from python.helpers.helper_pyval import extract_files
from python.utilities.data_loader import (
    get_dataframe,
    custom_cuts,
)
from python.plotters.fit_bw_cb import fit_bw_cb


def evaluate_systematics(data, mc, outfile):
    """
    Evaluate the systematics of the analysis.
    
    The study is performed by varying the R9, Et, and working point ID.
    The agreement between data and MC is calculated as the ratio of mu_data/mu_mc,
        where mu is the invariant mass peak obtained by fitting a histogram with
        a Breit-Wigner convoluted with a Crystal Ball function.
    The systematic uncertainty from a variation is taken as 
        1 - (mu_data_variation/mu_mc_variation)/(mu_data_nominal/mu_mc_nominal).

    Args:
        data (str): path to the data file
        mc (str): path to the mc file
        outfile (str): path to the output file
    Returns:
        None
    """

    data_df = get_dataframe(


    cuts = {
        "EBin": {
            "HighR9": {
                "eta_cuts": ((-1, 1), (-1, 1)),
                "r9_cuts": ((dc.R9_boundary, -1), (dc.R9_boundary, -1)),
                "et_cuts": (dc.MIN_PT_LEAD, dc.MIN_PT_SUB),
            },
            "LowR9": {
                "eta_cuts": ((-1, 1), (-1, 1)),
                "r9_cuts": ((-1, dc.R9_boundary), (-1, dc.R9_boundary)),
                "et_cuts": (dc.MIN_PT_LEAD, dc.MIN_PT_SUB),
            }
        },
        "EBout": {
            "HighR9": {
                "eta_cuts": ((1, 1.4442), (1, 1.4442)),
                "r9_cuts": ((dc.R9_boundary, -1), (dc.R9_boundary, -1)),
                "et_cuts": (dc.MIN_PT_LEAD, dc.MIN_PT_SUB),
            },
            "LowR9": {
                "eta_cuts": ((-1, 1), (-1, 1)),
                "r9_cuts": ((-1, dc.R9_boundary), (-1, dc.R9_boundary)),
                "et_cuts": (dc.MIN_PT_LEAD, dc.MIN_PT_SUB),
            }
        },
        "EEin": {
            "HighR9": {
                "eta_cuts": ((1.566, 2), (1.566, 2)),
                "r9_cuts": ((dc.R9_boundary, -1), (dc.R9_boundary, -1)),
                "et_cuts": (dc.MIN_PT_LEAD, dc.MIN_PT_SUB),
            },
            "LowR9": {
                "eta_cuts": ((1.566, 2), (1.566, 2)),
                "r9_cuts": ((-1, dc.R9_boundary), (-1, dc.R9_boundary)),
                "et_cuts": (dc.MIN_PT_LEAD, dc.MIN_PT_SUB),
            }
        },
        "EEout": {
            "HighR9": {
                "eta_cuts": ((2, 2.5), (2,2.5)),
                "r9_cuts": ((dc.R9_boundary, -1), (dc.R9_boundary, -1)),
                "et_cuts": (dc.MIN_PT_LEAD, dc.MIN_PT_SUB),
            },
            "LowR9": {
                "eta_cuts": ((2, 2.5), (2,2.5)),
                "r9_cuts": ((-1, dc.R9_boundary), (-1, dc.R9_boundary)),
                "et_cuts": (dc.MIN_PT_LEAD, dc.MIN_PT_SUB),
            }
        },
    }

    mids = [80.125 + 0.25*i for i in range(80)]

    nominal_hists = {
        eta_key: {r9_key: np.histogram(
                        custom_cuts(base_df, **cuts[eta_key][r9_key])[dc.INVMASS].values
                        )[0] 
                   for r9_key in cuts[eta_key]}
            for eta_key in cuts.keys()
    }
    
    r9_up_hists = {
        eta_key: {r9_key: np.histogram(
                        custom_cuts(base_df, 
                                    eta_cuts = cuts[eta_key][r9_key]["eta_cuts"],
                                    r9_cuts = ((0.965, -1), (0.965, -1)) if r9_key == "HighR9" else ((-1, 0.965), (-1, 0.965)),
                                    et_cuts=cuts[eta_key][r9_key]["et_cuts"],
                                    )[dc.INVMASS].values
                        )[0] 
                   for r9_key in cuts[eta_key]}
            for eta_key in cuts.keys()
    }
    r9_down_hists = {
        eta_key: {r9_key: np.histogram(
                        custom_cuts(base_df, 
                                    eta_cuts = cuts[eta_key][r9_key]["eta_cuts"],
                                    r9_cuts = ((0.955, -1), (0.955, -1)) if r9_key == "HighR9" else ((-1, 0.935), (-1, 0.935)),
                                    et_cuts=cuts[eta_key][r9_key]["et_cuts"],
                                    )[dc.INVMASS].values
                        )[0] 
                   for r9_key in cuts[eta_key]}
            for eta_key in cuts.keys()
    }
    et_up_hists = {
        eta_key: {r9_key: np.histogram(
                        custom_cuts(base_df, 
                                    eta_cuts = cuts[eta_key][r9_key]["eta_cuts"],
                                    r9_cuts = cuts[eta_key][r9_key]["r9_cuts"],
                                    et_cuts=(dc.MIN_PT_LEAD+2, dc.MIN_PT_SUB),
                                    )[dc.INVMASS].values
                        )[0] 
                   for r9_key in cuts[eta_key]}
            for eta_key in cuts.keys()
    }
    et_down_hists = {
        eta_key: {r9_key: np.histogram(
                        custom_cuts(base_df, 
                                    eta_cuts = cuts[eta_key][r9_key]["eta_cuts"],
                                    r9_cuts = cuts[eta_key][r9_key]["r9_cuts"],
                                    et_cuts=(dc.MIN_PT_LEAD-2, dc.MIN_PT_SUB),
                                    )[dc.INVMASS].values
                        )[0]
                   for r9_key in cuts[eta_key]}
            for eta_key in cuts.keys()
    }
    medium_id_hists = {
        eta_key: {r9_key: np.histogram(
                        custom_cuts(base_df, 
                                    eta_cuts = cuts[eta_key][r9_key]["eta_cuts"],
                                    r9_cuts = cuts[eta_key][r9_key]["r9_cuts"],
                                    et_cuts=cuts[eta_key][r9_key]["et_cuts"],
                                    working_point="medium",
                                    )[dc.INVMASS].values
                        )[0]
                   for r9_key in cuts[eta_key]}
            for eta_key in cuts.keys()
    }
    tight_id_hists = {
        eta_key: {r9_key: np.histogram(
                        custom_cuts(base_df, 
                                    eta_cuts = cuts[eta_key][r9_key]["eta_cuts"],
                                    r9_cuts = cuts[eta_key][r9_key]["r9_cuts"],
                                    et_cuts=cuts[eta_key][r9_key]["et_cuts"],
                                    working_point="tight",
                                    )[dc.INVMASS].values
                        )[0] 
                   for r9_key in cuts[eta_key]}
            for eta_key in cuts.keys()
    }

    # we need histograms with the invariant mass for each variation
    