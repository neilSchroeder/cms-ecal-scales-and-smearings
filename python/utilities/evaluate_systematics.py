"""Script to evaluate the systematics of the analysis."""

import numpy as np

from python.classes.constant_classes import PyValConstants as pvc
from python.classes.constant_classes import DataConstants as dc
from python.classes.config_class import SSConfig
global_config = SSConfig()

from python.plotters.fit_bw_cb import fit_bw_cb
from python.utilities.data_loader import (
    custom_cuts,
)
from python.utilities.write_files import write_systematics


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
    print(f"[INFO][evaluate_systematics] Evaluating systematics")
    cuts = dc.SYST_CUTS
    num_bins = 80

    mids = [80.125 + 0.25*i for i in range(80)]

    nominal_hists = {
        eta_key: {
            r9_key: [
                np.histogram(
                    custom_cuts(data, **cuts[eta_key][r9_key])[dc.INVMASS].values, bins=num_bins
                )[0],
                np.histogram(
                    custom_cuts(mc, **cuts[eta_key][r9_key])[dc.INVMASS].values, bins=num_bins
                )[0]
            ]
            for r9_key in cuts[eta_key]
        }
        for eta_key in cuts.keys()
    }
    
    r9_up_hists = {
        eta_key: {
            r9_key: [
                np.histogram(
                    custom_cuts(
                        data, 
                        eta_cuts = cuts[eta_key][r9_key]["eta_cuts"],
                        r9_cuts = ((0.965, -1), (0.965, -1)) if r9_key == "HighR9" else ((-1, 0.965), (-1, 0.965)),
                        et_cuts=cuts[eta_key][r9_key]["et_cuts"],
                    )[dc.INVMASS].values, bins=num_bins
                )[0],
                np.histogram(
                    custom_cuts(
                        mc,
                        eta_cuts = cuts[eta_key][r9_key]["eta_cuts"],
                        r9_cuts = ((0.965, -1), (0.965, -1)) if r9_key == "HighR9" else ((-1, 0.965), (-1, 0.965)),
                        et_cuts=cuts[eta_key][r9_key]["et_cuts"]
                    )[dc.INVMASS].values, bins=num_bins
                )[0]
            ] 
            for r9_key in cuts[eta_key]
        }
        for eta_key in cuts.keys()
    }
    r9_down_hists = {
        eta_key: {r9_key: [np.histogram(
                        custom_cuts(data, 
                                    eta_cuts = cuts[eta_key][r9_key]["eta_cuts"],
                                    r9_cuts = ((0.955, -1), (0.955, -1)) if r9_key == "HighR9" else ((-1, 0.935), (-1, 0.935)),
                                    et_cuts=cuts[eta_key][r9_key]["et_cuts"],
                                    )[dc.INVMASS].values, bins=num_bins
                        )[0],
                        np.histogram(custom_cuts(mc, 
                                    eta_cuts = cuts[eta_key][r9_key]["eta_cuts"],
                                    r9_cuts = ((0.955, -1), (0.955, -1)) if r9_key == "HighR9" else ((-1, 0.935), (-1, 0.935)),
                                    et_cuts=cuts[eta_key][r9_key]["et_cuts"],
                                    )[dc.INVMASS].values, bins=num_bins
                        )[0]]
                    for r9_key in cuts[eta_key]}
            for eta_key in cuts.keys()
    }
    et_up_hists = {
        eta_key: {r9_key: [np.histogram(
                        custom_cuts(data, 
                                    eta_cuts = cuts[eta_key][r9_key]["eta_cuts"],
                                    r9_cuts = cuts[eta_key][r9_key]["r9_cuts"],
                                    et_cuts=(dc.MIN_PT_LEAD+2, dc.MIN_PT_SUB),
                                    )[dc.INVMASS].values, bins=num_bins
                        )[0],
                        np.histogram(
                        custom_cuts(mc, 
                                    eta_cuts = cuts[eta_key][r9_key]["eta_cuts"],
                                    r9_cuts = cuts[eta_key][r9_key]["r9_cuts"],
                                    et_cuts=(dc.MIN_PT_LEAD+2, dc.MIN_PT_SUB),
                                    )[dc.INVMASS].values, bins=num_bins
                        )[0]] 
                    for r9_key in cuts[eta_key]}
            for eta_key in cuts.keys()
    }
    et_down_hists = {
        eta_key: {r9_key: [np.histogram(
                        custom_cuts(data, 
                                    eta_cuts = cuts[eta_key][r9_key]["eta_cuts"],
                                    r9_cuts = cuts[eta_key][r9_key]["r9_cuts"],
                                    et_cuts=(dc.MIN_PT_LEAD-2, dc.MIN_PT_SUB),
                                    )[dc.INVMASS].values, bins=num_bins
                        )[0],
                        np.histogram(
                        custom_cuts(mc, 
                                    eta_cuts = cuts[eta_key][r9_key]["eta_cuts"],
                                    r9_cuts = cuts[eta_key][r9_key]["r9_cuts"],
                                    et_cuts=(dc.MIN_PT_LEAD-2, dc.MIN_PT_SUB),
                                    )[dc.INVMASS].values, bins=num_bins
                        )[0]]
                    for r9_key in cuts[eta_key]}
            for eta_key in cuts.keys()
    }
    medium_id_hists = {
        eta_key: {r9_key: [np.histogram(
                        custom_cuts(data, 
                                    eta_cuts = cuts[eta_key][r9_key]["eta_cuts"],
                                    r9_cuts = cuts[eta_key][r9_key]["r9_cuts"],
                                    et_cuts=cuts[eta_key][r9_key]["et_cuts"],
                                    working_point="medium",
                                    )[dc.INVMASS].values, bins=num_bins
                        )[0],
                        np.histogram(
                        custom_cuts(mc, 
                                    eta_cuts = cuts[eta_key][r9_key]["eta_cuts"],
                                    r9_cuts = cuts[eta_key][r9_key]["r9_cuts"],
                                    et_cuts=cuts[eta_key][r9_key]["et_cuts"],
                                    working_point="medium",
                                    )[dc.INVMASS].values, bins=num_bins
                        )[0]]
                    for r9_key in cuts[eta_key]}
            for eta_key in cuts.keys()
    }
    tight_id_hists = {
        eta_key: {r9_key: [np.histogram(
                        custom_cuts(data, 
                                    eta_cuts = cuts[eta_key][r9_key]["eta_cuts"],
                                    r9_cuts = cuts[eta_key][r9_key]["r9_cuts"],
                                    et_cuts=cuts[eta_key][r9_key]["et_cuts"],
                                    working_point="tight",
                                    )[dc.INVMASS].values, bins=num_bins
                        )[0],
                        np.histogram(
                        custom_cuts(mc, 
                                    eta_cuts = cuts[eta_key][r9_key]["eta_cuts"],
                                    r9_cuts = cuts[eta_key][r9_key]["r9_cuts"],
                                    et_cuts=cuts[eta_key][r9_key]["et_cuts"],
                                    working_point="tight",
                                    )[dc.INVMASS].values, bins=num_bins
                        )[0]]
                    for r9_key in cuts[eta_key]}
            for eta_key in cuts.keys()
    }

    # add all hists to a list for easier looping
    hists = [
        nominal_hists,
        r9_down_hists,
        r9_up_hists,
        et_down_hists,
        et_up_hists,
        medium_id_hists,
        tight_id_hists
    ]

    # now we can fit and take a double ratio for each
    # variation
    for i, hist in enumerate(hists):
        for eta_key in hist:
            for r9_key in hist[eta_key]:
                print(f"{i= } | {eta_key= } | {r9_key= }")
                # check for zero weights
                if sum(hist[eta_key][r9_key][0]) == 0 or sum(hist[eta_key][r9_key][1]) == 0:
                    print(f"Zero weights for {eta_key} {r9_key}")
                    print(hist[eta_key][r9_key][0])
                    print(hist[eta_key][r9_key][1])
                mean_data = np.average(mids, weights=hist[eta_key][r9_key][0])
                mean_mc = np.average(mids, weights=hist[eta_key][r9_key][1])
                sigma_data = np.sqrt(np.average((mids-mean_data)**2, weights=hist[eta_key][r9_key][0]))
                sigma_mc = np.sqrt(np.average((mids-mean_mc)**2, weights=hist[eta_key][r9_key][1]))
                data_fit = fit_bw_cb(mids, hist[eta_key][r9_key][0],
                                                [1.424, 1.86, mean_data-dc.TARGET_MASS, sigma_data]
                                                )
                mc_fit = fit_bw_cb(mids, hist[eta_key][r9_key][1],
                                            [1.424, 1.86, mean_mc-dc.TARGET_MASS, sigma_mc]
                                            )
                hist[eta_key][r9_key] = data_fit['mu']/mc_fit['mu']

    systematics_categories = {
        "R9": (hists[1], hists[2]),
        "Et": (hists[3], hists[4]),
        "ID": (hists[5], hists[6])
    }
    nominal_hists = hists.pop(0)
    systematics = {
        eta_key: {
            r9_key: { 
                syst_key: 
                    max([1 - ((systematics_categories[syst_key][0][eta_key][r9_key] / systematics_categories[syst_key][1][eta_key][r9_key]) / (nominal_hists[eta_key][r9_key] / nominal_hists[eta_key][r9_key])) for syst_key in systematics_categories.keys()])
                    for syst_key in systematics_categories.keys()
            } for r9_key in cuts[eta_key]
        } for eta_key in cuts.keys()
    }

    write_systematics(systematics, outfile)