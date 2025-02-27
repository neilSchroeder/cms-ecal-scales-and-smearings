import gc
import logging
import os

import numpy as np
import pandas as pd

import python.helpers.helper_minimizer as helper_minimizer
import python.plotters.plot_cats as plotter
import python.tools.data_loader as data_loader
from python.classes.constant_classes import CategoryConstants as cc
from python.classes.constant_classes import DataConstants as dc
from python.tools.target_function import adaptive_scan_nll as scan_nll
from python.tools.target_function import fast_gradient as calculate_gradient
from python.tools.target_function import minimize as minz
from python.tools.target_function import target_function_wrapper

__num_scales__ = 0
__num_smears__ = 0

"""
Author: 
    Neil Schroeder, schr1077@umn.edu, neil.raymond.schroeder@cern.ch
"""


def minimize(data, mc, cats_df, args):
    """
    Main function for the minimization of the scales and smearings
    using the zcat class defined in python/zcat_class.py
    The strategy is to extract the invariant mass disributions in each category
    for data and MC and make zcats from them. These are then handed off to the
    loss function while the scipy.optimize.minimze function minimizes the loss
    function by varying the scales in the defined categories.
    -------------------------------------------------------------
    Args:
        data: dataset to be treated as data
        mc: dataset to be treated as simulation
        cats_df: a dataframe containing the single electron categories from
            which to derive the scales
        **args: optional arguments
            ignore_cats: path to a tsv containing pairs of single electron
                categories to ignore
            hist_min: minimum range of the histogram window defined for the Z
                inv mass line shape
            hist_max: maximum range of the histogram window defined for the Z
                inv mass line shape
            hist_bin_size: optional method to set the binning size to a defined
                value. _kAutoBin should be set to false.
            start_style: specifies the method by which the scales and
                smearings are seeded. Default is via a 1D scan.
            scan_min: minimum range of the 1D scan
                (true value not delta P, i.e. 0.99 not -0.01)
            scan_max: maximum range of the 1D scan
                (true value not delta P, i.e. 1.01 not 0.01)
            scan_step: step size of the 1D scan
            _kClosure: bool indicating whether this is a closure test or not
            _scales_: path to a tsv file containing scales applied the data
            _kPlot: activates the plotting feature (currently deprecated).
            plot_dir: directory to write the plots
            _kTestMethodAccuracy: option for validating scales. random
                scales/smearings will be injected, the minimizer will attempt
                to recover them.
            _kScan: activates the 1D scan feature (currently deprecated).
            _specify_file_: path to a tsv of single electron categories and
                the desired starting values of the scales in the case of
                start_style='specify'
            _kAutoBin: used to deliberately deactivate the auto_binning feature
                in the zcat invariant mass distributions.
            _kFixScales: used to prevent the scales from floating in the fit,
                they will be held at 1. and only the smearings will be allowed
                to float
    -------------------------------------------------------------
    Returns:
        optimum.x: the optimal values of the scales and smearings
    -------------------------------------------------------------
    """

    ignore_cats = args["ignore"]  # categories to ignore
    hist_min = round(float(args["hist_min"]), 2)  # bottom edge of histogram
    hist_max = round(float(args["hist_max"]), 2)  # top edge of histogram
    bin_size = round(float(args["bin_size"]), 2)  # bin size
    start_style = args["start_style"]  # seed method for scales/smearings
    scan_min = args["scan_min"]  # min value in scan
    scan_max = args["scan_max"]  # max value in scan
    scan_step = args["scan_step"]  # step size of scan
    min_step = args["min_step_size"]  # step size of minimizer
    _kClosure = args["_kClosure"]  # closure flag
    scales = args["scales"]  # scales file
    _kPlot = args["_kPlot"]  # plot flag
    plot_dir = args["plot_dir"]  # directory to put plots
    _kTestMethodAccuracy = args["_kTestMethodAccuracy"]  # test method flag
    _kScanNLL = args["_kScanNLL"]  # scan nll flag

    # don't let the minimizer start with a bad start_style
    allowed_start_styles = ("scan", "random", "specify")
    if start_style not in allowed_start_styles:
        print("[ERROR][python/nll.py][minimize] Designated start style not recognized.")
        print(
            f"[ERROR][python/nll.py][minimize] The allowed options are {allowed_start_styles}."
        )
        return []

    # count the number of categories which are a scale or a smearing
    __num_scales__ = np.sum(cats_df.iloc[:, 0].values == "scale")
    __num_smears__ = np.sum(cats_df.iloc[:, 0].values == "smear")

    if _kClosure:
        __num_smears__ = 0

    # check to see if transverse energy columns need to be added
    data, mc = data_loader.clean_up(data, mc, cats_df)

    print("[INFO][python/nll] extracting lists from category definitions")

    # extract the categories
    __ZCATS__ = data_loader.extract_cats(
        data, mc, cats_df, num_scales=__num_scales__, num_smears=__num_smears__, **args
    )

    __ZCATS__ = [
        cat for cat in __ZCATS__ if cat.valid
    ]  # remove any initially bad categories

    # once categories are extracted, data and mc can be released to make more room.
    del data
    del mc
    gc.collect()

    # set up boundaries on starting location of scales
    bounds = set_bounds(
        cats_df, num_scales=__num_scales__, num_smears=__num_smears__, **args
    )

    # it is important to test the accuracy with which a known scale can be recovered,
    # here we assign the known scales and inject them.
    scales_to_inject = []
    smearings_to_inject = []
    if _kTestMethodAccuracy:
        scales_to_inject = (
            np.random.uniform(
                low=0.99 * np.ones(__num_scales__), high=1.01 * np.ones(__num_scales__)
            )
            .ravel()
            .tolist()
        )
        scales_to_inject = np.array([round(x, 6) for x in scales_to_inject]).tolist()
        if __num_smears__ > 0:
            smearings_to_inject = (
                np.random.uniform(
                    low=0.009 * np.ones(__num_smears__),
                    high=0.011 * np.ones(__num_smears__),
                )
                .ravel()
                .tolist()
            )
            scales_to_inject.extend(smearings_to_inject)
        print(
            "[INFO][python/nll] the injected scales and smearings are: {}".format(
                scales_to_inject
            )
        )
        for cat in __ZCATS__:
            if cat.valid:
                if __num_smears__ > 0:
                    cat.inject(
                        scales_to_inject[cat.lead_index],
                        scales_to_inject[cat.sublead_index],
                        scales_to_inject[cat.lead_smear_index],
                        scales_to_inject[cat.sublead_smear_index],
                    )
                else:
                    cat.inject(
                        scales_to_inject[cat.lead_index],
                        scales_to_inject[cat.sublead_index],
                        0,
                        0,
                    )

    # deactivate invalid categories
    helper_minimizer.deactivate_cats(__ZCATS__, ignore_cats)

    # set up and run a basic nll scan for the initial guess
    guess = [1 for x in range(__num_scales__)] + [0.001 for x in range(__num_smears__)]
    empty_guess = [0 for x in guess]
    loss_function, reset_initial_guess = target_function_wrapper(empty_guess, __ZCATS__)

    # It is sometimes necessary to demonstrate a likelihood scan.
    if _kScanNLL:
        loss_function(
            guess, empty_guess, __ZCATS__, __num_scales__, __num_smears__, verbose=True
        )
        plotter.plot_1Dscan(plot_dir, scales, __ZCATS__)
        return [], []

    if start_style != "scan":
        if start_style == "random":
            xlow_scales = [0.99 for i in range(__num_scales__)]
            xhigh_scales = [1.01 for i in range(__num_scales__)]
            xlow_smears = [0.008 for i in range(__num_smears__)]
            xhigh_smears = [0.025 for i in range(__num_smears__)]

            # set the initial guess: random for a regular derivation and unity
            # for a closure derivation
            guess_scales = (
                np.random.uniform(low=xlow_scales, high=xhigh_scales).ravel().tolist()
            )
            if _kClosure:
                guess_scales = [1.0 for i in range(__num_scales__)]
            guess_smears = (
                np.random.uniform(low=xlow_smears, high=xhigh_smears).ravel().tolist()
            )
            if _kClosure or _kTestMethodAccuracy:
                guess_smears = [0.0 for i in range(__num_smears__)]
            if _kTestMethodAccuracy or not _kClosure:
                guess_scales.extend(guess_smears)
            guess = guess_scales

        if start_style == "specify":
            scan_file_df = pd.read_csv(scales, sep="\t", header=None)
            guess = scan_file_df.loc[:, 9].values
            guess = np.append(guess, [0.005 for x in range(__num_smears__)])

    else:  # this is the default initialization
        print(
            "[INFO][python/utilities/minimizer][minimize] You've selected scan start. Beginning scan:"
        )

        guess = scan_nll(
            guess,
            zcats=__ZCATS__,
            __GUESS__=empty_guess,
            cats=cats_df,
            num_smears=__num_smears__,
            num_scales=__num_scales__,
            **args,
        )

    print(
        "[INFO][python/nll] the initial guess is {} with nll {}".format(
            guess,
            loss_function(
                guess, empty_guess, __ZCATS__, __num_scales__, __num_smears__
            ),
        )
    )

    # increase the number of iterations

    # minimize
    import cProfile
    import io
    import pstats

    pr = cProfile.Profile()
    pr.enable()

    optimum = minz(
        loss_function,
        np.array(guess),
        args=(empty_guess, __ZCATS__, __num_scales__, __num_smears__),
        method=dc.MINIMIZATION_STRATEGY,  # might be interesting to try Nelder-Mead
        bounds=bounds,
        jac=calculate_gradient,
        options={
            "maxiter": 100000,
            "maxfun": 100000,
            "eps": 0.000001 if min_step is None else float(min_step),
        },
    )

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()
    print(s.getvalue())

    print(
        "[INFO][python/nll] the optimal values returned by scypi.optimize.minimize are:"
    )
    print(optimum)

    if _kPlot:
        for cat in __ZCATS__:
            cat.plot_history()

    if not optimum.success:
        print("#" * 40)
        print("#" * 40)
        print("[ERROR] MINIMIZATION DID NOT SUCCEED")
        print("[ERROR] Please review the output and resubmit")
        print("#" * 40)
        print("#" * 40)
        # plot the history in each category
        for cat in __ZCATS__:
            cat.plot_history()
        return optimum.x

    if _kTestMethodAccuracy:
        ret = optimum.x
        for i in range(len(scales_to_inject)):
            if i < __num_scales__:
                print(
                    "[INFO][ACCURACY TEST] The injected scale was recovered to {} %".format(
                        (scales_to_inject[i] * ret[i] - 1) * 100
                    )
                )
                ret[i] = 100 * (ret[i] * scales_to_inject[i] - 1)
            else:
                print(
                    "[INFO][ACCURACY TEST] The injected smearing was {}, the recovered smearing was {}".format(
                        scales_to_inject[i], ret[i]
                    )
                )
                ret[i] = 100 * (ret[i] / scales_to_inject[i] - 1)
        return ret
    return optimum.x


def set_bounds(cats, **options):
    """
    Set the bounds for the minimizer.

    Args:
        cats (pandas.DataFrame): dataframe containing the categories
        **options: keyword arguments, which contain the following:
            _kClosure (bool): whether or not this is a closure test
            _kTestMethodAdcuracy (bool): whether or not this is a test method adcuracy test
            _kFixScales (bool): whether or not to fix the scales
            num_scales (int): number of scales to be derived
            num_smears (int): number of smearings to be derived
    Returns:
        bounds (list): list of bounds for the minimizer
    """
    bounds = []
    if options["_kClosure"]:
        bounds = [(0.99, 1.01) for i in range(options["num_scales"])]
        if cats.iloc[1, cc.i_r9_min] != cc.empty or cats.iloc[1, cc.i_gain] != cc.empty:
            bounds = [(0.95, 1.05) for i in range(options["num_scales"])]
    elif options["_kTestMethodAccuracy"]:
        bounds = [(0.96, 1.04) for i in range(options["num_scales"])]
        bounds += [(0.000001, 0.03) for i in range(options["num_smears"])]
    elif options["_kFixScales"]:
        bounds = [(0.999999999, 1.000000001) for i in range(options["num_scales"])]
        bounds += [(0.000001, 0.03) for i in range(options["num_smears"])]
    else:
        bounds = [(0.96, 1.04) for i in range(options["num_scales"])]
        bounds += [(0.000001, 0.03) for i in range(options["num_smears"])]

    return bounds
