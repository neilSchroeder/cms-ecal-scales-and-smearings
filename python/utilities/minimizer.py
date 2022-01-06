
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot as up
import statistics as stats
import time
from scipy.optimize import basinhopping as basinHop
from scipy.optimize import  differential_evolution as diffEvolve
from scipy.optimize import minimize as minz
from scipy.special import xlogy
from scipy import stats as scistat 

import python.helpers.helper_minimizer as helper_minimizer
import python.plotters.plot_cats as plotter
from python.classes.zcat_class import zcat
from python.utilities.smear_mc import smear

__num_scales__ = 0
__num_smears__ = 0

"""
Author: 
    Neil Schroeder, schr1077@umn.edu, neil.raymond.schroeder@cern.ch

About:
    This suite of functions will perform the minimization of the scales and 
    smearings using the zcat class defined in python/zcat_class.py 
    The strategy is to extract the invariant mass disributions in each category
    for data and MC and make zcats from them. These are then handed off to the
    loss function while the scipy.optimize.minimze function minimizes the loss
    function by varying the scales in the defined categories. 

    The main function in minimize(...):
    the required arguments are:
        data -> dataset to be treated as data
        mc   -> dataset to be treated as simulation
        cats -> a dataframe containing the single electron categories from 
                which to derive the scales
    the options are:
        ignore_cats -> path to a tsv containing pairs of single electron 
                       categories to ignore
        hist_min -> minimum range of the histogram window defined for the Z 
                    inv mass line shape
        hist_max -> maximum range of the histogram window defined for the Z 
                    inv mass line shape
        hist_bin_size -> optional method to set the binning size to a defined 
                         value. _kAutoBin should be set to false.
        start_style -> specifies the method by which the scales and smearings
                       are seeded. Default is via a 1D scan.
        scan_min -> minimum range of the 1D scan 
                    (true value not delta P, i.e. 0.99 not -0.01)
        scan_max -> maximum range of the 1D scan 
                    (true value not delta P, i.e. 1.01 not 0.01)
        scan_step -> step size of the 1D scan
        _kClosure -> bool indicating whether this is a closure test or not
        _scales_ -> path to a tsv file containing scales applied the data
        _kPlot -> activates the plotting feature (currently deprecated).
        plot_dir -> directory to write the plots
        _kTestMethodAccuracy -> option for validating scales. random 
                                scales/smearings will be injected, the 
                                minimizer will attempt to recover them.
        _kScan -> activates the 1D scan feature (currently deprecated).
        _specify_file_ -> path to a tsv of single electron categories and 
                          the desired starting values of the scales in the 
                          case of start_style='specify'
        _kAutoBin -> used to deliberately deactivate the auto_binning feature
                     in the zcat invariant mass distributions.
        _kFixScales -> used to prevent the scales from floating in the fit, 
                        they will be held at 1. and only the smearings will
                        be allowed to float
        
"""


def minimize(data, mc, cats, **options):

    #don't let the minimizer start with a bad start_style
    allowed_start_styles = ('scan', 'random', 'specify')
    if options['start_style'] not in allowed_start_styles: 
        print("[ERROR][python/nll.py][minimize] Designated start style not recognized.")
        print("[ERROR][python/nll.py][minimize] The allowed options are {}.".format(allowed_start_styles))
        return []

    #count the number of categories which are a scale or a smearing
    __num_scales__ = np.sum( cats.iloc[:,0].values == 'scale')
    __num_smears__ = np.sum( cats.iloc[:,0].values == 'smear')

    if options['_kClosure']:
        __num_smears__ = 0

    #check to see if transverse energy columns need to be added
    data, mc = helper_minimizer.clean_up(data,mc, cats)
    
    print("[INFO][python/nll] extracting lists from category definitions")

    #extract the categories
    __ZCATS__ = helper_minimizer.extract_cats(data, mc, cats, 
                                              num_scales=__num_scales__, num_smears=__num_smears__, 
                                              **options
                                              )

    #once categories are extracted, data and mc can be released to make more room.
    del data
    del mc

    #set up boundaries on starting location of scales
    bounds = helper_minimizer.set_bounds(cats, num_scales=__num_scales__, num_smears=__num_smears__, **options)

    #it is important to test the accuracy with which a known scale can be recovered,
    #here we assign the known scales and inject them.
    scales_to_inject = []
    smearings_to_inject = []
    if options['_kTestMethodAccuracy']:
        scales_to_inject = np.random.uniform(low=0.99*np.ones(__num_scales__),
                                             high=1.01*np.ones(__num_scales__)
                                             ).ravel().tolist()
        scales_to_inject = np.array([round(x,6) for x in scales_to_inject]).tolist()
        if __num_smears__ > 0:
            smearings_to_inject = np.random.uniform(low=0.009*np.ones(__num_smears__),
                                                    high=0.011*np.ones(__num_smears__)
                                                    ).ravel().tolist()
            scales_to_inject.extend(smearings_to_inject)
        print("[INFO][python/nll] the injected scales and smearings are: {}".format(scales_to_inject))
        for cat in __ZCATS__:
            if cat.valid:
                if __num_smears__ > 0:
                    cat.inject(scales_to_inject[cat.lead_index], 
                               scales_to_inject[cat.sublead_index], 
                               scales_to_inject[cat.lead_smear_index], 
                               scales_to_inject[cat.sublead_smear_index])
                else:
                    cat.inject(scales_to_inject[cat.lead_index], 
                               scales_to_inject[cat.sublead_index],0,0)


    #deactivate invalid categories
    helper_minimizer.deactivate_cats(__ZCATS__, options['ignore_cats']) 

    #set up and run a basic nll scan for the initial guess
    guess = [1 for x in range(__num_scales__)] + [0.00 for x in range(__num_smears__)]
    __GUESS__ = [0 for x in guess]
    helper_minimizer.target_function(guess, __GUESS__,__ZCATS__, __num_scales__, __num_smears__) #initializes the categories

    #if we're plotting, just plot and return, don't run a minimization
    if options['_kPlot']:
        helper_minimizer.target_function(guess, __GUESS__, __ZCATS__, __num_scales__, __num_smears__)
        plotter.plot_cats(plot_dir, __ZCATS__, __CATS__)
        return [], []

    #It is sometimes necessary to demonstrate a likelihood scan. 
    if options['_kScanNLL']:
        helper_minimizer.target_function(guess, __GUESS__, __ZCATS__, __num_scales__, __num_smears__, verbose=True)
        plotter.plot_1Dscan(plot_dir, _specify_file_, __ZCATS__)
        return [], []

    if options['start_style'] != 'scan':
        if options['start_style'] == 'random':
            xlow_scales = [0.99 for i in range(__num_scales__)]
            xhigh_scales = [1.01 for i in range(__num_scales__)]
            xlow_smears = [0.008 for i in range(__num_smears__)]
            xhigh_smears = [0.025 for i in range(__num_smears__)]

            #set the initial guess: random for a regular derivation and unity 
            #for a closure derivation
            guess_scales = np.random.uniform(low=xlow_scales,
                    high=xhigh_scales).ravel().tolist()
            if options['_kClosure']: 
                guess_scales = [1. for i in range(__num_scales__)]
            guess_smears = np.random.uniform(low=xlow_smears,
                    high=xhigh_smears).ravel().tolist()
            if options['_kClosure'] or options['_kTestMethodAccuracy']: 
                guess_smears = [0.0 for i in range(__num_smears__)]
            if options['_kTestMethodAccuracy'] or not options['_kClosure']: 
                guess_scales.extend(guess_smears)
            guess = guess_scales

        if options['start_style'] == 'specify':
            scan_file_df = pd.read_csv(_specify_file_,sep='\t',header=None)
            guess = scan_file_df.loc[:,9].values
            guess = np.append(guess,[0.005 for x in range(__num_smears__)])

    else: #this is the default initialization
        print("[INFO][python/nll.py][minimize] You've selected scan start. Beginning scan:")
        
        guess = helper_minimizer.scan_nll(guess,
                                          zcats=__ZCATS__,
                                          __GUESS__=__GUESS__,
                                          cats=cats,
                                          num_smears=__num_smears__,
                                          num_scales=__num_scales__,
                                          **options
                                          )

    print("[INFO][python/nll] the initial guess is {} with nll {}".format(guess, 
        helper_minimizer.target_function(guess, __GUESS__,__ZCATS__,__num_scales__, __num_smears__)))
    min_step_dict = {}
    if options['min_step'] is not None:
        min_step_dict = {"eps":float(options['min_step'])}
    else:
        min_step_dict = {"eps":0.00001}

    #minimize
    optimum = minz(helper_minimizer.target_function,
                   np.array(guess), 
                   args=(__GUESS__,__ZCATS__,__num_scales__, __num_smears__),
                   method="L-BFGS-B", 
                   bounds=bounds,
                   options=min_step_dict) 

    print("[INFO][python/nll] the optimal values returned by scypi.optimize.minimize are:")
    print(optimum)

    if not optimum.success: 
        print("#"*40)
        print("#"*40)
        print("[ERROR] MINIMIZATION DID NOT SUCCEED")
        print("[ERROR] Please review the output and resubmit")
        print("#"*40)
        print("#"*40)
        return optimum.x

    if options['_kTestMethodAccuracy']:
        ret = optimum.x
        for i in range(len(scales_to_inject)): 
            if i < __num_scales__:
                print("[INFO][ACCURACY TEST] The injected scale was recovered to {} %".format((scales_to_inject[i]*ret[i]-1)*100))
                ret[i] = 100*(ret[i]*scales_to_inject[i] - 1)
            else:
                print("[INFO][ACCURACY TEST] The injected smearing was {}, the recovered smearing was {}".format(scales_to_inject[i], ret[i]))
                ret[i] = 100*(ret[i]/scales_to_inject[i] - 1)
        return ret
    return optimum.x
