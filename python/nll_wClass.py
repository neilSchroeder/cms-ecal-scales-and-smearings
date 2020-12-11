"""
Author: Neil Schroeder, schr1077@umn.edu, neil.raymond.schroeder@cern.ch
About:
    This suite of functions will perform the minimization of the scales and smearings
    using the zcat class defined in python/zcat_class.py 
    The strategy is to extract the invariant mass disributions in each category for
    data and MC and make zcats from them. These are then handed off the the loss function
    while the scipy.optimize.minimze function minimizes the loss function by varying the scales
    in the defined categories. 

    The main function in minimize(...):
    the required arguments are:
        data -> dataset to be treated as data
        mc   -> dataset to be treated as simulation
        cats -> a dataframe containing the single electron categories from which to derive the scales
    the options are:
        ignore_cats -> path to a tsv containing pairs of single electron categories to ignore
        hist_min -> minimum range of the histogram window defined for the Z inv mass line shape
        hist_max -> maximum range of the histogram window defined for the Z inv mass line shape
        hist_bin_size -> optional method to set the binning size to a defined value. _kAutoBin must be set to false.
        start_style -> specifies the method by which the scales and smearings are seeded. Default is via a 1D scan.
        scan_min -> minimum range of the 1D scan (true value not delta P, i.e. 0.99 not -0.01)
        scan_max -> maximum range of the 1D scan (true value not delta P, i.e. 1.01 not 0.01)
        scan_step -> step size of the 1D scan
        _kClosure -> bool indicating whether this is a closure test or not
        _scales_ -> path to a tsv file containing scales applied the data
        _kPlot -> activates the plotting feature (currently deprecated).
        plot_dir -> directory to write the plots
        _kTestMethodAccuracy -> option for validating scales. random scales/smearings will be injected, the minimizer will attempt to recover them.
        _kScan -> activates the 1D scan feature (currently deprecated).
        _specify_file_ -> path to a tsv of single electron categories and the desired starting values of the scales in the case of start_style='specify'
        _kAutoBin -> used to deliberately deactivate the auto_binning feature in the zcat invariant mass distributions.
        
"""
##################################################################################################################
import gc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot as up
import statistics as stats
from scipy.optimize import basinhopping as basinHop
from scipy.optimize import  differential_evolution as diffEvolve
from scipy.optimize import minimize as minz
from scipy.special import xlogy
from scipy import stats as scistat 

import plot_masses
from zcat_class import zcat

##################################################################################################################
__num_scales__ = 0
__num_smears__ = 0

__ZCATS__ = []
__CATS__ = []
__GUESS__ = []
__AUTO_BIN__ = True

##################################################################################################################
def add_transverse_energy(data,mc):
    #add a transverse energy column for data and mc 
    energy_0 = np.array(data.loc[:,'energy_ECAL_ele[0]'].values)
    energy_1 = np.array(data.loc[:,'energy_ECAL_ele[1]'].values)
    eta_0 = np.array(data.loc[:,'etaEle[0]'].values)
    eta_1 = np.array(data.loc[:,'etaEle[1]'].values)
    data['transverse_energy[0]'] = np.divide(energy_0,np.cosh(eta_0))
    data['transverse_energy[1]'] = np.divide(energy_1,np.cosh(eta_1))
    energy_0 = np.array(mc.loc[:,'energy_ECAL_ele[0]'].values)
    energy_1 = np.array(mc.loc[:,'energy_ECAL_ele[1]'].values)
    eta_0 = np.array(mc.loc[:,'etaEle[0]'].values)
    eta_1 = np.array(mc.loc[:,'etaEle[1]'].values)
    mc['transverse_energy[0]'] = np.divide(energy_0,np.cosh(eta_0))
    mc['transverse_energy[1]'] = np.divide(energy_1,np.cosh(eta_1))
    drop_list = ['energy_ECAL_ele[0]', 'energy_ECAL_ele[1]', 'R9Ele[0]', 'R9Ele[1]', 'gainSeedSC[0]', 'gainSeedSC[1]', 'runNumber']
    data.drop(drop_list, axis=1, inplace=True)
    mc.drop(drop_list, axis=1, inplace=True)
    #impose an et cut of 32 on leading and 20 on subleading
    mask_lead = data['transverse_energy[0]'].between(32, 99999) & data['transverse_energy[1]'].between(20, 99999)
    data = data.loc[mask_lead]
    mask_lead = mc['transverse_energy[0]'].between(32, 99999) & mc['transverse_energy[1]'].between(20, 99999)
    mc = mc.loc[mask_lead]
    return data,mc

##################################################################################################################
def get_smearing_index(cat_index):
    #this function takes in a category index and returns the associated smearing index
    r9_min = 0 if __CATS__.iloc[int(cat_index),3] < 0.96 else 0.96
    r9_max = 0.96 if __CATS__.iloc[int(cat_index),4] <= 0.96 else 10
    eta_min = __CATS__.iloc[int(cat_index),1]
    eta_max = __CATS__.iloc[int(cat_index),2]
    truth_type = __CATS__.loc[:,0] == 'smear'
    truth_eta_min = __CATS__.loc[:,1] == eta_min
    truth_eta_max = __CATS__.loc[:,2] == eta_max
    truth_r9_min = __CATS__.loc[:,3] == r9_min
    truth_r9_max = __CATS__.loc[:,4] == r9_max
    if __CATS__.iloc[int(cat_index),4] == -1:
        #there are no r9 categories in an et dependent scale minimization
        return __CATS__.loc[truth_type&truth_eta_min&truth_eta_max].index[0]
    else:
        return __CATS__.loc[truth_type&truth_eta_min&truth_eta_max&truth_r9_min&truth_r9_max].index[0]

##################################################################################################################
def smear_mc(mc,smearings):
    print("[INFO][python/nll][smear_mc] smearing the mc with {} for fixed smearings".format(smearings))
    smear_df = pd.read_csv(smearings, delimiter='\t', header=None, comment='#')
    #format is category, emain, err_mean, rho, err_rho, phi, err_phi
    for i,row in smear_df.iterrows():
        cat = row[0]
        cat_list = cat.split("-") #cat_list[0] is eta, cat_list[1] is r9
        eta_list = cat_list[0].split("_")
        r9_list = cat_list[1].split("_")

        eta_min, eta_max = float(eta_list[1]), float(eta_list[2])
        r9_min, r9_max = float(r9_list[1]), float(r9_list[2])

        mask_eta0 = mc['etaEle[0]'].between(eta_min, eta_max)
        mask_eta1 = mc['etaEle[1]'].between(eta_min, eta_max)
        mask_r90 = mc['R9Ele[0]'].between(r9_min, r9_max)
        mask_r91 = mc['R9Ele[1]'].between(r9_min, r9_max)
        mask0 = np.array(mask_eta0&mask_r90)
        mask1 = np.array(mask_eta1&mask_r91)

        energy_0 = np.array(mc['energy_ECAL_ele[0]'].values)
        energy_1 = np.array(mc['energy_ECAL_ele[1]'].values)
        invMass = np.array(mc['invMass_ECAL_ele'].values)

        smears_0 = np.multiply(mask0, np.random.normal(1, float(row[3]), len(mask0)))
        smears_1 = np.multiply(mask1, np.random.normal(1, float(row[3]), len(mask1)))
        smears_0[smears_0 == 0] = 1
        smears_1[smears_1 == 0] = 1
        energy_0 = np.multiply(energy_0, smears_0)
        energy_1 = np.multiply(energy_1, smears_1)
        invMass = np.multiply(invMass, np.sqrt(np.multiply(smears_0,smears_1)))

        mc['energy_ECAL_ele[0]'] = energy_0
        mc['energy_ECAL_ele[1]'] = energy_1
        mc['invMass_ECAL_ele'] = invMass

    return mc

##################################################################################################################
def extract_cats(data,mc):
    global __MIN_RANGE__
    global __MAX_RANGE__
    global __BIN_SIZE__
    global __AUTO_BIN__
    global __ZCATS__
    for index1 in range(__num_scales__):
        for index2 in range(index1+1):
            cat1 = __CATS__.iloc[index1]
            cat2 = __CATS__.iloc[index2]
            #thisCat should have the form: type etaMin etaMax r9Min r9Max gain etMin etMax
            entries_eta = data['etaEle[0]'].between(cat1[1],cat1[2]) & data['etaEle[1]'].between(cat2[1],cat2[2])
            entries_eta = entries_eta | (data['etaEle[1]'].between(cat1[1],cat1[2])&data['etaEle[0]'].between(cat2[1],cat2[2]))
            #need to handle the gain and et scales 
            entries_r9OrEt = []
            if cat1[5] != -1:
                gainlow1 = 0
                gainhigh1 = 0
                gainlow2 = 0
                gainhigh2 = 0
                if cat1[5] == 6:
                    gainlow1 = 1
                    gainhigh1 = 1
                if cat1[5] == 1:
                    gainlow1 = 2
                    gainhigh1 = 99999
                if cat2[5] == 6:
                    gainlow2 = 1
                    gainhigh2 = 1
                if cat2[5] == 1:
                    gainlow2 = 2
                    gainhigh2 = 99999
                entries_r9OrEt = data['gainSeedSC[0]'].between(gainlow1,gainhigh1)&data['gainSeedSC[1]'].between(gainlow2,gainhigh2)
                entries_r9OrEt = entries_r9OrEt | (data['gainSeedSC[1]'].between(gainlow1,gainhigh1)&data['gainSeedSC[0]'].between(gainlow2,gainhigh2))
            elif cat1[3] != -1 and cat1[5] == -1: #if r9Min is -1, this is an et dependent scale derivation
                entries_r9OrEt = data['R9Ele[0]'].between(cat1[3],cat1[4])&data['R9Ele[1]'].between(cat2[3],cat2[4])
                entries_r9OrEt = entries_r9OrEt | (data['R9Ele[1]'].between(cat1[3],cat1[4])&data['R9Ele[0]'].between(cat2[3],cat2[4]))
            elif cat1[3] == -1 and cat1[5] == -1:
                entries_r9OrEt = data['transverse_energy[0]'].between(cat1[6], cat1[7])&data['transverse_energy[1]'].between(cat2[6], cat2[7])
            else:
                print("[INFO][python/nll][extract_cats] Something has gone wrong in the category definitions. Please review and try again")
                raise KeyboardInterrupt

            df = data[entries_eta&entries_r9OrEt]
            mass_list_data = np.array(df['invMass_ECAL_ele'])
            del df
            del entries_eta
            del entries_r9OrEt
            gc.collect()

            entries_eta = mc['etaEle[0]'].between(cat1[1],cat1[2]) & mc['etaEle[1]'].between(cat2[1],cat2[2])
            entries_eta = entries_eta | (mc['etaEle[1]'].between(cat1[1],cat1[2])&mc['etaEle[0]'].between(cat2[1],cat2[2]))
            entries_r9OrEt = []

            #now do the same thing for MC
            if cat1[5] != -1:
                gainlow1 = 0
                gainhigh1 = 0
                gainlow2 = 0
                gainhigh2 = 0
                if cat1[5] == 6:
                    gainlow1 = 1
                    gainhigh1 = 1
                if cat1[5] == 1:
                    gainlow1 = 2
                    gainhigh1 = 99999
                if cat2[5] == 6:
                    gainlow2 = 1
                    gainhigh2 = 1
                if cat2[5] == 1:
                    gainlow2 = 2
                    gainhigh2 = 99999
                entries_r9OrEt = mc['gainSeedSC[0]'].between(gainlow1,gainhigh1)&mc['gainSeedSC[1]'].between(gainlow2,gainhigh2)
                entries_r9OrEt = entries_r9OrEt | (mc['gainSeedSC[1]'].between(gainlow1,gainhigh1)&mc['gainSeedSC[0]'].between(gainlow2,gainhigh2))
            elif cat1[3] != -1 and cat1[5] == -1: #if r9Min is -1, this is an et dependent scale derivation
                entries_r9OrEt = mc['R9Ele[0]'].between(cat1[3],cat1[4])&mc['R9Ele[1]'].between(cat2[3],cat2[4])
                entries_r9OrEt = entries_r9OrEt | (mc['R9Ele[1]'].between(cat1[3],cat1[4])&mc['R9Ele[0]'].between(cat2[3],cat2[4]))
            elif cat1[3] == -1 and cat1[5] == -1:
                entries_r9OrEt = mc['transverse_energy[0]'].between(cat1[6], cat1[7])&mc['transverse_energy[1]'].between(cat2[6], cat2[7])
            else:
                print("[INFO][python/nll][extract_cats] Something has gone wrong in the category definitions. Please review and try again")
                raise KeyboardInterrupt

            df = mc[entries_eta&entries_r9OrEt]
            mass_list_mc = np.array(df['invMass_ECAL_ele'])
            #MC needs to be over smeared in order to have good "resolution" on the scales and smearings
            while len(mass_list_mc) < 10*len(mass_list_data) and ((len(mass_list_mc) >= 1000 and index1 == index2) or (len(mass_list_mc) >= 2000 and index1 != index2)):
                mass_list_mc = np.append(mass_list_mc,mass_list_mc)

            #drop any "bad" entries
            mass_list_data = mass_list_data[~np.isnan(mass_list_data)]
            mass_list_mc = mass_list_mc[~np.isnan(mass_list_mc)]
            if __num_smears__ > 0:
                __ZCATS__.append(zcat(index1, index2, get_smearing_index(index1), get_smearing_index(index2), mass_list_data.copy(), mass_list_mc.copy(), __MIN_RANGE__, __MAX_RANGE__, __AUTO_BIN__, __BIN_SIZE__))
            else:
                __ZCATS__.append(zcat(index1, index2, -1,-1, mass_list_data.copy(), mass_list_mc.copy(), __MIN_RANGE__, __MAX_RANGE__, __AUTO_BIN__, __BIN_SIZE__))
            del df
            del entries_eta
            del entries_r9OrEt
            gc.collect()

##################################################################################################################
def target_function(x, verbose=False):
    """ 
    This is the target function, which returns an event weighted -2*Delta NLL
    This function features a small verbose option for debugging purposes.
    target_function accepts an iterable of floats and uses them to evaluate the NLL in each category.
    Some 'smart' checks prevent the function from evaluating all N(N+1)/2 categories unless absolutely necessary
    """
    global __ZCATS__
    global __GUESS__
    updated_scales = [i for i in range(len(x)) if __GUESS__[i] != x[i]]
    __GUESS__ = x
    for cat in __ZCATS__:
        if cat.valid:
            if cat.lead_index in updated_scales or cat.sublead_index in updated_scales or cat.lead_smear_index in updated_scales or cat.sublead_smear_index in updated_scales:
                if not cat.updated: 
                    if __num_smears__ == 0:
                        cat.update(x[cat.lead_index],x[cat.sublead_index])
                    else:
                        cat.update(x[cat.lead_index],x[cat.sublead_index],x[cat.lead_smear_index],x[cat.sublead_smear_index])

                    if verbose:
                        print("------------- zcat info -------------")
                        cat.print()
                        print("-------------------------------------")
                        print()

    tot = sum([cat.weight for cat in __ZCATS__ if cat.valid])
    ret = sum([cat.NLL*cat.weight for cat in __ZCATS__ if cat.valid])
    for cat in __ZCATS__: cat.reset()
    if verbose:
        print("------------- total info -------------")
        print("weighted nll:",ret/tot)
        print("diagonal nll vals:", [cat.NLL*cat.weight/tot for cat in __ZCATS__ if cat.lead_index == cat.sublead_index and cat.valid])
        print("using scales:",x)
        print("--------------------------------------")
    return ret/tot

##################################################################################################################
def scan_nll(x, scan_min, scan_max, scan_step):
    global __ZCATS__
    global __num_scales__
    global __num_smears__
    guess = x
    scanned = []
    while len(scanned) < __num_scales__:
        #find "worst" category and scan that first
        tot = np.sum([cat.NLL*cat.weight for cat in __ZCATS__ if cat.valid])/np.sum([cat.weight for cat in __ZCATS__ if cat.valid])
        max_index = -1
        max_nll = 0
        for cat in __ZCATS__:
            if cat.NLL*cat.weight/tot > max_nll and cat.valid and cat.lead_index not in scanned:
                max_index = cat.lead_index
                max_nll = cat.NLL*cat.weight/tot
        scanned.append(max_index)
        x = np.arange(scan_min,scan_max,scan_step)
        my_guesses = []
        #generate a few guesses             
        for j,val in enumerate(x): 
            guess[max_index] = val
            my_guesses.append(guess.copy())
        #evaluate nll for each guess
        nll_vals = np.array([ target_function(g) for g in my_guesses])
        mask = [y > 0 for y in nll_vals] #addresses edge cases of scale being too large/small
        x = x[mask]
        nll_vals = nll_vals[mask]
        guess[max_index] = x[nll_vals.argmin()]
        print("[INFO][python/nll] best guess for scale {} is {}".format(max_index, guess[max_index]))

    if __num_smears__ > 0:
        for i in range(__num_scales__,__num_scales__+__num_smears__,1):
            #smearings are different, so use different values for low,high,step 
            low = 0.005
            high = 0.025
            step = 0.0005
            x = np.arange(low,high,step)
            my_guesses = []
            #generate a few guesses             
            for j,val in enumerate(x): 
                guess[i] = val
                my_guesses.append(guess.copy())
            #evaluate nll for each guess
            nll_vals = np.array([ target_function(g) for g in my_guesses])
            mask = [y > 0 for y in nll_vals] #addresses edge cases of scale being too large/small
            x = x[mask]
            nll_vals = nll_vals[mask]
            guess[i] = x[nll_vals.argmin()]
            print("[INFO][python/nll] best guess for smearing {} is {}".format(i, guess[i]))

    return guess

##################################################################################################################
def minimize(data, mc, cats, ingore_cats='', hist_min=80, hist_max=100, hist_bin_size=0.25, start_style='scan', scan_min=0.98, scan_max=1.02, scan_step=0.001, _kClosure=False, _scales_='', _kPlot=False, plot_dir='./', _kTestMethodAccuracy=False, _kScan=False, _specify_file_ = '', _kAutoBin=True):
    """ 
    This is the control/main function for minimizing global scales and smearings 
    """

    #don't let the minimizer start with a bad start_style
    allowed_start_styles = ('scan', 'random', 'specify')
    if start_style not in allowed_start_styles: 
        print("[ERROR][python/nll.py][minimize] Designated start style not recognized.")
        print("[ERROR][python/nll.py][minimize] The allowed options are {}.".format(allowed_start_styles))
        return []

    #this is the main function used to handle the minimization
    global __CATS__
    global __num_scales__
    global __num_smears__
    global __MIN_RANGE__
    global __MAX_RANGE__
    global __BIN_SIZE__
    global __IGNORE__
    global __AUTO_BIN__
    global __GUESS__
    global __ZCATS__

    __CATS__ = cats
    __MIN_RANGE__ = hist_min
    __MAX_RANGE__ = hist_max
    __BIN_SIZE__ = hist_bin_size
    
    for i,row in cats.iterrows():
        if row[0] == 'scale':
            __num_scales__ += 1
        else:
            __num_smears__ += 1

    #if this a closure step smear the mc with static smearings and derive back only the scales
    if _kClosure:
        _smears_ = _scales_.replace('scales','smearings')
        mc = smear_mc(mc, _smears_)
        __num_smears__ = 0

    #check to see if transverse energy columns need to be added
    if cats.iloc[1, 3] == -1 and cats.iloc[1, 5] == -1:
        data,mc = add_transverse_energy(data, mc)
        gc.collect()
    else:
        if cats.iloc[0,3] != -1 and cats.iloc[0,5] == -1:
            drop_list = ['energy_ECAL_ele[0]', 'energy_ECAL_ele[1]', 'gainSeedSC[0]', 'gainSeedSC[1]', 'runNumber']
            print("[INFO][python/nll] dropping {}".format(drop_list))
            data.drop(drop_list, axis=1, inplace=True)
            mc.drop(drop_list, axis=1, inplace=True)
        else:
            drop_list = ['energy_ECAL_ele[0]', 'energy_ECAL_ele[1]', 'R9Ele[0]', 'R9Ele[1]', 'runNumber']
            print("[INFO][python/nll] dropping {}".format(drop_list))
            data.drop(drop_list, axis=1, inplace=True)
            mc.drop(drop_list, axis=1, inplace=True)
    
    print("[INFO][python/nll] extracting lists from category definitions")

    gc.collect()
    extract_cats(data, mc)

    #if we're plotting, just plot and return, don't run a minimization
    if _kPlot:
        target_function([1.0 for i in range(__num_scales__)].extend([0 for i in range(__num_smears__)]))
        plot_masses.plot_cats(plot_dir, __ZCATS__, __CATS__)
        return []

    #set up boundaries on starting location of scales
    bounds = []
    if _kClosure: bounds = [(0.99,1.01) for i in range(__num_scales__)]# + [(0., 0.03) for i in range(__num_smears__)]
    elif _kTestMethodAccuracy: bounds = [(0.96,1.04) for i in range(__num_scales__)] + [(0., 0.05) for i in range(__num_smears__)]
    else: bounds = [(0.96,1.04) for i in range(__num_scales__)] + [(0.002, 0.05) for i in range(__num_smears__)]

    #it is important to test the accuracy with which a known scale can be recovered,
    #here we assign the known scales and inject them.
    scales_to_inject = []
    smearings_to_inject = []
    if _kTestMethodAccuracy:
        scales_to_inject = np.random.uniform(low=0.99*np.ones(__num_scales__),high=1.01*np.ones(__num_scales__)).ravel().tolist()
        scales_to_inject = np.array([round(x,6) for x in scales_to_inject]).tolist()
        if __num_smears__ > 0:
            smearings_to_inject = np.random.uniform(low=0.009*np.ones(__num_smears__),high=0.011*np.ones(__num_smears__)).ravel().tolist()
            scales_to_inject.extend(smearings_to_inject)
        print("[INFO][python/nll] the injected scales and smearings are: {}".format(scales_to_inject))
        for cat in __ZCATS__:
            if cat.valid:
                if __num_smears__ > 0:
                    cat.inject(scales_to_inject[cat.lead_index], scales_to_inject[cat.sublead_index], scales_to_inject[cat.lead_smear_index], scales_to_inject[cat.sublead_smear_index])
                else:
                    cat.inject(scales_to_inject[cat.lead_index], scales_to_inject[cat.sublead_index],0,0)


    #deactivate invalid categories
    if ingore_cats != '':
        df_ignore = pd.read_csv(ingore_cats, sep="\t", header=None)
        for cat in __ZCATS__:
            for row in df_ingore.iterrows():
                if row[0] == cat.lead_index and row[1] == cat.sublead_index:
                    cat.valid=False


    #once categories are extracted, data and mc can be released to make more room.
    del data
    del mc
    gc.collect()

    #It is sometimes necessary to demonstrate a likelihood scan. 
    #the following code will implement such a scan, and write it to a format which can be plotted.
    #the_scan = []
    #if _kScan:

    #set up and run a basic nll scan for the initial guess
    guess = [1 for x in range(__num_scales__)] + [0.01 for x in range(__num_smears__)]
    __GUESS__ = [0 for x in guess]
    target_function(guess) #initializes the categories

    if start_style == 'scan':
        print("[INFO][python/nll.py][minimize] You've selected scan start. Beginning scan:")
        guess = scan_nll(guess, scan_min, scan_max, scan_step)

    if start_style == 'random':
        xlow_scales = [0.99 for i in range(__num_scales__)]
        xhigh_scales = [1.01 for i in range(__num_scales__)]
        xlow_smears = [0.008 for i in range(__num_smears__)]
        xhigh_smears = [0.025 for i in range(__num_smears__)]

        #set the initial guess: random for a regular derivation and unity for a closure derivation
        guess_scales = np.random.uniform(low=xlow_scales,high=xhigh_scales).ravel().tolist()
        if _kClosure: guess_scales = [1. for i in range(__num_scales__)]
        guess_smears = np.random.uniform(low=xlow_smears,high=xhigh_smears).ravel().tolist()
        if _kClosure or _kTestMethodAccuracy: guess_smears = [0.0 for i in range(__num_smears__)]
        if _kTestMethodAccuracy or not _kClosure: guess_scales.extend(guess_smears)
        guess = guess_scales

    if start_style == 'specify':
        scan_file_df = pd.read_csv(_specify_file_,sep='\t',header=None)
        guess = scan_file_df.loc[:,9].values
        guess = np.append(guess,[0.005 for x in range(__num_smears__)])
        
    print("[INFO][python/nll] the initial guess is {} with nll {}".format(guess,target_function(guess)))

    min_step_size = 0.00001 if not _kClosure else 0.000001
    #optimum = minz(target_function, np.array(guess), method="L-BFGS-B", bounds=bounds, options={"eps":min_step_size}) 
    optimum = minz(target_function, np.array(guess), method="L-BFGS-B", bounds=bounds) 

    print("[INFO][python/nll] the optimal values returned by scypi.optimize.minimize are:")
    print(optimum)

    if not optimum.success: 
        print("[ERROR] MINIMIZATION DID NOT SUCCEED")
        return []
    if _kTestMethodAccuracy:
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
