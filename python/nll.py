"""
Author: Neil Schroeder
About:
    This suite of functions will perform the minimization of the scales and smearings

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

import plot_masses

##################################################################################################################
__num_scales__ = 0
__num_smears__ = 0

__DATA__ = pd.DataFrame()
__MC__ = pd.DataFrame()
__CATS__ = []
__SEED__ = 0
__MASS_MC__ = []
__MASS_DATA__ = []

##################################################################################################################
def extract_cats(__DATA__,__MC__):
    for index1 in range(__num_scales__):
        mass_list_data = []
        mass_list_mc = []
        #print("[INFO][python/nll][extract_cats] extracting category with index {}".format(index1))
        for index2 in range(index1+1):
            cat1 = __CATS__.iloc[index1]
            cat2 = __CATS__.iloc[index2]
            #thisCat should have the form: type etaMin etaMax r9Min r9Max gain etMin etMax
            entries_eta = __DATA__['etaEle[0]'].between(cat1[1],cat1[2]) & __DATA__['etaEle[1]'].between(cat2[1],cat2[2])
            entries_eta = entries_eta | (__DATA__['etaEle[1]'].between(cat1[1],cat1[2])&__DATA__['etaEle[0]'].between(cat2[1],cat2[2]))
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
                entries_r9OrEt = __DATA__['gainSeedSC[0]'].between(gainlow1,gainhigh1)&__DATA__['gainSeedSC[1]'].between(gainlow2,gainhigh2)
                entries_r9OrEt = entries_r9OrEt | (__DATA__['gainSeedSC[1]'].between(gainlow1,gainhigh1)&__DATA__['gainSeedSC[0]'].between(gainlow2,gainhigh2))
            elif cat1[3] != -1 and cat1[5] == -1: #if r9Min is -1, this is an et dependent scale derivation
                entries_r9OrEt = __DATA__['R9Ele[0]'].between(cat1[3],cat1[4])&__DATA__['R9Ele[1]'].between(cat2[3],cat2[4])
                entries_r9OrEt = entries_r9OrEt | (__DATA__['R9Ele[1]'].between(cat1[3],cat1[4])&__DATA__['R9Ele[0]'].between(cat2[3],cat2[4]))
            elif cat1[3] == -1 and cat1[5] == -1:
                entries_r9OrEt = __DATA__['transverse_energy[0]'].between(cat1[6], cat1[7])&__DATA__['transverse_energy[1]'].between(cat2[6], cat2[7])
            else:
                print("[INFO][python/nll][extract_cats] Something has gone wrong in the category definitions. Please review and try again")
                raise KeyboardInterrupt

            df = __DATA__[entries_eta&entries_r9OrEt]
            mass_list_data.append(np.array(df['invMass_ECAL_ele']))
            del df
            del entries_eta
            del entries_r9OrEt
            gc.collect()

            entries_eta = __MC__['etaEle[0]'].between(cat1[1],cat1[2]) & __MC__['etaEle[1]'].between(cat2[1],cat2[2])
            entries_eta = entries_eta | (__MC__['etaEle[1]'].between(cat1[1],cat1[2])&__MC__['etaEle[0]'].between(cat2[1],cat2[2]))
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
                entries_r9OrEt = __MC__['gainSeedSC[0]'].between(gainlow1,gainhigh1)&__MC__['gainSeedSC[1]'].between(gainlow2,gainhigh2)
                entries_r9OrEt = entries_r9OrEt | (__MC__['gainSeedSC[1]'].between(gainlow1,gainhigh1)&__MC__['gainSeedSC[0]'].between(gainlow2,gainhigh2))
            elif cat1[3] != -1 and cat1[5] == -1: #if r9Min is -1, this is an et dependent scale derivation
                entries_r9OrEt = __MC__['R9Ele[0]'].between(cat1[3],cat1[4])&__MC__['R9Ele[1]'].between(cat2[3],cat2[4])
                entries_r9OrEt = entries_r9OrEt | (__MC__['R9Ele[1]'].between(cat1[3],cat1[4])&__MC__['R9Ele[0]'].between(cat2[3],cat2[4]))
            elif cat1[3] == -1 and cat1[5] == -1:
                entries_r9OrEt = __MC__['transverse_energy[0]'].between(cat1[6], cat1[7])&__MC__['transverse_energy[1]'].between(cat2[6], cat2[7])
            else:
                print("[INFO][python/nll][extract_cats] Something has gone wrong in the category definitions. Please review and try again")
                raise KeyboardInterrupt

            df = __MC__[entries_eta&entries_r9OrEt]
            mass_list_mc.append(np.array(df['invMass_ECAL_ele']))
            #MC needs to be over smeared in order to have good "resolution" on the scales and smearings
            while len(mass_list_mc[-1]) < 7000 and ((len(mass_list_mc[-1]) >= 1000 and index1 == index2) or (len(mass_list_mc[-1]) >= 2000 and index1 != index2)):
                mass_list_mc[-1] = np.append(mass_list_mc[-1],mass_list_mc[-1])

            #drop any "bad" entries
            mass_list_data[-1] = mass_list_data[-1][~np.isnan(mass_list_data[-1])]
            mass_list_mc[-1] = mass_list_mc[-1][~np.isnan(mass_list_mc[-1])]
            del df
            del entries_eta
            del entries_r9OrEt
            gc.collect()

        __MASS_DATA__.append(mass_list_data)
        __MASS_MC__.append(mass_list_mc)
        del mass_list_data
        del mass_list_mc
        gc.collect()


##################################################################################################################
def get_nll( data, mc):
    #this function takes in two numpy arrays and returns the binned negative log likelihood between the two
    if len(data) < 10 or len(mc) < 1000: return 0
    bins = np.arange(__MIN_RANGE__, __MAX_RANGE__, __BIN_SIZE__)
    mids = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]

    invMass_binned_data, edges_data = np.histogram(data, bins=bins)
    invMass_binned_mc, edges_mc = np.histogram(mc, bins=bins)
    
    if np.sum(invMass_binned_mc) < 1000 or np.sum(invMass_binned_data) < 10: return 0
    invMass_binned_mc[invMass_binned_mc == 0] = 1e-15 #there are complaints about log(0) being ill-defined, so here's a fix
    invMass_binned_normalized_mc = invMass_binned_mc / np.sum(invMass_binned_mc) #mc is normalized and becomes a template for data

    nll = xlogy(invMass_binned_data, invMass_binned_normalized_mc)
    nll[nll==-np.inf] = 0
    nll = np.sum(nll)/len(nll)
    penalty = xlogy( np.sum(invMass_binned_data) - invMass_binned_data, 1-invMass_binned_normalized_mc)
    penalty[penalty==-np.inf] = 0
    penalty = np.sum(penalty)/len(penalty)

    return -2*(nll + penalty)

##################################################################################################################
def apply_parameter( masses, par1, par2, kIsScale):
    #this function takes in a dataframe, category, and scale and returns an array of the scaled invariant masses
    ret = np.copy(masses)
    par1 = round(par1,5)
    par2 = round(par2,5)

    if kIsScale:
        ret = ret * np.sqrt(par1*par2)
    else:
        np.random.seed(__SEED__)
        par1_list = np.random.normal(1,np.abs(par1), len(ret))
        par2_list = np.random.normal(1,np.abs(par2), len(ret))
        ret = np.multiply( ret, np.sqrt(np.multiply(par1_list, par2_list)))

    return ret

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
def target_function(x, verbose=False):
    #this uses the scales to build a global nll value, which is our minimization target
    bad_et_cats_coarse=[(10,5),(10,10),(11,10),(14,5),(14,10),(15,5),(15,10),(15,15),(16,15),(17,5),(18,5),(19,5),(19,10),(20,1),(20,9),(20,10),(20,15),(20,20),(21,1),(21,9),(21,9),(22,1),(22,9),(23,9),(23,15),(24,9),(24,15)]
    bad_et_cats_fine=[(10,2),(10,10),(11,3),(11,10),(12,9),(18,18),(19,2),(19,10),(19,11),(19,18),(19,19),(20,3),(20,9),(20,10),(20,12),(20,18),(20,19),(21,9),(21,18),(21,21),(27,9),(27,10),(27,11),(27,18),(27,19),(27,20),(27,27),(28,9),(28,10),(28,11),(28,18),(28,19),(28,20),(28,27),(28,28),(29,9),(29,18),(29,27),(29,28),(30,27),(36,9),(36,10),(36,18),(36,19),(36,20),(36,27),(36,28),(36,29),(36,36),(37,9),(37,18),(37,27),(37,28),(37,36),(37,37),(38,18),(38,27),(38,28),(38,30),(38,36),(38,37),(39,20),(39,39)]
    """TO DO
    Put bad cats into files and read bad cats from file via option handling
    """
    ret_array = []
    data_masses = np.array([0])
    mc_masses = np.array([0])
    for i in range(__num_scales__):
        for j in range(i+1):
            mc_diag = len(__MASS_MC__[i][j]) > 1000 and i == j
            mc_offdiag = len(__MASS_MC__[i][j]) > 2000 and i != j
            good_cat = True
            if __IGNORE__ is not None:
                good_cat = not sum([(i,j) == x for x in __IGNORE__])
            if len(__MASS_DATA__[i][j] > 10) and (mc_diag or mc_offdiag) and good_cat:
                data_masses = apply_parameter( __MASS_DATA__[i][j], 1, 1, True)
                mc_masses = apply_parameter( __MASS_MC__[i][j], 1/x[i], 1/x[j], True)
                if __num_smears__ > 0: mc_masses = apply_parameter( mc_masses, x[get_smearing_index(i)], x[get_smearing_index(j)], False)
                ret_array.append(get_nll(data_masses, mc_masses))
                if verbose: 
                    print(j,i,round(np.mean(data_masses),4), round(np.mean(mc_masses),4), round(ret_array[-1],6))

    ret_array = np.array(ret_array)
    ret = np.sum(ret_array)
    if verbose:
        print(x)
        print(ret)
    return ret

##################################################################################################################
def add_transverse_energy(__DATA__,__MC__):
    #add a transverse energy column for data and mc 
    energy_0 = np.array(__DATA__.loc[:,'energy_ECAL_ele[0]'].values)
    energy_1 = np.array(__DATA__.loc[:,'energy_ECAL_ele[1]'].values)
    eta_0 = np.array(__DATA__.loc[:,'etaEle[0]'].values)
    eta_1 = np.array(__DATA__.loc[:,'etaEle[1]'].values)
    __DATA__['transverse_energy[0]'] = np.divide(energy_0,np.cosh(eta_0))
    __DATA__['transverse_energy[1]'] = np.divide(energy_1,np.cosh(eta_1))
    energy_0 = np.array(__MC__.loc[:,'energy_ECAL_ele[0]'].values)
    energy_1 = np.array(__MC__.loc[:,'energy_ECAL_ele[1]'].values)
    eta_0 = np.array(__MC__.loc[:,'etaEle[0]'].values)
    eta_1 = np.array(__MC__.loc[:,'etaEle[1]'].values)
    __MC__['transverse_energy[0]'] = np.divide(energy_0,np.cosh(eta_0))
    __MC__['transverse_energy[1]'] = np.divide(energy_1,np.cosh(eta_1))
    drop_list = ['energy_ECAL_ele[0]', 'energy_ECAL_ele[1]', 'R9Ele[0]', 'R9Ele[1]', 'gainSeedSC[0]', 'gainSeedSC[1]', 'runNumber']
    __DATA__.drop(drop_list, axis=1, inplace=True)
    __MC__.drop(drop_list, axis=1, inplace=True)
#impose an et cut of 32 on leading and 20 on subleading
    mask_lead = __DATA__['transverse_energy[0]'].between(32, 99999) & __DATA__['transverse_energy[1]'].between(20, 99999)
    __DATA__ = __DATA__.loc[mask_lead]
    mask_lead = __MC__['transverse_energy[0]'].between(32, 99999) & __MC__['transverse_energy[1]'].between(20, 99999)
    __MC__ = __MC__.loc[mask_lead]
    return __DATA__,__MC__

##################################################################################################################
def smear_mc(__MC__,smearings):
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

        mask_eta0 = __MC__['etaEle[0]'].between(eta_min, eta_max)
        mask_eta1 = __MC__['etaEle[1]'].between(eta_min, eta_max)
        mask_r90 = __MC__['R9Ele[0]'].between(r9_min, r9_max)
        mask_r91 = __MC__['R9Ele[1]'].between(r9_min, r9_max)
        mask0 = np.array(mask_eta0&mask_r90)
        mask1 = np.array(mask_eta1&mask_r91)

        energy_0 = np.array(__MC__['energy_ECAL_ele[0]'].values)
        energy_1 = np.array(__MC__['energy_ECAL_ele[1]'].values)
        invMass = np.array(__MC__['invMass_ECAL_ele'].values)

        smears_0 = np.multiply(mask0, np.random.normal(1, float(row[3]), len(mask0)))
        smears_1 = np.multiply(mask1, np.random.normal(1, float(row[3]), len(mask1)))
        smears_0[smears_0 == 0] = 1
        smears_1[smears_1 == 0] = 1
        energy_0 = np.multiply(energy_0, smears_0)
        energy_1 = np.multiply(energy_1, smears_1)
        invMass = np.multiply(invMass, np.sqrt(np.multiply(smears_0,smears_1)))

        __MC__['energy_ECAL_ele[0]'] = energy_0
        __MC__['energy_ECAL_ele[1]'] = energy_1
        __MC__['invMass_ECAL_ele'] = invMass

    return __MC__

##################################################################################################################
def minimize(data, mc, cats, ingore_cats='', hist_min=80, hist_max=100, hist_bin_size=0.25, scan_min=0.98, scan_max=1.02, scan_step=0.001, _closure_=False, _scales_='', _kPlot=False, _kTestMethodAccuracy=False, _kScan=False, _scan_file_ = '', _kGuessRandom=False):
    #this is the main function used to handle the minimization
    global __CATS__
    global __num_scales__
    global __num_smears__
    global __SEED__
    global __MIN_RANGE__
    global __MAX_RANGE__
    global __BIN_SIZE__
    global __IGNORE__

    #2D array of all mass arrays using these categories
    global __MASS_MC__ 
    global __MASS_DATA__

    __CATS__ = cats
    __SEED__ = np.random.randint(2**32 - 2)
    __MIN_RANGE__ = hist_min
    __MAX_RANGE__ = hist_max
    __BIN_SIZE__ = hist_bin_size
    
    if ingore_cats != '':
        df_ignore = pd.read_csv(ingore_cats, sep="\t", header=None)
        __IGNORE__ = [(row[0],row[1]) for row in df_ignore.iterrows()]
    else:
        __IGNORE__ = None

    for i,row in cats.iterrows():
        if row[0] == 'scale':
            __num_scales__ += 1
        else:
            __num_smears__ += 1

    #if this a closure step smear the mc with static smearings and derive back only the scales
    if _closure_:
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
        plot_masses.plot_cats(__MASS_DATA__,__MASS_MC__, __CATS__)
        return []

    #once categories are extracted, data and mc can be released to make more room.
    del data
    del mc
    gc.collect()

    #It is sometimes necessary to demonstrate a likelihood scan. 
    #the following code will implement such a scan, and write it to a format which is can be plotted.
    the_scan = []
    if _kScan:
        scan_scales = [1 for x in range(__num_scales__)]+[0 for x in range(__num_smears__)]
        if _scan_file_ != '':
            scan_file_df = pd.read_csv(_scan_file_,sep='\t',header=None)
            scan_scales = scan_file_df.loc[:,9].values
            scan_scales = np.append(scan_scales,[0.015 for x in range(__num_smears__)])
        scan_scales = np.array(scan_scales)
        print(scan_scales)
        for i in range(len(scan_scales)):
            initial_value = scan_scales[i]
            if i < __num_scales__:
                this_scan = []
                for x in np.arange(scan_scales[i]-0.025, scan_scales[i]+0.025, 0.0001):
                    scan_scales[i] = x
                    this_scan.append((x,target_function(scan_scales)))
                scan_scales[i] = initial_value
                this_scan = np.array(this_scan)
                output_file = _scan_file_
                output_file.replace("scales.dat", "scan_scale_{}".format(i))
                np.savetxt(output_file,this_scan,delimiter='\t')
            if i >= __num_scales__:
                this_scan=[]
                for x in np.arange(0.005, 0.025,0.0001):
                    scan_scales[i] = x
                    this_scan.append((x, target_function(scan_scales)))
                scan_scales[i] = initial_value
                this_scan = np.array(this_scan)
                output_file = _scan_file_
                output_file.replace("scales.dat", "scan_smearing_{}".format(i))
                np.savetxt(output_file,this_scan,delimiter='\t')
        return []

    #set up boundaries on starting location of scales
    bounds = []
    if _closure_: bounds = [(0.99,1.01) for i in range(__num_scales__)]# + [(0., 0.03) for i in range(__num_smears__)]
    elif _kTestMethodAccuracy: bounds = [(0.96,1.04) for i in range(__num_scales__)] + [(0., 0.00005) for i in range(__num_smears__)]
    else: bounds = [(0.96,1.04) for i in range(__num_scales__)] + [(0.002, 0.05) for i in range(__num_smears__)]
    
    #set up and run a basic nll scan for the initial guess
    guess = [1 for x in range(__num_scales__)] + [0.01 for x in range(__num_smears__)]
    for i in range(__num_scales__+__num_smears__):
        min_val = target_function(guess)
        low, high, step = scan_min, scan_max, scan_step
        if i >= __num_scales__:
            low = 0.005
            high = 0.025
            step = 0.0005
        for x in np.arange(low, high, step):
            initial_value = guess[i]
            guess[i] = x
            check = target_function(guess)
            if check < min_val:
                min_val = check
            else: guess[i] = initial_value

        print("[INFO][python/nll] best guess for {} {} is {}".format("scale" if i < __num_scales__ else "smearing", i, guess[i]))
    
    if _kGuessRandom:
        xlow_scales = [0.995 for i in range(__num_scales__)]
        xhigh_scales = [1.001 for i in range(__num_scales__)]
        xlow_smears = [0.008 for i in range(__num_smears__)]
        xhigh_smears = [0.025 for i in range(__num_smears__)]

    #set the initial guess: random for a regular derivation and unity for a closure derivation
        guess_scales = np.random.uniform(low=xlow_scales,high=xhigh_scales).ravel().tolist()
        if _closure_: guess_scales = [1. for i in range(__num_scales__)]
        guess_smears = np.random.uniform(low=xlow_smears,high=xhigh_smears).ravel().tolist()
        if _closure_ or _kTestMethodAccuracy: guess_smears = [0.0 for i in range(__num_smears__)]
        if _kTestMethodAccuracy or not _closure_: guess_scales.extend(guess_smears)
        guess = guess_scales

    #it is important to test the accuracy with which a known scale can be recovered,
    #here we assign the known scales and inject them.
    scales_to_inject = []
    if _kTestMethodAccuracy:
        scales_to_inject = np.random.uniform(low=xlow_scales,high=xhigh_scales).ravel().tolist()
        for i in range(__num_scales__):
            for j in range(i+1):
                __MASS_DATA__[i][j] = apply_parameter(__MASS_DATA__[i][j], scales_to_inject[i], scales_to_inject[j], True)

    print("[INFO][python/nll] the initial guess is {}".format(guess))

    min_step_size = 0.0001 if not _closure_ else 0.00001
    optimum = minz(target_function, np.array(guess), method="L-BFGS-B", bounds=bounds, options={"eps":min_step_size}) 

    print("[INFO][python/nll] the optimal values returned by scypi.optimize.minimize are:")
    print(optimum)
    print(optimum.success)
    
    if not optimum.success: 
        print("[ERROR] MINIMIZATION DID NOT SUCCEED")
        return []
    if _kTestMethodAccuracy:
        ret = optimum.x
        for i in range(len(scales_to_inject)): 
            print("[INFO][ACCURACY TEST] The injected scale was {}, the recovered scale was {}".format(scales_to_inject[i], 1./ret[i]))
            ret[i] *= scales_to_inject[i]
        return ret
    return optimum.x
