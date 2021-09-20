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

import python.classes.const_class as constants
import python.plotters.plot_cats as plotter
from python.utilities.smear_mc import smear
from python.classes.zcat_class import zcat

c = constants.const()

def add_transverse_energy(data,mc):
    #add a transverse energy column for data and mc 
    energy_0 = np.array(data[c.E_LEAD].values)
    energy_1 = np.array(data[c.E_SUB].values)
    eta_0 = np.array(data[c.ETA_LEAD].values)
    eta_1 = np.array(data[c.ETA_SUB].values)
    data['transverse_energy[0]'] = np.divide(energy_0,np.cosh(eta_0))
    data['transverse_energy[1]'] = np.divide(energy_1,np.cosh(eta_1))
    energy_0 = np.array(mc[c.E_LEAD].values)
    energy_1 = np.array(mc[c.E_SUB].values)
    eta_0 = np.array(mc[c.ETA_LEAD].values)
    eta_1 = np.array(mc[c.ETA_SUB].values)
    mc['transverse_energy[0]'] = np.divide(energy_0,np.cosh(eta_0))
    mc['transverse_energy[1]'] = np.divide(energy_1,np.cosh(eta_1))
    drop_list = [c.E_LEAD, c.E_SUB, c.GAIN_LEAD, c.GAIN_SUB, c.RUN]
    data.drop(drop_list, axis=1, inplace=True)
    mc.drop(drop_list, axis=1, inplace=True)
    #impose an et cut of 32 on leading and 20 on subleading
    mask_lead = data['transverse_energy[0]'].between(32, 99999) & data['transverse_energy[1]'].between(20, 99999)
    data = data[mask_lead]
    mask_lead = mc['transverse_energy[0]'].between(32, 99999) & mc['transverse_energy[1]'].between(20, 99999)
    mc = mc[mask_lead]
    return data,mc

def get_smearing_index(cats, cat_index):
    #this function takes in a category index and returns the associated smearing index
    eta_min = cats.iloc[int(cat_index),1]
    eta_max = cats.iloc[int(cat_index),2]
    r9_min = cats.iloc[int(cat_index),3]
    r9_max = cats.iloc[int(cat_index),4]
    et_min = cats.iloc[int(cat_index),6]
    et_max = cats.iloc[int(cat_index),7]
    truth_type = cats.loc[:,0] == 'smear'
    truth_eta_min = np.array([True for x in truth_type])
    truth_eta_max = np.array([True for x in truth_type])
    truth_r9_min = np.array([True for x in truth_type])
    truth_r9_max = np.array([True for x in truth_type])
    truth_et_min = np.array([True for x in truth_type])
    truth_et_max = np.array([True for x in truth_type])
    if eta_min != -1 and eta_max != -1:
        truth_eta_min = cats.loc[:,1] <= eta_min
        truth_eta_max = cats.loc[:,2] >= eta_max
    if r9_min != -1 and r9_max != - 1:
        truth_r9_min = cats.loc[:,3] <= r9_min
        truth_r9_max = cats.loc[:,4] >= r9_max
    if et_min != -1 and et_max != -1:
        truth_et_min = cats.loc[:,6] <= et_min
        truth_et_max = cats.loc[:,7] >= et_max

    truth = truth_type&truth_eta_min&truth_eta_max&truth_r9_min&truth_r9_max&truth_et_min&truth_et_max
    return cats.loc[truth].index[0]

def clean_up(data, mc, cats):
    #cleans up the dataframes, adds transverse energy if necessary, drops unnecessary columns
    if (cats.iloc[1, 3] == -1 and cats.iloc[1, 5] == -1) or (cats.iloc[1,3] != -1 and cats.iloc[1,6] != -1):
        data,mc = add_transverse_energy(data, mc)
        gc.collect()

    else:
        if cats.iloc[0,3] != -1 and cats.iloc[0,5] == -1:
            drop_list = [c.E_LEAD, 
                         c.E_SUB, 
                         c.GAIN_LEAD, 
                         c.GAIN_SUB, 
                         c.RUN]

            print("[INFO][python/nll] dropping {}".format(drop_list))
            data.drop(drop_list, axis=1, inplace=True)
            mc.drop(drop_list, axis=1, inplace=True)
        else:
            drop_list = [c.E_LEAD, 
                         c.E_SUB, 
                         c.R9_LEAD, 
                         c.R9_SUB, 
                         c.RUN]
            print("[INFO][python/nll] dropping {}".format(drop_list))
            data.drop(drop_list, axis=1, inplace=True)
            mc.drop(drop_list, axis=1, inplace=True)

    return data, mc
    

def extract_cats( data, mc, cats, **options):
    #builds zcat classes with data and mc for each category
    __ZCATS__ = []
    for index1 in range(options['num_scales']):
        for index2 in range(index1+1):
            cat1 = cats.iloc[index1]
            cat2 = cats.iloc[index2]
            #thisCat should have the form: type etaMin etaMax r9Min r9Max gain etMin etMax
            entries_eta = data[c.ETA_LEAD].between(cat1[1],cat1[2]) & data[c.ETA_SUB].between(cat2[1],cat2[2])
            entries_eta = entries_eta | (data[c.ETA_SUB].between(cat1[1],cat1[2])&data[c.ETA_LEAD].between(cat2[1],cat2[2]))
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
                entries_r9OrEt = data[c.GAIN_LEAD].between(gainlow1,gainhigh1)&data[c.GAIN_SUB].between(gainlow2,gainhigh2)
                entries_r9OrEt = entries_r9OrEt | (data[c.GAIN_SUB].between(gainlow1,gainhigh1)&data[c.GAIN_LEAD].between(gainlow2,gainhigh2))
            elif cat1[3] != -1 and cat1[5] == -1 and cat1[6] == -1: 
                #this is for R9 dependent scale derivation
                entries_r9OrEt = data[c.R9_LEAD].between(cat1[3],cat1[4])&data[c.R9_SUB].between(cat2[3],cat2[4])
                entries_r9OrEt = entries_r9OrEt | (data[c.R9_SUB].between(cat1[3],cat1[4])&data[c.R9_LEAD].between(cat2[3],cat2[4]))
            elif cat1[3] == -1 and cat1[5] == -1 and cat1[6] != -1:
                #this is for et dependent scale derivation
                entries_r9OrEt = data['transverse_energy[0]'].between(cat1[6], cat1[7])&data['transverse_energy[1]'].between(cat2[6], cat2[7])
            elif cat1[3] != -1 and cat1[5] == -1 and cat1[6] != -1:
                #this is specifically for stochastic smearings
                entries_r9OrEt = data[c.R9_LEAD].between(cat1[3],cat1[4])&data[c.R9_SUB].between(cat2[3],cat2[4])
                entries_r9OrEt = entries_r9OrEt | (data[c.R9_SUB].between(cat1[3],cat1[4])&data[c.R9_LEAD].between(cat2[3],cat2[4]))
                entries_r9OrEt = entries_r9OrEt & (data['transverse_energy[0]'].between(cat1[6], cat1[7])&data['transverse_energy[1]'].between(cat2[6], cat2[7]))
            else:
                print("[INFO][python/nll][extract_cats] Something has gone wrong in the category definitions. Please review and try again")
                #please forgive my sin of sloth
                raise KeyboardInterrupt

            df = data[entries_eta&entries_r9OrEt]
            mass_list_data = np.array(df[c.INVMASS])
            del df
            del entries_eta
            del entries_r9OrEt
            gc.collect()

            entries_eta = mc[c.ETA_LEAD].between(cat1[1],cat1[2]) & mc[c.ETA_SUB].between(cat2[1],cat2[2])
            entries_eta = entries_eta | (mc[c.ETA_SUB].between(cat1[1],cat1[2])&mc[c.ETA_LEAD].between(cat2[1],cat2[2]))
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
                entries_r9OrEt = mc[c.GAIN_LEAD].between(gainlow1,gainhigh1)&mc[c.GAIN_SUB].between(gainlow2,gainhigh2)
                entries_r9OrEt = entries_r9OrEt | (mc[c.GAIN_SUB].between(gainlow1,gainhigh1)&mc[c.GAIN_LEAD].between(gainlow2,gainhigh2))
            elif cat1[3] != -1 and cat1[5] == -1 and cat1[6] == -1: 
                #this is for R9 dependent scale derivation
                entries_r9OrEt = mc[c.R9_LEAD].between(cat1[3],cat1[4])&mc[c.R9_SUB].between(cat2[3],cat2[4])
                entries_r9OrEt = entries_r9OrEt | (mc[c.R9_SUB].between(cat1[3],cat1[4])&mc[c.R9_LEAD].between(cat2[3],cat2[4]))
            elif cat1[3] == -1 and cat1[5] == -1 and cat1[6] != -1:
                #this is for et dependent scale derivation
                entries_r9OrEt = mc['transverse_energy[0]'].between(cat1[6], cat1[7])&mc['transverse_energy[1]'].between(cat2[6], cat2[7])
            elif cat1[3] != -1 and cat1[5] == -1 and cat1[6] != -1:
                #this is specifically for stochastic smearings
                entries_r9OrEt = mc[c.R9_LEAD].between(cat1[3],cat1[4])&mc[c.R9_SUB].between(cat2[3],cat2[4])
                entries_r9OrEt = entries_r9OrEt | (mc[c.R9_SUB].between(cat1[3],cat1[4])&mc[c.R9_LEAD].between(cat2[3],cat2[4]))
                entries_r9OrEt = entries_r9OrEt & (mc['transverse_energy[0]'].between(cat1[6], cat1[7])&mc['transverse_energy[1]'].between(cat2[6], cat2[7]))
            else:
                print("[INFO][python/nll][extract_cats] Something has gone wrong in the category definitions. Please review and try again")
                raise KeyboardInterrupt

            df = mc[entries_eta&entries_r9OrEt]
            mass_list_mc = np.array(df[c.INVMASS].values, dtype=np.float32)
            weight_list_mc = np.array(df['pty_weight'].values, dtype=np.float32) if 'pty_weight' in df.columns else np.ones(len(mass_list_mc))
            #MC needs to be over smeared in order to have good "resolution" on the scales and smearings
            while len(mass_list_mc) < max(100*len(mass_list_data),200000) and len(mass_list_mc) > 0 and len(mass_list_data) > 10 and len(mass_list_mc) < 10000000:
                mass_list_mc = np.append(mass_list_mc,mass_list_mc)
                weight_list_mc = np.append(weight_list_mc,weight_list_mc)

            #drop any "bad" entries
            mass_list_data = mass_list_data[~np.isnan(mass_list_data)]
            weight_list_mc = weight_list_mc[~np.isnan(mass_list_mc)]
            mass_list_mc = mass_list_mc[~np.isnan(mass_list_mc)]
            
            if options['num_smears'] > 0:
                __ZCATS__.append(
                        zcat(
                            index1, index2, mass_list_data.copy(), mass_list_mc.copy(), weight_list_mc.copy(), 
                            smear_i=get_smearing_index(cats,index1), smear_j=get_smearing_index(cats,index2), 
                            **options
                            )
                        )
            else:
                __ZCATS__.append(
                        zcat(
                            index1, index2, #no smearing categories, so no smearing indices
                            mass_list_data.copy(), mass_list_mc.copy(), weight_list_mc.copy(),
                            **options
                            )
                        )

            del df
            del entries_eta
            del entries_r9OrEt
            gc.collect()

    return __ZCATS__

def set_bounds(cats, **options):
    #sets the rectangular bounds for the scales and smearings derivation

    bounds = []
    if options['_kClosure']:
        bounds = [(0.99,1.01) for i in range(options['num_scales'])]
        if cats.iloc[1,3] != -1 or cats.iloc[1,5] != -1:
            bounds=[(0.95,1.05) for i in range(options['num_scales'])]
    elif options['_kTestMethodAccuracy']:
        bounds = [(0.96,1.04) for i in range(options['num_scales'])]
        bounds += [(0., 0.05) for i in range(options['num_smears'])]
    elif options['_kFixScales']:
        bounds = [(0.999999999,1.000000001) for i in range(options['num_scales'])]
        bounds += [(0., 0.05) for i in range(options['num_smears'])]
    else:
        bounds = [(0.96,1.04) for i in range(options['num_scales'])]
        bounds += [(0.000, 0.05) for i in range(options['num_smears'])]
        
    return bounds



def deactivate_cats(__ZCATS__, ignore_cats):

    if ignore_cats is not None:
        df_ignore = pd.read_csv(ignore_cats, sep="\t", header=None)
        for cat in __ZCATS__:
            for row in df_ignore.iterrows():
                if row[0] == cat.lead_index and row[1] == cat.sublead_index:
                    cat.valid=False

def target_function(x, *args, verbose=False, **options):
    """ 
    This is the target function, which returns an event weighted -2*Delta NLL
    This function features a small verbose option for debugging purposes.
    target_function accepts an iterable of floats and uses them to evaluate the NLL in each category.
    Some 'smart' checks prevent the function from evaluating all N(N+1)/2 categories unless absolutely necessary
    """
    
    #unpack args
    (__GUESS__, __ZCATS__, __num_scales, __num_smears__) = args

    updated_scales = [i for i in range(len(x)) if __GUESS__[i] != x[i]]
    __GUESS__ = x
    for cat in __ZCATS__:
        if cat.valid:
            if cat.lead_index in updated_scales or cat.sublead_index in updated_scales or cat.lead_smear_index in updated_scales or cat.sublead_smear_index in updated_scales:
                if not cat.updated: 
                    if __num_smears__ == 0:
                        cat.update(x[cat.lead_index],
                                   x[cat.sublead_index])
                    else:
                        cat.update(x[cat.lead_index],
                                   x[cat.sublead_index],
                                   x[cat.lead_smear_index],
                                   x[cat.sublead_smear_index])

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
       #print("weighted nll:",ret/tot)
        print("diagonal nll vals:", [cat.NLL*cat.weight/tot for cat in __ZCATS__ if cat.lead_index == cat.sublead_index and cat.valid])
        print("using scales:",x)
        print("--------------------------------------")
    return ret/tot if tot != 0 else 9e30

def scan_nll(x, **options):
    #performs the NLL scan to initialize the variables
    __ZCATS__ = options['zcats']
    __GUESS__ = options['__GUESS__']
    guess = x
    scanned = []
    #find most sensitive category and scan that first
    weights = [(cat.weight, cat.lead_index) for cat in __ZCATS__ if cat.valid and cat.lead_index == cat.sublead_index]
    weights.sort(key=lambda x: x[0])
    if not options['_kFixScales']:
        while len(scanned) < options['num_scales']:
            max_index = -1
            for tup in weights:
                if tup[1] not in scanned:
                    max_index = tup[1]
                    scanned.append(tup[1])
                    break
            if max_index != -1:
                x = np.arange(options['scan_min'],options['scan_max'],options['scan_step'])
                my_guesses = []
                #generate a few guesses             
                for j,val in enumerate(x): 
                    guess[max_index] = val
                    my_guesses.append(guess.copy())
                #evaluate nll for each guess
                nll_vals = np.array([ target_function(g, __GUESS__, __ZCATS__, options['num_scales'], options['num_smears']) for g in my_guesses])
                mask = [y > 0 for y in nll_vals] #addresses edge cases of scale being too large/small
                x = x[mask]
                nll_vals = nll_vals[mask]
                if len(nll_vals) > 0:
                    guess[max_index] = x[nll_vals.argmin()]
                    print("[INFO][python/nll] best guess for scale {} is {}".format(max_index, guess[max_index]))

    print("[INFO][python/nll] scanning smearings:")
    scanned = []
    weights = [(cat.weight, cat.lead_smear_index) for cat in __ZCATS__ if cat.valid and cat.lead_smear_index == cat.sublead_smear_index]
    weights.sort(key=lambda x: x[0])
    if options['num_smears'] > 0:
        while len(scanned) < options['num_smears']:
            max_index = -1
            for tup in weights:
                if tup[1] not in scanned:
                    max_index = tup[1]
                    scanned.append(tup[1])
                    break
            #smearings are different, so use different values for low,high,step 
            if max_index != -1:
                low = 0.000
                high = 0.025
                step = 0.00025
                x = np.arange(low,high,step)
                my_guesses = []
                #generate a few guesses             
                for j,val in enumerate(x): 
                    guess[options['num_scales']+max_index] = val
                    my_guesses.append(guess.copy())
                #evaluate nll for each guess
                nll_vals = np.array([ target_function(g, __GUESS__, __ZCATS__, options['num_scales'], options['num_smears']) for g in my_guesses])
                mask = [y > 0 for y in nll_vals] #addresses edge cases of scale being too large/small
                x = x[mask]
                nll_vals = nll_vals[mask]
                if len(nll_vals) > 0:
                    guess[options['num_scales']+max_index] = x[nll_vals.argmin()]
                    print("[INFO][python/nll] best guess for smearing {} is {}".format(i, guess[i]))

    print("[INFO][python/nll] scan complete")
    return guess
