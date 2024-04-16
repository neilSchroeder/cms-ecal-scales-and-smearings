import gc
import numpy as np
import pandas as pd

from python.classes.constant_classes import DataConstants as dc
from python.classes.constant_classes import CategoryConstants as cc
from python.classes.zcat_class import zcat

def add_transverse_energy(data,mc):
    """
    Adds a transverse energy column to the data and mc dataframes.

    Args:
        data (pandas.DataFrame): data dataframe
        mc (pandas.DataFrame): mc dataframe
    Returns:
        data (pandas.DataFrame): data dataframe with transverse energy column
        mc (pandas.DataFrame): mc dataframe with transverse energy column
    """
    energy_0 = np.array(data[dc.E_LEAD].values)
    energy_1 = np.array(data[dc.E_SUB].values)
    eta_0 = np.array(data[dc.ETA_LEAD].values)
    eta_1 = np.array(data[dc.ETA_SUB].values)
    data[dc.ET_LEAD] = np.divide(energy_0,np.cosh(eta_0))
    data[dc.ET_SUB] = np.divide(energy_1,np.cosh(eta_1))
    energy_0 = np.array(mc[dc.E_LEAD].values)
    energy_1 = np.array(mc[dc.E_SUB].values)
    eta_0 = np.array(mc[dc.ETA_LEAD].values)
    eta_1 = np.array(mc[dc.ETA_SUB].values)
    mc[dc.ET_LEAD] = np.divide(energy_0,np.cosh(eta_0))
    mc[dc.ET_SUB] = np.divide(energy_1,np.cosh(eta_1))

    drop_list = [dc.E_LEAD, dc.E_SUB, dc.GAIN_LEAD, dc.GAIN_SUB, dc.RUN]
    data.drop(drop_list, axis=1, inplace=True)
    mc.drop(drop_list, axis=1, inplace=True)

    #  impose an et cut of 32 on leading and 20 on subleading
    mask_lead = data[dc.ET_LEAD].between(dc.MIN_ET_LEAD, dc.MAX_ET_LEAD) \
        & data[dc.ET_SUB].between(dc.MAX_ET_SUB, dc.MAX_ET_SUB)
    data = data[mask_lead]
    mask_lead = mc[dc.ET_LEAD].between(dc.MIN_ET_LEAD, dc.MAX_ET_LEAD) \
        & mc[dc.ET_SUB].between(dc.MIN_ET_SUB, dc.MAX_ET_SUB)
    mc = mc[mask_lead]
    return data,mc

def get_smearing_index(cats, cat_index):
    """
    Return the index of the smearing category that corresponds to the given category index
    
    Args:
        cats (pandas.DataFrame): dataframe containing the categories
        cat_index (int): index of the category
    Returns:
        (int): index of the smearing category that corresponds to the given category index
    """ 

    eta_min = cats.iloc[int(cat_index),cc.i_eta_min]
    eta_max = cats.iloc[int(cat_index),cc.i_eta_max]
    r9_min = cats.iloc[int(cat_index),cc.i_r9_min]
    r9_max = cats.iloc[int(cat_index),cc.i_r9_max]
    et_min = cats.iloc[int(cat_index),cc.i_et_min]
    et_max = cats.iloc[int(cat_index),cc.i_et_max]

    truth_type = cats.loc[:,cc.i_type] == 'smear'
    truth_eta_min = np.array([True for x in truth_type])
    truth_eta_max = np.array([True for x in truth_type])
    truth_r9_min = np.array([True for x in truth_type])
    truth_r9_max = np.array([True for x in truth_type])
    truth_et_min = np.array([True for x in truth_type])
    truth_et_max = np.array([True for x in truth_type])
    if eta_min != cc.empty and eta_max != cc.empty:
        truth_eta_min = cats.loc[:,cc.i_eta_min] <= eta_min
        truth_eta_max = cats.loc[:,cc.i_eta_max] >= eta_max
    if r9_min != cc.empty and r9_max != - 1:
        truth_r9_min = cats.loc[:,cc.i_r9_min] <= r9_min
        truth_r9_max = cats.loc[:,cc.i_r9_max] >= r9_max
    if et_min != cc.empty and et_max != cc.empty:
        truth_et_min = cats.loc[:,cc.i_et_min] <= et_min
        truth_et_max = cats.loc[:,cc.i_et_max] >= et_max

    truth = truth_type&truth_eta_min&truth_eta_max&truth_r9_min&truth_r9_max&truth_et_min&truth_et_max
    return cats.loc[truth].index[0]

def clean_up(data, mc, cats):
    """
    Clean up dataframes, add transverse energy if necessary, and drop unnecessary columns.

    Args:
        data (pandas.DataFrame): data dataframe
        mc (pandas.DataFrame): mc dataframe
        cats (pandas.DataFrame): dataframe containing the categories
    Returns:
        data (pandas.DataFrame): cleaned data dataframe
        mc (pandas.DataFrame): cleaned mc dataframe
    """
    if cats.iloc[0,cc.i_et_min] != cc.empty:
        data, mc = add_transverse_energy(data, mc)

    drop_list = [dc.E_LEAD, 
                dc.E_SUB, 
                dc.RUN]
    if cats.iloc[0,cc.i_gain] == cc.empty:
        drop_list += [dc.GAIN_LEAD, dc.GAIN_SUB]

    print("[INFO][python/nll] dropping {}".format(drop_list))
    data.drop(drop_list, axis=1, inplace=True)
    mc.drop(drop_list, axis=1, inplace=True)

    return data, mc

def extract_cats( data, mc, cats_df, **options):
    """
    Extract the dielectron categories from the data and mc dataframes.

    Args:
        data (pandas.DataFrame): data dataframe
        mc (pandas.DataFrame): mc dataframe
        cats_df (pandas.DataFrame): dataframe containing the categories
        **options: keyword arguments, which contain the following:
            num_scales (int): number of scales to be derived
            num_smears (int): number of smearings to be derived
    
    Returns:
        __ZCATS__ (list): list of zcat objects, each representing a dielectron category
    """
    # check for empty data
    if len(data) == 0:
        print("[INFO][python/nll] no data, returning")
        return []
    if len(mc) == 0:
        print("[INFO][python/nll] no mc, returning")
        return []
    
    __ZCATS__ = []
    for index1 in range(options['num_scales']):
        for index2 in range(index1+1):
            cat1 = cats_df.iloc[index1]
            cat2 = cats_df.iloc[index2]
            # thisCat should have the form: type etaMin etaMax r9Min r9Max gain etMin etMax

            eta_mask = np.ones(len(data), dtype=bool)
            if cat1[cc.i_eta_min] != cc.empty:
                eta_mask = data[dc.ETA_LEAD].between(cat1[cc.i_eta_min],cat1[cc.i_eta_max]) \
                    & data[dc.ETA_SUB].between(cat2[cc.i_eta_min],cat2[cc.i_eta_max])
                eta_mask = eta_mask | (data[dc.ETA_SUB].between(cat1[cc.i_eta_min],cat1[cc.i_eta_max])\
                                    &data[dc.ETA_LEAD].between(cat2[cc.i_eta_min],cat2[cc.i_eta_max]))
            
            r9_mask = np.ones(len(data), dtype=bool)
            if cat1[cc.i_r9_min] != cc.empty:
                r9_mask = data[dc.R9_LEAD].between(cat1[cc.i_r9_min],cat1[cc.i_r9_max])\
                    &data[dc.R9_SUB].between(cat2[cc.i_r9_min],cat2[cc.i_r9_max])
                r9_mask = r9_mask | (data[dc.R9_SUB].between(cat1[cc.i_r9_min],cat1[cc.i_r9_max])\
                                    &data[dc.R9_LEAD].between(cat2[cc.i_r9_min],cat2[cc.i_r9_max]))
            
            et_mask = np.ones(len(data), dtype=bool)
            if cat1[cc.i_et_min] != cc.empty:
                et_mask = data[dc.ET_LEAD].between(cat1[cc.i_et_min],cat1[cc.i_et_max])\
                    &data[dc.ET_SUB].between(cat2[cc.i_et_min],cat2[cc.i_et_max])
                et_mask = et_mask | (data[dc.ET_SUB].between(cat1[cc.i_et_min],cat1[cc.i_et_max])\
                                    &data[dc.ET_LEAD].between(cat2[cc.i_et_min],cat2[cc.i_et_max]))
            
            # gain mask should be all true if gain is not specified
            gain_mask = np.ones(len(data), dtype=bool)
            if cat1[cc.i_gain] != cc.empty:
                # possible gain values are 12, 6, 1
                # if gain is 12, then gainSeedSC is 0
                # if gain is 6, then gainSeedSC is 1
                # if gain is 1, then gainSeedSC is greater than 1
                gainlow1, gainhigh1, gainlow2, gainhigh2 = 0, 0, 0, 0
                if cat1[cc.i_gain] == 6: gainlow1, gainhigh1 = 1, 1
                if cat1[cc.i_gain] == 1: gainlow1, gainhigh1 = 2, 99999
                if cat2[cc.i_gain] == 6: gainlow2, gainhigh2 = 1, 1
                if cat2[cc.i_gain] == 1: gainlow2, gainhigh2 = 2, 99999
                gain_mask = data[dc.GAIN_LEAD].between(gainlow1,gainhigh1)\
                    &data[dc.GAIN_SUB].between(gainlow2,gainhigh2)
                gain_mask = gain_mask | (data[dc.GAIN_SUB].between(gainlow1,gainhigh1)\
                                        &data[dc.GAIN_LEAD].between(gainlow2,gainhigh2))
            
            df = data[eta_mask&r9_mask&et_mask&gain_mask]
            mass_list_data = np.array(df[dc.INVMASS])

            eta_mask = np.ones(len(mc), dtype=bool)
            if cat1[cc.i_eta_min] != cc.empty:
                eta_mask = mc[dc.ETA_LEAD].between(cat1[cc.i_eta_min],cat1[cc.i_eta_max]) \
                    & mc[dc.ETA_SUB].between(cat2[cc.i_eta_min],cat2[cc.i_eta_max])
                eta_mask = eta_mask | (mc[dc.ETA_SUB].between(cat1[cc.i_eta_min],cat1[cc.i_eta_max])\
                                    &mc[dc.ETA_LEAD].between(cat2[cc.i_eta_min],cat2[cc.i_eta_max]))
            
            r9_mask = np.ones(len(mc), dtype=bool)
            if cat1[cc.i_r9_min] != cc.empty:
                r9_mask = mc[dc.R9_LEAD].between(cat1[cc.i_r9_min],cat1[cc.i_r9_max])\
                    &mc[dc.R9_SUB].between(cat2[cc.i_r9_min],cat2[cc.i_r9_max])
                r9_mask = r9_mask | (mc[dc.R9_SUB].between(cat1[cc.i_r9_min],cat1[cc.i_r9_max])\
                                    &mc[dc.R9_LEAD].between(cat2[cc.i_r9_min],cat2[cc.i_r9_max]))
            
            et_mask = np.ones(len(mc), dtype=bool)
            if cat1[cc.i_et_min] != cc.empty:
                et_mask = mc[dc.ET_LEAD].between(cat1[cc.i_et_min],cat1[cc.i_et_max])\
                    &mc[dc.ET_SUB].between(cat2[cc.i_et_min],cat2[cc.i_et_max])
                et_mask = et_mask | (mc[dc.ET_SUB].between(cat1[cc.i_et_min],cat1[cc.i_et_max])\
                                    &mc[dc.ET_LEAD].between(cat2[cc.i_et_min],cat2[cc.i_et_max]))

            gain_mask = np.ones(len(mc), dtype=bool)
            if cat1[cc.i_gain] != cc.empty:
                gainlow1, gainhigh1, gainlow2, gainhigh2 = 0, 0, 0, 0
                if cat1[cc.i_gain] == 6: gainlow1, gainhigh1 = 1, 1
                if cat1[cc.i_gain] == 1: gainlow1, gainhigh1 = 2, 99999
                if cat2[cc.i_gain] == 6: gainlow2, gainhigh2 = 1, 1
                if cat2[cc.i_gain] == 1: gainlow2, gainhigh2 = 2, 99999
                gain_mask = mc[dc.GAIN_LEAD].between(gainlow1,gainhigh1)\
                    &mc[dc.GAIN_SUB].between(gainlow2,gainhigh2)
                gain_mask = gain_mask | (mc[dc.GAIN_SUB].between(gainlow1,gainhigh1)\
                                        &mc[dc.GAIN_LEAD].between(gainlow2,gainhigh2))

            df = mc[eta_mask&r9_mask&et_mask&gain_mask]
            mass_list_mc = np.array(df[dc.INVMASS].values, dtype=np.float32)
            weight_list_mc = np.array(df['pty_weight'].values, dtype=np.float32) if 'pty_weight' in df.columns else np.ones(len(mass_list_mc))
            # MC needs to be over smeared in order to have good "resolution" on the scales and smearings
            while len(mass_list_mc) < max(50*len(mass_list_data),50000) and len(mass_list_mc) > 100 and len(mass_list_data) > 10 and len(mass_list_mc) < 1000000:
                mass_list_mc = np.append(mass_list_mc,mass_list_mc)
                weight_list_mc = np.append(weight_list_mc,weight_list_mc)

            # drop any "bad" entries
            mass_list_data = mass_list_data[~np.isnan(mass_list_data)]
            weight_list_mc = weight_list_mc[~np.isnan(mass_list_mc)]
            mass_list_mc = mass_list_mc[~np.isnan(mass_list_mc)]
            
            if options['num_smears'] > 0:
                __ZCATS__.append(
                        zcat(
                            index1, index2, mass_list_data.copy(), mass_list_mc.copy(), weight_list_mc.copy(), 
                            smear_i=get_smearing_index(cats_df,index1), smear_j=get_smearing_index(cats_df,index2), 
                            **options
                            )
                        )
            else:
                __ZCATS__.append(
                        zcat(
                            index1, index2, # no smearing categories, so no smearing indices
                            mass_list_data.copy(), mass_list_mc.copy(), weight_list_mc.copy(),
                            **options
                            )
                        )

    return __ZCATS__

def set_bounds(cats, **options):
    """
    Set the bounds for the minimizer.

    Args:
        cats (pandas.DataFrame): dataframe containing the categories
        **options: keyword arguments, which contain the following:
            _kClosure (bool): whether or not this is a closure test
            _kTestMethodAccuracy (bool): whether or not this is a test method accuracy test
            _kFixScales (bool): whether or not to fix the scales
            num_scales (int): number of scales to be derived
            num_smears (int): number of smearings to be derived
    Returns:
        bounds (list): list of bounds for the minimizer
    """
    bounds = []
    if options['_kClosure']:
        bounds = [(0.99,1.01) for i in range(options['num_scales'])]
        if cats.iloc[1,cc.i_r9_min] != cc.empty or cats.iloc[1,cc.i_gain] != cc.empty:
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
                if row[cc.i_type] == cat.lead_index and row[cc.i_eta_min] == cat.sublead_index:
                    cat.valid=False


def update_cat(*args, **options):
    """
    Update z categories with new scales and smearings.
    
    Args:
        *args: a tuple of arguments, which contains the following:
            cat (zcat): zcat object, representing a dielectron category
            x (iterable): iterable of floats, representing the scales and smearings chosen by the minimizer
            updated_scales (list): list of updated scales
            __num_smears__ (int): number of smearings to be derived
            verbose (bool): whether or not to print verbose output
        **options: keyword arguments, which contain the following:
            verbose (bool): whether or not to print verbose output
    Returns:
        None
    """
    #  unpack args
    (cat, x, updated_scales, __num_smears__, verbose) = args
    if cat.valid:
        if cat.lead_index in updated_scales or cat.sublead_index in updated_scales or cat.lead_smear_index in updated_scales or cat.sublead_smear_index in updated_scales:
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


def target_function(x, *args, verbose=False, **options):
    """ 
    This is the target function, which returns an event weighted -2*Delta NLL
    This function features a small verbose option for debugging purposes.
    target_function accepts an iterable of floats and uses them to evaluate the NLL in each category.
    Some 'smart' checks prevent the function from evaluating all N(N+1)/2 categories unless absolutely necessary.

    Args:
        x (iterable): iterable of floats, representing the scales and smearings chosen by the minimizer
        *args: a tuple of arguments, which contains the following:
            __GUESS__ (iterable): iterable of floats, representing the initial guess for the scales and smearings
            __ZCATS__ (list): list of zcat objects, each representing a category
            __num_scales__ (int): number of scales to be derived
            __num_smears__ (int): number of smearings to be derived
    """
    
    # unpack args
    (__GUESS__, __ZCATS__, __num_scales__, __num_smears__) = args

    updated_scales = [i for i in range(len(x)) if __GUESS__[i] != x[i]]
    __GUESS__ = x

    for cat in __ZCATS__:
        if cat.valid:
            if cat.lead_index in updated_scales or cat.sublead_index in updated_scales or cat.lead_smear_index in updated_scales or cat.sublead_smear_index in updated_scales:
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


    if verbose:
        print("------------- total info -------------")
        # print("weighted nll:",ret/tot)
        print("diagonal nll vals:", [cat.NLL*cat.weight/tot for cat in __ZCATS__ if cat.lead_index == cat.sublead_index and cat.valid])
        print("using scales:",x)
        print("--------------------------------------")
    return ret/tot if tot != 0 else 9e30

def scan_nll(x, **options):
    """
    Performs the NLL scan to initialize the variables.

    Args:
        x (iterable): iterable of floats, representing the scales and smearings chosen by the minimizer
        **options: keyword arguments, which contain the following:
            __GUESS__ (iterable): iterable of floats, representing the initial guess for the scales and smearings
            __ZCATS__ (list): list of zcat objects, each representing a category
            _kFixScales (bool): whether or not to fix the scales
            num_scales (int): number of scales to be derived
            num_smears (int): number of smearings to be derived
            scan_min (float): minimum value for the scan
            scan_max (float): maximum value for the scan
            scan_step (float): step size for the scan
    Returns:
        guess (iterable): iterable of floats, representing the scales and smearings chosen by the minimizer
    """
    __ZCATS__ = options['zcats']
    __GUESS__ = options['__GUESS__']
    guess = x
    scanned = []

    # find most sensitive category and scan that first
    print("[INFO][python/helper_minimizer/scan_ll] scanning scales")
    weights = [(cat.weight, cat.lead_index) for cat in __ZCATS__ if cat.valid and cat.lead_index == cat.sublead_index]
    weights.sort(key=lambda x: x[0])

    if not options['_kFixScales']:
        while weights: 
            max_index = cc.empty
            tup = weights.pop(0)

            if tup[cc.i_eta_min] not in scanned:
                max_index = tup[cc.i_eta_min]
                scanned.append(tup[cc.i_eta_min])

            if max_index != cc.empty:
                x = np.arange(options['scan_min'],options['scan_max'],options['scan_step'])
                my_guesses = []

                # generate a few guesses             
                for j,val in enumerate(x): 
                    guess[max_index] = val
                    my_guesses.append(guess.copy())

                # evaluate nll for each guess
                nll_vals = np.array([ target_function(g, __GUESS__, __ZCATS__, options['num_scales'], options['num_smears']) for g in my_guesses])
                mask = [y > 0 for y in nll_vals] # addresses edge cases of scale being too large/small
                x = x[mask]
                nll_vals = nll_vals[mask]

                if len(nll_vals) > 0:
                    guess[max_index] = x[nll_vals.argmin()]
                    print("[INFO][python/nll] best guess for scale {} is {}".format(max_index, guess[max_index]))

    print("[INFO][python/helper_minimizer/scan_nll] scanning smearings:")
    scanned = []
    weights = [(cat.weight, cat.lead_smear_index) for cat in __ZCATS__ if cat.valid and cat.lead_smear_index == cat.sublead_smear_index]
    weights.sort(key=lambda x: x[0])

    if options['num_smears'] > 0:
        while weights:
            max_index = cc.empty
            tup = weights.pop(0)

            if tup[cc.i_eta_min] not in scanned:
                max_index = tup[cc.i_eta_min]
                scanned.append(tup[cc.i_eta_min])

            # smearings are different, so use different values for low,high,step 
            if max_index != cc.empty:
                low = 0.000
                high = 0.025
                step = 0.00025
                x = np.arange(low,high,step)
                my_guesses = []

                # generate a few guesses             
                for j,val in enumerate(x): 
                    guess[max_index] = val
                    my_guesses.append(guess.copy())

                # evaluate nll for each guess
                nll_vals = np.array([ target_function(g, __GUESS__, __ZCATS__, options['num_scales'], options['num_smears']) for g in my_guesses])
                mask = [y > 0 for y in nll_vals] # addresses edge cases of scale being too large/small
                x = x[mask]
                nll_vals = nll_vals[mask]
                if len(nll_vals) > 0:
                    guess[max_index] = x[nll_vals.argmin()]
                    print(f"[INFO][python/nll] best guess for smearing {max_index} is {guess[max_index]}")

    print("[INFO][python/nll] scan complete")
    return guess