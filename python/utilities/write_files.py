import json
import numpy as np
import os
import pandas as pd
from collections import OrderedDict

from python.classes.constant_classes import DataConstants as dc
from python.classes.config_class import SSConfig
ss_config = SSConfig()

def congruentCategories(last, this, nameLast, nameThis):
    """
    Determine if this category is congruent with the last category
    ----------
    Args:
        last: last category
        this: this category
        nameLast: name of last category
        nameThis: name of this category
    ----------
    Returns:
        ret: true if congruent, false otherwise
    ----------
    """
    ret = False
    """
    0: runMin
    1: runMax
    2: etaMin
    3: etaMax
    4: r9Min
    5: r9Max
    6: EtMin
    7: EtMax
    8: Gain
    9: Scale
    10: Err
    """

    congruent = False
    run = False
    eta = False
    r9  = False
    et  = False
    gain = False

    # check if the run ranges are the same
    if last[0] == this[0] and last[1] == this[1]:
        run = True
    elif last[0] <= this[0] and last[1] >= this[1]:
        # this is the case where the last category is a superset of this category
        run = True
    elif last[0] >= this[0] and last[1] <= this[1]:
        # this is the case where this category is a superset of the last category
        run = True
    else:
        # print(f"[FATAL ERROR][python/write_files][congruentCategories] run ranges are not the same: {last[0]} != {this[0]}")
        # print(f"[FATAL ERROR][python/write_files][congruentCategories] run ranges are not the same: {last[1]} != {this[1]}")
        return False

    # check if the eta ranges are the same
    if round(last[2],4) == round(this[2],4) and round(last[3],4) == round(this[3],4):
        eta = True
    elif round(last[2],4) <= round(this[2],4) and round(last[3],4) >= round(this[3],4):
        # this is the case where the last category is a superset of this category
        eta = True
    elif round(last[2],4) >= round(this[2],4) and round(last[3],4) <= round(this[3],4):
        # this is the case where this category is a superset of the last category
        eta = True
    else:   
        # print(f"[FATAL ERROR][python/write_files][congruentCategories] eta ranges are not the same: {last[2]} != {this[2]}")
        # print(f"[FATAL ERROR][python/write_files][congruentCategories] eta ranges are not the same: {last[3]} != {this[3]}")
        return False
    
    # check if the r9 ranges are the same
    if round(last[4],4) == round(this[4],4) and round(last[5],4) == round(this[5],4):
        r9 = True
    elif round(last[4],4) <= round(this[4],4) and round(last[5],4) >= round(this[5],4):
        # this is the case where the last category is a superset of this category
        r9 = True
    elif round(last[4],4) >= round(this[4],4) and round(last[5],4) <= round(this[5],4):
        # this is the case where this category is a superset of the last category
        r9 = True
    else:
        # print(f"[FATAL ERROR][python/write_files][congruentCategories] r9 ranges are not the same: {last[4]} != {this[4]}")
        # print(f"[FATAL ERROR][python/write_files][congruentCategories] r9 ranges are not the same: {last[5]} != {this[5]}")
        return False
    
    # check if the et ranges are the same
    if round(last[6],4) == round(this[6],4) and round(last[7],4) == round(this[7],4):
        et = True
    elif round(last[6],4) <= round(this[6],4) and round(last[7],4) >= round(this[7],4):
        # this is the case where the last category is a superset of this category
        et = True
    elif round(last[6],4) >= round(this[6],4) and round(last[7],4) <= round(this[7],4):
        # this is the case where this category is a superset of the last category
        et = True
    else:
        # print(f"[FATAL ERROR][python/write_files][congruentCategories] et ranges are not the same: {last[6]} != {this[6]}")
        # print(f"[FATAL ERROR][python/write_files][congruentCategories] et ranges are not the same: {last[7]} != {this[7]}")
        return False
    
    # check if the gain ranges are the same
    if last[8] == this[8]:
        gain = True
    elif last[8] == 0 and this[8] != 0:
        # going from no gain scales to gain scales
        gain = True
    elif last[8] != 0 and this[8] == 0:
        # non gain scales applied on top of gain scales
        gain = True
    else:   
        # print(f"[FATAL ERROR][python/write_files][congruentCategories] gain scales are not the same: {last[8]} != {this[8]}")
        return False

    # if all of the ranges are the same, then the categories are congruent
    return True


def addNewCategory(rowLast, rowThis, thisDict, lastStep, thisStep):
    """
    Add a new category to the dictionary
    ----------
    Args:  
        rowLast: last row
        rowThis: this row
        thisDict: dictionary to add to
        lastStep: last step
        thisStep: this step
    ----------
    Returns:
        thisDict: dictionary with new category added
    ----------
    """

    thisDict['runMin'].append(int(rowLast[0]))
    thisDict['runMax'].append(int(rowLast[1]))

    if rowThis[3]-rowThis[2] < rowLast[3]-rowLast[2]:
        thisDict['etaMin'].append(round(rowThis[2],4))
        thisDict['etaMax'].append(round(rowThis[3],4))
    else:
        thisDict['etaMin'].append(round(rowLast[2],4))
        thisDict['etaMax'].append(round(rowLast[3],4))

    if round(rowLast[4],4) <= round(rowThis[4],4) and round(rowLast[5],4) >= round(rowThis[5],4):
        thisDict['r9Min'].append(round(rowThis[4],4))
        thisDict['r9Max'].append(round(rowThis[5],4))
    else:
        thisDict['r9Min'].append(round(rowLast[4],4))
        thisDict['r9Max'].append(round(rowLast[5],4))

    if round(rowLast[6],4) <= round(rowThis[6],4) and round(rowLast[7],4) >= round(rowThis[7],4):
        thisDict['etMin'].append(round(rowThis[6],4))
        thisDict['etMax'].append(round(rowThis[7],4))
    else:
        thisDict['etMin'].append(round(rowLast[6],4))
        thisDict['etMax'].append(round(rowLast[7],4))
    
    if rowLast[8] != 0:
        thisDict['gain'].append(int(rowLast[8]))
    else: 
        thisDict['gain'].append(int(rowThis[8]))
    thisDict['scale'].append(round(float(rowThis[9])*float(rowLast[9]),6))
    thisDict['err'].append(rowThis[10])


def writeJsonFromDF(thisDF,outFile):

    #Takes the dictionary built in [combine] and writes a json file
    outFile = outFile.replace('.dat','.json')
    print("[INFO][python/write_files][writeJsonFromDF] producing json file in {}".format(outFile))
    thisDict = {}

    for i,row in thisDF.iterrows():
        key_run = 'run:[{},{}]'.format(int(row['runMin']),int(row['runMax']))
        if key_run not in thisDict: thisDict[key_run] = OrderedDict()
        key_eta = 'eta:[{},{}]'.format(row['etaMin'],row['etaMax'])
        if key_eta not in thisDict[key_run]: thisDict[key_run][key_eta] = OrderedDict()
        key_r9 =  'r9:[{},{}]'.format(row['r9Min'],row['r9Max'])
        if key_r9 not in thisDict[key_run][key_eta]: thisDict[key_run][key_eta][key_r9] = OrderedDict()
        key_pt =  'pt:[{},{}]'.format(row['etMin'],row['etMax'])
        if key_pt not in thisDict[key_run][key_eta][key_r9]: thisDict[key_run][key_eta][key_r9][key_pt] = OrderedDict()
        key_gain = 'gain:{}'.format(int(row['gain']))
        if key_gain not in thisDict[key_run][key_eta][key_r9][key_pt]: thisDict[key_run][key_eta][key_r9][key_pt][key_gain] = OrderedDict()

        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]['runMin'] = int(row['runMin'])
        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]['runMax'] = int(row['runMax'])
        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]['etaMin'] = row['etaMin']
        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]['etaMax'] = row['etaMax']
        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]['r9Min'] = row['r9Min']
        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]['r9Max'] = row['r9Max']
        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]['ptMin'] = row['etMin']
        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]['ptMax'] = row['etMax']
        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]['gain'] = int(row['gain'])
        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]['scale'] = row['scale']
        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]['scaleErr'] = row['err']

    with open(outFile,'w') as out:
        json.dump(thisDict,out,indent='\t')

    return

def combine(thisStep, lastStep, outFile):
    """
    Combines the scales from the last step with the current step
    --------------------------------
    Args:
        thisStep: the current step
        lastStep: the last step
        outFile: the output file
    --------------------------------
    Returns:
        None
    --------------------------------
    """
    
    print("[INFO][python/write_files][combine] producing combined scales file from {} and {}".format(thisStep, lastStep))
    print("[INFO][python/write_files][combine] this output will be written to {}".format(outFile))
    dfThisStep = pd.read_csv(thisStep, delimiter='	', header=None,dtype=float)
    dfLastStep = pd.read_csv(lastStep, delimiter='	', header=None,dtype=float)
    headers = ['runMin', 'runMax', 'etaMin', 'etaMax', 'r9Min', 'r9Max', 'etMin', 'etMax', 'gain', 'scale', 'err']
    dictForDf = OrderedDict.fromkeys(headers) #python hates you and your dictionaries
    for col in headers:
        dictForDf[col] = []

    #format is: runMin runMax etaMin etaMax r9Min r9Max etMin etMax gain val err
    for iLast,rowLast in dfLastStep.iterrows():
        for iThis,rowThis in dfThisStep.iterrows():
            #only build an entry if the two categories are congruent
            kCongruent = congruentCategories(rowLast, rowThis, lastStep, thisStep)
            if int(kCongruent) == -999:
                #now you've done it
                print("[FATAL ERROR][python/write_files][combine] Since this isn't working, let's just stop")
                return
            #builds the new categories by reference in dictForDf
            if kCongruent:
                addNewCategory(rowLast, rowThis, dictForDf, lastStep, thisStep)

    dfOut = pd.DataFrame(dictForDf)
    dfOut.to_csv(outFile, sep='	', header=False,index=False)

    writeJsonFromDF(dfOut,outFile)

    return


def write_scales(scales, cats, out):
    """
    Writes the scales to a file
    --------------------------------
    Args:
        scales: the scales (list)
        cats: the categories (pandas dataframe)
        out: the output file (string)
    --------------------------------
    Returns:
        None
    --------------------------------
    """
    #format of onlystepX files is 
    #000000 999999 etaMin etaMax r9Min r9Max etMin etMax gain val err
    headers = ['runMin', 'runMax', 'etaMin', 'etaMax', 'r9Min', 'r9Max', 'etMin', 'etMax', 'gain', 'scale', 'err']
    dictForDf = OrderedDict.fromkeys(headers) #python hates you and your dictionaries
    for col in headers:
        dictForDf[col] = []
    
    print(scales)
    print(len(scales),sum(cats.loc[:,0] == 'scale'))
    for index,row in cats.iterrows():
        if row[0] != 'smear':
            dictForDf['runMin'].append('000000')
            dictForDf['runMax'].append('999999')
            dictForDf['etaMin'].append(row[1])
            dictForDf['etaMax'].append(row[2])
            dictForDf['r9Min'].append(row[3] if row[3] != -1 else 0)
            dictForDf['r9Max'].append(row[4] if row[4] != -1 else 10)
            dictForDf['etMin'].append(row[6] if row[6] != -1 else 0)
            dictForDf['etMax'].append(row[7] if row[7] != -1 else 14000)
            dictForDf['gain'].append(row[5] if row[5] != -1 else 0)
            dictForDf['scale'].append(scales[index])
            dictForDf['err'].append(5e-05)

    dfOut = pd.DataFrame(dictForDf)
    dfOut.to_csv(out, sep='\t',header=False,index=False)
    

def write_smearings(smears, cats, out):
    """
    Writes the smearings to a file
    --------------------------------
    Args:
        smears: the smearings (list)
        cats: the categories (pandas dataframe)
        out: the output file (string)
    --------------------------------
    Returns:
        None
    --------------------------------
    """
    #format of smearings files is:
    #category       Emean   err_Emean   rho err_rho     phi err_phi
    headers = ['#category', 'Emean', 'err_Emean', 'rho', 'err_rho', 'phi', 'err_phi']
    dictForDf = OrderedDict.fromkeys(headers) #python hates you and your dictionaries
    for col in headers:
        dictForDf[col] = []

    smear_mask = cats.loc[:,0] == 'smear'
    
    for index,row in cats[smear_mask].iterrows():
        if row[0] != 'scale':
            if row[3] == -1: 
                row[3] = 0
                row[4] = 10
            if row[6] != -1:
                dictForDf['#category'].append(f"absEta_{row[1]}_{row[2]}-R9_{round(row[3],4)}_{row[4]}-Et_{row[6]}_{row[7]}")
            else:
                dictForDf['#category'].append(f"absEta_{row[1]}_{row[2]}-R9_{round(row[3],4)}_{row[4]}")
            dictForDf['Emean'].append(6.6)
            dictForDf['err_Emean'].append(0.0)
            dictForDf['rho'].append(round(smears[index],5))
            dictForDf['err_rho'].append(round(smears[index]*0.005,5))
            dictForDf['phi'].append('M_PI_2')
            dictForDf['err_phi'].append('M_PI_2')
    
    dfOut = pd.DataFrame(dictForDf)
    dfOut.to_csv(out, sep='\t',header=True,index=False)


def rewrite_smearings(cats, out):
    """
    Rewrites the smearings to a file
    --------------------------------
    Args:
        cats: the categories (pandas dataframe)
        out: the output file (string)
    --------------------------------
    Returns:
        None
    --------------------------------
    """
    #format of smearings files is:
    #category       Emean   err_Emean   rho err_rho     phi err_phi
    headers = ['#category', 'Emean', 'err_Emean', 'rho', 'err_rho', 'phi', 'err_phi']
    dictForDf = OrderedDict.fromkeys(headers) #python hates you and your dictionaries
    for col in headers:
        dictForDf[col] = []

    _cats = pd.read_csv(cats, delimiter='\t', header=None, comment='#')
    mask_smears = (_cats.loc[:,0] == 'smear') 
    smear_df = pd.read_csv(out, delimiter='\t', header=None, comment='#')
    smears = np.array(smear_df.loc[:,3].values)
    num_scales = np.sum(~mask_smears)
    
    for index,row in _cats.loc[_cats[:][0]=='smear'].iterrows():
        if row[0] != 'scale':
            if row[3] == -1: 
                row[3] = 0
                row[4] = 10
            dictForDf['#category'].append(str("absEta_"+str(row[1])+"_"+str(row[2])+"-R9_"+str(round(row[3],4))+"_"+str(row[4])))
            if row[6] != -1:
                dictForDf['#category'][-1] = str("absEta_"+str(row[1])+"_"+str(row[2])+"-R9_"+str(round(row[3],4))+"_"+str(row[4])+"-Et_"+str(row[6])+"_"+str(row[7]))
            dictForDf['Emean'].append(6.6)
            dictForDf['err_Emean'].append(0.0)
            dictForDf['rho'].append(smears[index-num_scales])
            dictForDf['err_rho'].append(0.00005)
            dictForDf['phi'].append('M_PI_2')
            dictForDf['err_phi'].append('M_PI_2')

    
    dfOut = pd.DataFrame(dictForDf)
    dfOut.to_csv(out, sep='\t',header=True,index=False)


def write_runs(runs, out):
    """
    Writes the runs to a file
    --------------------------------
    Args:
        runs: the runs (list)
        out: the output file (string)
    --------------------------------
    Returns:
        None
    --------------------------------
    """
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(out) # catch if the datFiles/ directory doesn't exist 


    headers = ['runMin', 'runMax']
    dictForDf = OrderedDict.fromkeys(headers) # python hates you and your dictionaries
    for col in headers:
        dictForDf[col] = []
    
    for pair in runs:
        dictForDf['runMin'].append(pair[0])
        dictForDf['runMax'].append(pair[1])

    dfOut = pd.DataFrame(dictForDf)
    dfOut.to_csv(out, sep='\t',header=False,index=False)


def write_time_stability(scales, runs, outFile):
    """
    Writes the time stability scales to a file
    --------------------------------
    Args:
        scales: the scales (list)
        runs: the runs (list)
        outFile: the output file (string)
    --------------------------------
    Returns:
        None
    --------------------------------
    """
    if outFile.find('scales') == -1:
        outFile.replace(".dat","_scales.dat")
    print("[INFO][python/write_files][write_time_stability] Writing time stability scales to {}".format(outFile))

    headers = ['runMin', 'runMax', 'etaMin', 'etaMax', 'r9Min', 'r9Max', 'etMin', 'etMax', 'gain', 'scale', 'err']
    dictForDf = OrderedDict.fromkeys(headers) #python hates you and your dictionaries
    for col in headers:
        dictForDf[col] = []
    cats = pd.read_csv(runs, delimiter='\t', header=None) 
    for index,row in cats.iterrows():
        if row[0] != 'smear':
            for i in range(len(scales)):
                dictForDf['runMin'].append(row[0])
                dictForDf['runMax'].append(row[1])
                dictForDf['etaMin'].append(dc.time_stability_eta_bins_low[i])
                dictForDf['etaMax'].append(dc.time_stability_eta_bins_high[i])
                dictForDf['r9Min'].append(0)
                dictForDf['r9Max'].append(10)
                dictForDf['etMin'].append(0)
                dictForDf['etMax'].append(14000)
                dictForDf['gain'].append(0)
                dictForDf['scale'].append(scales[i][index])
                dictForDf['err'].append(0.00005)

    dfOut = pd.DataFrame(dictForDf)
    dfOut.to_csv(outFile, sep='\t',header=False,index=False)


def write_weights(basename, weights, x_edges, y_edges):
    """ 
    writes weights to a tsv file:
    ----------
    Args:
        basename: name of the file to write to
        weights: weights to write
        x_edges: x edges of the weights
        y_edges: y edges of the weights
    ----------
    Returns:
        out: path to the file written
    ----------
    """
    headers = dc.PTY_WEIGHT_HEADERS
    dictForDf = OrderedDict.fromkeys(headers) #python hates you and your dictionaries
    for col in headers:
        dictForDf[col] = []

    for i,row in enumerate(weights):
        row = np.ravel(row)
        for j,weight in enumerate(row):
            dictForDf[dc.YMIN].append(x_edges[i])
            dictForDf[dc.YMAX].append(x_edges[i+1])
            dictForDf[dc.PTMIN].append(y_edges[j])
            dictForDf[dc.PTMAX].append(y_edges[j+1])
            dictForDf[dc.WEIGHT].append(weight)

    out = f"{ss_config.DEFAULT_WRITE_FILES_PATH}ptz_x_rapidity_weights_"+basename+".tsv"
    df_out = pd.DataFrame(dictForDf)
    df_out.to_csv(out, sep='\t', index=False)
    return out