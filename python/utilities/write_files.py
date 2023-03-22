import json
import numpy as np
import os
import pandas as pd
from collections import OrderedDict


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

    #this is finer than last categories:
    ret_eta = False
    if float(round(last[2],4)) <= float(round(this[2],4)) and float(round(last[3],4)) >= float(round(this[3],4)):
        ret_eta=True
        if nameLast.find('step1') != -1:
            return True
    elif float(round(last[2],4)) >= float(round(this[2],4)) and float(round(last[3],4)) <= float(round(this[3],4)):
        ret_eta=True
    else: pass
    
    ret_r9 = False
    if float(round(last[4],4)) <= float(round(this[4],4)) and float(round(last[5],5)) >= float(round(this[5],4)):
        ret_r9 = True
    elif float(round(last[4],4)) >= float(round(this[4],4)) and float(round(last[5],5)) <= float(round(this[5],4)):
        ret_r9 = True
    else: pass

    ret_gain = True
    if float(round(last[8],4)) != 0:
        ret_gain = False
        if float(round(last[8],4)) == float(round(this[8],4)):
            ret_gain = True

    ret_et = False
    if float(round(last[6],4)) <= float(round(this[6],4)):
        if float(round(last[7],4)) >= float(round(this[7],4)):
                ret_et=True
    if float(round(last[6],4)) >= float(round(this[6],4)):
        if float(round(last[7],4)) <= float(round(this[7],4)):
                ret_et=True

    if 'step2' in nameLast or 'step3' in nameLast or 'RunEtaR9_' in nameLast:
        return ret_r9 and ret_eta

    if 'stochastic' in nameThis or 'step4' in nameLast or 'EtaR9Et_' in nameLast:
        return ret_eta and ret_r9 and ret_et

    if nameThis.find('gain') != -1 or nameThis.find('Gain') != -1:
        if nameLast.find('gain') != -1 or nameLast.find('Gain') != -1:
            return ret_eta and ret_gain
        else:
            return ret_eta
        
    return False


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

##################################################################################################################
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

    smear_mask = cats['#type'] == 'smear'
    
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
                eta_min, eta_max = 0,1
                if i == 1: eta_min, eta_max = 1, 1.4442
                if i == 2: eta_min, eta_max = 1.566, 2
                if i == 3: eta_min, eta_max = 2, 2.5
                dictForDf['etaMin'].append(eta_min)
                dictForDf['etaMax'].append(eta_max)
                dictForDf['r9Min'].append(0)
                dictForDf['r9Max'].append(10)
                dictForDf['etMin'].append(0)
                dictForDf['etMax'].append(14000)
                dictForDf['gain'].append(0)
                dictForDf['scale'].append(scales[i][index])
                dictForDf['err'].append(0.00005)

    dfOut = pd.DataFrame(dictForDf)
    dfOut.to_csv(outFile, sep='\t',header=False,index=False)

