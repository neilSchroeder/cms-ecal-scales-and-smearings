import numpy as np
import pandas as pd
from optparse import OptionParser
from collections import OrderedDict

##################################################################################################################
def congruentCategories(last, this, nameLast, nameThis):
    ret = False
    if nameLast.find('step1') != -1:
        #compare eta
        if float(round(last[2],4)) <= float(round(this[2],4)):
            if float(round(last[3],4)) >= float(round(this[3],4)):
                return True
    elif nameLast.find('step2') != -1:
        #compare eta
        etaRet = False
        if float(round(last[2],4)) <= float(round(this[2],4)):
            if float(round(last[3],4)) >= float(round(this[3],4)):
                etaRet = True
        #compare R9
        r9Ret = False
        if float(round(last[4],4)) <= float(round(this[4],4)):
            if float(round(last[5],4)) >= float(round(this[5],4)):
                r9Ret = True
        ret = etaRet and r9Ret
    elif nameLast.find('step3') != -1 or nameLast.find('step5') != -1:
        if nameThis.find('gain') != -1:
            if float(round(this[2],4)) <= float(round(last[2],4)):
                if float(round(this[3],4)) >= float(round(last[3],4)):
                    return True
        else:
            etaRet = False
            if float(round(last[2],4)) <= float(round(this[2],4)):
                if float(round(last[3],4)) >= float(round(this[3],4)):
                    etaRet = True
            #compare R9
            r9Ret = False
            if float(round(this[4],4)) <= float(round(last[4],4)):
                if float(round(this[5],4)) >= float(round(last[5],4)):
                    r9Ret = True

            return r9Ret and etaRet

    elif nameLast.find('step4') != -1 and nameThis.find('step4') != -1:
        if float(round(this[2],4)) <= float(round(last[2],4)) and float(round(this[3],4)) >= float(round(last[3],4)):
            if float(round(this[6],4)) <= float(round(last[6],4)) and float(round(this[7],4)) >= float(round(last[7],4)):
                return True

        
    elif nameThis.find('step5') != -1:
        if float(round(last[2],4)) <= float(round(this[2])) and float(round(last[3],4)) <= float(round(this[3],4)):
            if float(round(last[6],4)) == float(round(this[6],4)) and float(round(last[7],4)) >= float(round(this[7],4)):
                return True
    else:
        print("[ERROR] function 'congruentCategories()' in module 'python/combineSteps.py' not configured for this setup.")
        ret = -999
    
    return ret

##################################################################################################################
def addNewCategory(rowLast, rowThis, thisDict, lastStep, thisStep):
    if thisStep.find('step2closure') != -1 and lastStep.find('step2') == -1: #not setep 4 
        thisDict['runMin'].append(int(rowLast[0]))
        thisDict['runMax'].append(int(rowLast[1]))
        
        thisDict['etaMin'].append(round(rowThis[2],4))
        thisDict['etaMax'].append(round(rowThis[3],4))
        
        thisDict['r9Min'].append(rowThis[4])
        if round(float(rowThis[5]),4) == 1.00:
            thisDict['r9Max'].append('100.000')
        else:
            thisDict['r9Max'].append(rowThis[5])
        
        thisDict['etMin'].append(rowThis[6])
        thisDict['etMax'].append(rowThis[7])
        
        thisDict['gain'].append(int(rowThis[8]))
        
        thisDict['scale'].append(round(float(rowThis[9])*float(rowLast[9]),6))
        
        if thisStep.find('step2') != -1:
            thisDict['err'].append(round(np.sqrt(float(rowThis[10])**2 + float(rowLast[10]/100)**2),6))
        else:
            thisDict['err'].append(round(np.sqrt(float(rowThis[10])**2 + float(rowLast[10]/100)**2),6))

    elif thisStep.find('step3') != -1:
        thisDict['runMin'].append(int(rowLast[0]))
        thisDict['runMax'].append(int(rowLast[1]))
        
        thisDict['etaMin'].append(round(rowThis[2],4))
        thisDict['etaMax'].append(round(rowThis[3],4))
        
        thisDict['r9Min'].append(rowThis[4])
        thisDict['r9Max'].append(rowThis[5])
        
        thisDict['etMin'].append(rowThis[6])
        thisDict['etMax'].append(rowThis[7])
        
        thisDict['gain'].append(int(rowThis[8]))
        
        thisDict['scale'].append(round(float(rowThis[9])*float(rowLast[9]),6))
        
        thisDict['err'].append(round(np.sqrt(float(rowThis[10])**2 + float(rowLast[10]/100)**2),6))

    elif thisStep.find('step6') != -1 and lastStep.find('step4') != -1: #not setep 4 
        thisDict['runMin'].append(int(rowLast[0]))
        thisDict['runMax'].append(int(rowLast[1]))
        
        thisDict['etaMin'].append(round(rowLast[2],4))
        thisDict['etaMax'].append(round(rowLast[3],4))
        
        thisDict['r9Min'].append(rowLast[4])
        if round(float(rowLast[5]),4) == 1.00:
            thisDict['r9Max'].append('100.000')
        else:
            thisDict['r9Max'].append(rowLast[5])
        
        thisDict['etMin'].append(rowLast[6])
        thisDict['etMax'].append(rowLast[7])
        
        thisDict['gain'].append(int(rowThis[8]))
        
        thisDict['scale'].append(round(float(rowThis[9])*float(rowLast[9]),6))
        
        if thisStep.find('step2') != -1:
            thisDict['err'].append(round(np.sqrt(float(rowThis[10])**2 + float(rowLast[10]/100)**2),6))
        else:
            thisDict['err'].append(round(np.sqrt(float(rowThis[10])**2 + float(rowLast[10]/100)**2),6))


    elif (thisStep.find('step4') != -1 ) and thisStep.find('et') != -1 and lastStep.find('step3') != -1: #step 4 
        thisDict['runMin'].append(int(rowLast[0]))
        thisDict['runMax'].append(int(rowLast[1]))
        
        thisDict['etaMin'].append(round(rowLast[2],4))
        thisDict['etaMax'].append(round(rowLast[3],4))
        
        thisDict['r9Min'].append(rowLast[4])
        thisDict['r9Max'].append(rowLast[5])
        
        thisDict['etMin'].append(rowThis[6])
        thisDict['etMax'].append(rowThis[7])
        
        thisDict['gain'].append(int(rowThis[8]))
        
        thisDict['scale'].append(float(rowThis[9])*float(rowLast[9]))
        thisDict['err'].append(round(np.sqrt(float(rowThis[10])**2 + float(rowLast[10]/100)**2),6))

    elif (thisStep.find('step4') != -1 ) and thisStep.find('et') != -1 and lastStep.find('step4') != -1: #step 4 
        thisDict['runMin'].append(int(rowLast[0]))
        thisDict['runMax'].append(int(rowLast[1]))
        
        thisDict['etaMin'].append(round(rowLast[2],4))
        thisDict['etaMax'].append(round(rowLast[3],4))
        
        thisDict['r9Min'].append(rowLast[4])
        thisDict['r9Max'].append(rowLast[5])
        
        thisDict['etMin'].append(rowLast[6])
        thisDict['etMax'].append(rowLast[7])
        
        thisDict['gain'].append(int(rowThis[8]))
        
        thisDict['scale'].append(float(rowThis[9])*float(rowLast[9]))
        thisDict['err'].append(round(np.sqrt(float(rowThis[10])**2 + float(rowLast[10]/100)**2),6))

    elif (thisStep.find('step5') != -1 ) and thisStep.find('et') != -1: #step 4 
        thisDict['runMin'].append(int(rowLast[0]))
        thisDict['runMax'].append(int(rowLast[1]))
        
        thisDict['etaMin'].append(round(rowLast[2],4))
        thisDict['etaMax'].append(round(rowLast[3],4))
        
        thisDict['r9Min'].append(rowLast[4])
        thisDict['r9Max'].append(rowLast[5])
        
        thisDict['etMin'].append(rowLast[6])
        thisDict['etMax'].append(rowLast[7])
        
        thisDict['gain'].append(int(rowLast[8]))
        
        thisDict['scale'].append(float(rowThis[9])*float(rowLast[9]))
        thisDict['err'].append(round(np.sqrt(float(rowThis[10])**2 + float(rowLast[10]/100)**2),6))

    elif thisStep.find('step4') != -1 and thisStep.find('gain') != -1: #step 4 
        thisDict['runMin'].append(int(rowLast[0]))
        thisDict['runMax'].append(int(rowLast[1]))
        
        thisDict['etaMin'].append(round(rowLast[2],4))
        thisDict['etaMax'].append(round(rowLast[3],4))
        
        thisDict['r9Min'].append(rowLast[4])
        thisDict['r9Max'].append(rowLast[5])
        
        thisDict['etMin'].append(rowLast[6])
        thisDict['etMax'].append(rowLast[7])
        
        thisDict['gain'].append(int(rowThis[8]))
        
        thisDict['scale'].append(float(rowThis[9])*float(rowLast[9]))
        thisDict['err'].append(round(np.sqrt(float(rowThis[10])**2 + float(rowLast[10]/100)**2),6))

    else:
        #for step4 you have to use the previous step's categories everywhere
        #this is because step4 is coarser than step2 or step3
        thisDict['runMin'].append(int(rowLast[0]))
        thisDict['runMax'].append(int(rowLast[1]))
        
        thisDict['etaMin'].append(round(rowThis[2],4))
        thisDict['etaMax'].append(round(rowThis[3],4))
        
        thisDict['r9Min'].append(rowThis[4])
        if round(float(rowThis[5]),4) == 1.00:
            thisDict['r9Max'].append('10.000')
        else:
            thisDict['r9Max'].append(rowThis[5])
        
        thisDict['etMin'].append(rowThis[6])
        thisDict['etMax'].append(rowThis[7])
        
        thisDict['gain'].append(int(rowThis[8]))
        
        thisDict['scale'].append(round(float(rowLast[9])*float(rowThis[9]),6))
        
        if thisStep.find('step2') != -1:
            thisDict['err'].append(round(np.sqrt(float(rowThis[10])**2 + float(rowLast[10]/100)**2),6))
        else:
            thisDict['err'].append(round(np.sqrt(float(rowThis[10])**2 + float(rowLast[10]/100)**2),6))


##################################################################################################################
def combine(thisStep, lastStep, outFile):
    
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


##################################################################################################################
def write_scales(scales, cats, out):
#format of onlystepX files is 
#000000 999999 etaMin etaMax r9Min r9Max etMin etMax gain val err
    headers = ['runMin', 'runMax', 'etaMin', 'etaMax', 'r9Min', 'r9Max', 'etMin', 'etMax', 'gain', 'scale', 'err']
    dictForDf = OrderedDict.fromkeys(headers) #python hates you and your dictionaries
    for col in headers:
        dictForDf[col] = []
    
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
            dictForDf['err'].append(0.00005)

    dfOut = pd.DataFrame(dictForDf)
    dfOut.to_csv(out, sep='\t',header=False,index=False)
    
##################################################################################################################
def write_smearings(smears, cats, out):
    #format of smearings files is:
#category       Emean   err_Emean   rho err_rho     phi err_phi
    headers = ['#category', 'Emean', 'err_Emean', 'rho', 'err_rho', 'phi', 'err_phi']
    dictForDf = OrderedDict.fromkeys(headers) #python hates you and your dictionaries
    for col in headers:
        dictForDf[col] = []
    
    for index,row in cats.loc[cats[:][0]=='smear'].iterrows():
        if row[0] != 'scale':
            if row[3] == -1: 
                row[3] = 0
                row[4] = 10
            dictForDf['#category'].append(str("absEta_"+str(row[1])+"_"+str(row[2])+"-R9_"+str(round(row[3],4))+"_"+str(row[4])))
            dictForDf['Emean'].append(6.6)
            dictForDf['err_Emean'].append(0.0)
            dictForDf['rho'].append(smears[index])
            dictForDf['err_rho'].append(0.00005)
            dictForDf['phi'].append('M_PI_2')
            dictForDf['err_phi'].append('M_PI_2')
    
    dfOut = pd.DataFrame(dictForDf)
    dfOut.to_csv(out, sep='\t',header=True,index=False)

##################################################################################################################
def write_runs(runs, out):
    headers = ['runMin', 'runMax']
    dictForDf = OrderedDict.fromkeys(headers) #python hates you and your dictionaries
    for col in headers:
        dictForDf[col] = []
    
    for pair in runs:
        dictForDf['runMin'].append(pair[0])
        dictForDf['runMax'].append(pair[1])

    dfOut = pd.DataFrame(dictForDf)
    dfOut.to_csv(out, sep='\t',header=False,index=False)

##################################################################################################################
def write_time_stability(scales, runs, outFile):
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

