import numpy as np
import pandas as pd
import uproot as up

import python.classes.const_class as constants

c = constants.const()

def prune(files, out, out_dir):
    #files is a text file with a list of data and mc root files
    print("[INFO][python/pruner][prune] You've chose to prune the files listed in {}".format(files))
    print("[INFO][python/pruner][prune] The resulting csv files will be given a name based on {}".format(out))
    print("[INFO][python/pruner][prune] The resulting csv files will be written to the directory {}".format(out_dir))
    files = open(files, 'r').readlines()
    files = [x.strip() for x in files]

    keep_cols = ['R9Ele', 'energy_ECAL_ele', 'etaEle', 'phiEle', 'gainSeedSC', 'invMass_ECAL_ele', 'runNumber']
    mc_files = []
    data_files = []

    for line in files:
        #The file format is 
        #data treeName filename
        #sim treeName filename
        #each entry is separated by a \t
        line_list = line.split('\t')
        print("[INFO][python/pruner][prune] Opening {} as a pandas dataframe".format(line_list[2]))
        df = up.open(line_list[2])[line_list[1]].pandas.df(keep_cols)
        
        #drop events in the transition region or outside the tracker
        df[c.ETA_LEAD] = np.abs(df[c.ETA_LEAD].values)
        df[c.ETA_SUB] = np.abs(df[c.ETA_SUB].values)

        mask_lead = np.logical_or(df[c.ETA_LEAD].values < c.MAX_EB, c.MIN_EE < df[c.ETA_LEAD].values)
        mask_lead = np.logical_and(mask_lead, df[c.ETA_LEAD].values <= 2.5)
        mask_sub = np.logical_or(df[c.ETA_SUB].values < c.MAX_EB, c.MIN_EE < df[c.ETA_SUB].values)
        mask_sub = np.logical_and(mask_sub, df[c.ETA_SUB].values <= 2.5)

        df = df[np.logical_and(mask_lead,mask_sub)]

        #drop events which are non-sensical
        energy_mask = np.logical_and( df[c.E_LEAD].values > 0, df[c.E_LEAD].values < 14000)
        energy_mask = np.logical_and( energy_mask, np.logical_and( df[c.E_SUB].values > 0, df[c.E_SUB].values < 14000))
        df = df[energy_mask]

        #drop events with invmass less than 60 or greater than 120
        invmass_mask = np.logical_and(c.invmass_min < df[c.INVMASS].values, df[c.INVMASS].values < c.invmass_max)
        df = df[invmass_mask]
        drop_list = ['R9Ele[2]', 'energy_ECAL_ele[2]', 'etaEle[2]', 'phiEle[2]', 'gainSeedSC[2]']
        df.drop(drop_list,axis=1,inplace=True)

        if line_list[0] == 'data': data_files.append(df)
        else: mc_files.append(df)


    data = pd.concat(data_files)
    mc = pd.concat(mc_files)

    #write the files into csv files
    print("[INFO][python/pruner][prune] Writing files")
    data.to_csv(str(out_dir+out+"_data.csv"), sep='\t', header=True, index=False)
    mc.to_csv(str(out_dir+out+"_mc.csv"), sep='\t', header=True, index=False)
