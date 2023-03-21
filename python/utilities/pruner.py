import numpy as np
import pandas as pd
import uproot as up

from python.classes.constant_classes import DataConstants as dc

def prune(files, out, out_dir):
    """ 
    prunes the files listed in files and writes the resulting csv files to out_dir using out as a tag 
    --------------------------
    Input:
    files: a text file containing the files to be pruned
    out: a string to be used as a tag for the resulting csv files
    out_dir: the directory to write the resulting csv files to
    --------------------------
    Returns:
    None
    --------------------------
    """
    INFO = "[INFO][python/pruner][prune]"

    print(f"{INFO} You've chose to prune the files listed in {files}")
    print(f"{INFO} The resulting csv files will be given a name based on {out}")
    print(f"{INFO} The resulting csv files will be written to the directory {out_dir}")
    files = open(files, 'r').readlines()
    files = [x.strip() for x in files]

    keep_cols = dc.KEEP_COLS
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
        df[dc.ETA_LEAD] = np.abs(df[dc.ETA_LEAD].values)
        df[dc.ETA_SUB] = np.abs(df[dc.ETA_SUB].values)

        mask_lead = np.logical_or(df[dc.ETA_LEAD].values < dc.MAX_EB, dc.MIN_EE < df[dc.ETA_LEAD].values)
        mask_lead = np.logical_and(mask_lead, df[dc.ETA_LEAD].values <= dc.MAX_EE)
        mask_sub = np.logical_or(df[dc.ETA_SUB].values < dc.MAX_EB, dc.MIN_EE < df[dc.ETA_SUB].values)
        mask_sub = np.logical_and(mask_sub, df[dc.ETA_SUB].values <= dc.MAX_EE)

        df = df[np.logical_and(mask_lead,mask_sub)]

        #drop events which are non-sensical
        energy_mask = np.logical_and( df[dc.E_LEAD].values > dc.MIN_E, df[dc.E_LEAD].values < dc.MAX_E)
        energy_mask = np.logical_and( energy_mask, 
                                     np.logical_and( df[dc.E_SUB].values > dc.MIN_E, df[dc.E_SUB].values < dc.MAX_E))
        df = df[energy_mask]

        #drop events with invmass less than 60 or greater than 120
        invmass_mask = np.logical_and(dc.invmass_min < df[dc.INVMASS].values, df[dc.INVMASS].values < dc.invmass_max)
        df = df[invmass_mask]
        drop_list = dc.DROP_LIST
        df.drop(drop_list,axis=1,inplace=True)

        if line_list[0] == 'data': data_files.append(df)
        else: mc_files.append(df)


    data = pd.concat(data_files)
    mc = pd.concat(mc_files)

    #write the files into csv files
    print("[INFO][python/pruner][prune] Writing files")
    data.to_csv(str(out_dir+out+"_data.csv"), sep='\t', header=True, index=False)
    mc.to_csv(str(out_dir+out+"_mdc.csv"), sep='\t', header=True, index=False)
