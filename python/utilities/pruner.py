import numpy as np
import os
import pandas as pd
import uproot3 as up

from python.classes.constant_classes import DataConstants as dc

def prune(files, out, out_dir):
    """ 
    prunes the files listed in files and writes the resulting csv files to out_dir using out as a tag 
    --------------------------
    Args:
        files: a text file containing the files to be pruned
        out: a string to be used as a tag for the resulting csv files
        out_dir: the directory to write the resulting csv files to
    --------------------------
    Returns:
        None
    --------------------------
    """
    INFO = "[INFO][python/pruner][prune]"

    print(f"[WARNING] This script requires a lot of memory")
    print(f"[WARNING] if you get a KILLED message, try switching to a machine with more available memory")

    print(f"{INFO} You've chose to prune the files listed in {files}")
    print(f"{INFO} The resulting csv files will be given a name based on {out}")
    print(f"{INFO} The resulting csv files will be written to the directory {out_dir}")
    files = open(files, 'r').readlines()
    files = [x.strip() for x in files]

    keep_cols = dc.KEEP_COLS
    data_index = 0
    mc_index = 0

    for line in files:
        #The file format is 
        #data treeName filename
        #sim treeName filename
        #each entry is separated by a \t
        line_list = line.split('\t')
        print("{} Opening {} as a pandas dataframe".format(INFO,line_list[2]))
        with up.open(line_list[2]) as f:
            df = f[line_list[1]].pandas.df(keep_cols)
            
            # absolute value of eta for convenience
            df[dc.ETA_LEAD] = np.abs(df[dc.ETA_LEAD].values)
            df[dc.ETA_SUB] = np.abs(df[dc.ETA_SUB].values)

            #drop events in the transition region or outside the tracker
            mask_lead = np.logical_or(df[dc.ETA_LEAD].values < dc.MAX_EB, dc.MIN_EE < df[dc.ETA_LEAD].values)
            mask_lead = np.logical_and(mask_lead, df[dc.ETA_LEAD].values <= dc.MAX_EE)
            mask_sub = np.logical_or(df[dc.ETA_SUB].values < dc.MAX_EB, dc.MIN_EE < df[dc.ETA_SUB].values)
            mask_sub = np.logical_and(mask_sub, df[dc.ETA_SUB].values <= dc.MAX_EE)

            df = df[np.logical_and(mask_lead,mask_sub)]

            #drop events which are not in the energy range
            energy_mask = np.logical_and( df[dc.E_LEAD].values > dc.MIN_E, df[dc.E_LEAD].values < dc.MAX_E)
            energy_mask = np.logical_and( energy_mask, 
                                        np.logical_and( df[dc.E_SUB].values > dc.MIN_E, df[dc.E_SUB].values < dc.MAX_E))
            df = df[energy_mask]

            #drop events with invmass less than 60 or greater than 120
            invmass_mask = np.logical_and(dc.invmass_min < df[dc.INVMASS].values, df[dc.INVMASS].values < dc.invmass_max)
            df = df[invmass_mask]
            drop_list = dc.DROP_LIST
            df.drop(drop_list,axis=1,inplace=True)

            data_mc = "data" if line_list[0] == 'data' else 'mc'
            index = data_index if line_list[0] == 'data' else mc_index

            print(f"{INFO} writing to {out_dir}{out}_{data_mc}_{index}.csv")
            df.to_csv(f"{out_dir}{out}_{data_mc}_{index}.csv", sep='\t', header=True, index=False)

            del df

            if line_list[0] == 'data':
                data_index += 1
            else:
                mc_index += 1

    # merge the csv files
    print(f"{INFO} merging csv files")
    with open(f"{out_dir}{out}_data.csv", 'w') as outfile:
        for index in range(data_index):
            with open(f"{out_dir}{out}_data_{index}.csv") as infile:
                for line in infile:
                    if index != 0 and line.startswith("R9"):
                        continue # skip the header
                    outfile.write(line)
            os.remove(f"{out_dir}{out}_data_{index}.csv")
    
    with open(f"{out_dir}{out}_mc.csv", 'w') as outfile:
        for index in range(mc_index):
            with open(f"{out_dir}{out}_mc_{index}.csv") as infile:
                for line in infile:
                    if index != 0 and line.startswith("R9"):
                        continue # skip the header
                    outfile.write(line)
            os.remove(f"{out_dir}{out}_mc_{index}.csv")

    print(f"{INFO} writing config files to config/{out}.cfg")
    with open(f"config/{out}.cfg", 'w') as outfile:
        outfile.write(f"{out_dir}{out}_data.csv\n")
        outfile.write(f"{out_dir}{out}_mc.csv")

    print(f"{INFO} done")
