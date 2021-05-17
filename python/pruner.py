import numpy as np
import pandas as pd
import uproot as up

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
        transition_mask_0 = ~df['etaEle[0]'].between(1.4442, 1.566)&(~df['etaEle[0]'].between(-1.566, -1.4442))
        transition_mask_1 = ~df['etaEle[1]'].between(1.4442, 1.566)&(~df['etaEle[1]'].between(-1.566, -1.4442))
        tracker_mask_0 = ~df['etaEle[0]'].between(2.5, 10)&(~df['etaEle[0]'].between(-10, -2.5)) 
        tracker_mask_1 = ~df['etaEle[1]'].between(2.5, 10)&(~df['etaEle[1]'].between(-10, -2.5))
        df = df.loc[transition_mask_0&transition_mask_1&tracker_mask_0&tracker_mask_1]

        #drop events which are non-sensical
        sublead_energy_mask = df['energy_ECAL_ele[0]'].between(0, 14000)&df['energy_ECAL_ele[1]'].between(0, 14000)
        df = df.loc[sublead_energy_mask]

        #drop events with invmass less than 60 or greater than 120
        invmass_mask = df['invMass_ECAL_ele'].values
        invmass_mask = [60 < val and val < 120 for val in invmass_mask]
        df = df.loc[invmass_mask]
        drop_list = ['R9Ele[2]', 'energy_ECAL_ele[2]', 'etaEle[2]', 'gainSeedSC[2]']
        df.drop(drop_list,axis=1,inplace=True)

        if line_list[0] == 'data': data_files.append(df)
        else: mc_files.append(df)


    data = pd.concat(data_files)
    mc = pd.concat(mc_files)

    #write the files into csv files
    print("[INFO][python/pruner][prune] Writing files")
    data.to_csv(str(out_dir+out+"_data.csv"), sep='\t', header=True, index=False)
    mc.to_csv(str(out_dir+out+"_mc.csv"), sep='\t', header=True, index=False)
