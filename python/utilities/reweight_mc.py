import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import multiprocessing as mp
from collections import OrderedDict

import python.classes.const_class_pyval as constants

def get_zpt(df):
    #calculates the transverse momentum of the dielectron event

    #constants
    c = constants.const()

    theta_lead = 2*np.arctan(np.exp(-1*np.array(df[c.ETA_LEAD].values)))
    theta_sub = 2*np.arctan(np.exp(-1*np.array(df[c.ETA_SUB].values)))
    p_lead_x = np.multiply(np.array(df[c.E_LEAD].values), 
            np.multiply(np.sin(theta_lead),np.cos(np.array(df[c.PHI_LEAD].values))))
    p_lead_y = np.multiply(df[c.E_LEAD].values, 
            np.multiply(np.sin(theta_lead),np.sin(df[c.PHI_LEAD].values)))
    p_sub_x = np.multiply(df[c.E_LEAD].values, 
            np.multiply(np.sin(theta_sub),np.cos(df[c.PHI_SUB].values)))
    p_sub_y = np.multiply(df[c.E_SUB].values, np.multiply(np.sin(theta_sub),np.sin(df[c.PHI_SUB].values)))

    return np.sqrt( np.add( np.multiply( np.add(p_lead_x,p_sub_x), np.add(p_lead_x,p_sub_x)), np.multiply( np.add(p_lead_y,p_sub_y), np.add(p_lead_y,p_sub_y))))

def get_rapidity(df):
    #calculates the rapidity of the dielectron event
    #constants
    c = constants.const()

    theta_lead = 2*np.arctan(np.exp(-1*np.array(df[c.ETA_LEAD].values)))
    theta_sub = 2*np.arctan(np.exp(-1*np.array(df[c.ETA_SUB].values)))

    p_lead_z = np.multiply(df[c.E_LEAD].values,np.cos(theta_lead))
    p_sub_z = np.multiply(df[c.E_SUB].values,np.cos(theta_sub))

    z_pz = np.add(p_lead_z,p_sub_z)
    z_energy = np.add(df[c.E_LEAD].values, df[c.E_SUB].values)

    return np.abs(0.5*np.log( np.divide( np.add(z_energy, z_pz), np.subtract(z_energy, z_pz))))

def write_weights(basename, weights, x_edges, y_edges):
    #writes the weights into a tsv
    headers = ['y_min', 'y_max', 'pt_min', 'pt_max', 'weight']
    dictForDf = OrderedDict.fromkeys(headers) #python hates you and your dictionaries
    for col in headers:
        dictForDf[col] = []

    for i,row in enumerate(weights):
        row = np.ravel(row)
        for j,weight in enumerate(row):
            dictForDf['y_min'].append(x_edges[i])
            dictForDf['y_max'].append(x_edges[i+1])
            dictForDf['pt_min'].append(y_edges[j])
            dictForDf['pt_max'].append(y_edges[j+1])
            dictForDf['weight'].append(weight)

    out = "datFiles/ptz_x_rapidity_weights_"+basename+".tsv"
    df_out = pd.DataFrame(dictForDf)
    df_out.to_csv(out, sep='\t', index=False)
    return out
    
def derive_pt_y_weights(df_data, df_mc, basename):
    #derives and writes the 2D Y(Z),Pt(Z) weights
    print("[INFO][python/reweight_pt_y][derive_pt_y_weights] deriving pt y weights")
    
    ptz_bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35,36,37,38,39,40,45,50,55,60,80,100,14000]
    yz_bins = [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5]

    #calculate pt(z) and y(z) for each event
    zpt_data = get_zpt(df_data)
    zpt_mc = get_zpt(df_mc)

    y_data = get_rapidity(df_data)
    y_mc = get_rapidity(df_mc)

    pt_hist, x_edges_pt = np.histogram(zpt_data, bins=ptz_bins)
    yz_hist, x_edges_y = np.histogram(y_data, bins=yz_bins)

    d_hist, d_hist_x_edges, d_hist_y_edges = np.histogram2d(y_data, zpt_data, [yz_bins,ptz_bins])
    m_hist, m_hist_x_edges, m_hist_y_edges = np.histogram2d(y_mc, zpt_mc, [yz_bins,ptz_bins])

    d_hist /= np.sum(d_hist)
    m_hist /= np.sum(m_hist)

    weights = np.divide(d_hist, m_hist)
    weights /= np.sum(weights)
    
    return write_weights(basename, weights, d_hist_x_edges, d_hist_y_edges)

def add(arg): 
    #adds the pt x rapidity weights to the df
    df, weights = arg

    i_rapidity_min = 0
    i_rapidity_max = 1
    i_ptz_min = 2
    i_ptz_max = 3
    i_weight = 4

    def find_weight(ptz):
        mask_ptz = (weights[:,i_ptz_min] <= ptz) & (ptz < weights[:,i_ptz_max])
        return np.ravel(weights[mask_ptz])[i_weight] if any(mask_ptz) else 0.

    return df.apply(find_weight)
        
def add_pt_y_weights(df, weight_file):
    #constants
    c = constants.const()

    print(min(df['invMass_ECAL_ele'].values))
    print(max(df['invMass_ECAL_ele'].values))
    #adds the pt x y weight as a column to the df
    print("[INFO][python/reweight_pt_y][add_pt_y_weights] applying weights from {}".format(weight_file))
    ptz = np.array(get_zpt(df))
    rapidity = np.array(get_rapidity(df))
    rapidity[np.isinf(rapidity)] = -999
    rapidity[np.isnan(rapidity)] = -999
    df['ptZ'] = ptz
    df['rapidity'] = rapidity

    df.drop([c.PHI_LEAD, c.PHI_SUB], axis=1, inplace=True)

    df_weight = pd.read_csv(weight_file, delimiter='\t', dtype=np.float32)
    df['pty_weight'] = np.zeros(len(df.iloc[:,0].values))

    #split by rapidity
    y_low = df_weight.loc[:,'y_min'].unique().tolist()
    y_high = df_weight.loc[:,'y_max'].unique().tolist()
    divided_df = [(df.loc[(df['rapidity'].values >= y_low[i]) & (df['rapidity'].values < y_high[i])])['ptZ'] for i in range(len(y_low))]

    #pack up the divided dataframe and the corresponding weights
    divided_weights = [(divided_df[i],df_weight.loc[df_weight.loc[:,'y_min'] == y_low[i]].values) for i in range(len(y_low))]

    #ship them off to multiple cores
    processors = mp.cpu_count() - 1
    pool = mp.Pool(processes=processors) 
    scaled_data=pool.map(add, divided_weights) 
    pool.close()
    pool.join()

    df['pty_weight'] = pd.concat(scaled_data).values
    df.drop(['ptZ', 'rapidity'], axis=1, inplace=True)

    return df
