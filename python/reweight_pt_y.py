import pandas as pd
import numpy as np

def get_zpt(df):
    #calculates the transverse momentum of the dielectron event
    p_lead_x = np.multiply(df['energy_ECAL_ele[0]'].values, np.cos(df['phiEle[0]'].values))
    p_lead_y = np.multiply(df['energy_ECAL_ele[0]'].values, np.sin(df['phiEle[0]'].values))
    p_sub_x = np.multiply(df['energy_ECAL_ele[1]'].values, np.cos(df['phiEle[1]'].values))
    p_sub_y = np.multiply(df['energy_ECAL_ele[1]'].values, np.sin(df['phiEle[1]'].values))

    return np.sqrt( np.add( np.multiply( np.add(p_lead_x,p_sub_x), np.add(p_lead_x,p_sub_x)), np.multiply( np.add(p_lead_y,p_sub_y), np.add(p_lead_y,p_sub_y))))

def get_rapidity(df):
    #calculates the rapidity of the dielectron event
    p_lead_z = np.multiply(df['energy_ECAL_ele[0]'].values, np.sinh(df['etaEle[0]'].values))
    p_sub_z = np.multiply(df['energy_ECAL_ele[1]'].values, np.sinh(df['etaEle[1]'].values))

    z_pz = np.add(p_lead_z,p_sub_z)
    z_energy = np.add(df['energy_ECAL_ele[0]'].values, df['energy_ECAL_ele[1]'].values)

    return 0.5*np.log( np.divide( np.add(z_energy, z_pz), np.subtract(z_energy, z_pz)))

def write_weights(basename, weights, x_edges, y_edges):
    #writes the weights into a tsv
    headers = ['y_min', 'y_max', 'pt_min', 'pt_max', 'weight']
    dictForDf = OrderedDict.fromkeys(headers) #python hates you and your dictionaries
    for col in headers:
        dictForDf[col] = []

    for i,row in enumerate(weights):
        row = np.ravel(row)
        for j,weight in enumerate(row):
            dictForDf['y_min'] = x_edges[i]
            dictForDf['y_max'] = x_edges[i+1]
            dictForDf['pt_min'] = y_edges[j]
            dictForDf['pt_min'] = y_edges[j+1]
            dictForDf['weight'] = weight

    out = "datFiles/ptz_x_rapidity_weights_"+basename
    df_out.to_csv(out, sep='\t', index=False)
    return out
    
def derive_pt_y_weights(df_data, df_mc, basename):
    #derives and writes the 2D Y(Z),Pt(Z) weights
    
    ptz_bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35,36,37,38,39,40,45,50,55,60,80,100,14000]
    yz_bins = [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5]

    #calculate pt(z) and y(z) for each event
    zpt_data = get_zpt(df_data)
    zpt_mc = get_zpt(df_mc)

    y_data = get_rapidity(df_data)
    y_mc = get_rapidity(df_mc)

    d_hist, d_hist_x_edges, d_hist_y_edges = np.histogram2d(y_data, zpt_data, [yz_bins,ptz_bins])
    m_hist, m_hist_x_edges, m_hist_y_edges = np.histogram2d(y_mc, zpt_mc, [yz_bins,ptz_bins])

    d_hist /= np.sum(d_hist)
    m_hist /= np.sum(m_hist)

    weights = np.divide(d_hist, m_hist)
    weights /= np.sum(weights)
    
    return write_weights(basename, weights, d_hist_x_edges, d_hist_y_edges)
        
def add_pt_y_weights(df, weight_file):
    #adds the pt x y weight as a column to the df
    rapidity = np.array(get_rapidity(df))
    ptz = np.array(get_zpt(df))

    df_weight = pd.read_csv(weight_file, delimiter='\t', dtype=float)
    df['pty_weight'] = np.zeros(len(df.loc[0,:].values))

    for i,row in df_weight.iterrows():
        mask_rapidity = (rapidity > row[0])&(rapidity < row[1])
        mask_ptz = (ptz > row[2])&(ptz < row[3])
        df.loc[mask_rapidity&mask_ptz,'pty_weight'] = row[4]

    return df


