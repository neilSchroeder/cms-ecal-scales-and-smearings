import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import multiprocessing as mp
from collections import OrderedDict

from python.classes.constant_classes import DataConstants as dc

from python.classes.config_class import SSConfig
ss_config = SSConfig()

def get_zpt(df):
    """ 
    calculates the transverse momentum of the dielectron event 
    ----------
    Args:
        df: dataframe of the event
    ----------
    Returns:
        z_pt: transverse momentum of the event
    ----------
    """

    theta_lead = 2*np.arctan(np.exp(-1*np.array(df[dc.ETA_LEAD].values)))
    theta_sub = 2*np.arctan(np.exp(-1*np.array(df[dc.ETA_SUB].values)))
    p_lead_x = np.multiply(np.array(df[dc.E_LEAD].values), 
            np.multiply(np.sin(theta_lead),np.cos(np.array(df[dc.PHI_LEAD].values))))
    p_lead_y = np.multiply(df[dc.E_LEAD].values, 
            np.multiply(np.sin(theta_lead),np.sin(df[dc.PHI_LEAD].values)))
    p_sub_x = np.multiply(df[dc.E_LEAD].values, 
            np.multiply(np.sin(theta_sub),np.cos(df[dc.PHI_SUB].values)))
    p_sub_y = np.multiply(df[dc.E_SUB].values, np.multiply(np.sin(theta_sub),np.sin(df[dc.PHI_SUB].values)))

    return np.sqrt( np.add( np.multiply( np.add(p_lead_x,p_sub_x), np.add(p_lead_x,p_sub_x)), np.multiply( np.add(p_lead_y,p_sub_y), np.add(p_lead_y,p_sub_y))))

def get_rapidity(df):
    """ 
    calculates the rapidity of the dielectron event 
    ----------
    Args:
        df: dataframe of the event
    ----------
    Returns:
        z_rapidity: rapidity of the event
    ----------
    """

    theta_lead = 2*np.arctan(np.exp(-1*np.array(df[dc.ETA_LEAD].values)))
    theta_sub = 2*np.arctan(np.exp(-1*np.array(df[dc.ETA_SUB].values)))

    p_lead_z = np.multiply(df[dc.E_LEAD].values,np.cos(theta_lead))
    p_sub_z = np.multiply(df[dc.E_SUB].values,np.cos(theta_sub))

    z_pz = np.add(p_lead_z,p_sub_z)
    z_energy = np.add(df[dc.E_LEAD].values, df[dc.E_SUB].values)

    return np.abs(0.5*np.log( np.divide( np.add(z_energy, z_pz), np.subtract(z_energy, z_pz))))

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
    
def derive_pt_y_weights(df_data, df_mc, basename):
    """ 
    derives and writes the 2D Y(Z),Pt(Z) weights
    ----------
    Args:
        df_data: dataframe of data
        df_mc: dataframe of mc
        basename: name of the file to write to
    ----------
    Returns:
        out: path to the file written
    ----------
    """
    #derives and writes the 2D Y(Z),Pt(Z) weights
    print("[INFO][python/reweight_pt_y][derive_pt_y_weights] deriving pt y weights")
    
    ptz_bins = dc.PTZ_BINS
    yz_bins = dc.YZ_BINS

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
    """ 
    adds the pt x rapidity weights to the df 
    ----------
    Args:
        arg: tuple of (df, weights)
    ----------
    Returns:
        df: dataframe with pt x rapidity weights added
    ----------
    """
    df, weights = arg

    i_ptz_min = 2
    i_ptz_max = 3
    i_weight = 4

    def find_weight(ptz):
        """ 
        finds the corresponding weight by ptZ 
        weights are divided by rapidity before being provided as arguments, so no need to check rapidity compatibility
        """
        mask_ptz = (weights[:,i_ptz_min] <= ptz) & (ptz < weights[:,i_ptz_max])
        return np.ravel(weights[mask_ptz])[i_weight] if any(mask_ptz) else 0.

    return df.apply(find_weight)
        
def add_pt_y_weights(df, weight_file):
    """
    Adds the pt x y weight as a column to the df 
    ----------
    Args:
        df: dataframe to add the weights to
        weight_file: path to the weights file
    ----------
    Returns:
        df: dataframe with pt x y weights added
    ----------
    """

    print(f"[INFO][python/reweight_pt_y][add_pt_y_weights] applying weights from {weight_file}")
    ptz = np.array(get_zpt(df))
    rapidity = np.array(get_rapidity(df))
    rapidity[np.isinf(rapidity)] = -999
    rapidity[np.isnan(rapidity)] = -999
    df[dc.PTZ] = ptz
    df[dc.RAPIDITY] = rapidity

    df.drop([dc.PHI_LEAD, dc.PHI_SUB], axis=1, inplace=True)

    df_weight = pd.read_csv(weight_file, delimiter='\t', dtype=np.float32)
    df[dc.PTY_WEIGHT] = np.zeros(len(df.iloc[:,0].values))

    # split by rapidity
    y_low = df_weight.loc[:,dc.YMIN].unique().tolist()
    y_high = df_weight.loc[:,dc.YMAX].unique().tolist()
    divided_df = [(df.loc[(df[dc.RAPIDITY].values >= y_low[i]) & (df[dc.RAPIDITY].values < y_high[i])])[dc.PTZ] for i in range(len(y_low))]

    # pack up the divided dataframe and the corresponding weights
    divided_weights = [(divided_df[i],df_weight.loc[df_weight.loc[:,dc.YMIN] == y_low[i]].values) for i in range(len(y_low))]

    # ship them off to multiple cores
    processors = mp.cpu_count() - 1
    pool = mp.Pool(processes=processors) 
    scaled_data=pool.map(add, divided_weights) 
    pool.close()
    pool.join()

    df[dc.PTY_WEIGHT] = pd.concat(scaled_data).values
    df.drop([dc.PTZ, dc.RAPIDITY], axis=1, inplace=True)

    return df
