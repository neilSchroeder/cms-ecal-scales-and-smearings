import pandas as pd

pd.options.mode.chained_assignment = None
import multiprocessing as mp
from collections import OrderedDict

import numpy as np

from python.classes.config_class import SSConfig
from python.classes.constant_classes import DataConstants as dc
from python.utilities.write_files import write_weights

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

    theta_lead = 2 * np.arctan(np.exp(-1 * np.array(df[dc.ETA_LEAD].values)))
    theta_sub = 2 * np.arctan(np.exp(-1 * np.array(df[dc.ETA_SUB].values)))
    p_lead_x = np.multiply(
        np.array(df[dc.E_LEAD].values),
        np.multiply(np.sin(theta_lead), np.cos(np.array(df[dc.PHI_LEAD].values))),
    )
    p_lead_y = np.multiply(
        df[dc.E_LEAD].values,
        np.multiply(np.sin(theta_lead), np.sin(df[dc.PHI_LEAD].values)),
    )
    # Fix p_sub_x to use E_SUB instead of E_LEAD
    p_sub_x = np.multiply(
        df[dc.E_SUB].values,
        np.multiply(np.sin(theta_sub), np.cos(df[dc.PHI_SUB].values)),
    )
    p_sub_y = np.multiply(
        df[dc.E_SUB].values,
        np.multiply(np.sin(theta_sub), np.sin(df[dc.PHI_SUB].values)),
    )

    return np.sqrt(
        np.add(
            np.multiply(np.add(p_lead_x, p_sub_x), np.add(p_lead_x, p_sub_x)),
            np.multiply(np.add(p_lead_y, p_sub_y), np.add(p_lead_y, p_sub_y)),
        )
    )


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

    theta_lead = 2 * np.arctan(np.exp(-1 * np.array(df[dc.ETA_LEAD].values)))
    theta_sub = 2 * np.arctan(np.exp(-1 * np.array(df[dc.ETA_SUB].values)))

    p_lead_z = np.multiply(df[dc.E_LEAD].values, np.cos(theta_lead))
    p_sub_z = np.multiply(df[dc.E_SUB].values, np.cos(theta_sub))

    z_pz = np.add(p_lead_z, p_sub_z)
    z_energy = np.add(df[dc.E_LEAD].values, df[dc.E_SUB].values)

    return np.abs(
        0.5 * np.log(np.divide(np.add(z_energy, z_pz), np.subtract(z_energy, z_pz)))
    )


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
    # derives and writes the 2D Y(Z),Pt(Z) weights
    print("[INFO][python/reweight_pt_y][derive_pt_y_weights] deriving pt y weights")

    ptz_bins = dc.PTZ_BINS
    yz_bins = dc.YZ_BINS

    # calculate pt(z) and y(z) for each event
    zpt_data = get_zpt(df_data)
    zpt_mc = get_zpt(df_mc)

    y_data = get_rapidity(df_data)
    y_mc = get_rapidity(df_mc)

    d_hist, d_hist_x_edges, d_hist_y_edges = np.histogram2d(
        y_data, zpt_data, [yz_bins, ptz_bins]
    )
    m_hist, m_hist_x_edges, m_hist_y_edges = np.histogram2d(
        y_mc, zpt_mc, [yz_bins, ptz_bins]
    )

    d_hist /= np.sum(d_hist)
    m_hist /= np.sum(m_hist)

    weights = np.divide(d_hist, m_hist)
    weights /= np.sum(weights)

    return write_weights(basename, weights, d_hist_x_edges, d_hist_y_edges)


def add_weights_to_df(arg):
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
        weights are divided by rapidity before being provided as arguments,
        so no need to check rapidity compatibility
        """
        mask_ptz = (weights[:, i_ptz_min] <= ptz) & (ptz < weights[:, i_ptz_max])
        # Access the weight directly from the correct column of matching rows
        return weights[mask_ptz, i_weight][0] if any(mask_ptz) else 0.0

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

    print(
        f"[INFO][python/reweight_pt_y][add_pt_y_weights] applying weights from {weight_file}"
    )
    ptz = np.array(get_zpt(df))
    rapidity = np.array(get_rapidity(df))
    rapidity[np.isinf(rapidity)] = -999
    rapidity[np.isnan(rapidity)] = -999
    df[dc.PTZ] = ptz
    df[dc.RAPIDITY] = rapidity

    df.drop([dc.PHI_LEAD, dc.PHI_SUB], axis=1, inplace=True)

    df_weight = pd.read_csv(weight_file, delimiter="\t", dtype=np.float32)
    df[dc.PTY_WEIGHT] = np.zeros(len(df.iloc[:, 0].values))

    # split by rapidity
    y_low = df_weight.loc[:, dc.YMIN].unique().tolist()
    y_high = df_weight.loc[:, dc.YMAX].unique().tolist()
    divided_df = [
        (
            df.loc[
                (df[dc.RAPIDITY].values >= y_low[i])
                & (df[dc.RAPIDITY].values < y_high[i])
            ]
        )[dc.PTZ]
        for i in range(len(y_low))
    ]

    # pack up the divided dataframe and the corresponding weights
    divided_weights = [
        (divided_df[i], df_weight.loc[df_weight.loc[:, dc.YMIN] == y_low[i]].values)
        for i in range(len(y_low))
    ]

    # ship them off to multiple cores
    processors = mp.cpu_count() - 1
    pool = mp.Pool(processes=processors)
    scaled_data = pool.map(add_weights_to_df, divided_weights)
    pool.close()
    pool.join()

    # Create a Series to hold all weights
    all_weights = pd.Series(np.zeros(len(df)), index=df.index)

    # Update weights for each rapidity bin, preserving the original indices
    for weights_series in scaled_data:
        all_weights.update(weights_series)

    # Assign weights to the dataframe
    df[dc.PTY_WEIGHT] = all_weights.values
    print(df)
    df.drop([dc.PTZ, dc.RAPIDITY], axis=1, inplace=True)

    return df
