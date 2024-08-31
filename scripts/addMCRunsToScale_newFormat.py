#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse as ap

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


def apply_systs(df, systs):
    """
    Applies the systematics to the dataframe
    ----------
    Args:
        df: dataframe of the scale and smearings
        systs: dataframe of the systematics
    ----------
    Returns:
        None
    ----------
    """

    for i, row in systs.iterrows():
        eta_mask_low = df.loc[:, 2] >= row[0]
        eta_mask_high = df.loc[:, 3] <= row[1]
        r9_mask_low = [True for x in eta_mask_high]
        r9_mask_high = [True for x in eta_mask_high]
        if row[4] != -1:
            r9_mask_low = df.loc[:, 4] >= row[2]
            r9_mask_high = df.loc[:, 5] <= row[3]
        gain_mask = [True for x in eta_mask_high]
        if row[6] != -1:  # gain mask
            gain_mask = df.loc[:, 8] == row[6]

        df.loc[
            eta_mask_low & eta_mask_high & r9_mask_low & r9_mask_high & gain_mask, 10
        ] = (row[7] * df.loc[eta_mask_low & eta_mask_high & gain_mask, 9] / 100.0)

    return df


def main():
    """
    Adds MC run entries to new format scales files
    ----------
    Args:
        -i, --inFile: input file to add MC run entries to
        -s, --syst: systematics file to use
    ----------
    Returns:
        None
    ----------
    """

    parser = ap.ArgumentParser(
        description="Adds MC run entries to new format scales files"
    )
    parser.add_argument(
        "-i", "--inFile", required=True, help="input file to add MC run entries to"
    )
    parser.add_argument("-s", "--syst", required=True, help="systematics file to use")
    arg = parser.parse_args()

    user_input = input("This script will overwrite the input file. Continue? (y/n): ")
    if user_input != "y":
        return

    df = pd.read_csv(arg.inFile, sep="\t", header=None)
    df_syst = pd.read_csv(arg.syst, sep="\t", header=None)

    min_run = np.min(df.loc[:, 0].values)
    max_run = np.max(df.loc[:, 1].values)

    df_run1 = df.loc[df[0] == min_run]
    df_run2 = df.loc[df[0] == min_run]
    df_run3 = df.loc[df[1] == max_run]

    # build run 1 2
    df_run1.loc[:, 0] = np.array([1 for i in range(len(df_run1[0].values))])
    df_run1.loc[:, 1] = np.array([2 for i in range(len(df_run1[1].values))])
    df_run1.loc[:, 9] = np.array([1.0 for i in range(len(df_run1[9].values))])
    df_run1 = apply_systs(df_run1, df_syst)

    # build run 3 minRun
    df_run2.loc[:, 0] = np.array([3 for i in range(len(df_run2[0].values))])
    df_run2.loc[:, 1] = np.array([min_run - 1 for i in range(len(df_run2[0].values))])
    df_run2.loc[:, 9] = np.array([1.0 for i in range(len(df_run2[9].values))])
    df_run2 = apply_systs(df_run2, df_syst)

    # build run maxRun 999999
    df_run3.loc[:, 0] = np.array([max_run + 1 for i in range(len(df_run3[0].values))])
    df_run3.loc[:, 1] = np.array([999999 for i in range(len(df_run3[1].values))])
    df_run3.loc[:, 9] = np.array([1.0 for i in range(len(df_run3[9].values))])
    df_run3 = apply_systs(df_run3, df_syst)

    df_out = pd.concat([df_run1, df_run2, df, df_run3])

    df_out.to_csv(arg.inFile, sep="\t", header=False, index=False)


if __name__ == "__main__":
    main()
