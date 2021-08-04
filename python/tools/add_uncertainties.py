import pandas as pd
import argparse as ap
import numpy as np

"""
adds systematic uncertainties by category
input file must have the following format
min-eta	max-eta	min-r9	max-r9	min-et	max-et	gain	syst[%]
"""

def main():
    parser = ap.ArgumentParser(description="add systematic uncertainties by category")

    parser.add_argument("-i","--inputFile", required=True,
                        help="Scales file which we will add uncertainties to")
    parser.add_argument("-u","--uncFile", required=True,
                        help="Config file containing the uncertainties per category to add")

    args = parser.parse_args()

    scales = pd.read_csv(args.inputFile, sep='\t', header=None)
    unc = pd.read_csv(args.uncFile, sep='\t', header=None)
    
    #make necessary masks
    for i,row in unc.iterrows():
        eta_mask_low = scales.loc[:,2] >= row[0]
        eta_mask_high = scales.loc[:,3] <= row[1]
        r9_mask_low = [True for x in eta_mask_low]
        r9_mask_high = [True for x in eta_mask_high]
        if row[4] != -1:
            r9_mask_low = scales.loc[:,4] >= row[2]
            r9_mask_high = scales.loc[:,5] <= row[3]
        gain_mask = [True for x in eta_mask_high]
        if row[6] != -1: #gain mask
            gain_mask = scales.loc[:,8] == row[6]

        scales.loc[eta_mask_low&eta_mask_high&r9_mask_low&r9_mask_high&gain_mask,10] = row[7]*scales.loc[eta_mask_low&eta_mask_high&gain_mask,9]/100.

    scales.to_csv(args.inputFile, sep='\t', header=False,index=False)


main()
