import pandas as pd
import argparse as ap
import numpy as np

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
        eta_mask_low = scales.loc[:,2] >= row[2]
        eta_mask_high = scales.loc[:,3] <= row[3]
        scales.loc[eta_mask_low&eta_mask_high,10] = row[9]*scales.loc[eta_mask_low&eta_mask_high,9]/100.

    scales.to_csv(args.inputFile, sep='\t', header=False,index=False)


main()