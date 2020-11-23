import pandas as pd
import numpy as np
import argparse as ap

import write_files

def main():
    parser = ap.ArgumentParser()

    parser.add_argument("--target", help="Target scales file from which we will do the extracting")
    parser.add_argument("--previous", help="previous scales file from which target scales were derived")
    parser.add_argument("--cats", help="Categories for extraction")
    parser.add_argument("--output", help="output string")

    args = parser.parse_args()

    df_target = pd.read_csv(args.target, sep='\t', header=None, dtype=float)
    df_origin = pd.read_csv(args.previous, sep='\t', header=None, dtype=float)
    df_cats = pd.read_csv(args.cats, sep='\t', header=None, comment='#',dtype={0:str,1:float,2:float,3:float,4:float,5:float,6:float,7:float})
    df_cats.loc[:,1] = round(df_cats.loc[:,1],6)

    extracted_scales = []
    for i_cat,row_cat in df_cats.iterrows():
        for index,row in df_origin.iterrows():
            for index2,row2 in df_target.iterrows():
                if all(row[:4].values == row2[:4].values) and row[4] == 0. and row[0] == df_origin.loc[0,0] and row2[0] == df_origin.loc[0,0]:
                    if any(row2[6:8].values == row_cat[6:].values) and any(row2[2:4].values == row_cat[1:3].values):
                       extracted_scales.append(round(row2[9]/row[9],6))
                       print(round(row2[9]/row[9],6))
                       break
                       break
                if row2[0] != df_origin.iloc[0,0]: break
            if row[0] != df_origin.iloc[0,0]: break

    write_files.write_scales(extracted_scales, df_cats, args.output)

main()
