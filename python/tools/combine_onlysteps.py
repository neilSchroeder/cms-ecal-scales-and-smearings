import pandas as pd
import argparse as ap

def main():
    """
    Combine two only step files
    ----------
    Args:
        --new: new file, this will be mixed into the old file
        --old: old file, this will be the base, the new file will be mixed in
        --out: output file
    ----------
    Returns:
        None
    ----------
    """
    parser = ap.ArgumentParser(description="combine two only step files")

    parser.add_argument("--new", default='',
            help="new file, this will be mixed into the old file")
    parser.add_argument("--old", default='',
            help="old file, this will be the base, the new file will be mixed in")
    parser.add_argument("--out", default='',
            help="output file")

    args = parser.parse_args()

    df_new = pd.read_csv(args.new,delimiter='\t',header=None)
    df_old = pd.read_csv(args.old,delimiter='\t',header=None)
    
    for i,row_new in df_new.iterrows():
        for j,row_old in df_old.iterrows():
            if row_new[2] <= row_old[2] and row_new[3] >= row_old[3]:
                if row_new[6] <= row_old[6] and row_new[7] >= row_old[7]:
                    df_old.iloc[j,9] *= df_new.iloc[i,9]

    df_old.to_csv(args.out, sep='\t', index=False, header=False)

main()

