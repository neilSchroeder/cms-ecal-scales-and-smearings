"""Adds a flat scale to the provided scales file."""

import pandas as pd
import argparse as ap
import numpy as np

def main():
    """
    Add a flat scale to the provided scales file
    ----------
    Args:
        --scales: scales file
        --scale: scale to add in percent (i.e. 0.2 to add 0.2% to all scales)
        --out: output file
    ----------
    Returns:
        None
    ----------
    """
    parser = ap.ArgumentParser(description="add a flat scale to the provided scales file")

    parser.add_argument(
        "--scales", default="", help="scales file"
    )
    parser.add_argument(
        "--scale",
        default=0,
        type=float,
        help="scale to multiply by",
    )
    parser.add_argument("--out", default="", help="output file")

    args = parser.parse_args()

    df_scales = pd.read_csv(args.scales, delimiter="\t", header=None)
    scale = np.float64(args.scale/100)
    for i, row in df_scales.iterrows():
        df_scales.iloc[i, 9] = df_scales.iloc[i, 9] + scale

    df_scales.to_csv(args.out, sep="\t", index=False, header=False)


if __name__ == "__main__":
    main()