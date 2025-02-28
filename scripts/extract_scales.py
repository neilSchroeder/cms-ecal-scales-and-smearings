import pandas as pd
import argparse as ap

import src.tools.write_files as write_files


def congruent(cat, origin, target):
    """
    determines if the two categories are congruent\
    ----------
    Args:
        cat: category to check
        origin: original category
        target: target category
    ----------
    Returns:
        True if congruent, False otherwise
    ----------
    """
    i_run_min = 0
    i_run_max = 1
    i_eta_min = 2
    i_eta_max = 3
    i_r9_min = 4
    i_r9_max = 5

    ret_origin = False
    ret_origin_eta = False
    ret_origin_r9 = False
    ret_target = False
    ret_target_eta = False
    ret_target_r9 = False

    if (
        origin[i_eta_min] <= cat[i_eta_min - 1]
        and origin[i_eta_max] >= cat[i_eta_max - 1]
    ):
        ret_origin_eta = True
        if (
            origin[i_r9_min] <= cat[i_r9_min - 1]
            and origin[i_r9_max] >= cat[i_r9_max - 1]
        ):
            ret_origin_r9 = True
        else:
            return False
    else:
        return False

    if (
        round(target[i_eta_min], 4) == cat[i_eta_min - 1]
        and round(target[i_eta_max], 4) == cat[i_eta_max - 1]
    ):
        ret_target_eta = True
        if (
            target[i_r9_min] == cat[i_r9_min - 1]
            and target[i_r9_max] == cat[i_r9_max - 1]
        ):
            ret_target_r9 = True
        else:
            return False
    else:
        return False

    return ret_target_eta and ret_target_r9 and ret_origin_eta and ret_origin_r9


def main():
    """
    Extracts the scales from a previous scales file to a new scales file
    ----------
    Args:
        --target: Target scales file from which we will do the extracting
        --previous: previous scales file from which target scales were derived
        --cats: Categories for extraction
        --output: output string
    ----------
    Returns:
        None
    ----------
    """
    parser = ap.ArgumentParser()

    parser.add_argument(
        "--target", help="Target scales file from which we will do the extracting"
    )
    parser.add_argument(
        "--previous", help="previous scales file from which target scales were derived"
    )
    parser.add_argument("--cats", help="Categories for extraction")
    parser.add_argument("--output", help="output string")

    args = parser.parse_args()

    df_target = pd.read_csv(args.target, sep="\t", header=None, dtype=float)
    df_origin = pd.read_csv(args.previous, sep="\t", header=None, dtype=float)
    run_mask = df_origin.loc[:, 0] == df_origin.iloc[0, 0]
    df_origin = df_origin.loc[run_mask]
    df_target = df_target.loc[(df_target.loc[:, 0] == df_origin.iloc[0, 0])]
    df_cats = pd.read_csv(
        args.cats,
        sep="\t",
        header=None,
        comment="#",
        dtype={
            0: str,
            1: float,
            2: float,
            3: float,
            4: float,
            5: float,
            6: float,
            7: float,
        },
    )
    df_cats.loc[:, 1] = round(df_cats.loc[:, 1], 6)

    extracted_scales = []
    for i_cat, row_cat in df_cats.iterrows():
        for index, row in df_origin.iterrows():
            for index2, row2 in df_target.iterrows():
                if congruent(row_cat, row, row2):
                    extracted_scales.append(round(row2[9] / row[9], 6))
                    break
                    break

    write_files.write_scales(extracted_scales, df_cats, args.output)


main()
