import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from tqdm import tqdm

from python.classes.config_class import SSConfig
from python.classes.constant_classes import DataConstants as dc

ss_config = SSConfig()

from collections import OrderedDict
from typing import Any, Dict, List, Tuple


def is_range_compatible(
    range1: Tuple[float, float], range2: Tuple[float, float]
) -> bool:
    """
    Check if two ranges are compatible (either equal or one contains the other).

    Args:
        range1: Tuple of (min, max) for first range
        range2: Tuple of (min, max) for second range

    Returns:
        bool: True if ranges are compatible, False otherwise
    """
    min1, max1 = range1
    min2, max2 = range2

    # Handle exact equality
    if min1 == min2 and max1 == max2:
        return True

    # Check if range1 contains range2
    if min1 <= min2 and max1 >= max2:
        return True

    # Check if range2 contains range1
    if min2 <= min1 and max2 >= max1:
        return True

    return False


def are_categories_congruent(last: List[float], this: List[float]) -> bool:
    """
    Determine if two categories are congruent (compatible for combining).

    Args:
        last: List containing parameters for last category
              [runMin, runMax, etaMin, etaMax, r9Min, r9Max, etMin, etMax, gain, scale, err]
        this: List containing parameters for current category
              [runMin, runMax, etaMin, etaMax, r9Min, r9Max, etMin, etMax, gain, scale, err]

    Returns:
        bool: True if categories are congruent, False otherwise
    """
    # Extract ranges to check
    ranges_to_check = [
        ((last[0], last[1]), (this[0], this[1])),  # run range
        (
            (round(last[2], 4), round(last[3], 4)),
            (round(this[2], 4), round(this[3], 4)),
        ),  # eta range
        (
            (round(last[4], 4), round(last[5], 4)),
            (round(this[4], 4), round(this[5], 4)),
        ),  # r9 range
        (
            (round(last[6], 4), round(last[7], 4)),
            (round(this[6], 4), round(this[7], 4)),
        ),  # et range
    ]

    # Check all ranges are compatible
    if not all(
        is_range_compatible(range1, range2) for range1, range2 in ranges_to_check
    ):
        return False

    # Special handling for gain
    last_gain, this_gain = last[8], this[8]
    if last_gain == this_gain:
        return True
    if last_gain == 0 or this_gain == 0:  # One is no-gain scale
        return True

    return False


def get_smaller_range(
    range1: Tuple[float, float], range2: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Get the smaller of two ranges.

    Args:
        range1: Tuple of (min, max) for first range
        range2: Tuple of (min, max) for second range

    Returns:
        Tuple[float, float]: The smaller range
    """
    span1 = range1[1] - range1[0]
    span2 = range2[1] - range2[0]
    return range2 if span2 < span1 else range1


def combine_categories(last: List[float], this: List[float]) -> Dict[str, Any]:
    """
    Combine two compatible categories into a new category.

    Args:
        last: List containing parameters for last category
        this: List containing parameters for current category

    Returns:
        Dict containing the combined category parameters
    """
    result = {}

    # Handle run range
    result["runMin"] = int(last[0])
    result["runMax"] = int(last[1])

    # Handle all ranges - take the smaller range for each
    for idx, key in [(2, "eta"), (4, "r9"), (6, "et")]:
        range1 = (round(last[idx], 4), round(last[idx + 1], 4))
        range2 = (round(this[idx], 4), round(this[idx + 1], 4))
        smaller = get_smaller_range(range1, range2)
        result[f"{key}Min"] = smaller[0]
        result[f"{key}Max"] = smaller[1]

    # Handle gain - prefer non-zero gain
    result["gain"] = int(last[8] if last[8] != 0 else this[8])

    # Combine scales and error
    result["scale"] = round(float(this[9]) * float(last[9]), 6)
    result["err"] = this[10]

    return result


def combine_scale_steps(this_step: str, last_step: str, out_file: str) -> None:
    """
    Combines the scales from two consecutive steps and writes the result to a file.

    Args:
        this_step: Path to the current step's scale file
        last_step: Path to the previous step's scale file
        out_file: Path where the combined output should be written
    """
    print(f"[INFO] Combining scales from {this_step} and {last_step}")
    print(f"[INFO] Output will be written to {out_file}")

    # Read input files
    df_this = pd.read_csv(this_step, delimiter="\t", header=None, dtype=float)
    df_last = pd.read_csv(last_step, delimiter="\t", header=None, dtype=float)

    # Initialize output dictionary
    headers = [
        "runMin",
        "runMax",
        "etaMin",
        "etaMax",
        "r9Min",
        "r9Max",
        "etMin",
        "etMax",
        "gain",
        "scale",
        "err",
    ]
    out_dict = OrderedDict((col, []) for col in headers)

    # Combine categories with faster iteration using itertuples
    for row_last in tqdm(
        df_last.itertuples(index=False),
        total=len(df_last),
        desc=f"Checking {len(df_last)} categories against {len(df_this)}",
    ):
        for row_this in df_this.itertuples(index=False):
            if are_categories_congruent(row_last, row_this):
                combined = combine_categories(row_last, row_this)
                for key in headers:
                    out_dict[key].append(combined[key])

    # Write output
    df_out = pd.DataFrame(out_dict)
    df_out.to_csv(out_file, sep="\t", header=False, index=False)

    # Write JSON version if requested
    if out_file.endswith(".dat"):
        json_file = out_file.replace(".dat", ".json")
        writeJsonFromDF(df_out, json_file)  # Assuming this function exists


def addNewCategory(rowLast, rowThis, thisDict, lastStep, thisStep):
    """
    Add a new category to the dictionary
    ----------
    Args:
        rowLast: last row
        rowThis: this row
        thisDict: dictionary to add to
        lastStep: last step
        thisStep: this step
    ----------
    Returns:
        thisDict: dictionary with new category added
    ----------
    """

    thisDict["runMin"].append(int(rowLast[0]))
    thisDict["runMax"].append(int(rowLast[1]))

    if rowThis[3] - rowThis[2] < rowLast[3] - rowLast[2]:
        thisDict["etaMin"].append(round(rowThis[2], 4))
        thisDict["etaMax"].append(round(rowThis[3], 4))
    else:
        thisDict["etaMin"].append(round(rowLast[2], 4))
        thisDict["etaMax"].append(round(rowLast[3], 4))

    if round(rowLast[4], 4) <= round(rowThis[4], 4) and round(rowLast[5], 4) >= round(
        rowThis[5], 4
    ):
        thisDict["r9Min"].append(round(rowThis[4], 4))
        thisDict["r9Max"].append(round(rowThis[5], 4))
    else:
        thisDict["r9Min"].append(round(rowLast[4], 4))
        thisDict["r9Max"].append(round(rowLast[5], 4))

    if round(rowLast[6], 4) <= round(rowThis[6], 4) and round(rowLast[7], 4) >= round(
        rowThis[7], 4
    ):
        thisDict["etMin"].append(round(rowThis[6], 4))
        thisDict["etMax"].append(round(rowThis[7], 4))
    else:
        thisDict["etMin"].append(round(rowLast[6], 4))
        thisDict["etMax"].append(round(rowLast[7], 4))

    if rowLast[8] != 0:
        thisDict["gain"].append(int(rowLast[8]))
    else:
        thisDict["gain"].append(int(rowThis[8]))
    thisDict["scale"].append(round(float(rowThis[9]) * float(rowLast[9]), 6))
    thisDict["err"].append(rowThis[10])


def writeJsonFromDF(thisDF, outFile):

    # Takes the dictionary built in [combine] and writes a json file
    outFile = outFile.replace(".dat", ".json")
    print(
        "[INFO][python/write_files][writeJsonFromDF] producing json file in {}".format(
            outFile
        )
    )
    thisDict = {}

    for i, row in thisDF.iterrows():
        key_run = "run:[{},{}]".format(int(row["runMin"]), int(row["runMax"]))
        if key_run not in thisDict:
            thisDict[key_run] = OrderedDict()
        key_eta = "eta:[{},{}]".format(row["etaMin"], row["etaMax"])
        if key_eta not in thisDict[key_run]:
            thisDict[key_run][key_eta] = OrderedDict()
        key_r9 = "r9:[{},{}]".format(row["r9Min"], row["r9Max"])
        if key_r9 not in thisDict[key_run][key_eta]:
            thisDict[key_run][key_eta][key_r9] = OrderedDict()
        key_pt = "pt:[{},{}]".format(row["etMin"], row["etMax"])
        if key_pt not in thisDict[key_run][key_eta][key_r9]:
            thisDict[key_run][key_eta][key_r9][key_pt] = OrderedDict()
        key_gain = "gain:{}".format(int(row["gain"]))
        if key_gain not in thisDict[key_run][key_eta][key_r9][key_pt]:
            thisDict[key_run][key_eta][key_r9][key_pt][key_gain] = OrderedDict()

        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]["runMin"] = int(
            row["runMin"]
        )
        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]["runMax"] = int(
            row["runMax"]
        )
        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]["etaMin"] = row["etaMin"]
        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]["etaMax"] = row["etaMax"]
        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]["r9Min"] = row["r9Min"]
        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]["r9Max"] = row["r9Max"]
        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]["ptMin"] = row["etMin"]
        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]["ptMax"] = row["etMax"]
        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]["gain"] = int(row["gain"])
        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]["scale"] = row["scale"]
        thisDict[key_run][key_eta][key_r9][key_pt][key_gain]["scaleErr"] = row["err"]

    with open(outFile, "w") as out:
        json.dump(thisDict, out, indent="\t")

    return


def write_scales(scales, cats, out):
    """
    Writes the scales to a file
    --------------------------------
    Args:
        scales: the scales (list)
        cats: the categories (pandas dataframe)
        out: the output file (string)
    --------------------------------
    Returns:
        None
    --------------------------------
    """
    # format of onlystepX files is
    # 000000 999999 etaMin etaMax r9Min r9Max etMin etMax gain val err
    headers = [
        "runMin",
        "runMax",
        "etaMin",
        "etaMax",
        "r9Min",
        "r9Max",
        "etMin",
        "etMax",
        "gain",
        "scale",
        "err",
    ]
    dictForDf = OrderedDict.fromkeys(headers)  # python hates you and your dictionaries
    for col in headers:
        dictForDf[col] = []

    print(scales)
    print(len(scales), sum(cats.loc[:, 0] == "scale"))
    for index, row in cats.iterrows():
        if row[0] != "smear":
            dictForDf["runMin"].append("000000")
            dictForDf["runMax"].append("999999")
            dictForDf["etaMin"].append(row[1])
            dictForDf["etaMax"].append(row[2])
            dictForDf["r9Min"].append(row[3] if row[3] != -1 else 0)
            dictForDf["r9Max"].append(row[4] if row[4] != -1 else 10)
            dictForDf["etMin"].append(row[6] if row[6] != -1 else 0)
            dictForDf["etMax"].append(row[7] if row[7] != -1 else 14000)
            dictForDf["gain"].append(row[5] if row[5] != -1 else 0)
            dictForDf["scale"].append(scales[index])
            dictForDf["err"].append(5e-05)

    dfOut = pd.DataFrame(dictForDf)
    dfOut.to_csv(out, sep="\t", header=False, index=False)


def write_smearings(smears, cats, out):
    """
    Writes the smearings to a file
    --------------------------------
    Args:
        smears: the smearings (list)
        cats: the categories (pandas dataframe)
        out: the output file (string)
    --------------------------------
    Returns:
        None
    --------------------------------
    """
    # format of smearings files is:
    # category       Emean   err_Emean   rho err_rho     phi err_phi
    headers = ["#category", "Emean", "err_Emean", "rho", "err_rho", "phi", "err_phi"]
    dictForDf = OrderedDict.fromkeys(headers)  # python hates you and your dictionaries
    for col in headers:
        dictForDf[col] = []

    smear_mask = cats.loc[:, 0] == "smear"

    for index, row in cats[smear_mask].iterrows():
        if row[0] != "scale":
            if row[3] == -1:
                row[3] = 0
                row[4] = 10
            if row[6] != -1:
                dictForDf["#category"].append(
                    f"absEta_{row[1]}_{row[2]}-R9_{round(row[3],4)}_{row[4]}-Et_{row[6]}_{row[7]}"
                )
            else:
                dictForDf["#category"].append(
                    f"absEta_{row[1]}_{row[2]}-R9_{round(row[3],4)}_{row[4]}"
                )
            dictForDf["Emean"].append(6.6)
            dictForDf["err_Emean"].append(0.0)
            dictForDf["rho"].append(round(smears[index], 5))
            dictForDf["err_rho"].append(round(smears[index] * 0.005, 5))
            dictForDf["phi"].append("M_PI_2")
            dictForDf["err_phi"].append("M_PI_2")

    dfOut = pd.DataFrame(dictForDf)
    dfOut.to_csv(out, sep="\t", header=True, index=False)


def rewrite_smearings(cats, out):
    """
    Rewrites the smearings to a file
    --------------------------------
    Args:
        cats: the categories (pandas dataframe)
        out: the output file (string)
    --------------------------------
    Returns:
        None
    --------------------------------
    """
    # format of smearings files is:
    # category       Emean   err_Emean   rho err_rho     phi err_phi
    headers = ["#category", "Emean", "err_Emean", "rho", "err_rho", "phi", "err_phi"]
    dictForDf = OrderedDict.fromkeys(headers)  # python hates you and your dictionaries
    for col in headers:
        dictForDf[col] = []

    _cats = pd.read_csv(cats, delimiter="\t", header=None, comment="#")
    mask_smears = _cats.loc[:, 0] == "smear"
    smear_df = pd.read_csv(out, delimiter="\t", header=None, comment="#")
    smears = np.array(smear_df.loc[:, 3].values)
    num_scales = np.sum(~mask_smears)

    for index, row in _cats.loc[_cats[:][0] == "smear"].iterrows():
        if row[0] != "scale":
            if row[3] == -1:
                row[3] = 0
                row[4] = 10
            dictForDf["#category"].append(
                str(
                    "absEta_"
                    + str(row[1])
                    + "_"
                    + str(row[2])
                    + "-R9_"
                    + str(round(row[3], 4))
                    + "_"
                    + str(row[4])
                )
            )
            if row[6] != -1:
                dictForDf["#category"][-1] = str(
                    "absEta_"
                    + str(row[1])
                    + "_"
                    + str(row[2])
                    + "-R9_"
                    + str(round(row[3], 4))
                    + "_"
                    + str(row[4])
                    + "-Et_"
                    + str(row[6])
                    + "_"
                    + str(row[7])
                )
            dictForDf["Emean"].append(6.6)
            dictForDf["err_Emean"].append(0.0)
            dictForDf["rho"].append(smears[index - num_scales])
            dictForDf["err_rho"].append(0.00005)
            dictForDf["phi"].append("M_PI_2")
            dictForDf["err_phi"].append("M_PI_2")

    dfOut = pd.DataFrame(dictForDf)
    dfOut.to_csv(out, sep="\t", header=True, index=False)


def write_runs(runs, out):
    """
    Writes the runs to a file
    --------------------------------
    Args:
        runs: the runs (list)
        out: the output file (string)
    --------------------------------
    Returns:
        None
    --------------------------------
    """
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(out)  # catch if the datFiles/ directory doesn't exist

    headers = ["runMin", "runMax"]
    dictForDf = OrderedDict.fromkeys(headers)  # python hates you and your dictionaries
    for col in headers:
        dictForDf[col] = []

    for pair in runs:
        dictForDf["runMin"].append(pair[0])
        dictForDf["runMax"].append(pair[1])

    dfOut = pd.DataFrame(dictForDf)
    dfOut.to_csv(out, sep="\t", header=False, index=False)


def write_time_stability(scales, runs, outFile):
    """
    Writes the time stability scales to a file
    --------------------------------
    Args:
        scales: the scales (list)
        runs: the runs (list)
        outFile: the output file (string)
    --------------------------------
    Returns:
        None
    --------------------------------
    """
    if outFile.find("scales") == -1:
        outFile.replace(".dat", "_scales.dat")
    print(
        "[INFO][python/write_files][write_time_stability] Writing time stability scales to {}".format(
            outFile
        )
    )

    headers = [
        "runMin",
        "runMax",
        "etaMin",
        "etaMax",
        "r9Min",
        "r9Max",
        "etMin",
        "etMax",
        "gain",
        "scale",
        "err",
    ]
    dictForDf = OrderedDict.fromkeys(headers)  # python hates you and your dictionaries
    for col in headers:
        dictForDf[col] = []
    cats = pd.read_csv(runs, delimiter="\t", header=None)
    for index, row in cats.iterrows():
        if row[0] != "smear":
            for i in range(len(scales)):
                dictForDf["runMin"].append(row[0])
                dictForDf["runMax"].append(row[1])
                dictForDf["etaMin"].append(dc.time_stability_eta_bins_low[i])
                dictForDf["etaMax"].append(dc.time_stability_eta_bins_high[i])
                dictForDf["r9Min"].append(0)
                dictForDf["r9Max"].append(10)
                dictForDf["etMin"].append(0)
                dictForDf["etMax"].append(14000)
                dictForDf["gain"].append(0)
                dictForDf["scale"].append(scales[i][index])
                dictForDf["err"].append(0.00005)

    dfOut = pd.DataFrame(dictForDf)
    dfOut.to_csv(outFile, sep="\t", header=False, index=False)


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
    dictForDf = OrderedDict.fromkeys(headers)  # python hates you and your dictionaries
    for col in headers:
        dictForDf[col] = []

    for i, row in enumerate(weights):
        row = np.ravel(row)
        for j, weight in enumerate(row):
            dictForDf[dc.YMIN].append(x_edges[i])
            dictForDf[dc.YMAX].append(x_edges[i + 1])
            dictForDf[dc.PTMIN].append(y_edges[j])
            dictForDf[dc.PTMAX].append(y_edges[j + 1])
            dictForDf[dc.WEIGHT].append(weight)

    out = (
        f"{ss_config.DEFAULT_WRITE_FILES_PATH}ptz_x_rapidity_weights_"
        + basename
        + ".tsv"
    )
    df_out = pd.DataFrame(dictForDf)
    df_out.to_csv(out, sep="\t", index=False)
    return out


def write_systematics(systematics, output_tag):
    """
    Writes the systematics to a file
    --------------------------------
    Args:
        systematics (dict): the systematics
    --------------------------------
    Returns:
        None
    --------------------------------
    """
    out = f"{ss_config.DEFAULT_WRITE_FILES_PATH}systematics_{output_tag}.dat"
    print(f"[INFO][python/write_files][write_systematics] Writing systematics to {out}")

    headers = dc.SCALES_HEADERS
    dictForDf = OrderedDict.fromkeys(headers)  # python hates you and your dictionaries

    for eta_key in systematics.keys():
        for r9_key in systematics.keys():
            dictForDf[dc.SCALES_HEADERS[2]].append(
                dc.SYST_CUTS[eta_key][r9_key]["ele_cats"][0]
            )
            dictForDf[dc.SCALES_HEADERS[3]].append(
                dc.SYST_CUTS[eta_key][r9_key]["ele_cats"][1]
            )
            dictForDf[dc.SCALES_HEADERS[4]].append(
                dc.SYST_CUTS[eta_key][r9_key]["ele_cats"][2]
            )
            dictForDf[dc.SCALES_HEADERS[5]].append(
                dc.SYST_CUTS[eta_key][r9_key]["ele_cats"][3]
            )
            dictForDf[dc.SCALES_HEADERS[6]].append(
                dc.SYST_CUTS[eta_key][r9_key]["ele_cats"][4]
            )
            dictForDf[dc.SCALES_HEADERS[7]].append(
                dc.SYST_CUTS[eta_key][r9_key]["ele_cats"][5]
            )
            dictForDf[dc.SCALES_HEADERS[8]].append(
                dc.SYST_CUTS[eta_key][r9_key]["ele_cats"][6]
            )
            dictForDf[dc.SCALES_HEADERS[9]].append(systematics[eta_key][r9_key])

    dfOut = pd.DataFrame(dictForDf)
    dfOut.to_csv(out, sep="	", header=False, index=False)
    return
