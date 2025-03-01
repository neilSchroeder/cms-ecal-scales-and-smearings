"""Script to evaluate the systematics of the analysis."""

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.classes.config_class import SSConfig
from src.classes.constant_classes import DataConstants as dc
from src.classes.constant_classes import PyValConstants as pvc

global_config = SSConfig()

from src.plotters.fit_bw_cb import fit_bw_cb
from src.tools.data_loader import apply_custom_event_selection
from src.tools.write_files import write_systematics


def create_histogram(df: pd.DataFrame, cuts: Dict, num_bins: int) -> np.ndarray:
    """
    Create a histogram from the dataframe based on given cuts.

    Args:
        df: pandas DataFrame containing the data.
        cuts: dictionary containing the cuts to be applied to the dataframe.
        num_bins: integer, number of bins for the histogram.

    Returns:
        histogram: numpy array representing the histogram of the invariant mass.

    Raises:
        None

    Prints:
        None
    """
    return np.histogram(
        apply_custom_event_selection(df, **cuts)[dc.INVMASS].values, bins=num_bins
    )[0]


def create_histograms(
    data: pd.DataFrame, mc: pd.DataFrame, cuts: Dict, num_bins: int
) -> Dict:
    """
    Create histograms for all combinations of cuts.

    Args:
        data: pandas DataFrame containing the data.
        mc: pandas DataFrame containing the Monte Carlo simulation data.
        cuts: dictionary containing the cuts for each eta and r9 category.
        num_bins: integer, number of bins for the histograms.

    Returns:
        histograms: dictionary containing the histograms for data and MC for each combination of cuts.

    Raises:
        None

    Prints:
        None
    """
    return {
        eta_key: {
            r9_key: [
                create_histogram(data, cuts[eta_key][r9_key], num_bins),
                create_histogram(mc, cuts[eta_key][r9_key], num_bins),
            ]
            for r9_key in cuts[eta_key]
        }
        for eta_key in cuts.keys()
    }


def apply_variation(cuts: Dict, variation_func: Callable) -> Dict:
    """
    Apply a variation function to the cuts.

    Args:
        cuts: dictionary containing the cuts for each eta and r9 category.
        variation_func: callable function that applies a variation to the cuts.

    Returns:
        varied_cuts: dictionary containing the varied cuts for each eta and r9 category.

    Raises:
        None

    Prints:
        None
    """
    return {
        eta_key: {
            r9_key: variation_func(cuts[eta_key][r9_key]) for r9_key in cuts[eta_key]
        }
        for eta_key in cuts.keys()
    }


def fit_histograms(hists: Dict, mids: List[float]) -> Dict:
    """
    Fit histograms and calculate ratios.

    Args:
        hists: dictionary containing histograms for data and MC for each combination of eta and r9 cuts.
        mids: list of float values representing the midpoints of the histogram bins.

    Returns:
        fit_results: dictionary containing the fit results for data and MC for each combination of eta and r9 cuts.

    Raises:
        None

    Prints:
        Information about any eta and r9 combinations with zero weights.
    """
    for eta_key in hists:
        for r9_key in hists[eta_key]:
            if (
                sum(hists[eta_key][r9_key][0]) == 0
                or sum(hists[eta_key][r9_key][1]) == 0
            ):
                print(f"Zero weights for {eta_key} {r9_key}")
                continue

            mean_data = np.average(mids, weights=hists[eta_key][r9_key][0])
            mean_mc = np.average(mids, weights=hists[eta_key][r9_key][1])
            sigma_data = np.sqrt(
                np.average((mids - mean_data) ** 2, weights=hists[eta_key][r9_key][0])
            )
            sigma_mc = np.sqrt(
                np.average((mids - mean_mc) ** 2, weights=hists[eta_key][r9_key][1])
            )

            data_fit = fit_bw_cb(
                mids,
                hists[eta_key][r9_key][0],
                [1.424, 1.86, mean_data - dc.TARGET_MASS, sigma_data],
            )
            mc_fit = fit_bw_cb(
                mids,
                hists[eta_key][r9_key][1],
                [1.424, 1.86, mean_mc - dc.TARGET_MASS, sigma_mc],
            )

            hists[eta_key][r9_key] = data_fit["mu"] / mc_fit["mu"]

    return hists


def calculate_systematics(
    nominal: Dict, variations: Dict[str, Tuple[Dict, Dict]]
) -> Dict:
    """
    Calculate systematic uncertainties based on nominal and variation histograms.

    Args:
        nominal: dictionary containing the nominal histograms for data and MC.
        variations: dictionary where each key is a systematic variation name and each value is a tuple
                    containing two dictionaries: the first for data histograms and the second for MC histograms.

    Returns:
        systematics: dictionary containing the calculated systematic uncertainties for each combination of eta and r9 cuts.

    Raises:
        None

    Prints:
        None
    """
    return {
        eta_key: {
            r9_key: {
                syst_key: max(
                    1
                    - (
                        (
                            variations[syst_key][0][eta_key][r9_key]
                            / variations[syst_key][1][eta_key][r9_key]
                        )
                        / (nominal[eta_key][r9_key] / nominal[eta_key][r9_key])
                    )
                    for syst_key in variations.keys()
                )
                for syst_key in variations.keys()
            }
            for r9_key in nominal[eta_key]
        }
        for eta_key in nominal.keys()
    }


def evaluate_systematics(data: pd.DataFrame, mc: pd.DataFrame, outfile: str) -> None:
    """
    Evaluate the systematics of the analysis.

    The study is performed by varying the R9, Et, and working point ID.
    The agreement between data and MC is calculated as the ratio of mu_data/mu_mc,
    where mu is the invariant mass peak obtained by fitting a histogram with
    a Breit-Wigner convoluted with a Crystal Ball function.
    The systematic uncertainty from a variation is taken as
    1 - (mu_data_variation/mu_mc_variation)/(mu_data_nominal/mu_mc_nominal).

    Args:
        data: pandas DataFrame containing the data.
        mc: pandas DataFrame containing the Monte Carlo simulation data.
        outfile: string, path to the output file.

    Returns:
        None

    Raises:
        None

    Prints:
        Information about the evaluation process.
    """
    print("[INFO][evaluate_systematics] Evaluating systematics")
    cuts = dc.SYST_CUTS
    num_bins = 80
    mids = [80.125 + 0.25 * i for i in range(80)]

    # Create nominal histograms
    nominal_hists = create_histograms(data, mc, cuts, num_bins)

    # Define variations
    variations = {
        "R9_up": lambda c, r9_key: {
            **c,
            "r9_cuts": (
                ((0.965, -1), (0.965, -1))
                if r9_key == "HighR9"
                else ((-1, 0.965), (-1, 0.965))
            ),
        },
        "R9_down": lambda c, r9_key: {
            **c,
            "r9_cuts": (
                ((0.955, -1), (0.955, -1))
                if r9_key == "HighR9"
                else ((-1, 0.955), (-1, 0.955))
            ),
        },
        "Et_up": lambda c: {**c, "et_cuts": (dc.MIN_PT_LEAD + 2, dc.MIN_PT_SUB)},
        "Et_down": lambda c: {**c, "et_cuts": (dc.MIN_PT_LEAD - 2, dc.MIN_PT_SUB)},
        "ID_medium": lambda c: {**c, "working_point": "medium"},
        "ID_tight": lambda c: {**c, "working_point": "tight"},
    }

    # Create variation histograms
    variation_hists = {
        var_name: create_histograms(data, mc, apply_variation(cuts, var_func), num_bins)
        for var_name, var_func in variations.items()
    }

    # Fit histograms
    nominal_hists = fit_histograms(nominal_hists, mids)
    variation_hists = {
        var_name: fit_histograms(var_hists, mids)
        for var_name, var_hists in variation_hists.items()
    }

    # Calculate systematics
    systematics_categories = {
        "R9": (variation_hists["R9_down"], variation_hists["R9_up"]),
        "Et": (variation_hists["Et_down"], variation_hists["Et_up"]),
        "ID": (variation_hists["ID_medium"], variation_hists["ID_tight"]),
    }
    systematics = calculate_systematics(nominal_hists, systematics_categories)

    # Write results
    write_systematics(systematics, outfile)
