"""Shared masking utilities for building dielectron category selections."""

import numpy as np
import pandas as pd

from python.classes.constant_classes import (
    DataConstants as dc,
    CategoryConstants as cc,
)


def _gain_bounds(gain_value):
    """Convert a gain category value (12, 6, 1) to gainSeedSC bounds.

    The detector encodes gains as:
      gain 12 -> gainSeedSC == 0
      gain 6  -> gainSeedSC == 1
      gain 1  -> gainSeedSC >= 2
    """
    if gain_value == 6:
        return 1, 1
    if gain_value == 1:
        return 2, 99999
    # gain 12 (default)
    return 0, 0


def build_dielectron_mask(df, cat1, cat2):
    """Build a boolean mask selecting events where the lead+sub electrons
    match the (cat1, cat2) category pair *or* the symmetric (cat2, cat1).

    Args:
        df (pd.DataFrame): DataFrame with standard column names from DataConstants.
        cat1: Row (Series or array-like) from the categories TSV for one electron.
        cat2: Row for the other electron.

    Returns:
        np.ndarray[bool]: Boolean mask over df.
    """
    n = len(df)

    # --- eta ---
    eta_mask = np.ones(n, dtype=bool)
    if cat1[cc.i_eta_min] != cc.empty:
        fwd = df[dc.ETA_LEAD].between(cat1[cc.i_eta_min], cat1[cc.i_eta_max]) & df[
            dc.ETA_SUB
        ].between(cat2[cc.i_eta_min], cat2[cc.i_eta_max])
        rev = df[dc.ETA_SUB].between(cat1[cc.i_eta_min], cat1[cc.i_eta_max]) & df[
            dc.ETA_LEAD
        ].between(cat2[cc.i_eta_min], cat2[cc.i_eta_max])
        eta_mask = fwd | rev

    # --- R9 ---
    r9_mask = np.ones(n, dtype=bool)
    if cat1[cc.i_r9_min] != cc.empty:
        fwd = df[dc.R9_LEAD].between(cat1[cc.i_r9_min], cat1[cc.i_r9_max]) & df[
            dc.R9_SUB
        ].between(cat2[cc.i_r9_min], cat2[cc.i_r9_max])
        rev = df[dc.R9_SUB].between(cat1[cc.i_r9_min], cat1[cc.i_r9_max]) & df[
            dc.R9_LEAD
        ].between(cat2[cc.i_r9_min], cat2[cc.i_r9_max])
        r9_mask = fwd | rev

    # --- Et ---
    et_mask = np.ones(n, dtype=bool)
    if cat1[cc.i_et_min] != cc.empty:
        fwd = df[dc.ET_LEAD].between(cat1[cc.i_et_min], cat1[cc.i_et_max]) & df[
            dc.ET_SUB
        ].between(cat2[cc.i_et_min], cat2[cc.i_et_max])
        rev = df[dc.ET_SUB].between(cat1[cc.i_et_min], cat1[cc.i_et_max]) & df[
            dc.ET_LEAD
        ].between(cat2[cc.i_et_min], cat2[cc.i_et_max])
        et_mask = fwd | rev

    # --- gain ---
    gain_mask = np.ones(n, dtype=bool)
    if cat1[cc.i_gain] != cc.empty:
        low1, high1 = _gain_bounds(cat1[cc.i_gain])
        low2, high2 = _gain_bounds(cat2[cc.i_gain])
        fwd = df[dc.GAIN_LEAD].between(low1, high1) & df[dc.GAIN_SUB].between(
            low2, high2
        )
        rev = df[dc.GAIN_SUB].between(low1, high1) & df[dc.GAIN_LEAD].between(
            low2, high2
        )
        gain_mask = fwd | rev

    return eta_mask & r9_mask & et_mask & gain_mask
