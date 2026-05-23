"""Tests for the shared masking utility."""

import numpy as np
import pandas as pd
from python.classes.constant_classes import DataConstants as dc, CategoryConstants as cc
from python.utilities.masks import build_dielectron_mask, _gain_bounds


def _make_df(n=10):
    """Create a small synthetic DataFrame mimicking the real data schema."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            dc.ETA_LEAD: rng.uniform(0, 2.5, n).astype(np.float32),
            dc.ETA_SUB: rng.uniform(0, 2.5, n).astype(np.float32),
            dc.R9_LEAD: rng.uniform(0.5, 1.0, n).astype(np.float32),
            dc.R9_SUB: rng.uniform(0.5, 1.0, n).astype(np.float32),
            dc.ET_LEAD: rng.uniform(30, 200, n).astype(np.float32),
            dc.ET_SUB: rng.uniform(20, 150, n).astype(np.float32),
            dc.GAIN_LEAD: np.zeros(n, dtype=np.int16),
            dc.GAIN_SUB: np.zeros(n, dtype=np.int16),
            dc.INVMASS: rng.uniform(80, 100, n).astype(np.float32),
        }
    )


def _make_cat(eta_min, eta_max, r9_min, r9_max, gain=-1, et_min=-1, et_max=-1):
    """Create a category row as a pandas Series matching the TSV format."""
    return pd.Series(
        {
            cc.i_type: "scale",
            cc.i_eta_min: eta_min,
            cc.i_eta_max: eta_max,
            cc.i_r9_min: r9_min,
            cc.i_r9_max: r9_max,
            cc.i_gain: gain,
            cc.i_et_min: et_min,
            cc.i_et_max: et_max,
        }
    )


def test_all_pass_when_no_cuts():
    """When all category bounds are -1 (empty), every event passes."""
    df = _make_df(20)
    cat = _make_cat(-1, -1, -1, -1)
    mask = build_dielectron_mask(df, cat, cat)
    assert mask.all(), "All events should pass when no cuts are applied"


def test_eta_cut_selects_barrel():
    """Only barrel events (eta < 1.0) should pass."""
    df = _make_df(1000)
    cat = _make_cat(0.0, 1.0, -1, -1)
    mask = build_dielectron_mask(df, cat, cat)
    selected = df[mask]
    assert len(selected) < len(df), "Should reject some events"
    # For a diagonal category the symmetric OR means both electrons
    # must be in [0, 1.0] (since cat1==cat2)
    assert (
        selected[dc.ETA_LEAD].between(0, 1.0) & selected[dc.ETA_SUB].between(0, 1.0)
    ).all()


def test_symmetric_selection():
    """A (barrel, endcap) pair should select events where lead↔sub can swap."""
    df = _make_df(1000)
    cat_barrel = _make_cat(0.0, 1.0, -1, -1)
    cat_endcap = _make_cat(1.566, 2.5, -1, -1)
    mask = build_dielectron_mask(df, cat_barrel, cat_endcap)
    selected = df[mask]
    for _, row in selected.iterrows():
        lead_barrel = 0.0 <= row[dc.ETA_LEAD] <= 1.0
        sub_endcap = 1.566 <= row[dc.ETA_SUB] <= 2.5
        lead_endcap = 1.566 <= row[dc.ETA_LEAD] <= 2.5
        sub_barrel = 0.0 <= row[dc.ETA_SUB] <= 1.0
        assert (lead_barrel and sub_endcap) or (lead_endcap and sub_barrel)


def test_gain_bounds():
    assert _gain_bounds(12) == (0, 0)
    assert _gain_bounds(6) == (1, 1)
    assert _gain_bounds(1) == (2, 99999)


def test_r9_and_eta_combined():
    """R9 and eta cuts should combine correctly."""
    df = _make_df(1000)
    cat = _make_cat(0.0, 1.0, 0.96, 10.0)  # barrel high-R9
    mask = build_dielectron_mask(df, cat, cat)
    selected = df[mask]
    for _, row in selected.iterrows():
        assert 0.0 <= row[dc.ETA_LEAD] <= 1.0
        assert 0.0 <= row[dc.ETA_SUB] <= 1.0
        assert row[dc.R9_LEAD] >= 0.96
        assert row[dc.R9_SUB] >= 0.96


def test_et_cut():
    """Et cut should restrict transverse energy range."""
    df = _make_df(1000)
    cat = _make_cat(-1, -1, -1, -1, et_min=50, et_max=100)
    mask = build_dielectron_mask(df, cat, cat)
    selected = df[mask]
    assert len(selected) < len(df), "Should reject some events outside Et range"
    for _, row in selected.iterrows():
        assert 50 <= row[dc.ET_LEAD] <= 100
        assert 50 <= row[dc.ET_SUB] <= 100


def test_gain_cut_gain12():
    """Gain=12 cut should select events with gainSeedSC==0."""
    df = _make_df(100)
    # Set some events to gain 0, some to gain 1
    df[dc.GAIN_LEAD] = np.repeat([0, 1], 50)
    df[dc.GAIN_SUB] = np.repeat([0, 1], 50)
    cat = _make_cat(-1, -1, -1, -1, gain=12)
    mask = build_dielectron_mask(df, cat, cat)
    selected = df[mask]
    assert len(selected) > 0
    for _, row in selected.iterrows():
        assert row[dc.GAIN_LEAD] == 0
        assert row[dc.GAIN_SUB] == 0


def test_gain_cut_gain6():
    """Gain=6 cut should select events with gainSeedSC==1."""
    df = _make_df(100)
    df[dc.GAIN_LEAD] = np.repeat([0, 1], 50)
    df[dc.GAIN_SUB] = np.repeat([0, 1], 50)
    cat = _make_cat(-1, -1, -1, -1, gain=6)
    mask = build_dielectron_mask(df, cat, cat)
    selected = df[mask]
    assert len(selected) > 0
    for _, row in selected.iterrows():
        assert row[dc.GAIN_LEAD] == 1
        assert row[dc.GAIN_SUB] == 1


if __name__ == "__main__":
    test_all_pass_when_no_cuts()
    test_eta_cut_selects_barrel()
    test_symmetric_selection()
    test_gain_bounds()
    test_r9_and_eta_combined()
    print("All mask tests passed!")
