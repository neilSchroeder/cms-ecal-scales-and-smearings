"""Tests for python/utilities/data_loader.py — cuts, Et computation, category extraction."""

import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Mock uproot before importing data_loader (uproot not installed in test env)
sys.modules.setdefault("uproot", MagicMock())

from python.classes.constant_classes import DataConstants as dc, CategoryConstants as cc
from python.utilities.data_loader import (
    standard_cuts,
    custom_cuts,
    add_transverse_energy,
    get_smearing_index,
    extract_cats,
)


def _make_dielectron_df(
    n,
    eta_lead=1.0,
    eta_sub=0.5,
    e_lead=50.0,
    e_sub=40.0,
    invmass=91.0,
    run=300000,
    r9_lead=0.95,
    r9_sub=0.85,
    gain_lead=0,
    gain_sub=0,
):
    """Build a minimal dielectron DataFrame."""
    return pd.DataFrame(
        {
            dc.ETA_LEAD: np.full(n, eta_lead, dtype=np.float32),
            dc.ETA_SUB: np.full(n, eta_sub, dtype=np.float32),
            dc.E_LEAD: np.full(n, e_lead, dtype=np.float32),
            dc.E_SUB: np.full(n, e_sub, dtype=np.float32),
            dc.INVMASS: np.full(n, invmass, dtype=np.float32),
            dc.RUN: np.full(n, run, dtype=np.int32),
            dc.R9_LEAD: np.full(n, r9_lead, dtype=np.float32),
            dc.R9_SUB: np.full(n, r9_sub, dtype=np.float32),
            dc.GAIN_LEAD: np.full(n, gain_lead, dtype=np.int16),
            dc.GAIN_SUB: np.full(n, gain_sub, dtype=np.int16),
            dc.PHI_LEAD: np.zeros(n, dtype=np.float32),
            dc.PHI_SUB: np.zeros(n, dtype=np.float32),
            dc.ID_LEAD: np.full(n, 0xFFFF, dtype=np.uint32),
            dc.ID_SUB: np.full(n, 0xFFFF, dtype=np.uint32),
        }
    )


# ---------------------------------------------------------------------------
# standard_cuts
# ---------------------------------------------------------------------------


class TestStandardCuts:
    """Test standard dielectron selection cuts."""

    def test_keeps_good_events(self):
        """Events inside all cuts should be kept."""
        df = _make_dielectron_df(100, eta_lead=1.0, eta_sub=0.5, invmass=91.0)
        result = standard_cuts(df)
        assert len(result) == 100

    def test_removes_transition_region(self):
        """Events in ECAL transition region should be removed."""
        df = _make_dielectron_df(100, eta_lead=1.5, eta_sub=0.5, invmass=91.0)
        result = standard_cuts(df)
        assert len(result) == 0

    def test_removes_low_invmass(self):
        """Events with invmass < 60 should be removed."""
        df = _make_dielectron_df(100, invmass=50.0)
        result = standard_cuts(df)
        assert len(result) == 0

    def test_removes_high_invmass(self):
        """Events with invmass > 120 should be removed."""
        df = _make_dielectron_df(100, invmass=130.0)
        result = standard_cuts(df)
        assert len(result) == 0

    def test_takes_absolute_eta(self):
        """Negative eta should be treated as positive."""
        df = _make_dielectron_df(100, eta_lead=-1.0, eta_sub=-0.5, invmass=91.0)
        result = standard_cuts(df)
        assert len(result) == 100


# ---------------------------------------------------------------------------
# custom_cuts
# ---------------------------------------------------------------------------


class TestCustomCuts:
    """Test custom cut application."""

    def test_eta_cuts_tuple(self):
        """Eta cuts as tuple-of-tuples should filter correctly."""
        df = _make_dielectron_df(100, eta_lead=1.0, eta_sub=0.5)
        # Only keep lead eta in [0.5, 1.5]
        result = custom_cuts(df, eta_cuts=((0.5, 1.5), (0.0, 1.0)))
        assert len(result) == 100

    def test_eta_cuts_exclude(self):
        """Events outside eta range should be removed."""
        df = _make_dielectron_df(100, eta_lead=2.0, eta_sub=0.5)
        result = custom_cuts(df, eta_cuts=((0.0, 1.5), (0.0, 1.0)))
        assert len(result) == 0

    def test_invmass_cuts(self):
        """Invariant mass cut should filter correctly."""
        df = _make_dielectron_df(100, invmass=91.0)
        result = custom_cuts(df, inv_mass_cuts=(85.0, 95.0))
        assert len(result) == 100

        df2 = _make_dielectron_df(100, invmass=80.0)
        result2 = custom_cuts(df2, inv_mass_cuts=(85.0, 95.0))
        assert len(result2) == 0

    def test_r9_cuts(self):
        """R9 cuts should filter correctly."""
        df = _make_dielectron_df(100, r9_lead=0.97, r9_sub=0.93)
        result = custom_cuts(df, r9_cuts=(0.96, 0.90))
        assert len(result) == 100

        df2 = _make_dielectron_df(100, r9_lead=0.90, r9_sub=0.80)
        result2 = custom_cuts(df2, r9_cuts=(0.96, 0.90))
        assert len(result2) == 0

    def test_no_cuts_keeps_all(self):
        """No cuts should keep all events."""
        df = _make_dielectron_df(100)
        result = custom_cuts(df)
        assert len(result) == 100

    def test_eta_cuts_4tuple(self):
        """Eta cuts as a flat 4-tuple should filter both eta windows."""
        # 4-tuple: two windows (0,1) and (1.566,2.5) for each electron
        df = _make_dielectron_df(100, eta_lead=0.5, eta_sub=0.5)
        result = custom_cuts(df, eta_cuts=(0.0, 1.0, 1.566, 2.5))
        assert len(result) == 100

    def test_eta_cuts_4tuple_excludes(self):
        """Events outside both eta windows should be removed."""
        df = _make_dielectron_df(100, eta_lead=1.3, eta_sub=0.5)
        result = custom_cuts(df, eta_cuts=(0.0, 1.0, 1.566, 2.5))
        assert len(result) == 0

    def test_et_cuts_simple(self):
        """Simple et_cuts as (min_lead, min_sub) should filter correctly."""
        # e_lead=50, eta_lead=0 -> et=50; needs et>40
        df = _make_dielectron_df(
            100, e_lead=50.0, eta_lead=0.0, e_sub=30.0, eta_sub=0.0
        )
        result = custom_cuts(df, et_cuts=(40.0, 25.0))
        assert len(result) == 100

    def test_et_cuts_tuple_of_tuples(self):
        """Et cuts as tuple-of-tuples should filter min and max per electron."""
        df = _make_dielectron_df(
            100, e_lead=50.0, eta_lead=0.0, e_sub=30.0, eta_sub=0.0
        )
        result = custom_cuts(df, et_cuts=((40.0, 60.0), (25.0, 35.0)))
        assert len(result) == 100

    def test_et_cuts_tuple_excludes(self):
        """Events outside Et max should be removed."""
        df = _make_dielectron_df(
            100, e_lead=50.0, eta_lead=0.0, e_sub=30.0, eta_sub=0.0
        )
        result = custom_cuts(df, et_cuts=((40.0, 45.0), (25.0, 35.0)))
        assert len(result) == 0

    def test_r9_cuts_tuple_of_tuples(self):
        """R9 cuts as tuple-of-tuples should filter min and max per electron."""
        df = _make_dielectron_df(100, r9_lead=0.97, r9_sub=0.93)
        result = custom_cuts(df, r9_cuts=((0.95, -1), (0.90, -1)))
        assert len(result) == 100

    def test_working_point_filter(self):
        """Working point filter should apply electron ID bitmask."""
        df = _make_dielectron_df(100)
        # Set IDs to a value that passes tight ID
        df[dc.ID_LEAD] = dc.TIGHT_ID
        df[dc.ID_SUB] = dc.TIGHT_ID
        result = custom_cuts(df, working_point="tight")
        assert len(result) == 100

    def test_working_point_rejects_loose(self):
        """Events with ID=0 should fail working point check."""
        df = _make_dielectron_df(100)
        df[dc.ID_LEAD] = 0
        df[dc.ID_SUB] = 0
        result = custom_cuts(df, working_point="tight")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# add_transverse_energy
# ---------------------------------------------------------------------------


class TestAddTransverseEnergy:
    """Test transverse energy column addition and Et cuts."""

    def test_adds_et_columns(self):
        """Et columns should be added to both data and MC."""
        data = _make_dielectron_df(50, e_lead=50.0, eta_lead=0.0)
        mc = _make_dielectron_df(50, e_lead=50.0, eta_lead=0.0)
        result_data, result_mc = add_transverse_energy(data, mc)
        assert dc.ET_LEAD in result_data.columns
        assert dc.ET_SUB in result_data.columns
        assert dc.ET_LEAD in result_mc.columns

    def test_et_computation(self):
        """Et = E / cosh(eta) should be computed correctly."""
        eta = 1.0
        e = 50.0
        data = _make_dielectron_df(10, e_lead=e, eta_lead=eta)
        mc = _make_dielectron_df(10, e_lead=e, eta_lead=eta)
        result_data, _ = add_transverse_energy(data, mc)
        expected_et = e / np.cosh(eta)
        np.testing.assert_allclose(
            result_data[dc.ET_LEAD].values, expected_et, rtol=1e-5
        )

    def test_et_cut_removes_low_pt(self):
        """Events with Et < 30 (lead) should be removed."""
        # e_lead=20, eta=0 -> Et=20 < 30
        data = _make_dielectron_df(
            50, e_lead=20.0, eta_lead=0.0, e_sub=25.0, eta_sub=0.0
        )
        mc = _make_dielectron_df(50, e_lead=20.0, eta_lead=0.0, e_sub=25.0, eta_sub=0.0)
        result_data, result_mc = add_transverse_energy(data, mc)
        assert len(result_data) == 0
        assert len(result_mc) == 0


# ---------------------------------------------------------------------------
# get_smearing_index
# ---------------------------------------------------------------------------


class TestGetSmearingIndex:
    """Test smearing category lookup."""

    def _make_cats(self, rows):
        """Build cats_df with all 8 columns in positional order."""
        data = {
            cc.i_type: [r[0] for r in rows],
            cc.i_eta_min: [r[1] for r in rows],
            cc.i_eta_max: [r[2] for r in rows],
            cc.i_r9_min: [r[3] for r in rows],
            cc.i_r9_max: [r[4] for r in rows],
            cc.i_gain: [r[5] for r in rows],
            cc.i_et_min: [r[6] for r in rows],
            cc.i_et_max: [r[7] for r in rows],
        }
        return pd.DataFrame(data)

    def test_finds_matching_smear(self):
        """Should find the smearing category that encloses the scale category."""
        cats_df = self._make_cats(
            [
                # type, eta_min, eta_max, r9_min, r9_max, gain, et_min, et_max
                ("scale", 0.0, 1.0, 0.0, 0.5, -1, -1, -1),
                ("scale", 1.0, 2.5, 0.0, 0.5, -1, -1, -1),
                ("smear", 0.0, 2.5, 0.0, 1.0, -1, -1, -1),
            ]
        )
        # Scale cat 0 (eta 0-1) should match smear cat 2 (eta 0-2.5)
        assert get_smearing_index(cats_df, 0) == 2
        # Scale cat 1 (eta 1-2.5) should also match smear cat 2
        assert get_smearing_index(cats_df, 1) == 2

    def test_narrow_smear_matches_subset(self):
        """When two smear categories exist, the narrowest enclosing one is matched."""
        cats_df = self._make_cats(
            [
                ("scale", 0.0, 1.0, 0.0, 0.5, -1, -1, -1),
                ("smear", 0.0, 1.0, 0.0, 1.0, -1, -1, -1),
                ("smear", 0.0, 2.5, 0.0, 1.0, -1, -1, -1),
            ]
        )
        # Scale cat 0 matches both smears, but smear at index 1 is narrower
        assert get_smearing_index(cats_df, 0) == 1


# ---------------------------------------------------------------------------
# extract_cats
# ---------------------------------------------------------------------------


class TestExtractCats:
    """Test dielectron category extraction."""

    def _make_cats_df(self):
        """Build a simple 2-scale categories frame."""
        return pd.DataFrame(
            {
                cc.i_type: ["scale", "scale"],
                cc.i_eta_min: [0.0, 0.0],
                cc.i_eta_max: [2.5, 2.5],
                cc.i_r9_min: [-1, -1],
                cc.i_r9_max: [-1, -1],
                cc.i_gain: [-1, -1],
                cc.i_et_min: [-1, -1],
                cc.i_et_max: [-1, -1],
            }
        )

    def test_returns_list_of_zcats(self):
        """extract_cats should return a list of zcat objects."""
        data = _make_dielectron_df(200)
        mc = _make_dielectron_df(1000)
        cats_df = self._make_cats_df()
        result = extract_cats(data, mc, cats_df, num_scales=2, num_smears=0)
        assert isinstance(result, list)
        # 2 scales → (0,0), (1,0), (1,1) = 3 categories
        assert len(result) == 3

    def test_empty_data_returns_empty(self):
        """Empty data should return empty list."""
        data = _make_dielectron_df(0)
        mc = _make_dielectron_df(100)
        cats_df = self._make_cats_df()
        result = extract_cats(data, mc, cats_df, num_scales=2, num_smears=0)
        assert result == []

    def test_empty_mc_returns_empty(self):
        """Empty MC should return empty list."""
        data = _make_dielectron_df(100)
        mc = _make_dielectron_df(0)
        cats_df = self._make_cats_df()
        result = extract_cats(data, mc, cats_df, num_scales=2, num_smears=0)
        assert result == []


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
