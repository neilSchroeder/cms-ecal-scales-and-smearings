"""Tests for python/utilities/reweight_pt_y.py — Z pT and rapidity calculations."""

import numpy as np
import pandas as pd
import sys
import os
import tempfile
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Mock SSConfig (instantiated at module level) and write_files
sys.modules.setdefault("uproot", MagicMock())
with patch("python.classes.config_class.SSConfig", MagicMock):
    from python.utilities.reweight_pt_y import (
        get_zpt,
        get_rapidity,
        derive_pt_y_weights,
        add_pt_y_weights,
    )

from python.classes.constant_classes import DataConstants as dc


def _make_event_df(
    n=1, eta_lead=0.0, eta_sub=0.0, e_lead=50.0, e_sub=50.0, phi_lead=0.0, phi_sub=np.pi
):
    """Build a minimal dielectron DataFrame for kinematic calculations."""
    return pd.DataFrame(
        {
            dc.ETA_LEAD: np.full(n, eta_lead),
            dc.ETA_SUB: np.full(n, eta_sub),
            dc.E_LEAD: np.full(n, e_lead),
            dc.E_SUB: np.full(n, e_sub),
            dc.PHI_LEAD: np.full(n, phi_lead),
            dc.PHI_SUB: np.full(n, phi_sub),
        }
    )


# ---------------------------------------------------------------------------
# get_zpt
# ---------------------------------------------------------------------------


class TestGetZpt:
    """Test Z boson transverse momentum computation."""

    def test_back_to_back_central(self):
        """Back-to-back electrons at eta=0 with equal E should give pT=0."""
        df = _make_event_df(
            10,
            eta_lead=0.0,
            eta_sub=0.0,
            e_lead=50.0,
            e_sub=50.0,
            phi_lead=0.0,
            phi_sub=np.pi,
        )
        zpt = get_zpt(df)
        np.testing.assert_allclose(zpt, 0.0, atol=1e-10)

    def test_same_direction_central(self):
        """Same-direction electrons at eta=0 should give pT = E1+E2."""
        df = _make_event_df(
            10,
            eta_lead=0.0,
            eta_sub=0.0,
            e_lead=50.0,
            e_sub=30.0,
            phi_lead=0.0,
            phi_sub=0.0,
        )
        zpt = get_zpt(df)
        # At eta=0, theta=pi/2, so pt = E*sin(theta) = E
        np.testing.assert_allclose(zpt, 80.0, atol=1e-10)

    def test_returns_array(self):
        """Should return an array of same length as input."""
        df = _make_event_df(25)
        zpt = get_zpt(df)
        assert len(zpt) == 25

    def test_non_negative(self):
        """pT should always be non-negative."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                dc.ETA_LEAD: rng.uniform(-2.5, 2.5, 100),
                dc.ETA_SUB: rng.uniform(-2.5, 2.5, 100),
                dc.E_LEAD: rng.uniform(20, 100, 100),
                dc.E_SUB: rng.uniform(20, 100, 100),
                dc.PHI_LEAD: rng.uniform(-np.pi, np.pi, 100),
                dc.PHI_SUB: rng.uniform(-np.pi, np.pi, 100),
            }
        )
        zpt = get_zpt(df)
        assert np.all(zpt >= 0)


# ---------------------------------------------------------------------------
# get_rapidity
# ---------------------------------------------------------------------------


class TestGetRapidity:
    """Test Z boson rapidity computation."""

    def test_central_symmetric(self):
        """Symmetric central events should have rapidity near 0."""
        df = _make_event_df(10, eta_lead=0.0, eta_sub=0.0, e_lead=50.0, e_sub=50.0)
        y = get_rapidity(df)
        np.testing.assert_allclose(y, 0.0, atol=1e-10)

    def test_forward_boosted(self):
        """Forward-boosted events should have positive rapidity (abs)."""
        df = _make_event_df(10, eta_lead=2.0, eta_sub=2.0, e_lead=50.0, e_sub=50.0)
        y = get_rapidity(df)
        assert np.all(y > 0)

    def test_returns_absolute_value(self):
        """Rapidity should be |y| (absolute value)."""
        df = _make_event_df(10, eta_lead=-2.0, eta_sub=-2.0, e_lead=50.0, e_sub=50.0)
        y = get_rapidity(df)
        assert np.all(y >= 0)

    def test_returns_array(self):
        """Should return an array of same length as input."""
        df = _make_event_df(30)
        y = get_rapidity(df)
        assert len(y) == 30

    def test_non_negative(self):
        """Rapidity (abs) should always be non-negative."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                dc.ETA_LEAD: rng.uniform(-2.5, 2.5, 100),
                dc.ETA_SUB: rng.uniform(-2.5, 2.5, 100),
                dc.E_LEAD: rng.uniform(20, 100, 100),
                dc.E_SUB: rng.uniform(20, 100, 100),
                dc.PHI_LEAD: rng.uniform(-np.pi, np.pi, 100),
                dc.PHI_SUB: rng.uniform(-np.pi, np.pi, 100),
            }
        )
        y = get_rapidity(df)
        assert np.all(y >= 0)


# ---------------------------------------------------------------------------
# derive_pt_y_weights
# ---------------------------------------------------------------------------


def _make_bulk_df(n=5000, seed=42):
    """Build a large dielectron DataFrame for histogram-based tests."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            dc.ETA_LEAD: rng.uniform(-2.5, 2.5, n),
            dc.ETA_SUB: rng.uniform(-2.5, 2.5, n),
            dc.E_LEAD: rng.uniform(30, 100, n),
            dc.E_SUB: rng.uniform(25, 80, n),
            dc.PHI_LEAD: rng.uniform(-np.pi, np.pi, n),
            dc.PHI_SUB: rng.uniform(-np.pi, np.pi, n),
        }
    )


class TestDerivePtYWeights:
    """Test weight derivation from data/MC histograms."""

    def test_returns_file_path(self, monkeypatch):
        """derive_pt_y_weights should return a path to the written file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setattr(
                "python.utilities.write_files.ss_config",
                type("C", (), {"DEFAULT_WRITE_FILES_PATH": tmpdir + "/"})(),
            )
            df_data = _make_bulk_df(2000, seed=1)
            df_mc = _make_bulk_df(5000, seed=2)
            outfile = derive_pt_y_weights(df_data, df_mc, "test")
            assert os.path.exists(outfile)
            df = pd.read_csv(outfile, sep="\t")
            assert "weight" in df.columns

    def test_weights_file_has_rows(self, monkeypatch):
        """Weight file should have at least one row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setattr(
                "python.utilities.write_files.ss_config",
                type("C", (), {"DEFAULT_WRITE_FILES_PATH": tmpdir + "/"})(),
            )
            df_data = _make_bulk_df(2000, seed=10)
            df_mc = _make_bulk_df(5000, seed=20)
            outfile = derive_pt_y_weights(df_data, df_mc, "fin_test")
            df = pd.read_csv(outfile, sep="\t")
            assert len(df) > 0


# ---------------------------------------------------------------------------
# add_pt_y_weights
# ---------------------------------------------------------------------------


class TestAddPtYWeights:
    """Test adding precalculated weights to a dataframe."""

    def test_adds_weight_column(self, monkeypatch):
        """After add_pt_y_weights, df should have a pty_weight column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setattr(
                "python.utilities.write_files.ss_config",
                type("C", (), {"DEFAULT_WRITE_FILES_PATH": tmpdir + "/"})(),
            )
            df_data = _make_bulk_df(2000, seed=1)
            df_mc = _make_bulk_df(5000, seed=2)
            wf = derive_pt_y_weights(df_data, df_mc, "addtest")

            # Now apply weights to MC
            df_mc2 = _make_bulk_df(500, seed=3)
            result = add_pt_y_weights(df_mc2, wf)
            assert dc.PTY_WEIGHT in result.columns
            assert len(result) == 500


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
