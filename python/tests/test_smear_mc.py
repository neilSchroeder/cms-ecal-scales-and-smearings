"""Tests for python/utilities/smear_mc.py — MC smearing function."""

import numpy as np
import pandas as pd
import pytest
import os
import tempfile

from python.classes.constant_classes import DataConstants as dc
from python.utilities.smear_mc import smear


def _make_mc(n=100, eta=0.5, r9=0.95, e_lead=50.0, e_sub=40.0, invmass=91.0):
    """Build a minimal MC DataFrame."""
    return pd.DataFrame(
        {
            dc.ETA_LEAD: np.full(n, eta, dtype=np.float32),
            dc.ETA_SUB: np.full(n, eta, dtype=np.float32),
            dc.E_LEAD: np.full(n, e_lead, dtype=np.float32),
            dc.E_SUB: np.full(n, e_sub, dtype=np.float32),
            dc.INVMASS: np.full(n, invmass, dtype=np.float32),
            dc.R9_LEAD: np.full(n, r9, dtype=np.float32),
            dc.R9_SUB: np.full(n, r9, dtype=np.float32),
        }
    )


def _write_smearings(path, rows):
    """Write a smearings TSV file.

    Each row should be a tuple: (category_str, Emean, err_Emean, rho, err_rho, phi, err_phi)
    """
    lines = []
    for row in rows:
        lines.append("\t".join(str(x) for x in row))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


class TestSmear:
    """Test the smear function."""

    def test_energy_changes(self):
        """Smearing should modify E_LEAD and E_SUB."""
        mc = _make_mc(500, eta=0.5, r9=0.95)
        original_e_lead = mc[dc.E_LEAD].values.copy()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            # category format: absEta_min_max-R9_min_max
            f.write(
                "absEta_0.0_1.0-R9_0.0_10.0\t6.6\t0.0\t0.01\t0.001\tM_PI_2\tM_PI_2\n"
            )
            tmpfile = f.name

        try:
            result = smear(mc, tmpfile)
            assert not np.allclose(result[dc.E_LEAD].values, original_e_lead)
        finally:
            os.unlink(tmpfile)

    def test_invmass_changes(self):
        """Smearing should modify invariant mass."""
        mc = _make_mc(500, eta=0.5, r9=0.95)
        original_invmass = mc[dc.INVMASS].values.copy()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(
                "absEta_0.0_1.0-R9_0.0_10.0\t6.6\t0.0\t0.01\t0.001\tM_PI_2\tM_PI_2\n"
            )
            tmpfile = f.name

        try:
            result = smear(mc, tmpfile)
            assert not np.allclose(result[dc.INVMASS].values, original_invmass)
        finally:
            os.unlink(tmpfile)

    def test_returns_dataframe(self):
        """Result should be a DataFrame with same length."""
        mc = _make_mc(100)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(
                "absEta_0.0_1.0-R9_0.0_10.0\t6.6\t0.0\t0.01\t0.001\tM_PI_2\tM_PI_2\n"
            )
            tmpfile = f.name

        try:
            result = smear(mc, tmpfile)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 100
        finally:
            os.unlink(tmpfile)

    def test_et_columns_dropped(self):
        """Temporary et_lead/et_sub columns should be dropped after smearing."""
        mc = _make_mc(100)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(
                "absEta_0.0_1.0-R9_0.0_10.0\t6.6\t0.0\t0.01\t0.001\tM_PI_2\tM_PI_2\n"
            )
            tmpfile = f.name

        try:
            result = smear(mc, tmpfile)
            assert "et_lead" not in result.columns
            assert "et_sub" not in result.columns
        finally:
            os.unlink(tmpfile)

    def test_unmatched_events_unchanged(self):
        """Events outside the smearing category shouldn't be smeared."""
        mc = _make_mc(100, eta=2.0, r9=0.95)  # eta=2.0
        original_e_lead = mc[dc.E_LEAD].values.copy()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            # Only smear eta 0-1
            f.write(
                "absEta_0.0_1.0-R9_0.0_10.0\t6.6\t0.0\t0.01\t0.001\tM_PI_2\tM_PI_2\n"
            )
            tmpfile = f.name

        try:
            result = smear(mc, tmpfile)
            np.testing.assert_array_equal(result[dc.E_LEAD].values, original_e_lead)
        finally:
            os.unlink(tmpfile)

    def test_et_dependent_smearing(self):
        """Smearing with Et bins should apply correctly."""
        # eta=0, E=50, Et=50/cosh(0)=50
        mc = _make_mc(200, eta=0.0, r9=0.95, e_lead=50.0, e_sub=40.0)
        original_e = mc[dc.E_LEAD].values.copy()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            # Et range 30-100 covers Et=50
            f.write(
                "absEta_0.0_1.0-R9_0.0_10.0-Et_30_100\t6.6\t0.0\t0.02\t0.001\tM_PI_2\tM_PI_2\n"
            )
            tmpfile = f.name

        try:
            result = smear(mc, tmpfile)
            assert not np.allclose(result[dc.E_LEAD].values, original_e)
        finally:
            os.unlink(tmpfile)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
