"""Tests for python/tools/scales_validator.py — coverage validation functions."""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from python.tools.scales_validator import (
    validate_et,
    validate_r9,
    validate_eta,
    validate_scales,
)


# ---------------------------------------------------------------------------
# validate_et
# ---------------------------------------------------------------------------


class TestValidateEt:
    """Test Et coverage validation."""

    def test_full_coverage(self):
        """Full Et range 0-14000 should pass."""
        assert validate_et([0], [14000]) is True

    def test_split_coverage(self):
        """Two bins summing to 14000 should pass."""
        assert validate_et([0, 7000], [7000, 14000]) is True

    def test_incomplete_coverage(self):
        """Coverage less than 14000 should fail."""
        assert validate_et([0], [5000]) is False

    def test_mismatched_lengths(self):
        """Mismatched min/max lengths should fail."""
        assert validate_et([0, 100], [14000]) is False


# ---------------------------------------------------------------------------
# validate_r9
# ---------------------------------------------------------------------------


class TestValidateR9:
    """Test R9 coverage validation."""

    def test_full_coverage(self):
        """Full R9 range 0-10 should pass."""
        assert validate_r9([0], [10]) is True

    def test_split_coverage(self):
        """Two bins summing to 10 should pass."""
        assert validate_r9([0, 0.94], [0.94, 10]) is True

    def test_incomplete_coverage(self):
        """Coverage less than 10 should fail."""
        assert validate_r9([0], [1.0]) is False

    def test_mismatched_lengths(self):
        """Mismatched min/max lengths should fail."""
        assert validate_r9([0, 0.5], [10]) is False


# ---------------------------------------------------------------------------
# validate_eta
# ---------------------------------------------------------------------------


class TestValidateEta:
    """Test eta coverage validation."""

    def test_full_eb_and_ee_coverage(self):
        """Full EB (0-1.4442) and EE (1.566-2.5) coverage should pass."""
        assert validate_eta([0, 1.566], [1.4442, 2.5]) is True

    def test_incomplete_eb(self):
        """Incomplete EB coverage should fail."""
        assert validate_eta([0, 1.566], [1.0, 2.5]) is False

    def test_incomplete_ee(self):
        """Incomplete EE coverage should fail."""
        assert validate_eta([0, 1.566], [1.4442, 2.0]) is False

    def test_mismatched_lengths(self):
        """Mismatched min/max lengths should fail."""
        assert validate_eta([0], [1.4442, 2.5]) is False


# ---------------------------------------------------------------------------
# validate_scales
# ---------------------------------------------------------------------------


class TestValidateScales:
    """Test full scales file validation."""

    def _make_scales_df(self, rows):
        """Build a scales DataFrame mimicking the TSV format.

        Columns: 0=runMin, 1=runMax, 2=etaMin, 3=etaMax, 4=r9Min, 5=r9Max,
                 6=etMin, 7=etMax, 8=scale (value)
        """
        return pd.DataFrame(rows)

    def test_valid_single_run(self):
        """A complete coverage in a single run bin should pass."""
        rows = [
            # run  run   eta     eta      r9     r9   et     et    scale
            [1, 100, 0.0, 1.4442, 0.0, 10.0, 0, 14000, 1.0],
            [1, 100, 1.566, 2.5, 0.0, 10.0, 0, 14000, 1.0],
        ]
        df = self._make_scales_df(rows)
        assert validate_scales(df) is True

    def test_invalid_eta_fails(self):
        """Missing EE coverage should fail."""
        rows = [
            [1, 100, 0.0, 1.4442, 0.0, 10.0, 0, 14000, 1.0],
            # Missing EE bin (1.566-2.5)
        ]
        df = self._make_scales_df(rows)
        assert validate_scales(df) is False

    def test_invalid_r9_fails(self):
        """Incomplete R9 coverage within an eta bin should fail."""
        rows = [
            [1, 100, 0.0, 1.4442, 0.0, 1.0, 0, 14000, 1.0],  # r9: 0-1, not 0-10
            [1, 100, 1.566, 2.5, 0.0, 10.0, 0, 14000, 1.0],
        ]
        df = self._make_scales_df(rows)
        assert validate_scales(df) is False


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
