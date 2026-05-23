"""Tests for Crystal Ball and Breit-Wigner PDF classes."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from python.classes.crystal_ball import cb
from python.classes.breit_wigner import bw


# ---------------------------------------------------------------------------
# Crystal Ball
# ---------------------------------------------------------------------------


class TestCrystalBall:
    """Test Crystal Ball distribution class."""

    def setup_method(self):
        """Create a standard Crystal Ball for testing."""
        self.x = np.linspace(80, 100, 200)
        # params: [alpha, n, mean, width]
        self.params = [1.5, 5.0, 0.0, 2.0]
        self.crystal = cb(self.x, self.params)

    def test_normalized(self):
        """PDF should sum to 1 (discrete normalization)."""
        np.testing.assert_allclose(np.sum(self.crystal.y), 1.0, rtol=1e-6)

    def test_non_negative(self):
        """PDF values should all be non-negative."""
        assert np.all(self.crystal.y >= 0)

    def test_peak_near_mean(self):
        """Peak should be near the mean of the distribution."""
        peak_idx = np.argmax(self.crystal.y)
        # x is shifted by -(min+max)/2 internally, mean=0 -> peak at center
        peak_x = self.x[peak_idx]
        expected_center = (self.x[0] + self.x[-1]) / 2.0
        assert abs(peak_x - expected_center) < 2.0

    def test_update_changes_shape(self):
        """Updating parameters should change the distribution."""
        old_y = self.crystal.y.copy()
        self.crystal.update([2.0, 3.0, 1.0, 3.0])
        self.crystal.getY()
        assert not np.allclose(old_y, self.crystal.y)

    def test_getY_returns_array(self):
        """getY should return a numpy array."""
        result = self.crystal.getY()
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.x)

    def test_wider_width_broader_peak(self):
        """Larger width should produce a broader distribution."""
        narrow = cb(self.x, [1.5, 5.0, 0.0, 1.0])
        wide = cb(self.x, [1.5, 5.0, 0.0, 4.0])
        # narrow peak should be taller than wide peak
        assert np.max(narrow.y) > np.max(wide.y)


# ---------------------------------------------------------------------------
# Breit-Wigner
# ---------------------------------------------------------------------------


class TestBreitWigner:
    """Test Relativistic Breit-Wigner distribution class."""

    def setup_method(self):
        """Create a Breit-Wigner for the Z mass region."""
        self.x = np.linspace(80, 100, 200)
        self.breit = bw(self.x)

    def test_normalized(self):
        """PDF should sum to 1 (discrete normalization)."""
        np.testing.assert_allclose(np.sum(self.breit.y), 1.0, rtol=1e-6)

    def test_non_negative(self):
        """PDF values should all be non-negative."""
        assert np.all(self.breit.y >= 0)

    def test_peak_near_z_mass(self):
        """Peak should be near the Z boson mass (91.188 GeV)."""
        peak_idx = np.argmax(self.breit.y)
        peak_x = self.x[peak_idx]
        assert abs(peak_x - 91.188) < 0.5

    def test_z_mass_and_width(self):
        """Default mean and width should match PDG Z values."""
        assert self.breit.mean == 91.188
        assert self.breit.width == 2.4952

    def test_getY_returns_array(self):
        """getY should return a numpy array."""
        result = self.breit.getY()
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.x)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
