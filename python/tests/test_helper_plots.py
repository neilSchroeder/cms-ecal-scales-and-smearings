"""Tests for python/helpers/helper_plots.py — statistical helpers for plotting."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from python.helpers.helper_plots import (
    get_bin_uncertainties,
    get_systematic_uncertainty,
    get_chi2,
    get_reduced_chi2,
)


class TestGetBinUncertainties:
    """Test weighted bin uncertainty computation."""

    def test_unit_weights(self):
        """Unit weights: uncertainty = sqrt(N) per bin."""
        bins = np.array([0.0, 1.0, 2.0])
        values = np.array([0.5, 0.5, 0.5, 1.5])
        weights = np.ones(4)
        result = get_bin_uncertainties(bins, values, weights)
        # bin 0: 3 entries -> sqrt(3*1^2) = sqrt(3)
        # bin 1: 1 entry -> sqrt(1) = 1
        np.testing.assert_allclose(result[0], np.sqrt(3.0))
        np.testing.assert_allclose(result[1], 1.0)

    def test_non_unit_weights(self):
        """Uncertainty should be sqrt(sum(w^2)) in each bin."""
        bins = np.array([0.0, 10.0])
        values = np.array([3.0, 5.0])
        weights = np.array([2.0, 3.0])
        result = get_bin_uncertainties(bins, values, weights)
        expected = np.sqrt(2.0**2 + 3.0**2)
        np.testing.assert_allclose(result[0], expected)

    def test_empty_bin(self):
        """Bins with no entries should have zero uncertainty."""
        bins = np.array([0.0, 1.0, 2.0])
        values = np.array([0.5])
        weights = np.array([1.0])
        result = get_bin_uncertainties(bins, values, weights)
        assert result[0] == 1.0
        assert result[1] == 0.0


class TestGetSystematicUncertainty:
    """Test systematic uncertainty envelope calculation."""

    def test_symmetric_shift(self):
        """Symmetric up/down shifts should give consistent envelope."""
        np.random.seed(42)
        data = np.random.normal(90, 2, 10000)
        data_up = data + 0.1
        data_down = data - 0.1
        bins = 40
        result = get_systematic_uncertainty(bins, data, data_up, data_down)
        assert len(result) == bins
        # envelope should be non-negative
        assert np.all(result >= 0)

    def test_no_shift(self):
        """No shift should give near-zero systematics."""
        np.random.seed(42)
        data = np.random.normal(90, 2, 5000)
        result = get_systematic_uncertainty(40, data, data, data)
        np.testing.assert_allclose(result, np.zeros(40), atol=1e-10)


class TestGetChi2:
    """Test chi-squared computation."""

    def test_perfect_agreement(self):
        """Identical data and MC should yield chi2 = 0."""
        data = np.array([10.0, 20.0, 30.0])
        mc = np.array([10.0, 20.0, 30.0])
        err = np.array([1.0, 1.0, 1.0])
        assert get_chi2(data, err, mc, err) == 0.0

    def test_known_chi2(self):
        """Known 1-sigma pull in one bin should give chi2 = 1."""
        data = np.array([11.0])
        mc = np.array([10.0])
        data_err = np.array([0.0])
        mc_err = np.array([1.0])
        # (11-10)^2 / (0^2 + 1^2) = 1
        np.testing.assert_allclose(get_chi2(data, data_err, mc, mc_err), 1.0)

    def test_multi_bin(self):
        """Chi2 should sum across bins."""
        data = np.array([12.0, 22.0])
        mc = np.array([10.0, 20.0])
        err = np.array([1.0, 1.0])
        # each bin contributes (2)^2 / (1+1) = 2, total = 4
        np.testing.assert_allclose(get_chi2(data, err, mc, err), 4.0)


class TestGetReducedChi2:
    """Test reduced chi-squared computation."""

    def test_ndf(self):
        """Reduced chi2 = chi2 / (N-1)."""
        data = np.array([12.0, 22.0, 32.0])
        mc = np.array([10.0, 20.0, 30.0])
        err_d = np.array([0.0, 0.0, 0.0])
        err_m = np.array([1.0, 1.0, 1.0])
        chi2 = get_chi2(data, err_d, mc, err_m)  # 4+4+4 = 12
        reduced = get_reduced_chi2(data, err_d, mc, err_m)
        np.testing.assert_allclose(reduced, chi2 / 2.0)  # ndf = 3-1 = 2


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
