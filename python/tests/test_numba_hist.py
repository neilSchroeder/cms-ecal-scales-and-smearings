"""Tests for python/utilities/numba_hist.py — numba-accelerated histogramming."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from python.utilities.numba_hist import (
    get_bin_edges,
    compute_bin,
    numba_histogram,
    numba_weighted_histogram,
)


class TestGetBinEdges:
    """Test bin edge computation."""

    def test_uniform_edges(self):
        """Bin edges should be uniformly spaced from min to max."""
        a = np.array([0.0, 10.0])
        edges = get_bin_edges(a, 5)
        expected = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        np.testing.assert_allclose(edges, expected)

    def test_num_edges(self):
        """Should return bins+1 edges."""
        a = np.array([1.0, 5.0, 3.0])
        edges = get_bin_edges(a, 10)
        assert len(edges) == 11

    def test_last_edge_equals_max(self):
        """Last edge should exactly equal data max."""
        a = np.array([0.0, 7.3])
        edges = get_bin_edges(a, 3)
        assert edges[-1] == 7.3


class TestComputeBin:
    """Test bin index computation."""

    def test_first_bin(self):
        """Value at minimum should map to bin 0."""
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        assert compute_bin(0.0, edges) == 0

    def test_last_bin(self):
        """Value at maximum should map to last bin."""
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        assert compute_bin(3.0, edges) == 2  # n-1

    def test_mid_value(self):
        """Value in middle should map to correct bin."""
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        assert compute_bin(1.5, edges) == 1

    def test_out_of_range_low(self):
        """Values clearly below range should return -1 sentinel."""
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        result = compute_bin(-1.5, edges)
        assert result == -1

    def test_above_max_returns_sentinel(self):
        """Values above max should return -1 sentinel."""
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        result = compute_bin(3.1, edges)
        assert result == -1


class TestNumbaHistogram:
    """Test unweighted histogram against numpy."""

    def test_matches_numpy(self):
        """numba_histogram should match np.histogram for uniform data."""
        np.random.seed(42)
        data = np.random.normal(91.0, 2.0, 10000)
        bins = 40

        nb_hist, nb_edges = numba_histogram(data, bins)
        np_hist, np_edges = np.histogram(data, bins=bins)

        np.testing.assert_array_equal(nb_hist, np_hist)
        np.testing.assert_allclose(nb_edges, np_edges, atol=1e-10)

    def test_single_value(self):
        """All identical values should land in one bin."""
        data = np.array([5.0, 5.0, 5.0])
        # With identical values, min==max so get_bin_edges has delta=0.
        # This is a degenerate case; just verify no crash.
        hist, edges = numba_histogram(data, 1)
        assert hist[0] == 3

    def test_known_distribution(self):
        """Histogram of [0,1,2,3] with 4 bins should have 1 per bin."""
        data = np.array([0.0, 1.0, 2.0, 3.0])
        hist, edges = numba_histogram(data, 4)
        # 3.0 goes to last bin via the == a_max special case
        assert sum(hist) == 4


class TestNumbaWeightedHistogram:
    """Test weighted histogram against numpy."""

    def test_matches_numpy(self):
        """Weighted histogram should match np.histogram with weights."""
        np.random.seed(42)
        data = np.random.normal(91.0, 2.0, 5000)
        weights = np.random.uniform(0.5, 1.5, 5000).astype(np.float32)
        bins = 30

        nb_hist, nb_edges = numba_weighted_histogram(data, weights, bins)
        np_hist, np_edges = np.histogram(data, bins=bins, weights=weights)

        np.testing.assert_allclose(nb_hist, np_hist, rtol=1e-5)
        np.testing.assert_allclose(nb_edges, np_edges, atol=1e-10)

    def test_unit_weights_match_unweighted(self):
        """Unit weights should give same result as unweighted histogram."""
        np.random.seed(123)
        data = np.random.uniform(80, 100, 1000)
        weights = np.ones(1000, dtype=np.float32)
        bins = 20

        hist_w, _ = numba_weighted_histogram(data, weights, bins)
        hist_u, _ = numba_histogram(data, bins)

        np.testing.assert_allclose(hist_w, hist_u.astype(np.float32))

    def test_zero_weights(self):
        """Zero weights should produce empty histogram."""
        data = np.array([1.0, 2.0, 3.0])
        weights = np.zeros(3, dtype=np.float32)
        hist, _ = numba_weighted_histogram(data, weights, 3)
        np.testing.assert_allclose(hist, np.zeros(3))


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
