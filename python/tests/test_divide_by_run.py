"""Tests for python/utilities/divide_by_run.py — run binning logic."""

import numpy as np
import pandas as pd
import pytest

from python.utilities.divide_by_run import divide


class TestDivide:
    """Test run division logic."""

    def _make_data(self, runs):
        """Build a DataFrame with a runNumber column from a list of run numbers."""
        return pd.DataFrame({"runNumber": runs})

    def test_single_large_run(self):
        """A single run exceeding min_num_events should produce one bin."""
        data = self._make_data([100] * 50)
        bins = divide(data, 10)
        assert bins == [(100, 100)]

    def test_multiple_large_runs(self):
        """Each run exceeding the threshold should get its own bin."""
        data = self._make_data([1] * 20 + [2] * 20 + [3] * 20)
        bins = divide(data, 10)
        assert bins == [(1, 1), (2, 2), (3, 3)]

    def test_small_runs_merge(self):
        """Runs below the threshold should be merged together."""
        data = self._make_data([1] * 3 + [2] * 3 + [3] * 3 + [4] * 3)
        bins = divide(data, 10)
        # 3+3+3+3=12 >= 10, so all runs merge into one bin
        assert len(bins) == 1
        assert bins[0] == (1, 4)

    def test_mixed_sizes(self):
        """A large run followed by several small runs that merge greedily."""
        data = self._make_data([1] * 15 + [2] * 3 + [3] * 3 + [4] * 15)
        bins = divide(data, 10)
        # Run 1 is large -> own bin; runs 2+3+4 merge greedily (3+3+15 >= 10)
        assert bins[0] == (1, 1)
        assert bins[1][0] == 2
        assert bins[1][1] == 4
        assert len(bins) == 2

    def test_all_single_events(self):
        """Many runs with one event each should merge into bins."""
        runs = list(range(1, 11))  # 10 runs, 1 event each
        data = self._make_data(runs)
        bins = divide(data, 5)
        # Total events=10, threshold=5, should get at most 2 bins
        assert len(bins) <= 2
        assert bins[0][0] == 1
        assert bins[-1][1] == 10

    def test_exact_threshold(self):
        """A run with exactly min_num_events should get its own bin."""
        data = self._make_data([1] * 10)
        bins = divide(data, 10)
        assert bins == [(1, 1)]

    def test_preserves_run_order(self):
        """Bins should be in ascending run order."""
        data = self._make_data([5] * 20 + [1] * 20 + [3] * 20)
        bins = divide(data, 10)
        run_starts = [b[0] for b in bins]
        assert run_starts == sorted(run_starts)
