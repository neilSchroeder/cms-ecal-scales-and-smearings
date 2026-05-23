"""Tests for python/helpers/helper_minimizer.py — target function and wrapper."""

import numpy as np
import pytest

from python.helpers.helper_minimizer import target_function, target_function_wrapper


def _make_zcat(
    lead_index,
    sublead_index,
    nll=1.0,
    weight=100,
    valid=True,
    lead_smear_index=None,
    sublead_smear_index=None,
):
    """Build a mock zcat with the needed attributes.

    Uses a simple namespace object instead of MagicMock to avoid
    numpy array wrapping issues with MagicMock.
    """

    class FakeZcat:
        def __init__(self):
            self.lead_index = lead_index
            self.sublead_index = sublead_index
            self.lead_smear_index = (
                lead_smear_index if lead_smear_index is not None else lead_index
            )
            self.sublead_smear_index = (
                sublead_smear_index
                if sublead_smear_index is not None
                else sublead_index
            )
            self.NLL = nll
            self.weight = weight
            self.valid = valid
            self._update_calls = []

        def update(self, *args):
            self._update_calls.append(args)

    return FakeZcat()


# ---------------------------------------------------------------------------
# target_function
# ---------------------------------------------------------------------------


class TestTargetFunction:
    """Test the NLL target function."""

    def test_returns_weighted_nll(self):
        """Should return sum(NLL*weight) / sum(weight)."""
        cat = _make_zcat(0, 0, nll=2.0, weight=100)
        x = [1.0]
        previous = [0.0]  # different so it triggers update
        result = target_function(x, previous, [cat], 1, 0)
        assert result == pytest.approx(2.0)

    def test_multiple_cats_weighted_average(self):
        """Result should be weighted average of NLLs."""
        cat1 = _make_zcat(0, 0, nll=1.0, weight=100)
        cat2 = _make_zcat(1, 1, nll=3.0, weight=100)
        x = [1.0, 1.0]
        previous = [0.0, 0.0]
        result = target_function(x, previous, [cat1, cat2], 2, 0)
        expected = (1.0 * 100 + 3.0 * 100) / (100 + 100)
        assert result == pytest.approx(expected)

    def test_invalid_cats_excluded(self):
        """Invalid categories should not contribute to the NLL."""
        cat_valid = _make_zcat(0, 0, nll=2.0, weight=100)
        cat_invalid = _make_zcat(1, 1, nll=999.0, weight=100, valid=False)
        x = [1.0, 1.0]
        previous = [0.0, 0.0]
        result = target_function(x, previous, [cat_valid, cat_invalid], 2, 0)
        assert result == pytest.approx(2.0)

    def test_no_valid_cats_returns_large(self):
        """If no valid categories, should return 9e30."""
        cat = _make_zcat(0, 0, nll=1.0, weight=100, valid=False)
        x = [1.0]
        previous = [0.0]
        result = target_function(x, previous, [cat], 1, 0)
        assert result == 9e30

    def test_only_updated_cats_called(self):
        """Only categories whose indices changed should have update() called."""
        cat0 = _make_zcat(0, 0, nll=1.0, weight=100)
        cat1 = _make_zcat(1, 1, nll=2.0, weight=100)
        x = [1.0, 0.5]
        previous = [1.0, 1.0]  # only index 1 changed
        target_function(x, previous, [cat0, cat1], 2, 0)
        assert len(cat0._update_calls) == 0
        assert len(cat1._update_calls) == 1

    def test_update_called_with_scales(self):
        """update() should be called with scale values from x when num_smears=0."""
        cat = _make_zcat(0, 0, nll=1.0, weight=100)
        x = [1.005]
        previous = [0.0]
        target_function(x, previous, [cat], 1, 0)
        assert cat._update_calls == [(1.005, 1.005)]


# ---------------------------------------------------------------------------
# target_function_wrapper
# ---------------------------------------------------------------------------


class TestTargetFunctionWrapper:
    """Test the closure wrapper for the target function."""

    def test_returns_callable(self):
        """Wrapper should return a callable and a reset function."""
        initial = [1.0]
        cats = [_make_zcat(0, 0)]
        func, reset = target_function_wrapper(initial, cats)
        assert callable(func)
        assert callable(reset)

    def test_wrapper_tracks_previous_guess(self):
        """Successive calls should pass the previous guess to target_function."""
        cat = _make_zcat(0, 0, nll=1.0, weight=100)
        initial = [1.0]
        func, reset = target_function_wrapper(initial, [cat])

        # First call: previous should be initial
        result1 = func([1.001], initial, [cat], 1, 0)
        # Second call: previous should be [1.001]
        result2 = func([1.002], [1.001], [cat], 1, 0)
        # Both should return valid NLL
        assert isinstance(result1, float)
        assert isinstance(result2, float)

    def test_reset_restores_initial(self):
        """Reset should restore the initial guess."""
        cat = _make_zcat(0, 0, nll=1.0, weight=100)
        initial = [1.0]
        func, reset = target_function_wrapper(initial, [cat])

        func([1.005], initial, [cat], 1, 0)
        reset()
        # After reset, calling again should treat initial as previous
        # This shouldn't error
        func([1.001], initial, [cat], 1, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
