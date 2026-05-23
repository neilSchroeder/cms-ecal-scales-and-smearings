"""Tests for python/classes/zcat_class.py — Z category class and standalone functions."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from python.classes.zcat_class import (
    xlogy,
    apply_smearing,
    compute_nll_chisqr,
    compute_earthmovers_distance,
    zcat,
)


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------


class TestXlogy:
    """Test xlogy(x, y) = x * log(y) with x==0 -> 0."""

    def test_basic(self):
        """Standard values should give x*log(y)."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([np.e, np.e, np.e])
        result = xlogy(x, y)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0], rtol=1e-6)

    def test_zero_x(self):
        """x==0 should return 0 regardless of y."""
        x = np.array([0.0, 0.0])
        y = np.array([1.0, 0.0])  # y=0 would be log(0)=-inf normally
        result = xlogy(x, y)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_mixed(self):
        """Mix of zero and nonzero x."""
        x = np.array([0.0, 2.0, 0.0, 1.0])
        y = np.array([5.0, np.e, 0.1, np.e**2])
        result = xlogy(x, y)
        expected = np.array([0.0, 2.0 * np.log(np.e), 0.0, 1.0 * np.log(np.e**2)])
        np.testing.assert_allclose(result, expected, rtol=1e-6)


class TestApplySmearing:
    """Test gaussian smearing of MC invariant mass."""

    def test_zero_smearing(self):
        """Zero smearing should leave MC unchanged (within random seed)."""
        mc = np.array([91.0, 91.0, 91.0])
        result = apply_smearing(mc, 0.0, 0.0, 42)
        # With sigma=0, normal(0,0) = 0 exactly, so 1+0=1, sqrt(1*1)=1
        np.testing.assert_allclose(result, mc, rtol=1e-6)

    def test_smearing_changes_values(self):
        """Nonzero smearing should perturb MC values."""
        mc = np.ones(10000) * 91.0
        result = apply_smearing(mc, 0.01, 0.01, 42)
        assert not np.allclose(result, mc)
        # Mean should still be close to 91
        np.testing.assert_allclose(np.mean(result), 91.0, atol=0.5)

    def test_deterministic_with_seed(self):
        """Same seed should give identical results."""
        mc = np.array([91.0, 92.0, 90.0])
        r1 = apply_smearing(mc, 0.01, 0.02, 12345)
        r2 = apply_smearing(mc, 0.01, 0.02, 12345)
        np.testing.assert_array_equal(r1, r2)


class TestComputeNllChisqr:
    """Test the combined NLL*chi-squared loss function."""

    def test_identical_distributions(self):
        """Identical data and MC should give low (near-zero) loss."""
        # Avoid zero-count bins — they cause NaN via log(0) and 0/0.
        # In production, zcat.update sets mc[mc==0]=1e-15 before calling this.
        data = np.array([1, 2, 10, 50, 100, 50, 10, 2, 1, 1], dtype=np.float64)
        mc_norm = data / np.sum(data)
        result = compute_nll_chisqr(data, mc_norm, num_bins=10)
        assert not np.isnan(result)
        assert abs(result) < 1.0

    def test_shifted_gives_larger_loss(self):
        """Shifted MC should give larger loss than matched MC."""
        data = np.array([1, 2, 5, 30, 100, 30, 5, 2, 1, 1], dtype=np.float64)
        mc_matched = data / np.sum(data)
        mc_shifted = np.array([1, 5, 30, 100, 30, 5, 1, 1, 1, 1], dtype=np.float64)
        mc_shifted = mc_shifted / np.sum(mc_shifted)
        loss_matched = compute_nll_chisqr(data, mc_matched, num_bins=10)
        loss_shifted = compute_nll_chisqr(data, mc_shifted, num_bins=10)
        assert loss_shifted > loss_matched

    def test_non_negative(self):
        """Loss should be non-negative for reasonable inputs."""
        np.random.seed(42)
        data = np.random.poisson(50, 20).astype(np.float64)
        data[data == 0] = 1  # avoid zero bins
        mc_norm = data / np.sum(data)
        mc_norm[mc_norm == 0] = 1e-15
        mc_norm = mc_norm / np.sum(mc_norm)
        result = compute_nll_chisqr(data, mc_norm, num_bins=20)
        assert result >= 0


class TestComputeEarthmoversDistance:
    """Test earth mover's distance between distributions."""

    def test_identical_zero(self):
        """EMD of identical distributions should be 0."""
        data = np.array([10.0, 20.0, 30.0, 20.0, 10.0])
        mc = data.copy()
        emd = compute_earthmovers_distance(data, mc)
        np.testing.assert_allclose(emd, 0.0, atol=1e-6)

    def test_shifted_positive(self):
        """Shifted distribution should give positive EMD."""
        data = np.array([0.0, 0.0, 100.0, 0.0, 0.0])
        mc = np.array([0.0, 0.0, 0.0, 100.0, 0.0])
        emd = compute_earthmovers_distance(data, mc)
        assert emd > 0


# ---------------------------------------------------------------------------
# zcat class
# ---------------------------------------------------------------------------


class TestZcatInit:
    """Test zcat initialization."""

    def _make_zcat(self, n_data=500, n_mc=5000, **kwargs):
        """Helper to build a zcat with realistic-ish data."""
        np.random.seed(42)
        data = np.random.normal(91.0, 2.0, n_data).astype(np.float32)
        mc = np.random.normal(91.0, 2.5, n_mc).astype(np.float32)
        weights = np.ones(n_mc, dtype=np.float32)
        defaults = dict(hist_min=80.0, hist_max=100.0, bin_size=0.25, _kAutoBin=False)
        defaults.update(kwargs)
        return zcat(0, 0, data, mc, weights, **defaults)

    def test_indices(self):
        """Lead and sublead indices should be set."""
        z = self._make_zcat()
        assert z.lead_index == 0
        assert z.sublead_index == 0

    def test_valid(self):
        """With enough events, zcat should be valid."""
        z = self._make_zcat()
        assert z.valid is True

    def test_hist_range(self):
        """Histogram range should match options."""
        z = self._make_zcat(hist_min=82.0, hist_max=98.0)
        assert z.hist_min == 82.0
        assert z.hist_max == 98.0

    def test_smear_indices(self):
        """Smearing indices should be set from options."""
        z = self._make_zcat(smear_i=3, smear_j=5)
        assert z.lead_smear_index == 3
        assert z.sublead_smear_index == 5

    def test_insufficient_data_invalidates(self):
        """Too few data events should mark zcat as invalid."""
        z = self._make_zcat(n_data=5, n_mc=5000, _kAutoBin=True)
        assert z.valid is False

    def test_diagonal_weight(self):
        """Diagonal categories (i==j) should have weight 1, off-diag 0.1."""
        z_diag = self._make_zcat()
        assert z_diag.weight == 1

        np.random.seed(42)
        data = np.random.normal(91.0, 2.0, 500).astype(np.float32)
        mc = np.random.normal(91.0, 2.5, 5000).astype(np.float32)
        weights = np.ones(5000, dtype=np.float32)
        z_off = zcat(
            0, 1, data, mc, weights, hist_min=80.0, hist_max=100.0, _kAutoBin=False
        )
        assert z_off.weight == 0.1


class TestZcatCheckInvalid:
    """Test validity checks."""

    def test_enough_events_valid(self):
        """Should return False (not invalid) with enough events."""
        np.random.seed(42)
        data = np.random.normal(91.0, 2.0, 500).astype(np.float32)
        mc = np.random.normal(91.0, 2.5, 5000).astype(np.float32)
        z = zcat(
            0,
            0,
            data,
            mc,
            np.ones(5000, dtype=np.float32),
            hist_min=80.0,
            hist_max=100.0,
            _kAutoBin=False,
        )
        assert z.check_invalid() is False

    def test_too_few_data(self):
        """Should return True (invalid) with < MIN_EVENTS_DATA."""
        np.random.seed(42)
        data = np.random.normal(91.0, 2.0, 5).astype(np.float32)
        mc = np.random.normal(91.0, 2.5, 5000).astype(np.float32)
        z = zcat(
            0,
            0,
            data,
            mc,
            np.ones(5000, dtype=np.float32),
            hist_min=80.0,
            hist_max=100.0,
            _kAutoBin=False,
        )
        assert z.check_invalid() is True


class TestZcatInject:
    """Test scale/smearing injection."""

    def test_scale_injection(self):
        """Injecting a pure scale should shift invmass by sqrt(s1*s2)."""
        np.random.seed(42)
        data = np.ones(100, dtype=np.float32) * 91.0
        mc = np.ones(5000, dtype=np.float32) * 91.0
        z = zcat(
            0,
            0,
            data,
            mc,
            np.ones(5000, dtype=np.float32),
            hist_min=80.0,
            hist_max=100.0,
            _kAutoBin=False,
        )
        z.inject(1.01, 1.01, 0, 0)
        expected = 91.0 * np.sqrt(1.01 * 1.01)
        np.testing.assert_allclose(z.data, expected, rtol=1e-5)


class TestZcatUpdate:
    """Test the update method (core of NLL evaluation)."""

    def test_nll_changes_with_scale(self):
        """NLL at correct scale should differ from NLL at wrong scale."""
        np.random.seed(42)
        data = np.random.normal(91.0, 2.0, 1000).astype(np.float32)
        mc = np.random.normal(91.0, 2.5, 10000).astype(np.float32)
        z = zcat(
            0,
            0,
            data,
            mc,
            np.ones(10000, dtype=np.float32),
            hist_min=80.0,
            hist_max=100.0,
            _kAutoBin=False,
            bin_size=0.5,
        )
        z.update(1.0, 1.0)
        nll_at_one = z.NLL
        z.update(1.05, 1.05)
        nll_at_shifted = z.NLL
        assert nll_at_one != nll_at_shifted

    def test_update_sets_updated_flag(self):
        """Updated flag should be True after update."""
        np.random.seed(42)
        data = np.random.normal(91.0, 2.0, 500).astype(np.float32)
        mc = np.random.normal(91.0, 2.5, 5000).astype(np.float32)
        z = zcat(
            0,
            0,
            data,
            mc,
            np.ones(5000, dtype=np.float32),
            hist_min=80.0,
            hist_max=100.0,
            _kAutoBin=False,
        )
        z.update(1.0, 1.0)
        assert z.updated is True

    def test_zero_scale_treated_as_one(self):
        """Scale of 0 should be treated as 1 (no scaling)."""
        np.random.seed(42)
        data = np.random.normal(91.0, 2.0, 500).astype(np.float32)
        mc = np.random.normal(91.0, 2.5, 5000).astype(np.float32)
        z1 = zcat(
            0,
            0,
            data.copy(),
            mc.copy(),
            np.ones(5000, dtype=np.float32),
            hist_min=80.0,
            hist_max=100.0,
            _kAutoBin=False,
            bin_size=0.5,
        )
        z2 = zcat(
            0,
            0,
            data.copy(),
            mc.copy(),
            np.ones(5000, dtype=np.float32),
            hist_min=80.0,
            hist_max=100.0,
            _kAutoBin=False,
            bin_size=0.5,
        )
        z1.update(0, 0)
        z2.update(1.0, 1.0)
        np.testing.assert_allclose(z1.NLL, z2.NLL, rtol=1e-5)


# ---------------------------------------------------------------------------
# Auto-bin initialization
# ---------------------------------------------------------------------------


class TestZcatAutoBin:
    """Test zcat auto-bin initialization path."""

    def test_auto_bin_sets_bin_size(self):
        """Auto-binning should compute a bin_size from the data."""
        np.random.seed(42)
        data = np.random.normal(91.0, 2.0, 2000).astype(np.float32)
        mc = np.random.normal(91.0, 2.5, 10000).astype(np.float32)
        z = zcat(
            0,
            0,
            data,
            mc,
            np.ones(len(mc), dtype=np.float32),
            hist_min=80.0,
            hist_max=100.0,
            _kAutoBin=True,
            bin_size=0.25,
        )
        assert z.bin_size != 0.25  # should have been auto-computed
        assert z.bin_size > 0

    def test_auto_bin_insufficient_data_cleans_up(self):
        """Auto-bin with too few events should deactivate the category."""
        data = np.array([91.0], dtype=np.float32)
        mc = np.array([91.0] * 5, dtype=np.float32)
        z = zcat(
            0,
            0,
            data,
            mc,
            np.ones(len(mc), dtype=np.float32),
            hist_min=80.0,
            hist_max=100.0,
            _kAutoBin=True,
            bin_size=0.25,
        )
        assert z.valid is False
        assert z.data is None


# ---------------------------------------------------------------------------
# clean_up
# ---------------------------------------------------------------------------


class TestZcatCleanUp:
    """Test the clean_up method."""

    def test_clean_up_nullifies(self):
        """clean_up should set data/mc/weights/bins to None and valid to False."""
        np.random.seed(42)
        data = np.random.normal(91.0, 2.0, 500).astype(np.float32)
        mc = np.random.normal(91.0, 2.5, 5000).astype(np.float32)
        z = zcat(
            0,
            0,
            data,
            mc,
            np.ones(len(mc), dtype=np.float32),
            hist_min=80.0,
            hist_max=100.0,
            _kAutoBin=False,
        )
        z.clean_up()
        assert z.valid is False
        assert z.data is None
        assert z.mc is None
        assert z.weights is None


# ---------------------------------------------------------------------------
# print
# ---------------------------------------------------------------------------


class TestZcatPrint:
    """Test the print method."""

    def test_print_does_not_error(self, capsys):
        """print() should output category info without raising."""
        np.random.seed(42)
        data = np.random.normal(91.0, 2.0, 500).astype(np.float32)
        mc = np.random.normal(91.0, 2.5, 5000).astype(np.float32)
        z = zcat(
            0,
            0,
            data,
            mc,
            np.ones(len(mc), dtype=np.float32),
            hist_min=80.0,
            hist_max=100.0,
            _kAutoBin=False,
        )
        z.print()
        captured = capsys.readouterr()
        assert "lead index: 0" in captured.out
        assert "valid: True" in captured.out


# ---------------------------------------------------------------------------
# get_smeared_mc
# ---------------------------------------------------------------------------


class TestZcatGetSmearedMc:
    """Test the get_smeared_mc method."""

    def test_zero_smear_returns_original(self):
        """Zero smearing should return unchanged mc."""
        np.random.seed(42)
        data = np.random.normal(91.0, 2.0, 500).astype(np.float32)
        mc = np.random.normal(91.0, 2.5, 5000).astype(np.float32)
        z = zcat(
            0,
            0,
            data,
            mc,
            np.ones(len(mc), dtype=np.float32),
            hist_min=80.0,
            hist_max=100.0,
            _kAutoBin=False,
        )
        result = z.get_smeared_mc(mc, 0.0, 0.0, 42)
        np.testing.assert_array_equal(result, mc)

    def test_nonzero_smear_changes_mc(self):
        """Nonzero smearing should change mc values."""
        np.random.seed(42)
        mc = np.random.normal(91.0, 2.5, 5000).astype(np.float32)
        data = np.random.normal(91.0, 2.0, 500).astype(np.float32)
        z = zcat(
            0,
            0,
            data,
            mc,
            np.ones(len(mc), dtype=np.float32),
            hist_min=80.0,
            hist_max=100.0,
            _kAutoBin=False,
        )
        result = z.get_smeared_mc(mc.copy(), 0.01, 0.01, 42)
        assert not np.allclose(result, mc)


# ---------------------------------------------------------------------------
# update with smearing
# ---------------------------------------------------------------------------


class TestZcatUpdateWithSmearing:
    """Test update method with smearing parameters."""

    def test_update_with_smearing(self):
        """Providing nonzero smearing should change NLL vs no smearing."""
        np.random.seed(42)
        data = np.random.normal(91.0, 2.0, 500).astype(np.float32)
        mc = np.random.normal(91.0, 2.5, 5000).astype(np.float32)
        z1 = zcat(
            0,
            0,
            data.copy(),
            mc.copy(),
            np.ones(len(mc), dtype=np.float32),
            hist_min=80.0,
            hist_max=100.0,
            _kAutoBin=False,
            bin_size=0.5,
        )
        z2 = zcat(
            0,
            0,
            data.copy(),
            mc.copy(),
            np.ones(len(mc), dtype=np.float32),
            hist_min=80.0,
            hist_max=100.0,
            _kAutoBin=False,
            bin_size=0.5,
        )
        z1.update(1.0, 1.0, 0.0, 0.0)
        z2.update(1.0, 1.0, 0.02, 0.02)
        assert z1.NLL != z2.NLL

    def test_inject_with_smearing(self):
        """inject() with nonzero smearing should further modify data."""
        np.random.seed(42)
        data = np.random.normal(91.0, 2.0, 500).astype(np.float32)
        mc = np.random.normal(91.0, 2.5, 5000).astype(np.float32)
        z = zcat(
            0,
            0,
            data.copy(),
            mc.copy(),
            np.ones(len(mc), dtype=np.float32),
            hist_min=80.0,
            hist_max=100.0,
            _kAutoBin=False,
        )
        original = z.data.copy()
        z.inject(1.0, 1.0, 0.02, 0.02)
        assert not np.allclose(z.data, original)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
