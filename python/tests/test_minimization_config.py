"""Tests for MinimizationConfig dataclass in python/classes/config_class.py."""

import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from python.classes.config_class import MinimizationConfig


class TestMinimizationConfigDefaults:
    """Test default values of MinimizationConfig."""

    def test_default_hist_range(self):
        """Default histogram range should be [80, 100] with 0.25 bin size."""
        config = MinimizationConfig()
        assert config.hist_min == 80.0
        assert config.hist_max == 100.0
        assert config.bin_size == 0.25

    def test_default_flags_false(self):
        """All workflow flags should default to False."""
        config = MinimizationConfig()
        assert config._kClosure is False
        assert config._kFixScales is False
        assert config._kPlot is False
        assert config._kTestMethodAccuracy is False
        assert config._kScanNLL is False
        assert config._kDebug is False

    def test_default_scan_range(self):
        """Default scan range should be [0.98, 1.02] with 0.001 step."""
        config = MinimizationConfig()
        assert config.scan_min == 0.98
        assert config.scan_max == 1.02
        assert config.scan_step == 0.001


class TestMinimizationConfigToDict:
    """Test to_dict() method."""

    def test_round_trip_keys(self):
        """to_dict should contain all field names."""
        config = MinimizationConfig()
        d = config.to_dict()
        for field_name in MinimizationConfig.__dataclass_fields__:
            assert field_name in d

    def test_values_match(self):
        """to_dict values should match attribute values."""
        config = MinimizationConfig(
            hist_min=82.0, _kClosure=True, scales="/path/to/scales.tsv"
        )
        d = config.to_dict()
        assert d["hist_min"] == 82.0
        assert d["_kClosure"] is True
        assert d["scales"] == "/path/to/scales.tsv"

    def test_dict_is_copy(self):
        """Modifying returned dict should not affect original."""
        config = MinimizationConfig()
        d = config.to_dict()
        d["hist_min"] = 999.0
        assert config.hist_min == 80.0


class TestMinimizationConfigFromArgs:
    """Test from_args() classmethod."""

    def _make_args(self, **overrides):
        """Build a fake argparse Namespace with all required fields."""
        defaults = dict(
            hist_min="80.0",
            hist_max="100.0",
            bin_size="0.25",
            start_style="scan",
            scan_min="0.98",
            scan_max="1.02",
            scan_step="0.001",
            min_step_size=None,
            base_seed=3543136929,
            _kClosure=False,
            _kFixScales=False,
            _kPlot=False,
            _kTestMethodAccuracy=False,
            _kScanNLL=False,
            scales=None,
            ignore=None,
            plot_dir="./",
            scale_bounds=(0.96, 1.04),
            smear_bounds=(0.0, 0.05),
            closure_scale_bounds=(0.99, 1.01),
            off_diag_weight_scheme="constant",
            off_diag_weight=0.1,
            loss_weighting="uniform",
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_basic_conversion(self):
        """from_args should convert string args to proper types."""
        args = self._make_args(hist_min="82.5", _kClosure=True)
        config = MinimizationConfig.from_args(args)
        assert config.hist_min == 82.5
        assert isinstance(config.hist_min, float)
        assert config._kClosure is True

    def test_auto_bin_default(self):
        """auto_bin should be True when _kNoAutoBin is absent."""
        args = self._make_args()
        config = MinimizationConfig.from_args(args)
        assert config.auto_bin is True

    def test_auto_bin_disabled(self):
        """auto_bin should be False when _kNoAutoBin is True."""
        args = self._make_args(_kNoAutoBin=True)
        config = MinimizationConfig.from_args(args)
        assert config.auto_bin is False

    def test_debug_default(self):
        """_kDebug should default to False when absent from args."""
        args = self._make_args()
        config = MinimizationConfig.from_args(args)
        assert config._kDebug is False

    def test_scales_path(self):
        """Scales path should pass through."""
        args = self._make_args(scales="/some/scales.tsv")
        config = MinimizationConfig.from_args(args)
        assert config.scales == "/some/scales.tsv"

    def test_from_args_to_dict_round_trip(self):
        """from_args -> to_dict should preserve all values."""
        args = self._make_args(hist_min="85.0", scan_step="0.005", _kPlot=True)
        config = MinimizationConfig.from_args(args)
        d = config.to_dict()
        assert d["hist_min"] == 85.0
        assert d["scan_step"] == 0.005
        assert d["_kPlot"] is True


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
