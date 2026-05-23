"""Tests for python/helpers/helper_pyval.py — check_args validation."""

import pytest
import os
import tempfile

from python.helpers.helper_pyval import check_args, extract_files


def _make_args(**overrides):
    """Build a minimal args namespace matching pyval CLI expectations."""

    class Args:
        input_file = None
        output_file = None
        data_title = None
        mc_title = None
        lumi_label = None
        bins = None
        write_location = None
        _kPlotFit = False
        _kFit = False
        no_reweight = False

    args = Args()
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class TestCheckArgs:
    """Test argument validation for pyval."""

    def test_missing_input_file(self):
        """Should raise ValueError when input_file is None."""
        args = _make_args(output_file="out", data_title="d", mc_title="m", bins=50)
        with pytest.raises(ValueError, match="input file not specified"):
            check_args(args)

    def test_nonexistent_input_file(self):
        """Should raise FileNotFoundError for a missing file."""
        args = _make_args(
            input_file="/nonexistent/path/file.txt",
            output_file="out",
            data_title="d",
            mc_title="m",
            bins=50,
        )
        with pytest.raises(FileNotFoundError):
            check_args(args)

    def test_missing_output_file(self):
        """Should raise ValueError when output_file is None."""
        args = _make_args(
            input_file=__file__,  # use this test file as a real path
            data_title="d",
            mc_title="m",
            bins=50,
        )
        with pytest.raises(ValueError, match="output file not specified"):
            check_args(args)

    def test_missing_data_title(self):
        """Should raise ValueError when data_title is None."""
        args = _make_args(
            input_file=__file__,
            output_file="out",
            mc_title="m",
            bins=50,
        )
        with pytest.raises(ValueError, match="data title not specified"):
            check_args(args)

    def test_missing_mc_title(self):
        """Should raise ValueError when mc_title is None."""
        args = _make_args(
            input_file=__file__,
            output_file="out",
            data_title="d",
            bins=50,
        )
        with pytest.raises(ValueError, match="mc title not specified"):
            check_args(args)

    def test_missing_bins(self):
        """Should raise ValueError when bins is None."""
        args = _make_args(
            input_file=__file__,
            output_file="out",
            data_title="d",
            mc_title="m",
        )
        with pytest.raises(ValueError, match="binning not specified"):
            check_args(args)

    def test_plot_fit_without_fit(self):
        """Should raise ValueError when _kPlotFit is True but _kFit is False."""
        args = _make_args(
            input_file=__file__,
            output_file="out",
            data_title="d",
            mc_title="m",
            bins=50,
            _kPlotFit=True,
            _kFit=False,
        )
        with pytest.raises(ValueError, match="cannot plot fit without fitting"):
            check_args(args)

    def test_valid_args_passes(self):
        """Valid args should not raise."""
        args = _make_args(
            input_file=__file__,
            output_file="out",
            data_title="d",
            mc_title="m",
            bins=50,
        )
        # Should not raise
        check_args(args)


# ---------------------------------------------------------------------------
# extract_files
# ---------------------------------------------------------------------------


class TestExtractFiles:
    """Test config file parsing."""

    def test_extracts_data_and_mc(self):
        """extract_files should return lists keyed by DATA, MC, etc."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy files referenced in config
            data_path = os.path.join(tmpdir, "data.root")
            mc_path = os.path.join(tmpdir, "mc.root")
            open(data_path, "w").close()
            open(mc_path, "w").close()

            config_path = os.path.join(tmpdir, "config.txt")
            with open(config_path, "w") as f:
                f.write(f"DATA\t{data_path}\n")
                f.write(f"MC\t{mc_path}\n")

            result = extract_files(config_path)
            assert data_path in result["DATA"]
            assert mc_path in result["MC"]

    def test_extracts_scales_and_smearings(self):
        """Scales and smearings entries should be parsed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scales_path = os.path.join(tmpdir, "scales.dat")
            smear_path = os.path.join(tmpdir, "smearings.dat")
            open(scales_path, "w").close()
            open(smear_path, "w").close()

            config_path = os.path.join(tmpdir, "config.txt")
            with open(config_path, "w") as f:
                f.write(f"SCALES\t{scales_path}\n")
                f.write(f"SMEARINGS\t{smear_path}\n")

            result = extract_files(config_path)
            assert scales_path in result["SCALES"]
            assert smear_path in result["SMEARINGS"]

    def test_nonexistent_file_raises(self):
        """Referencing a nonexistent file should raise RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.txt")
            with open(config_path, "w") as f:
                f.write("DATA\t/nonexistent/file.root\n")

            with pytest.raises(RuntimeError):
                extract_files(config_path)

    def test_comments_ignored(self):
        """Lines starting with # should be ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.root")
            open(data_path, "w").close()

            config_path = os.path.join(tmpdir, "config.txt")
            with open(config_path, "w") as f:
                f.write("# This is a comment\n")
                f.write(f"DATA\t{data_path}\n")

            result = extract_files(config_path)
            assert len(result["DATA"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
