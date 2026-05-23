"""Tests for python/utilities/write_files.py — congruentCategories, addNewCategory, write_scales."""

import json
import numpy as np
import os
import pandas as pd
import pytest
import tempfile
from collections import OrderedDict

from python.utilities.write_files import (
    congruentCategories,
    addNewCategory,
    write_scales,
    write_smearings,
    writeJsonFromDF,
    combine,
    write_weights,
    write_runs,
)


def _make_row(
    run_min=1,
    run_max=100,
    eta_min=0.0,
    eta_max=1.4442,
    r9_min=0.0,
    r9_max=10.0,
    et_min=0,
    et_max=14000,
    gain=0,
    scale=1.0,
    err=5e-5,
):
    """Build a row array matching the scales file format."""
    return [
        run_min,
        run_max,
        eta_min,
        eta_max,
        r9_min,
        r9_max,
        et_min,
        et_max,
        gain,
        scale,
        err,
    ]


# ---------------------------------------------------------------------------
# congruentCategories
# ---------------------------------------------------------------------------


class TestCongruentCategories:
    """Test category congruence checking."""

    def test_identical_categories(self):
        """Identical categories should be congruent."""
        row = _make_row()
        assert congruentCategories(row, row, "a", "b") is True

    def test_different_run_ranges(self):
        """Non-overlapping run ranges should not be congruent."""
        a = _make_row(run_min=1, run_max=50)
        b = _make_row(run_min=51, run_max=100)
        assert congruentCategories(a, b, "a", "b") is False

    def test_superset_run_range(self):
        """A superset run range should be congruent."""
        a = _make_row(run_min=1, run_max=200)
        b = _make_row(run_min=50, run_max=100)
        assert congruentCategories(a, b, "a", "b") is True

    def test_different_eta(self):
        """Different eta ranges should not be congruent."""
        a = _make_row(eta_min=0.0, eta_max=1.0)
        b = _make_row(eta_min=1.5, eta_max=2.5)
        assert congruentCategories(a, b, "a", "b") is False

    def test_subset_eta(self):
        """A subset eta range should be congruent."""
        a = _make_row(eta_min=0.0, eta_max=2.5)
        b = _make_row(eta_min=0.0, eta_max=1.0)
        assert congruentCategories(a, b, "a", "b") is True

    def test_different_r9(self):
        """Different R9 ranges should not be congruent."""
        a = _make_row(r9_min=0.0, r9_max=0.5)
        b = _make_row(r9_min=0.6, r9_max=1.0)
        assert congruentCategories(a, b, "a", "b") is False

    def test_superset_r9(self):
        """A superset R9 range should be congruent."""
        a = _make_row(r9_min=0.0, r9_max=10.0)
        b = _make_row(r9_min=0.0, r9_max=0.94)
        assert congruentCategories(a, b, "a", "b") is True

    def test_different_et(self):
        """Different Et ranges should not be congruent."""
        a = _make_row(et_min=0, et_max=50)
        b = _make_row(et_min=60, et_max=14000)
        assert congruentCategories(a, b, "a", "b") is False

    def test_gain_zero_and_nonzero(self):
        """gain=0 and gain!=0 should be congruent."""
        a = _make_row(gain=0)
        b = _make_row(gain=12)
        assert congruentCategories(a, b, "a", "b") is True

    def test_different_nonzero_gain(self):
        """Two different nonzero gains should not be congruent."""
        a = _make_row(gain=1)
        b = _make_row(gain=6)
        assert congruentCategories(a, b, "a", "b") is False


# ---------------------------------------------------------------------------
# addNewCategory
# ---------------------------------------------------------------------------


class TestAddNewCategory:
    """Test building combined category entries."""

    def _empty_dict(self):
        headers = [
            "runMin",
            "runMax",
            "etaMin",
            "etaMax",
            "r9Min",
            "r9Max",
            "etMin",
            "etMax",
            "gain",
            "scale",
            "err",
        ]
        d = OrderedDict.fromkeys(headers)
        for col in headers:
            d[col] = []
        return d

    def test_populates_all_keys(self):
        """All header keys should have one entry after addNewCategory."""
        d = self._empty_dict()
        a = _make_row(scale=1.001)
        b = _make_row(scale=0.999)
        addNewCategory(a, b, d, "last", "this")
        for key in d:
            assert len(d[key]) == 1

    def test_scales_multiply(self):
        """Combined scale should be product of both scales."""
        d = self._empty_dict()
        a = _make_row(scale=1.002)
        b = _make_row(scale=0.998)
        addNewCategory(a, b, d, "last", "this")
        expected = round(1.002 * 0.998, 6)
        assert d["scale"][0] == expected

    def test_narrower_eta_chosen(self):
        """The narrower eta range should be chosen."""
        d = self._empty_dict()
        a = _make_row(eta_min=0.0, eta_max=2.5)
        b = _make_row(eta_min=0.0, eta_max=1.0)
        addNewCategory(a, b, d, "last", "this")
        assert d["etaMin"][0] == 0.0
        assert d["etaMax"][0] == 1.0

    def test_run_from_last(self):
        """Run range should come from the 'last' row."""
        d = self._empty_dict()
        a = _make_row(run_min=200, run_max=300)
        b = _make_row(run_min=100, run_max=400)
        addNewCategory(a, b, d, "last", "this")
        assert d["runMin"][0] == 200
        assert d["runMax"][0] == 300


# ---------------------------------------------------------------------------
# write_scales
# ---------------------------------------------------------------------------


class TestWriteScales:
    """Test write_scales output file format."""

    def _make_cats(self):
        """Build a minimal categories dataframe with 2 scale rows."""
        return pd.DataFrame(
            [
                ["scale", 0.0, 1.4442, 0.0, 10.0, -1, -1, -1],
                ["scale", 1.566, 2.5, 0.94, 10.0, -1, -1, -1],
            ]
        )

    def test_writes_correct_number_of_rows(self):
        """Output should have one row per scale category."""
        cats = self._make_cats()
        scales = [1.001, 0.999]
        with tempfile.NamedTemporaryFile(suffix=".dat", mode="w", delete=False) as f:
            outpath = f.name
        try:
            write_scales(scales, cats, outpath)
            df = pd.read_csv(outpath, sep="\t", header=None)
            assert len(df) == 2
        finally:
            os.unlink(outpath)

    def test_scale_values_written(self):
        """Scale values should appear in column 9."""
        cats = self._make_cats()
        scales = [1.005, 0.995]
        with tempfile.NamedTemporaryFile(suffix=".dat", mode="w", delete=False) as f:
            outpath = f.name
        try:
            write_scales(scales, cats, outpath)
            df = pd.read_csv(outpath, sep="\t", header=None)
            np.testing.assert_allclose(df.iloc[:, 9].values, [1.005, 0.995])
        finally:
            os.unlink(outpath)

    def test_defaults_for_sentinel_values(self):
        """When category has -1, should use default values (r9 0/10, et 0/14000)."""
        cats = self._make_cats()
        scales = [1.0, 1.0]
        with tempfile.NamedTemporaryFile(suffix=".dat", mode="w", delete=False) as f:
            outpath = f.name
        try:
            write_scales(scales, cats, outpath)
            df = pd.read_csv(outpath, sep="\t", header=None)
            # r9Min=0, r9Max=10 (first row had -1/-1 → defaults)
            assert df.iloc[0, 4] == 0
            assert df.iloc[0, 5] == 10
            # et defaults
            assert df.iloc[0, 6] == 0
            assert df.iloc[0, 7] == 14000
        finally:
            os.unlink(outpath)

    def test_skips_smear_rows(self):
        """Rows labeled 'smear' should be skipped."""
        cats = pd.DataFrame(
            [
                ["scale", 0.0, 1.4442, 0.0, 10.0, 0, 0, 14000],
                ["smear", 0.0, 1.4442, 0.0, 10.0, 0, 0, 14000],
            ]
        )
        scales = [1.001, 0.02]
        with tempfile.NamedTemporaryFile(suffix=".dat", mode="w", delete=False) as f:
            outpath = f.name
        try:
            write_scales(scales, cats, outpath)
            df = pd.read_csv(outpath, sep="\t", header=None)
            assert len(df) == 1
        finally:
            os.unlink(outpath)


# ---------------------------------------------------------------------------
# write_smearings
# ---------------------------------------------------------------------------


class TestWriteSmearings:
    """Test write_smearings output file format."""

    def _make_cats(self):
        """Build categories with 1 scale + 1 smear row."""
        return pd.DataFrame(
            [
                ["scale", 0.0, 1.4442, 0.0, 10.0, 0, -1, -1],
                ["smear", 0.0, 1.4442, 0.94, 10.0, 0, -1, -1],
            ]
        )

    def test_writes_header(self):
        """Output should have a header row with #category."""
        cats = self._make_cats()
        smears = [1.0, 0.02]
        with tempfile.NamedTemporaryFile(suffix=".dat", mode="w", delete=False) as f:
            outpath = f.name
        try:
            write_smearings(smears, cats, outpath)
            with open(outpath) as fh:
                header = fh.readline()
            assert "#category" in header
        finally:
            os.unlink(outpath)

    def test_smear_value_correct(self):
        """Rho column should contain the smearing value."""
        cats = self._make_cats()
        smears = [1.0, 0.035]
        with tempfile.NamedTemporaryFile(suffix=".dat", mode="w", delete=False) as f:
            outpath = f.name
        try:
            write_smearings(smears, cats, outpath)
            df = pd.read_csv(outpath, sep="\t")
            assert df["rho"].iloc[0] == 0.035
        finally:
            os.unlink(outpath)

    def test_category_name_format(self):
        """Category name should include absEta and R9."""
        cats = self._make_cats()
        smears = [1.0, 0.02]
        with tempfile.NamedTemporaryFile(suffix=".dat", mode="w", delete=False) as f:
            outpath = f.name
        try:
            write_smearings(smears, cats, outpath)
            df = pd.read_csv(outpath, sep="\t")
            assert "absEta" in df["#category"].iloc[0]
            assert "R9" in df["#category"].iloc[0]
        finally:
            os.unlink(outpath)

    def test_et_in_category_name(self):
        """When Et is not -1, category name should include Et."""
        cats = pd.DataFrame(
            [
                ["smear", 0.0, 1.4442, 0.0, 10.0, 0, 32, 50],
            ]
        )
        smears = [0.025]
        with tempfile.NamedTemporaryFile(suffix=".dat", mode="w", delete=False) as f:
            outpath = f.name
        try:
            write_smearings(smears, cats, outpath)
            df = pd.read_csv(outpath, sep="\t")
            assert "Et_32" in df["#category"].iloc[0]
        finally:
            os.unlink(outpath)

    def test_r9_sentinel_replaced(self):
        """When r9 is -1, should default to 0/10."""
        cats = pd.DataFrame(
            [
                ["smear", 0.0, 1.4442, -1, -1, 0, -1, -1],
            ]
        )
        smears = [0.02]
        with tempfile.NamedTemporaryFile(suffix=".dat", mode="w", delete=False) as f:
            outpath = f.name
        try:
            write_smearings(smears, cats, outpath)
            df = pd.read_csv(outpath, sep="\t")
            cat = df["#category"].iloc[0]
            assert "R9_0_" in cat
        finally:
            os.unlink(outpath)


# ---------------------------------------------------------------------------
# writeJsonFromDF
# ---------------------------------------------------------------------------


class TestWriteJsonFromDF:
    """Test JSON output from DataFrame."""

    def test_produces_valid_json(self):
        """Output should be valid JSON."""
        df = pd.DataFrame(
            {
                "runMin": [1],
                "runMax": [100],
                "etaMin": [0.0],
                "etaMax": [1.4442],
                "r9Min": [0.0],
                "r9Max": [10.0],
                "etMin": [0],
                "etMax": [14000],
                "gain": [0],
                "scale": [1.001],
                "err": [5e-5],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".dat", mode="w", delete=False) as f:
            outpath = f.name
        json_path = outpath.replace(".dat", ".json")
        try:
            writeJsonFromDF(df, outpath)
            with open(json_path) as fh:
                data = json.load(fh)
            assert isinstance(data, dict)
        finally:
            for p in [outpath, json_path]:
                if os.path.exists(p):
                    os.unlink(p)

    def test_json_contains_scale(self):
        """JSON should contain scale value nested under the category keys."""
        df = pd.DataFrame(
            {
                "runMin": [200],
                "runMax": [300],
                "etaMin": [0.0],
                "etaMax": [1.4442],
                "r9Min": [0.0],
                "r9Max": [10.0],
                "etMin": [0],
                "etMax": [14000],
                "gain": [0],
                "scale": [1.005],
                "err": [5e-5],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".dat", mode="w", delete=False) as f:
            outpath = f.name
        json_path = outpath.replace(".dat", ".json")
        try:
            writeJsonFromDF(df, outpath)
            with open(json_path) as fh:
                data = json.load(fh)
            # Traverse nested dict to find scale
            run_key = list(data.keys())[0]
            eta_key = list(data[run_key].keys())[0]
            r9_key = list(data[run_key][eta_key].keys())[0]
            pt_key = list(data[run_key][eta_key][r9_key].keys())[0]
            gain_key = list(data[run_key][eta_key][r9_key][pt_key].keys())[0]
            assert data[run_key][eta_key][r9_key][pt_key][gain_key]["scale"] == 1.005
        finally:
            for p in [outpath, json_path]:
                if os.path.exists(p):
                    os.unlink(p)


# ---------------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------------


class TestCombine:
    """Test combining two scale files."""

    def _write_scale_file(self, path, rows):
        """Write a minimal scales TSV."""
        df = pd.DataFrame(rows)
        df.to_csv(path, sep="\t", header=False, index=False)

    def test_combines_two_files(self):
        """combine should produce a merged output."""
        row_a = [1, 100, 0.0, 1.4442, 0.0, 10.0, 0, 14000, 0, 1.002, 5e-5]
        row_b = [1, 100, 0.0, 1.4442, 0.0, 10.0, 0, 14000, 0, 0.998, 5e-5]
        with tempfile.TemporaryDirectory() as tmpdir:
            last = os.path.join(tmpdir, "last.dat")
            this = os.path.join(tmpdir, "this.dat")
            out = os.path.join(tmpdir, "out.dat")
            self._write_scale_file(last, [row_a])
            self._write_scale_file(this, [row_b])
            combine(this, last, out)
            assert os.path.exists(out)
            df = pd.read_csv(out, sep="\t", header=None)
            assert len(df) == 1
            np.testing.assert_allclose(
                df.iloc[0, 9], round(1.002 * 0.998, 6), rtol=1e-5
            )

    def test_combine_produces_json(self):
        """combine should also produce a .json file."""
        row = [1, 100, 0.0, 1.4442, 0.0, 10.0, 0, 14000, 0, 1.0, 5e-5]
        with tempfile.TemporaryDirectory() as tmpdir:
            last = os.path.join(tmpdir, "last.dat")
            this = os.path.join(tmpdir, "this.dat")
            out = os.path.join(tmpdir, "out.dat")
            self._write_scale_file(last, [row])
            self._write_scale_file(this, [row])
            combine(this, last, out)
            json_path = out.replace(".dat", ".json")
            assert os.path.exists(json_path)

    def test_no_congruent_produces_empty(self):
        """Non-overlapping categories produce an empty output."""
        row_a = [1, 50, 0.0, 1.0, 0.0, 10.0, 0, 14000, 0, 1.0, 5e-5]
        row_b = [51, 100, 1.5, 2.5, 0.0, 10.0, 0, 14000, 0, 1.0, 5e-5]
        with tempfile.TemporaryDirectory() as tmpdir:
            last = os.path.join(tmpdir, "last.dat")
            this = os.path.join(tmpdir, "this.dat")
            out = os.path.join(tmpdir, "out.dat")
            self._write_scale_file(last, [row_a])
            self._write_scale_file(this, [row_b])
            combine(this, last, out)
            assert os.path.exists(out)
            # Empty file (no congruent categories)
            assert os.path.getsize(out) == 0


# ---------------------------------------------------------------------------
# write_weights
# ---------------------------------------------------------------------------


class TestWriteWeights:
    """Test write_weights output."""

    def test_writes_tsv(self, monkeypatch):
        """write_weights should produce a TSV with correct dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setattr(
                "python.utilities.write_files.ss_config",
                type("FakeConfig", (), {"DEFAULT_WRITE_FILES_PATH": tmpdir + "/"})(),
            )
            weights = np.array([[1.0, 2.0], [3.0, 4.0]])
            x_edges = np.array([0.0, 1.0, 2.0])
            y_edges = np.array([0.0, 10.0, 20.0])
            out = write_weights("test", weights, x_edges, y_edges)
            assert os.path.exists(out)
            df = pd.read_csv(out, sep="\t")
            assert len(df) == 4  # 2x2 grid
            assert "weight" in df.columns


# ---------------------------------------------------------------------------
# write_runs
# ---------------------------------------------------------------------------


class TestWriteRuns:
    """Test write_runs output."""

    def test_writes_run_pairs(self):
        """write_runs should produce a TSV with run pairs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "runs.dat")
            runs = [(100, 200), (300, 400)]
            write_runs(runs, outpath)
            df = pd.read_csv(outpath, sep="\t", header=None)
            assert len(df) == 2
            assert df.iloc[0, 0] == 100
            assert df.iloc[1, 1] == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
