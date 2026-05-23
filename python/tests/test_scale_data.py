"""Tests for the vectorized scale_data.apply() function."""

import numpy as np
import pandas as pd
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from python.classes.constant_classes import DataConstants as dc, PyValConstants as pvc
from python.utilities.scale_data import apply, scale


def _make_scales_array(entries):
    """Build a scales numpy array from a list of dicts."""
    rows = []
    for e in entries:
        row = [0.0] * 11
        row[dc.i_run_min] = e.get("run_min", 0)
        row[dc.i_run_max] = e.get("run_max", 999999)
        row[dc.i_eta_min] = e.get("eta_min", 0.0)
        row[dc.i_eta_max] = e.get("eta_max", 2.5)
        row[dc.i_r9_min] = e.get("r9_min", 0.0)
        row[dc.i_r9_max] = e.get("r9_max", 1.0)
        row[dc.i_et_min] = e.get("et_min", dc.MIN_ET)
        row[dc.i_et_max] = e.get("et_max", dc.MAX_ET)
        row[dc.i_gain] = e.get("gain", 0)
        row[dc.i_scale] = e.get("scale", 1.0)
        row[dc.i_err] = e.get("err", 0.001)
        rows.append(row)
    return np.array(rows)


def _make_data(
    n,
    run=300000,
    eta_lead=1.0,
    eta_sub=0.5,
    r9_lead=0.95,
    r9_sub=0.85,
    e_lead=50.0,
    e_sub=40.0,
    gain_lead=0,
    gain_sub=0,
    invmass=91.0,
):
    """Build a minimal data DataFrame with n identical events."""
    return pd.DataFrame(
        {
            dc.RUN: np.full(n, run),
            dc.ETA_LEAD: np.full(n, eta_lead),
            dc.ETA_SUB: np.full(n, eta_sub),
            dc.R9_LEAD: np.full(n, r9_lead),
            dc.R9_SUB: np.full(n, r9_sub),
            dc.E_LEAD: np.full(n, e_lead),
            dc.E_SUB: np.full(n, e_sub),
            dc.GAIN_LEAD: np.full(n, gain_lead, dtype=int),
            dc.GAIN_SUB: np.full(n, gain_sub, dtype=int),
            dc.INVMASS: np.full(n, invmass),
        }
    )


def test_single_category_match():
    """Every event should match the single scale category and get scaled."""
    scales = _make_scales_array(
        [
            {
                "run_min": 0,
                "run_max": 999999,
                "eta_min": 0,
                "eta_max": 2.5,
                "r9_min": 0,
                "r9_max": 1.0,
                "scale": 1.005,
                "err": 0.001,
            }
        ]
    )
    data = _make_data(100)
    result = apply((data, scales))

    # E_LEAD should be scaled by 1.005
    expected_e_lead = 50.0 * 1.005
    np.testing.assert_allclose(result[dc.E_LEAD].values, expected_e_lead, rtol=1e-5)
    # invmass scaled by sqrt(lead_scale * sub_scale)
    expected_invmass = 91.0 * np.sqrt(1.005 * 1.005)
    np.testing.assert_allclose(result[dc.INVMASS].values, expected_invmass, rtol=1e-4)
    # up/down columns should exist
    assert pvc.KEY_INVMASS_UP in result.columns
    assert pvc.KEY_INVMASS_DOWN in result.columns
    print("  test_single_category_match: PASSED")


def test_no_matching_category():
    """Events outside all scale categories should get scale=0 (unscaled E=0)."""
    scales = _make_scales_array(
        [{"run_min": 100, "run_max": 200, "scale": 1.01, "err": 0.001}]
    )
    data = _make_data(10, run=300000)  # run 300000 not in [100,200]
    result = apply((data, scales))
    # scale=0 means E_LEAD *= 0
    np.testing.assert_allclose(result[dc.E_LEAD].values, 0.0, atol=1e-10)
    print("  test_no_matching_category: PASSED")


def test_et_dependent_scales():
    """Et-dependent categories should correctly select by transverse energy."""
    et_boundary = 50.0
    # Event with eta_lead=0 -> et = E/cosh(0) = E
    scales = _make_scales_array(
        [
            {"et_min": 0, "et_max": et_boundary, "scale": 0.99, "err": 0.001},
            {"et_min": et_boundary, "et_max": 14000, "scale": 1.01, "err": 0.001},
        ]
    )
    # e_lead=40 -> et=40/cosh(0)=40 < 50, should match first category
    data = _make_data(5, e_lead=40.0, e_sub=40.0, eta_lead=0.0, eta_sub=0.0)
    result = apply((data, scales))
    expected = 40.0 * 0.99
    np.testing.assert_allclose(result[dc.E_LEAD].values, expected, rtol=1e-5)

    # e_lead=60 -> et=60/cosh(0)=60 >= 50, should match second category
    data2 = _make_data(5, e_lead=60.0, e_sub=60.0, eta_lead=0.0, eta_sub=0.0)
    result2 = apply((data2, scales))
    expected2 = 60.0 * 1.01
    np.testing.assert_allclose(result2[dc.E_LEAD].values, expected2, rtol=1e-5)
    print("  test_et_dependent_scales: PASSED")


def test_gain_dependent_scales():
    """Gain-dependent categories should correctly map gain encoding."""
    scales = _make_scales_array(
        [
            {"gain": 12, "scale": 1.001, "err": 0.001},
            {"gain": 6, "scale": 1.002, "err": 0.001},
            {"gain": 1, "scale": 1.003, "err": 0.001},
        ]
    )
    # gain_lead=0 -> maps to gain 12; gain_sub=1 -> maps to gain 6
    data = _make_data(5, gain_lead=0, gain_sub=1)
    result = apply((data, scales))
    np.testing.assert_allclose(result[dc.E_LEAD].values, 50.0 * 1.001, rtol=1e-5)
    np.testing.assert_allclose(result[dc.E_SUB].values, 40.0 * 1.002, rtol=1e-5)

    # gain_lead=3 (>2) -> maps to gain 1
    data2 = _make_data(5, gain_lead=3, gain_sub=3)
    result2 = apply((data2, scales))
    np.testing.assert_allclose(result2[dc.E_LEAD].values, 50.0 * 1.003, rtol=1e-5)
    print("  test_gain_dependent_scales: PASSED")


def test_empty_data():
    """Empty data should return empty dataframe."""
    scales = _make_scales_array([{"scale": 1.0, "err": 0.001}])
    data = _make_data(0)
    result = apply((data, scales))
    assert len(result) == 0
    print("  test_empty_data: PASSED")


def test_empty_scales():
    """Empty scales array should return data unchanged."""
    data = _make_data(10)
    original_e = data[dc.E_LEAD].values.copy()
    result = apply((data, np.array([]).reshape(0, 11)))
    np.testing.assert_array_equal(result[dc.E_LEAD].values, original_e)
    print("  test_empty_scales: PASSED")


def test_multiple_eta_categories():
    """Different events matching different eta bins should get correct scales."""
    scales = _make_scales_array(
        [
            {"eta_min": 0.0, "eta_max": 1.0, "scale": 1.01, "err": 0.001},
            {"eta_min": 1.0, "eta_max": 2.5, "scale": 1.02, "err": 0.001},
        ]
    )
    data = pd.DataFrame(
        {
            dc.RUN: [300000, 300000],
            dc.ETA_LEAD: [0.5, 1.5],  # first in bin 0, second in bin 1
            dc.ETA_SUB: [0.5, 1.5],
            dc.R9_LEAD: [0.95, 0.95],
            dc.R9_SUB: [0.85, 0.85],
            dc.E_LEAD: [50.0, 50.0],
            dc.E_SUB: [40.0, 40.0],
            dc.GAIN_LEAD: [0, 0],
            dc.GAIN_SUB: [0, 0],
            dc.INVMASS: [91.0, 91.0],
        }
    )
    result = apply((data, scales))
    np.testing.assert_allclose(result[dc.E_LEAD].values[0], 50.0 * 1.01, rtol=1e-5)
    np.testing.assert_allclose(result[dc.E_LEAD].values[1], 50.0 * 1.02, rtol=1e-5)
    print("  test_multiple_eta_categories: PASSED")


def test_invmass_up_greater_than_nominal():
    """Invariant mass up should be >= nominal (scale + err)."""
    scales = _make_scales_array([{"scale": 1.005, "err": 0.002}])
    data = _make_data(10)
    result = apply((data, scales))
    assert (result[pvc.KEY_INVMASS_UP].values >= result[dc.INVMASS].values).all()


def test_invmass_down_less_than_nominal():
    """Invariant mass down should be <= nominal (scale - err)."""
    scales = _make_scales_array([{"scale": 1.005, "err": 0.002}])
    data = _make_data(10)
    result = apply((data, scales))
    assert (result[pvc.KEY_INVMASS_DOWN].values <= result[dc.INVMASS].values).all()


def test_scale_end_to_end():
    """scale() should read a scales file and apply to data via multiprocessing."""
    scales_row = "200000\t400000\t0.0\t2.5\t0.0\t1.0\t0\t14000\t0\t1.003\t0.001"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write(scales_row + "\n")
        scales_path = f.name
    try:
        data = _make_data(50)
        result = scale(data, scales_path)
        expected_e_lead = 50.0 * 1.003
        np.testing.assert_allclose(result[dc.E_LEAD].values, expected_e_lead, rtol=1e-4)
    finally:
        os.unlink(scales_path)


if __name__ == "__main__":
    print("Running scale_data tests...")
    test_single_category_match()
    test_no_matching_category()
    test_et_dependent_scales()
    test_gain_dependent_scales()
    test_empty_data()
    test_empty_scales()
    test_multiple_eta_categories()
    print("All scale_data tests passed!")
