import numpy as np
import pytest
from scipy import stats

from python.classes.constant_classes import CategoryConstants as cc
from python.classes.zcat_class import zcat

# Ensure default constants for testing if not defined
if not hasattr(cc, "MIN_EVENTS_DATA"):
    cc.MIN_EVENTS_DATA = 5
if not hasattr(cc, "MIN_EVENTS_MC_DIAG"):
    cc.MIN_EVENTS_MC_DIAG = 5
if not hasattr(cc, "MIN_EVENTS_MC_OFFDIAG"):
    cc.MIN_EVENTS_MC_OFFDIAG = 5


@pytest.fixture
def valid_zcat():
    # create sample data arrays within the allowed histogram range
    data = np.random.normal(91.188, 2.8, 100000).astype(np.float32)
    mc = np.random.normal(91.188, 2.8, 100000).astype(np.float32)
    weights = np.ones(100000, dtype=np.float32)
    # instantiate with auto_bin off to avoid bin size update complexities
    return zcat(
        0,
        0,
        data,
        mc,
        weights,
        hist_min=80.0,
        hist_max=100.0,
        bin_size=0.25,
        _kAutoBin=False,
    )


def test_create_zcat(valid_zcat):
    # check that the zcat object is created with the correct attributes
    assert valid_zcat.lead_index == 0
    assert valid_zcat.sublead_index == 0
    assert valid_zcat.lead_smear_index == -1
    assert valid_zcat.sublead_smear_index == -1
    # assert data, mc, and weights are float32 arrays
    assert valid_zcat.data.dtype == np.float32
    assert valid_zcat.mc.dtype == np.float32
    assert valid_zcat.weights.dtype == np.float32

    assert valid_zcat.hist_min == 80.0
    assert valid_zcat.hist_max == 100.0
    assert valid_zcat.auto_bin is False
    assert valid_zcat.bin_size == 0.25
    assert valid_zcat.valid is True

    # check bins and buffers
    assert isinstance(valid_zcat.num_bins, int)
    assert valid_zcat.num_bins > 0
    assert valid_zcat.temp_data is not None
    assert valid_zcat.temp_mc is not None

    # check other attributes
    assert valid_zcat.updated is False
    assert valid_zcat.NLL == 0
    assert valid_zcat.weight == 1
    assert valid_zcat.seed == 3543136929
    assert valid_zcat.history == []
    assert valid_zcat.lead_smear == 0
    assert valid_zcat.sublead_smear == 0
    assert valid_zcat.lead_scale == 1
    assert valid_zcat.sublead_scale == 1
    assert np.allclose(
        valid_zcat.top_and_bottom, [valid_zcat.hist_min, valid_zcat.hist_max]
    )


def test_update(valid_zcat):
    # call update with non-trivial scale and smearing values
    valid_zcat.update(1.1, 0.9, 0.05, 0.05)
    # verify update flag and history is appended
    assert valid_zcat.updated is True
    # check that NLL is computed and finite
    assert np.isfinite(valid_zcat.NLL)


def test_inject(valid_zcat):
    old_data = valid_zcat.data.copy()
    # apply injection
    valid_zcat.inject(1.2, 0.8, 0.03, 0.04)
    # check data has changed from the original values
    assert not np.allclose(old_data, valid_zcat.data)


def test_clean_up(valid_zcat):
    valid_zcat.clean_up()
    # after clean_up, data, mc, and weights should be set to None and valid flag should be False
    assert valid_zcat.data is None
    assert valid_zcat.mc is None
    assert valid_zcat.weights is None
    assert valid_zcat.valid is False


def test_check_invalid_with_counts(valid_zcat):
    # using check_invalid with counts lower than the constants should return True.
    # Here we simulate insufficient statistics by providing low counts.
    invalid = valid_zcat.check_invalid(
        cc.MIN_EVENTS_DATA - 1, cc.MIN_EVENTS_MC_DIAG - 1
    )
    assert invalid is True

    # For diag category (i==j), use numbers above the minimum
    valid = valid_zcat.check_invalid(cc.MIN_EVENTS_DATA + 1, cc.MIN_EVENTS_MC_DIAG + 1)
    assert valid is False


def test_set_bin_size():
    # Create an instance with auto_bin True to trigger set_bin_size
    data = np.random.normal(90, 5, 1000).astype(np.float32)
    mc = np.random.normal(90, 5, 1000).astype(np.float32)
    weights = np.ones(50, dtype=np.float32)
    instance = zcat(
        1,
        2,
        data,
        mc,
        weights,
        hist_min=80.0,
        hist_max=100.0,
        bin_size=0.25,
        _kAutoBin=True,
    )
    # Before setting bin size, bin_size is 0.25, but set_bin_size is called in __init__ if auto_bin is True.
    # We check that bin_size is updated (or remains >=0.25).
    assert instance.bin_size >= 0.25


def test_print_output(valid_zcat, capsys):
    # Capture the output of print()
    valid_zcat.print()
    captured = capsys.readouterr().out
    # Check that expected keywords are in output
    assert "lead index:" in captured
    assert "sublead index:" in captured
    assert "NLL:" in captured
