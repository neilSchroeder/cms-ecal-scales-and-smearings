import numpy as np
import pytest
from unittest.mock import patch
from src.plotters.fit_bw_cb import fit_bw_cb, NLL, target
from src.classes.breit_wigner import bw
from src.classes.crystal_ball import cb

def test_nll_function():
    """Test the negative log-likelihood function."""
    # Test with equal arrays
    a = np.ones(10)
    b = np.ones(10)
    nll_equal = NLL(a, b)
    assert isinstance(nll_equal, float)
    # NLL should be minimized when a and b are identical
    assert nll_equal >= 0
    
    # Test with different arrays
    a = np.array([10, 15, 20, 5, 0])
    b = np.array([12, 14, 18, 6, 1])
    nll_diff = NLL(a, b)
    assert isinstance(nll_diff, float)
    assert nll_diff > 0
    
    # Test with array containing zeros
    a = np.array([10, 0, 20, 5, 0])
    b = np.array([12, 1, 18, 6, 1])
    nll_zeros = NLL(a, b)
    assert isinstance(nll_zeros, float)
    assert np.isfinite(nll_zeros)
    
    # Test that NLL is higher for worse fits
    a = np.ones(10) * 10
    b1 = np.ones(10) * 10  # Perfect match
    b2 = np.ones(10) * 11  # Small difference
    b3 = np.ones(10) * 20  # Large difference
    assert NLL(b1, a) <= NLL(b2, a) <= NLL(b3, a)
    
    # Test with arrays of different lengths should raise an error
    with pytest.raises(ValueError):
        NLL(np.array([1, 2, 3]), np.array([1, 2]))


def test_target_function():
    """Test the target function used for optimization."""
    
    thisBW = bw(np.linspace(0, 100, 100))
    thisCB = cb(np.linspace(0, 100, 100), [1.4, 1.9, 0.0, 1.0])
    data = np.random.poisson(100, 100)
    
    # Target should return a scalar value
    result = target([1.4, 1.9, 0.0, 1.0], thisBW, thisCB, data)
    assert isinstance(result, float)
    
    # Target should be positive (it's returned from NLL which is a sum of log terms)
    assert result > 0


@patch('src.plotters.fit_bw_cb.minimize')
def test_fit_bw_cb_basic(mock_minimize):
    """Test basic functionality of fit_bw_cb."""
    # Mock minimize result
    mock_result = type('obj', (object,), {
        'x': [1.5, 2.0, 0.1, 1.0],
        'success': True,
        'fun': 100.0
    })
    mock_minimize.return_value = mock_result
    
    # Create simple histogram data
    mids = np.linspace(85, 95, 20)
    hist = np.random.poisson(100, 20)
    init_params = [1.4, 1.9, 0.0, 1.0]
    
    # Call fit_bw_cb
    result = fit_bw_cb(mids, hist, init_params)
    
    # Check that fit_bw_cb returns a dictionary with expected keys
    assert isinstance(result, dict)
    assert 'fit_params' in result
    assert 'fit_hist' in result
    
    # Check that fit_hist has the same length as input mids
    assert len(result['fit_hist']) == len(mids)
    
    # Check that fit_params contains the optimized parameters
    assert result['fit_params'] == mock_result.x


def test_fit_bw_cb_realistic():
    """Test fit_bw_cb with realistic Z peak data."""
    # Create Z-peak like histogram (normal distribution centered around 91 GeV)
    mids = np.linspace(80, 100, 40)
    # Generate mock Z peak data
    peak_center = 91.2
    peak_width = 2.5
    hist = np.random.normal(peak_center, peak_width, 10000)
    hist_counts, _ = np.histogram(hist, bins=40, range=(80, 100))
    
    # Initial parameters [alpha, n, mean, sigma]
    init_params = [1.4, 1.9, peak_center - 91.188, peak_width]
    
    # Call fit_bw_cb
    result = fit_bw_cb(mids, hist_counts, init_params)
    
    # Basic checks on the result
    assert isinstance(result, dict)
    assert 'fit_params' in result
    assert 'fit_hist' in result
    

def test_fit_bw_cb_with_empty_histogram():
    """Test fit_bw_cb with empty histogram."""
    mids = np.linspace(80, 100, 20)
    hist = np.zeros(20)  # Empty histogram
    init_params = [1.4, 1.9, 0.0, 1.0]
    
    # Should handle empty histogram gracefully
    result = fit_bw_cb(mids, hist, init_params)
    
    # Even with empty histogram, should return a dictionary
    assert isinstance(result, dict)