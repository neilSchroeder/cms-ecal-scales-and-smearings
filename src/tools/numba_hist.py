import numba
import numpy as np

"""
This code was pulled directly from https://numba.pydata.org/numba-examples/examples/density_estimation/histogram/results.html
It is a simplified implementation of a numpy histogram, with vastly increased speed
"""


@numba.jit(nopython=True)
def get_bin_edges(a, bins):
    bin_edges = np.zeros((bins + 1,), dtype=np.float64)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in range(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges


@numba.jit(nopython=True)
def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1  # a_max always in last bin

    bin = int(n * (x - a_min) / (a_max - a_min))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin


@numba.jit(nopython=True)
def slow_numba_histogram(a, bins):
    hist = np.zeros((bins,), dtype=np.intp)
    bin_edges = get_bin_edges(a, bins)
    for x in a.flat:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1

    return hist, bin_edges


@numba.jit(nopython=True)
def slow_weighted_histogram(a, weights, bins):
    hist = np.zeros((bins,), dtype=np.float32)
    bin_edges = get_bin_edges(a, bins)
    for i, x in enumerate(a.flat):
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += weights[i]

    return hist, bin_edges

@numba.njit(parallel=True, fastmath=True, cache=True)
def optimized_get_bin_edges(a_min, a_max, bins):
    """Pre-compute bin edges given min/max values"""
    bin_edges = np.linspace(a_min, a_max, bins + 1, dtype=np.float64)
    return bin_edges

@numba.njit(fastmath=True, cache=True)
def optimized_compute_bin(x, a_min, a_max, n):
    """Compute bin index without needing full bin_edges array"""
    # Handle edge cases more efficiently
    if x < a_min:
        return -1  # Out of bounds
    
    if x >= a_max:
        return n - 1  # Last bin
    
    # Fast computation without branches
    bin_idx = int(n * (x - a_min) / (a_max - a_min))
    return min(bin_idx, n - 1)  # Safety check for numerical edge cases

@numba.njit(parallel=True, fastmath=True, cache=True)
def numba_histogram(a, bins, a_min=None, a_max=None):
    """Parallelized histogram computation"""
    # Auto-detect min/max if not provided
    if a_min is None:
        a_min = a.min()
    if a_max is None:
        a_max = a.max()
    
    # Add a tiny amount to a_max to include the max value
    a_max = a_max * (1.0 + 1e-7)
    
    # Pre-allocate histogram
    hist = np.zeros(bins, dtype=np.int64)
    bin_edges = optimized_get_bin_edges(a_min, a_max, bins)
    
    # Compute bin for each element in parallel
    for i in numba.prange(len(a)):
        x = a[i]
        bin_idx = optimized_compute_bin(x, a_min, a_max, bins)
        if bin_idx >= 0:
            # Use atomic add for thread safety in parallel mode
            numba.atomic.add(hist, bin_idx, 1)
    
    return hist, bin_edges

@numba.njit(parallel=True, fastmath=True, cache=True)
def numba_weighted_histogram(a, weights, bins, a_min=None, a_max=None):
    """Parallelized weighted histogram computation"""
    # Auto-detect min/max if not provided
    if a_min is None:
        a_min = a.min()
    if a_max is None:
        a_max = a.max()
    
    # Add a tiny amount to a_max to include the max value
    a_max = a_max * (1.0 + 1e-7)
    
    # Pre-allocate histogram
    hist = np.zeros(bins, dtype=np.float64)
    bin_edges = optimized_get_bin_edges(a_min, a_max, bins)
    
    # For smaller arrays, use sequential processing to avoid atomic overhead
    if len(a) < 1000:
        for i in range(len(a)):
            x = a[i]
            bin_idx = optimized_compute_bin(x, a_min, a_max, bins)
            if bin_idx >= 0:
                hist[bin_idx] += weights[i]
    else:
        # For larger arrays, parallel processing with atomic operations
        # We'll use a chunking strategy to reduce atomic contention
        n_chunks = min(numba.get_num_threads(), 16)  # Limit chunks for efficiency
        chunk_size = (len(a) + n_chunks - 1) // n_chunks
        
        # Create local histograms for each chunk
        local_hists = np.zeros((n_chunks, bins), dtype=np.float64)
        
        # Process each chunk in parallel
        for chunk_id in numba.prange(n_chunks):
            start = chunk_id * chunk_size
            end = min(start + chunk_size, len(a))
            
            # Process this chunk
            for i in range(start, end):
                x = a[i]
                bin_idx = optimized_compute_bin(x, a_min, a_max, bins)
                if bin_idx >= 0:
                    local_hists[chunk_id, bin_idx] += weights[i]
        
        # Combine local histograms
        for chunk_id in range(n_chunks):
            hist += local_hists[chunk_id]
    
    return hist, bin_edges