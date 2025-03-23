import gc

import numba
import numpy as np
from numba import njit, prange
from numba.core import types
from numba.typed import Dict


# Optimized caching for SPSA - using tuple hash directly
class SPSACache:
    """Optimized cache implementation for SPSA gradients"""

    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}  # Track access frequency instead of order

    def get(self, x):
        """Get value from cache if it exists using a more efficient hash"""
        # Create a hash from the array values - more efficient than converting to tuple
        x_hash = hash(x.tobytes())

        if x_hash in self.cache:
            # Update access count
            self.access_count[x_hash] = self.access_count.get(x_hash, 0) + 1
            return self.cache[x_hash], True
        return None, False

    def set(self, x, value):
        """Add value to cache with frequency-based eviction policy"""
        x_hash = hash(x.tobytes())

        # If cache is full, remove least frequently used item
        if len(self.cache) >= self.max_size and x_hash not in self.cache:
            if self.access_count:
                # Find the key with minimum access count
                min_key = min(self.access_count.items(), key=lambda x: x[1])[0]
                del self.cache[min_key]
                del self.access_count[min_key]

        # Set the value and initialize/update access count
        self.cache[x_hash] = value
        self.access_count[x_hash] = self.access_count.get(x_hash, 0) + 1

    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_count.clear()


# Optimized target function wrapper with improved caching
def create_target_function_wrapper(target_function):
    """Creates a more efficient wrapper around the target function"""

    # Create cache instance
    cache = SPSACache(1000)

    def cached_target_function(x, *args, **kwargs):
        """Cache-enabled target function with numpy array caching"""
        # Check if array is contiguous for better performance
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)

        # Check if result is in cache
        result, found = cache.get(x)
        if found:
            return result

        # Call the actual target function
        result = target_function(x, *args, **kwargs)

        # Store in cache
        cache.set(x, result)
        return result

    def clear_cache():
        """Clear the cache"""
        cache.clear()

    return cached_target_function, clear_cache


# Core SPSA calculation with optimized Numba compilation
@njit(fastmath=True)
def _spsa_core_calculation(delta, ck, y_plus, y_minus):
    """Optimized core SPSA gradient calculation"""
    # Use fastmath for better performance
    return (y_plus - y_minus) / (2 * ck * delta)


# Optimized SPSA gradient estimation
def spsa_gradient_optimized(
    target_function,
    x,
    *args,
    c=1e-6,
    a=1.0,
    alpha=0.602,
    gamma=0.101,
    n_iter=2,
    cache_size=1000,
    **options,
):
    """
    Highly optimized SPSA gradient estimation.

    Args
    ----
    target_function : callable
        The target function to approximate the gradient for.
    x : array_like
        The parameter vector.
    args : tuple
        Additional arguments for the target function.
    c : float, optional
        The perturbation size for SPSA.
    a : float, optional
        The step size scaling factor.
    alpha : float, optional
        The decay factor for step size.
    gamma : float, optional
        The decay factor for perturbation size.
    n_iter : int, optional
        The number of SPSA iterations.
    cache_size : int, optional
        The maximum size of the function evaluation cache.
    options : dict
        Additional options for the target function.

    Returns
    -------
    array_like
        The approximated gradient vector.

    Notes
    -----
    This function is highly optimized for performance and uses Numba
    for accelerated numerical computations. It is suitable for high-
    dimensional optimization tasks.
    """
    # Ensure x is a numpy array with proper dtype
    x = np.asarray(x, dtype=np.float64)

    # Create cached version of target function
    cached_func, clear_cache = create_target_function_wrapper(target_function)

    # Pre-allocate memory for gradient estimate
    n_params = len(x)
    gradient_estimate = np.zeros(n_params, dtype=np.float64)

    try:
        # Pre-compute decay values for all iterations
        ck_values = np.array(
            [c / (k**gamma) for k in range(1, n_iter + 1)], dtype=np.float64
        )

        # Perform multiple SPSA iterations and average results
        for k in range(n_iter):
            ck = ck_values[k]
            delta = np.random.choice([-1.0, 1.0], size=n_params).astype(np.float64)

            x_plus = x + ck * delta
            x_minus = x - ck * delta

            # Evaluate function with error handling
            y_plus = cached_func(x_plus, *args, **options)
            y_minus = cached_func(x_minus, *args, **options)

            # Verify valid outputs
            if (
                y_plus is None
                or y_minus is None
                or np.isnan(y_plus)
                or np.isnan(y_minus)
            ):
                print(f"Warning: Invalid function values in SPSA iteration {k}")
                # Use a small default gradient instead of None
                return np.ones(n_params) * 1e-6
            print(delta, ck, y_plus, y_minus, gradient_estimate)
            try:
                gradient_iter = _spsa_core_calculation(delta, ck, y_plus, y_minus)
                gradient_estimate += gradient_iter

            except Exception as e:
                print(f"Error in SPSA gradient calculation: {e}")
                # Return a small default gradient rather than failing
                return np.ones(n_params) * 1e-6

            print(gradient_estimate)

        # Divide by number of iterations
        if n_iter > 0:
            gradient_estimate /= n_iter
        print(gradient_estimate)

    finally:
        clear_cache()

    print(gradient_estimate)
    return gradient_estimate


@njit(parallel=True)
def _compute_batch_gradient_numba(
    x, h, base_loss, batch_start, batch_end, function_values
):
    """
    Numba-accelerated batch gradient computation.

    Args:
        x: Parameter vector
        h: Step size
        base_loss: Function value at x
        batch_start: Start index of batch
        batch_end: End index of batch
        function_values: Array of function values for perturbed points

    Returns:
        Gradient for the batch
    """
    batch_size = batch_end - batch_start
    batch_grad = np.zeros(batch_size)

    for i in prange(batch_size):  # Parallel loop with Numba
        # Calculate the gradient for this parameter using forward difference
        batch_grad[i] = (function_values[i] - base_loss) / h

    return batch_grad


# Fast finite difference gradient with Numba acceleration
def fast_gradient_optimized(
    target_function,
    x,
    *args,
    h=1e-6,
    batch_size=None,
    cache_size=1000,
    use_spsa=True,
    spsa_iterations=5,
    **options,
):
    """
    Optimized gradient calculation using either SPSA or finite differences.

    Args:
        target_function: Function to approximate gradient for
        x: Parameter vector
        args: Additional arguments for target_function
        h: Step size for finite difference
        batch_size: Size of parameter batches
        cache_size: Maximum size of function evaluation cache
        use_spsa: Whether to use SPSA gradient estimation
        spsa_iterations: Number of SPSA iterations
        options: Additional options for target_function

    Returns:
        Approximated gradient vector
    """
    # Use SPSA for higher dimensions or when explicitly requested
    if use_spsa:
        return spsa_gradient_optimized(
            target_function,
            x,
            *args,
            c=h,
            n_iter=spsa_iterations,
            cache_size=cache_size,
            **options,
        )

    # Create cached version of target function
    cached_func, clear_cache = create_target_function_wrapper(target_function)

    # Get number of parameters
    n_params = len(x)

    # Auto-determine batch size if not specified
    if batch_size is None:
        if n_params > 1000:
            batch_size = min(100, max(20, n_params // 20))
        else:
            batch_size = min(20, max(10, n_params // 10))

    # Create batch indices
    batch_indices = [
        (i, min(i + batch_size, n_params)) for i in range(0, n_params, batch_size)
    ]

    # Base function value at current point
    base_loss = cached_func(x, *args, **options)

    # Initialize gradient array
    gradient = np.zeros(n_params)

    # Process each batch
    for start, end in batch_indices:
        batch_size_actual = end - start

        # Function values for the perturbed parameter vectors
        function_values = np.zeros(batch_size_actual)

        # Evaluate function at perturbed points
        for i in range(batch_size_actual):
            # Create perturbed parameter vector
            x_perturbed = x.copy()
            x_perturbed[start + i] += h

            # Evaluate function at perturbed point
            function_values[i] = cached_func(x_perturbed, *args, **options)

        # Compute batch gradient using Numba-accelerated function
        batch_grad = _compute_batch_gradient_numba(
            x, h, base_loss, start, end, function_values
        )

        # Store results in gradient array
        gradient[start:end] = batch_grad

    # Clean up memory and clear cache
    clear_cache()
    gc.collect()

    return gradient


# Simplified decorator for the gradient function
def gradient_function(target_func):
    """
    Creates an optimized gradient function compatible with optimizers.

    Args:
        target_func: The target function to optimize

    Returns:
        A gradient function that matches the optimizer's expected interface
    """

    def grad_fn(x, *args, **kwargs):
        # Extract gradient-specific parameters with improved defaults
        h = kwargs.pop("h", 1e-6)
        use_spsa = kwargs.pop("use_spsa", True)
        spsa_iterations = kwargs.pop("spsa_iterations", 5)
        cache_size = kwargs.pop("cache_size", 1000)

        # Use only SPSA for gradient estimation
        if use_spsa:
            return spsa_gradient_optimized(
                target_func,
                x,
                *args,
                c=h,
                n_iter=spsa_iterations,
                cache_size=cache_size,
                **kwargs,
            )
        else:
            # Fallback to another gradient method if needed
            raise ValueError(
                "Only SPSA gradient estimation is supported in this optimized version"
            )

    return grad_fn


# Example usage:
# @gradient_function
# def my_objective_function(x, *args, **kwargs):
#     # Your objective function implementation
#     return function_value
