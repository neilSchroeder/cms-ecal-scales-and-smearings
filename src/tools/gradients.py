import numpy as np
import numba
from numba import njit, prange
from numba.typed import Dict
from numba.core import types
import gc

# Simple LRU cache implementation compatible with Numba
class NumbaCompatibleCache:
    """Thread-safe simple cache implementation that works with Numba"""
    
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.keys_order = []  # Simple LRU tracking
    
    def get(self, key):
        """Get value from cache if it exists"""
        if key in self.cache:
            # Move key to end (most recently used)
            self.keys_order.remove(key)
            self.keys_order.append(key)
            return self.cache[key], True
        return None, False
    
    def set(self, key, value):
        """Add value to cache with LRU eviction policy"""
        if key not in self.cache:
            # If cache is full, remove least recently used item
            if len(self.cache) >= self.max_size:
                if self.keys_order:
                    oldest_key = self.keys_order.pop(0)
                    if oldest_key in self.cache:
                        del self.cache[oldest_key]
            
            self.keys_order.append(key)
        else:
            # Move key to end (most recently used)
            self.keys_order.remove(key)
            self.keys_order.append(key)
            
        self.cache[key] = value
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.keys_order.clear()


# Numba-compatible target function wrapper
def create_target_function_wrapper(target_function):
    """Creates a wrapper around the target function that works with the cache"""
    
    # Create cache instance for this wrapper
    cache = NumbaCompatibleCache(1000)
    
    def cached_target_function(x, *args, **kwargs):
        """Cache-enabled target function"""
        # Convert numpy array to tuple for hashing
        try:
            x_tuple = tuple(float(v) for v in x)
            
            # Check if result is in cache
            result, found = cache.get(x_tuple)
            if found:
                return result
                
            # Call the actual target function
            result = target_function(x, *args, **kwargs)
            
            # Store in cache
            cache.set(x_tuple, result)
            return result
        except (TypeError, ValueError):
            # Fallback for unhashable types or other errors
            return target_function(x, *args, **kwargs)
    
    def clear_cache():
        """Clear the cache"""
        cache.clear()
        
    # Return the wrapped function and a method to clear the cache
    return cached_target_function, clear_cache


# Core SPSA calculation that can be JIT-compiled
@njit
def _spsa_core_calculation(x, delta, ck, y_plus, y_minus):
    """Core SPSA gradient calculation step"""
    # Calculate gradient using the SPSA formula
    gradient = (y_plus - y_minus) / (2 * ck * delta)
    return gradient


# SPSA gradient estimation with Numba optimization where possible
def spsa_gradient_optimized(target_function, x, *args, c=1e-6, a=1.0, alpha=0.602, 
                           gamma=0.101, n_iter=2, cache_size=1000, **options):
    """
    Optimized SPSA gradient estimation with partial Numba acceleration.
    
    Args:
        target_function: Function to approximate gradient for
        x: Parameter vector
        args: Additional arguments for target_function
        c: Perturbation size
        a: Step size for SPSA iterations
        alpha: Decay rate for step size
        gamma: Decay rate for perturbation size
        n_iter: Number of SPSA iterations to average
        cache_size: Maximum size of function evaluation cache
        
    Returns:
        Approximated gradient vector using SPSA
    """
    # Create cached version of target function
    cached_func, clear_cache = create_target_function_wrapper(target_function)
    
    # Get dimension of parameter vector
    n_params = len(x)
    
    # Initialize gradient estimate
    gradient_estimate = np.zeros(n_params)
    
    # Base function value at current point (not strictly needed for SPSA)
    base_loss = cached_func(x, *args, **options)
    
    # Perform multiple SPSA iterations and average results
    for k in range(1, n_iter + 1):
        # Decay perturbation size and step size with iteration
        ck = c / (k ** gamma)
        ak = a / (k ** alpha)
        
        # Generate random perturbation vector (±1 with equal probability)
        delta = np.random.choice([-1, 1], size=n_params)
        
        # Generate perturbed parameter vectors
        x_plus = x + ck * delta
        x_minus = x - ck * delta
        
        # Evaluate function at perturbed points
        y_plus = cached_func(x_plus, *args, **options)
        y_minus = cached_func(x_minus, *args, **options)
        
        # Use Numba-optimized core calculation
        gradient_iter = _spsa_core_calculation(x, delta, ck, y_plus, y_minus)
        
        # Update gradient estimate (average across iterations)
        gradient_estimate += gradient_iter / n_iter
    
    # Clean up memory and clear cache
    clear_cache()
    gc.collect()
    
    return gradient_estimate


# Optimized batch gradient computation with Numba
@njit(parallel=True)
def _compute_batch_gradient_numba(x, h, base_loss, batch_start, batch_end, function_values):
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
def fast_gradient_optimized(target_function, x, *args, h=1e-6, batch_size=None, 
                           cache_size=1000, use_spsa=True, spsa_iterations=5, **options):
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
        return spsa_gradient_optimized(target_function, x, *args, c=h, n_iter=spsa_iterations, 
                                      cache_size=cache_size, **options)
    
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
    batch_indices = [(i, min(i + batch_size, n_params)) 
                     for i in range(0, n_params, batch_size)]
    
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


# Decorator to make the gradient function compatible with the optimizer
def gradient_function(target_func):
    """
    Creates a gradient function compatible with the optimizer interface.
    
    Args:
        target_func: The target function to optimize
    
    Returns:
        A gradient function that matches the optimizer's expected interface
    """
    def grad_fn(x, *args, **kwargs):
        # Extract gradient-specific parameters
        h = kwargs.pop('h', 1e-6)
        use_spsa = kwargs.pop('use_spsa', True)
        spsa_iterations = kwargs.pop('spsa_iterations', 5)
        batch_size = kwargs.pop('batch_size', None)
        cache_size = kwargs.pop('cache_size', 1000)
        
        # Call the optimized gradient function
        return fast_gradient_optimized(
            target_func, x, *args, 
            h=h, 
            batch_size=batch_size,
            cache_size=cache_size,
            use_spsa=use_spsa, 
            spsa_iterations=spsa_iterations,
            **kwargs
        )
    
    return grad_fn


# Usage example:
# @gradient_function
# def my_objective_function(x, *args, **kwargs):
#     # Your objective function implementation
#     return function_value
#
# Then use with AdamW optimizer:
# optimizer = OptimizedAdamWMinimizer(...)
# result = optimizer.minimize(my_objective_function, x0, jac=my_objective_function)