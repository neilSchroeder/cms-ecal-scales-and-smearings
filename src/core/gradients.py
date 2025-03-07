import numpy as np
import numba
from numba import njit
from numba.typed import Dict
from numba.core import types
import gc

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
        if not x.flags['C_CONTIGUOUS']:
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
def spsa_gradient_optimized(target_function, x, *args, c=1e-6, a=1.0, alpha=0.602, 
                           gamma=0.101, n_iter=2, cache_size=1000, **options):
    """
    Highly optimized SPSA gradient estimation.
    
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
    # Ensure x is a numpy array with proper dtype
    x = np.asarray(x, dtype=np.float64)
    
    # Create cached version of target function
    cached_func, clear_cache = create_target_function_wrapper(target_function)
    
    # Pre-allocate memory for gradient estimate
    n_params = len(x)
    gradient_estimate = np.zeros(n_params, dtype=np.float64)
    
    # Pre-compute decay values for all iterations to avoid repeated calculations
    ck_values = np.array([c / (k ** gamma) for k in range(1, n_iter + 1)], dtype=np.float64)
    
    try:
        # Perform multiple SPSA iterations and average results
        for k in range(n_iter):
            # Get pre-computed perturbation size
            ck = ck_values[k]
            
            # Generate random perturbation vector (±1 with equal probability)
            # Pre-allocate for better memory efficiency
            delta = np.random.choice([-1.0, 1.0], size=n_params).astype(np.float64)
            
            # Generate perturbed parameter vectors efficiently
            # Use in-place operations to avoid extra memory allocations
            x_plus = x + ck * delta  # This creates a new array
            x_minus = x - ck * delta  # This creates a new array
            
            # Evaluate function at perturbed points
            y_plus = cached_func(x_plus, *args, **options)
            y_minus = cached_func(x_minus, *args, **options)
            
            # Use optimized core calculation
            # Calculate gradient for each parameter using vectorized operations
            gradient_iter = _spsa_core_calculation(delta, ck, y_plus, y_minus)
            
            # Accumulate gradient (divide by n_iter at the end for better precision)
            gradient_estimate += gradient_iter
            
        # Divide by number of iterations at the end (more numerically stable)
        if n_iter > 0:
            gradient_estimate /= n_iter
    finally:
        # Clean up memory and clear cache
        clear_cache()
        
    return gradient_estimate


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
        h = kwargs.pop('h', 1e-6)
        use_spsa = kwargs.pop('use_spsa', True)
        spsa_iterations = kwargs.pop('spsa_iterations', 5)
        cache_size = kwargs.pop('cache_size', 1000)
        
        # Use only SPSA for gradient estimation
        if use_spsa:
            return spsa_gradient_optimized(
                target_func, x, *args, 
                c=h, 
                n_iter=spsa_iterations,
                cache_size=cache_size,
                **kwargs
            )
        else:
            # Fallback to another gradient method if needed
            raise ValueError("Only SPSA gradient estimation is supported in this optimized version")
    
    return grad_fn


# Example usage:
# @gradient_function
# def my_objective_function(x, *args, **kwargs):
#     # Your objective function implementation
#     return function_value