import multiprocessing
from functools import lru_cache

import numpy as np
import scipy.optimize
from joblib import Parallel, delayed

from src.core.adamw import optimized_adamw_minimize

# Import the gradient optimization utilities
from src.core.gradients import fast_gradient_optimized


def enhanced_target_function_wrapper(initial_guess, ZCATS, **kwargs):
    """
    Enhanced wrapper for the target function with improved caching and gradient computation.

    Args:
        initial_guess: Initial parameter vector
        ZCATS: Category objects
        **kwargs: Additional keyword arguments

    Returns:
        A tuple containing:
        - The wrapped target function
        - A reset function
        - A function for gradient computation
    """
    previous_guess = [initial_guess.copy()]
    num_scales = kwargs.get("num_scales", 0)
    num_smears = kwargs.get("num_smears", 0)

    # Cache for the wrapped function
    function_cache = {}

    def wrapped_target_function(x, *args, **options):
        """Target function with caching and optimization"""
        # Try to use cache
        key = tuple(float(v) for v in x)
        if key in function_cache:
            return function_cache[key]

        # Extract whether verbose flag is set
        verbose = options.get("verbose", False)

        # Use the actual target function
        ret = target_function(
            x,
            previous_guess[0],
            ZCATS,
            num_scales,
            num_smears,
            verbose=verbose,
            **options,
        )

        # Update previous guess
        previous_guess[0] = x.copy()

        # Cache result
        if len(function_cache) > 1000:  # Limit cache size
            # Remove a random key (simple strategy)
            try:
                function_cache.pop(next(iter(function_cache)))
            except (StopIteration, KeyError):
                pass

        function_cache[key] = ret
        return ret

    def reset(x=None):
        """Reset the previous guess and clear cache"""
        previous_guess[0] = x.copy() if x is not None else initial_guess.copy()
        function_cache.clear()

    # Create a gradient computation function specifically for this target function
    def compute_gradient(x, *args, h=1e-6, use_spsa=True, spsa_iterations=3, **options):
        """
        Compute the gradient of the target function.

        Args:
            x: Parameter vector
            h: Step size for finite difference
            use_spsa: Whether to use SPSA for gradient estimation
            spsa_iterations: Number of SPSA iterations
            **options: Additional options

        Returns:
            Gradient vector
        """
        fast_gradient_optimized(
            wrapped_target_function,
            x,
            *args,
            h=h,
            use_spsa=use_spsa,
            spsa_iterations=spsa_iterations,
            **options,
        )

    return wrapped_target_function, reset, compute_gradient


def target_function(
    x, previous, ZCATS, num_scales, num_smears, verbose=False, **options
):
    """
    Optimized target function with faster category updates.

    Args:
        x: Current parameter vector
        previous: Previous parameter vector
        ZCATS: Category objects
        num_scales: Number of scales
        num_smears: Number of smears
        verbose: Whether to print verbose output
        **options: Additional options

    Returns:
        The computed negative log-likelihood
    """
    # Find where parameters have changed (using vectorized operations)
    diff_mask = np.abs(x - previous) > 1e-10
    if not np.any(diff_mask):
        # No changes, can return cached result if available
        if hasattr(x, "_cached_result"):
            return x._cached_result

    # Find which scales have updated
    updated_scales = np.where(diff_mask)[0]

    # Check if any updated scales affect each category
    mask = np.zeros(len(ZCATS), dtype=bool)
    for i, cat in enumerate(ZCATS):
        if not cat.valid:
            continue

        # Check if any parameter used by this category has changed
        indices = [cat.lead_index, cat.sublead_index]
        if num_smears > 0:
            indices.extend([cat.lead_smear_index, cat.sublead_smear_index])

        needs_update = any(idx in updated_scales for idx in indices)
        mask[i] = needs_update

    cats_to_update = np.array(ZCATS)[mask]

    # Update only the necessary categories (in parallel if possible)
    if len(cats_to_update) > 1000:  # Only use parallelization for enough work
        num_cores = min(multiprocessing.cpu_count(), len(cats_to_update))

        def update_cat(cat):
            if num_smears == 0:
                cat.update(x[cat.lead_index], x[cat.sublead_index])
            else:
                cat.update(
                    x[cat.lead_index],
                    x[cat.sublead_index],
                    lead_smear=x[cat.lead_smear_index],
                    sublead_smear=x[cat.sublead_smear_index],
                )
            return cat

        updated_cats = Parallel(n_jobs=num_cores)(
            delayed(update_cat)(cat) for cat in cats_to_update
        )

        # Put the updated cats back
        for i, updated in zip(np.where(mask)[0], updated_cats):
            ZCATS[i] = updated
    else:
        # Sequential update for small number of categories
        for cat in cats_to_update:
            if num_smears == 0:
                cat.update(x[cat.lead_index], x[cat.sublead_index])
            else:
                cat.update(
                    x[cat.lead_index],
                    x[cat.sublead_index],
                    lead_smear=x[cat.lead_smear_index],
                    sublead_smear=x[cat.sublead_smear_index],
                )

    # Calculate NLL with vectorized operations where possible
    valid_cats = [cat for cat in ZCATS if cat.valid]
    weights = np.array([cat.weight for cat in valid_cats])
    nlls = np.array([cat.NLL for cat in valid_cats])

    tot = np.sum(weights)
    weighted_nlls = weights * nlls
    ret = np.sum(weighted_nlls)

    final_value = ret / tot if tot != 0 else 9e30
    print(final_value, x)

    # Cache the result
    if hasattr(x, "_cached_result"):
        x._cached_result = final_value

    return final_value


def minimize(
    fun, x0, args=(), method="adamw", jac=None, bounds=None, callback=None, options=None
):
    """Enhanced minimize wrapper with optimized implementation."""
    if method.lower() == "adamw":
        # Default options
        default_options = {
            "lr": 1e-5,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 1e-6,
            "max_iter": 1000,
            "tol": 1e-5,
            "patience": 100,
            "lr_reduce_factor": 0.5,
            "lr_reduce_patience": 5,
            "verbose": False,
            "n_jobs": options.get("n_jobs", 1),
        }

        # Update with user options
        if options is not None:
            default_options.update(options)

        return optimized_adamw_minimize(
            fun, x0, args, jac, bounds, callback, **default_options
        )
    else:
        # Use scipy's implementation
        return scipy.optimize.minimize(
            fun,
            x0,
            args=args,
            method=method,
            jac=jac,
            bounds=bounds,
            callback=callback,
            options=options,
        )
