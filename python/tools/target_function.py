import multiprocessing
from functools import lru_cache

import numpy as np
import scipy.optimize
from joblib import Parallel, delayed
from scipy.optimize import OptimizeResult


# Add caching to expensive functions
def memoize_target_function(func):
    """Decorator to memoize the target function to avoid redundant calculations."""
    cache = {}

    def wrapper(x, *args, **kwargs):
        # Convert x to tuple for hashing
        key = tuple(x)
        if key in cache:
            return cache[key]

        result = func(x, *args, **kwargs)
        cache[key] = result
        return result

    # Clear cache method
    def clear_cache():
        cache.clear()

    wrapper.clear_cache = clear_cache
    return wrapper


def target_function_wrapper(initial_guess, __ZCATS__, *args, **kwargs):
    """
    Optimized wrapper for the target function with improved caching.
    """
    previous_guess = [initial_guess]

    # Apply memoization decorator
    @memoize_target_function
    def wrapped_target_function(x, *args, **options):
        (previous, __ZCATS__, __num_scales__, __num_smears__) = args
        ret = target_function(
            x, previous_guess[0], __ZCATS__, __num_scales__, __num_smears__, **options
        )
        previous_guess[0] = x.copy()  # Make sure to copy
        return ret

    def reset(x=None):
        previous_guess[0] = x.copy() if x is not None else initial_guess.copy()
        wrapped_target_function.clear_cache()

    return wrapped_target_function, reset


def target_function(x, *args, verbose=False, **options):
    """
    Optimized target function with faster category updates.
    """
    # Unpack args
    (previous, __ZCATS__, __num_scales__, __num_smears__) = args

    # Find where parameters have changed (using vectorized operations)
    diff_mask = np.abs(x - previous) > 1e-10
    if not np.any(diff_mask):
        # No changes, use cached result if available
        if hasattr(x, "_cached_result"):
            return x._cached_result

    updated_scales = np.where(diff_mask)[0]

    # Vectorized mask creation for categories that need updates
    cat_indices = np.array(
        [
            (
                cat.lead_index,
                cat.sublead_index,
                cat.lead_smear_index if __num_smears__ > 0 else -1,
                cat.sublead_smear_index if __num_smears__ > 0 else -1,
            )
            for cat in __ZCATS__
        ]
    )

    # Check if any updated scales affect each category
    mask = np.zeros(len(__ZCATS__), dtype=bool)
    for i, cat in enumerate(__ZCATS__):
        if not cat.valid:
            continue

        # Check if any parameter used by this category has changed
        indices = [cat.lead_index, cat.sublead_index]
        if __num_smears__ > 0:
            indices.extend([cat.lead_smear_index, cat.sublead_smear_index])

        needs_update = any(idx in updated_scales for idx in indices)
        mask[i] = needs_update

    cats_to_update = np.array(__ZCATS__)[mask]

    # Update only the necessary categories (in parallel if possible)
    if len(cats_to_update) > 10:  # Only use parallelization for enough work
        num_cores = min(multiprocessing.cpu_count(), len(cats_to_update))

        def update_cat(cat):
            if __num_smears__ == 0:
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
            __ZCATS__[i] = updated
    else:
        # Sequential update for small number of categories
        for cat in cats_to_update:
            if __num_smears__ == 0:
                cat.update(x[cat.lead_index], x[cat.sublead_index])
            else:
                cat.update(
                    x[cat.lead_index],
                    x[cat.sublead_index],
                    lead_smear=x[cat.lead_smear_index],
                    sublead_smear=x[cat.sublead_smear_index],
                )

    # Calculate NLL with vectorized operations where possible
    valid_cats = [cat for cat in __ZCATS__ if cat.valid]
    weights = np.array([cat.weight for cat in valid_cats])
    nlls = np.array([cat.NLL for cat in valid_cats])

    tot = np.sum(weights)
    weighted_nlls = weights * nlls
    ret = np.sum(weighted_nlls)

    final_value = ret / tot if tot != 0 else 9e30

    # Cache the result
    if hasattr(x, "_cached_result"):
        x._cached_result = final_value

    return final_value


def fast_gradient(x, *args, h=1e-6, n_jobs=-1, **options):
    """
    Optimized gradient calculation using parallel processing and forward differences.

    Args:
        x: Parameter vector
        args: Additional arguments for target_function
        h: Step size for finite difference
        n_jobs: Number of parallel jobs (-1 for all cores)
        options: Additional options for target_function

    Returns:
        gradient: Approximated gradient vector
    """
    # Base function value
    base_loss = target_function(x, *args, **options)

    # Number of parameters
    n_params = len(x)

    # Use forward difference instead of central difference (2x faster)
    def compute_partial_derivative(i):
        x_plus = x.copy()
        x_plus[i] += h

        loss_plus = target_function(x_plus, *args, **options)
        return (loss_plus - base_loss) / h

    # Parallel computation of gradients
    if n_jobs != 1:
        n_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        gradient = Parallel(n_jobs=n_cores)(
            delayed(compute_partial_derivative)(i) for i in range(n_params)
        )
    else:
        # Sequential computation
        gradient = [compute_partial_derivative(i) for i in range(n_params)]

    return np.array(gradient)


class OptimizedAdamWMinimizer:
    """
    Optimized AdamW implementation with adaptive learning rates and better convergence.
    """

    def __init__(
        self,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        max_iter=1000,
        tol=1e-5,
        patience=10,  # Added patience for early stopping
        lr_reduce_factor=0.5,  # For learning rate scheduling
        lr_reduce_patience=5,
        verbose=False,
        gradient_batch_size=None,  # For mini-batch gradients
    ):
        self.lr = lr
        self.initial_lr = lr  # Store for restarts
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience
        self.lr_reduce_factor = lr_reduce_factor
        self.lr_reduce_patience = lr_reduce_patience
        self.verbose = verbose
        self.gradient_batch_size = gradient_batch_size

        # States
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0  # Time step

    def _step(self, x, grad, func_val):
        """Optimized step function with adaptive learning rate."""
        # Initialize moments if first step
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)

        self.t += 1

        # Update moments with better numerical stability
        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * grad
        self.v = self.betas[1] * self.v + (1 - self.betas[1]) * (
            np.square(grad) + self.eps
        )

        # Bias correction
        m_hat = self.m / (1 - self.betas[0] ** self.t)
        v_hat = self.v / (1 - self.betas[1] ** self.t)

        # AdamW update with improved numerical stability
        # Apply weight decay directly to parameters
        x_wd = x * (1 - self.lr * self.weight_decay)

        # Adaptive step size based on gradient history
        update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        # Clip update for stability if needed
        if np.max(np.abs(update)) > 1.0:
            update = update * (1.0 / np.max(np.abs(update)))

        new_x = x_wd - update

        return new_x

    def minimize(
        self, fun, x0, args=(), jac=None, bounds=None, callback=None, n_jobs=-1
    ):
        """Minimization with early stopping and learning rate scheduling."""
        x = np.asarray(x0).copy()
        if bounds is not None:
            x = np.clip(x, *zip(*bounds))

        # Setup tracking variables
        best_x = x.copy()
        best_fun = float("inf")
        fun_history = []
        no_improve_count = 0
        lr_reduce_count = 0

        # Reset optimizer state
        self.m = None
        self.v = None
        self.t = 0

        # Setup gradient function
        if jac is None:
            grad_fn = lambda x_new: fast_gradient(x_new, *args, n_jobs=n_jobs)
        else:
            grad_fn = lambda x_new: jac(x_new, *args)

        # Initial evaluation
        f = fun(x, *args)
        g = grad_fn(x)
        fun_history.append(f)

        if f < best_fun:
            best_fun = f
            best_x = x.copy()

        if self.verbose:
            print(f"Initial loss: {f:.6f}")

        # Main optimization loop
        for i in range(self.max_iter):
            # Step with current parameters
            x_new = self._step(x, g, f)

            # Apply bounds
            if bounds is not None:
                x_new = np.clip(x_new, *zip(*bounds))

            # Evaluate at new point
            f_new = fun(x_new, *args)
            g_new = grad_fn(x_new)
            fun_history.append(f_new)

            # Update best solution
            if f_new < best_fun:
                improve_ratio = (best_fun - f_new) / (best_fun + self.eps)
                best_fun = f_new
                best_x = x_new.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

                # Learning rate scheduling
                if no_improve_count % self.lr_reduce_patience == 0:
                    self.lr *= self.lr_reduce_factor
                    lr_reduce_count += 1
                    if self.verbose:
                        print(f"Reducing learning rate to {self.lr:.8f}")

                    # Restart optimizer state for better convergence
                    if lr_reduce_count % 3 == 0:
                        self.m = None
                        self.v = None
                        self.t = 0

            # Early stopping
            if no_improve_count >= self.patience:
                if self.verbose:
                    print(
                        f"Early stopping after {i+1} iterations (no improvement for {self.patience} steps)"
                    )
                break

            # Callback
            if callback is not None:
                callback(x_new)

            # Convergence checks
            x_diff = np.linalg.norm(x_new - x)
            f_diff = abs(f_new - f)
            g_norm = np.linalg.norm(g_new)

            if self.verbose and (i % 10 == 0 or i == self.max_iter - 1):
                print(
                    f"Iter {i}: f={f_new:.6f}, |g|={g_norm:.6f}, |x_diff|={x_diff:.6f}, lr={self.lr:.8f}"
                )

            # Prepare for next iteration
            x = x_new
            f = f_new
            g = g_new

            # Strict convergence check
            if x_diff < self.tol and f_diff < self.tol and g_norm < self.tol:
                if self.verbose:
                    print(f"Converged after {i+1} iterations.")
                break

        # Return result
        result = OptimizeResult(
            x=best_x,
            fun=best_fun,
            jac=g,
            nit=i + 1,
            nfev=i + 1,
            success=(i < self.max_iter - 1) or (g_norm < self.tol),
            message=(
                "Optimization terminated successfully."
                if ((i < self.max_iter - 1) or (g_norm < self.tol))
                else "Maximum iterations reached."
            ),
            history=fun_history,
        )

        return result


def optimized_adamw_minimize(
    fun,
    x0,
    args=(),
    jac=None,
    bounds=None,
    callback=None,
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    max_iter=1000,
    tol=1e-5,
    patience=10,
    lr_reduce_factor=0.5,
    lr_reduce_patience=5,
    verbose=False,
    n_jobs=-1,
    **kwargs,
):
    """Optimized AdamW minimizer with better performance."""
    optimizer = OptimizedAdamWMinimizer(
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        max_iter=max_iter,
        tol=tol,
        patience=patience,
        lr_reduce_factor=lr_reduce_factor,
        lr_reduce_patience=lr_reduce_patience,
        verbose=verbose,
    )

    return optimizer.minimize(fun, x0, args, jac, bounds, callback, n_jobs=n_jobs)


def minimize(
    fun, x0, args=(), method="adamw", jac=None, bounds=None, callback=None, options=None
):
    """Enhanced minimize wrapper with optimized implementation."""
    if method.lower() == "adamw":
        # Default options
        default_options = {
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.01,
            "max_iter": 1000,
            "tol": 1e-5,
            "patience": 10,
            "lr_reduce_factor": 0.5,
            "lr_reduce_patience": 5,
            "verbose": False,
            "n_jobs": -1,  # Use all cores by default
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
