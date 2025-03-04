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
    if len(cats_to_update) > 1000:  # Only use parallelization for enough work
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
        patience=100,  # Added patience for early stopping
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

class HighPerformanceAdamWMinimizer:
    """
    High-performance AdamW implementation with advanced techniques for faster convergence.
    """

    def __init__(
        self,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        max_iter=1000,
        tol=1e-5,
        patience=100,
        lr_reduce_factor=0.5,
        lr_reduce_patience=5,
        verbose=False,
        gradient_batch_size=None,
        use_line_search=True,
        adaptive_params=True,
        coordinate_descent=False,
        coordinate_groups=None,
    ):
        self.lr = lr
        self.initial_lr = lr
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
        self.use_line_search = use_line_search
        self.adaptive_params = adaptive_params
        self.coordinate_descent = coordinate_descent
        self.coordinate_groups = coordinate_groups

        # States
        self.m = None
        self.v = None
        self.t = 0
        self.param_specific_lr = None
        self.param_types = None

    def _reset_state(self):
        """Reset optimizer state for better convergence after LR changes."""
        self.m = None
        self.v = None
        self.t = 0

    def _initialize_param_config(self, x, args=None):
        """Initialize parameter-specific configurations for better performance."""
        n_params = len(x)
        
        # Set up parameter-specific learning rates
        self.param_specific_lr = np.ones(n_params) * self.lr
        
        # Try to determine parameter types (scales vs smears)
        if args and len(args) > 1 and hasattr(args[1], "__iter__"):
            zcats = args[1]
            if zcats and hasattr(zcats[0], "lead_index"):
                # Identify parameter types based on ZCATS
                self.param_types = np.full(n_params, "other", dtype=object)
                
                # Identify scale parameters
                scale_indices = set()
                for cat in zcats:
                    if hasattr(cat, "lead_index") and cat.lead_index >= 0:
                        scale_indices.add(cat.lead_index)
                    if hasattr(cat, "sublead_index") and cat.sublead_index >= 0:
                        scale_indices.add(cat.sublead_index)
                
                # Identify smearing parameters
                smear_indices = set()
                for cat in zcats:
                    if hasattr(cat, "lead_smear_index") and cat.lead_smear_index >= 0:
                        smear_indices.add(cat.lead_smear_index)
                    if hasattr(cat, "sublead_smear_index") and cat.sublead_smear_index >= 0:
                        smear_indices.add(cat.sublead_smear_index)
                
                # Apply parameter-specific settings
                for idx in scale_indices:
                    self.param_types[idx] = "scale"
                    # Scales often need smaller learning rates
                    self.param_specific_lr[idx] = self.lr * 0.75
                
                for idx in smear_indices:
                    self.param_types[idx] = "smear"
                    # Smears often need larger learning rates
                    self.param_specific_lr[idx] = self.lr * 1.5
        
        # If coordinate descent is enabled, create parameter groups
        if self.coordinate_descent and self.coordinate_groups is None:
            # Auto-create coordinate groups based on parameter types or simple chunking
            if self.param_types is not None:
                # Group by parameter type
                self.coordinate_groups = {
                    "scales": np.where(self.param_types == "scale")[0],
                    "smears": np.where(self.param_types == "smear")[0],
                    "others": np.where(self.param_types == "other")[0]
                }
            else:
                # Simple chunking strategy
                group_size = min(50, max(10, n_params // 10))
                self.coordinate_groups = {}
                for i in range(0, n_params, group_size):
                    group_name = f"group_{i // group_size}"
                    self.coordinate_groups[group_name] = list(range(i, min(i + group_size, n_params)))

    def _step(self, x, grad, func_val, iteration):
        """Optimized step function with parameter-specific adaptive learning rates."""
        # Initialize moments if first step
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)

        # Initialize parameter-specific settings if needed
        if self.param_specific_lr is None:
            self.param_specific_lr = np.ones_like(x) * self.lr

        self.t += 1

        # Update moments with better numerical stability
        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * grad
        self.v = self.betas[1] * self.v + (1 - self.betas[1]) * (np.square(grad) + self.eps)

        # Bias correction
        m_hat = self.m / (1 - self.betas[0] ** self.t)
        v_hat = self.v / (1 - self.betas[1] ** self.t)

        # Adaptive gradient clipping based on norm
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1.0:
            m_hat = m_hat * (1.0 / grad_norm)

        # Apply weight decay directly to parameters with parameter-specific learning rates
        x_wd = x * (1 - self.param_specific_lr * self.weight_decay)

        # Compute update with parameter-specific learning rates
        update = self.param_specific_lr * m_hat / (np.sqrt(v_hat) + self.eps)

        # Apply warmup during early iterations
        if iteration < 10:
            warmup_factor = 0.5 + 0.5 * (iteration / 10.0)
            update = update * warmup_factor

        # Apply update
        new_x = x_wd - update

        return new_x

    def _perform_line_search(self, fun, x, new_x, f_x, grad, args):
        """Simple backtracking line search for better step sizes."""
        if not self.use_line_search:
            return new_x

        # Calculate the direction we're moving in
        direction = new_x - x
        
        # Check if we're moving in a descent direction
        directional_derivative = np.dot(grad, direction)
        if directional_derivative >= 0:
            # Not moving in a descent direction, stick with standard step
            return new_x

        # Armijo line search parameters
        alpha = 1.0
        c1 = 1e-4  # Sufficient decrease constant
        max_backtracks = 5

        # Initial candidate point is the standard step
        x_candidate = new_x
        f_candidate = fun(x_candidate, *args)

        # Try backtracking steps
        for i in range(max_backtracks):
            # Check Armijo condition (sufficient decrease)
            if f_candidate <= f_x + c1 * alpha * directional_derivative:
                # Condition satisfied, accept this step
                return x_candidate

            # Backtrack: reduce alpha
            alpha *= 0.5
            
            # Compute new candidate point
            x_candidate = x + alpha * direction
            f_candidate = fun(x_candidate, *args)

        # If we get here, no suitable step found, return the original proposal
        return new_x

    def _coordinate_step(self, fun, x, grad_fn, group_indices, args):
        """Perform a step focusing only on a subset of coordinates."""
        n_params = len(x)
        
        # Create a mask for the active parameters
        active_mask = np.zeros(n_params, dtype=bool)
        active_mask[group_indices] = True
        
        # Compute gradient just for these parameters
        full_grad = grad_fn(x)
        group_grad = np.zeros_like(full_grad)
        group_grad[active_mask] = full_grad[active_mask]
        
        # Take a step focusing only on these parameters
        x_new = self._step(x, group_grad, None, 0)
        
        # Apply the update only to the selected parameters
        result = x.copy()
        result[active_mask] = x_new[active_mask]
        
        return result, full_grad

    def minimize(
        self, fun, x0, args=(), jac=None, bounds=None, callback=None, n_jobs=-1
    ):
        """Minimization with enhanced convergence strategies."""
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
        self._reset_state()
        
        # Initialize parameter-specific configurations
        self._initialize_param_config(x, args)

        # Setup gradient function
        if jac is None:
            # Use enhanced gradient calculation
            def grad_fn(x_new):
                return fast_gradient(x_new, *args, n_jobs=n_jobs, h=1e-6)
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

        # Prepare for coordinate descent if enabled
        if self.coordinate_descent and self.coordinate_groups:
            coord_groups = list(self.coordinate_groups.values())
        else:
            coord_groups = None

        # Main optimization loop
        for i in range(self.max_iter):
            # Choose between full-space or coordinate descent step
            if coord_groups and i > 0 and i % len(coord_groups) != 0:
                # Take a coordinate descent step
                group_idx = (i % len(coord_groups))
                x_new, g_new = self._coordinate_step(fun, x, grad_fn, coord_groups[group_idx], args)
                f_new = fun(x_new, *args)
            else:
                # Standard full-space step
                x_new = self._step(x, g, f, i)
                
                # Apply line search if enabled
                if self.use_line_search and i > 3:  # Skip early iterations
                    x_new = self._perform_line_search(fun, x, x_new, f, g, args)

                # Apply bounds
                if bounds is not None:
                    x_new = np.clip(x_new, *zip(*bounds))

                # Evaluate at new point
                f_new = fun(x_new, *args)
                g_new = grad_fn(x_new)

            # Record history
            fun_history.append(f_new)

            # Update best solution
            if f_new < best_fun:
                improvement = best_fun - f_new
                relative_improvement = improvement / (abs(best_fun) + self.eps)
                
                # Update best known solution
                best_fun = f_new
                best_x = x_new.copy()
                no_improve_count = 0
                
                # Adapt learning rates based on parameter importance
                if self.adaptive_params and relative_improvement > 0.01 and i > 10:
                    # Identify parameters that contributed significantly to improvement
                    param_changes = np.abs(x_new - x)
                    significant_params = param_changes > np.percentile(param_changes, 75)
                    
                    # Boost learning rates for significant parameters
                    boost_factor = min(2.0, 1.0 + relative_improvement * 5)
                    self.param_specific_lr[significant_params] *= boost_factor
                    
                    # Cap learning rates to reasonable bounds
                    max_lr = self.initial_lr * 5
                    self.param_specific_lr = np.clip(self.param_specific_lr, 
                                                    self.initial_lr * 0.1, 
                                                    max_lr)
            else:
                no_improve_count += 1

                # Learning rate scheduling
                if no_improve_count % self.lr_reduce_patience == 0:
                    # Reduce learning rates
                    self.param_specific_lr *= self.lr_reduce_factor
                    lr_reduce_count += 1
                    
                    if self.verbose:
                        avg_lr = np.mean(self.param_specific_lr)
                        print(f"Reducing learning rates at iter {i}, average LR: {avg_lr:.8f}")

                    # Periodically reset optimizer state for better convergence
                    if lr_reduce_count % 3 == 0:
                        self._reset_state()

            # Early stopping
            if no_improve_count >= self.patience:
                if self.verbose:
                    print(f"Early stopping after {i+1} iterations (no improvement for {self.patience} steps)")
                break

            # Callback
            if callback is not None:
                callback(x_new)

            # Convergence checks
            x_diff = np.linalg.norm(x_new - x)
            f_diff = abs(f_new - f)
            g_norm = np.linalg.norm(g_new)

            if self.verbose and (i % 10 == 0 or i == self.max_iter - 1):
                avg_lr = np.mean(self.param_specific_lr)
                print(f"Iter {i}: f={f_new:.6f}, |g|={g_norm:.6f}, |x_diff|={x_diff:.6f}, avg_lr={avg_lr:.8f}")

            # Prepare for next iteration
            x = x_new
            f = f_new
            g = g_new

            # Strict convergence check
            if x_diff < self.tol and f_diff < self.tol and g_norm < self.tol:
                if self.verbose:
                    print(f"Converged after {i+1} iterations.")
                break

        # Final clip to ensure bounds are respected
        if bounds is not None:
            best_x = np.clip(best_x, *zip(*bounds))

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


def high_performance_adamw_minimize(
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
    use_line_search=True,
    adaptive_params=True,
    coordinate_descent=False,
    coordinate_groups=None,
    **kwargs,
):
    """High-performance AdamW minimizer with advanced optimization strategies."""
    optimizer = HighPerformanceAdamWMinimizer(
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
        use_line_search=use_line_search,
        adaptive_params=adaptive_params,
        coordinate_descent=coordinate_descent,
        coordinate_groups=coordinate_groups
    )

    return optimizer.minimize(fun, x0, args, jac, bounds, callback, n_jobs=n_jobs)


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
            "n_jobs": 1,  # Use no cores by default
        }

        # Update with user options
        if options is not None:
            default_options.update(options)

        return optimized_adamw_minimize(
            fun, x0, args, jac, bounds, callback, **default_options
        )
    elif method.lower() == "high_adamw" or method.lower() == "high_performance_adamw":
        # Default options for high performance variant
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
            "n_jobs": 1,
            "use_line_search": True,
            "adaptive_params": True,
            "coordinate_descent": True,
            "coordinate_groups": None,
        }

        # Update with user options
        if options is not None:
            default_options.update(options)

        return high_performance_adamw_minimize(
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


def optimized_scan_nll(x, **options):
    """
    Performs the NLL scan to initialize the variables in a parallelized and optimized way.

    Args:
        x (iterable): iterable of floats, representing the scales and smearings chosen by the minimizer
        **options: keyword arguments containing configuration parameters

    Returns:
        guess (numpy.ndarray): Optimized initial guess for scales and smearings
    """
    __ZCATS__ = options["zcats"]
    __GUESS__ = options["__GUESS__"]
    guess = np.array(x).copy()  # Ensure we have a numpy array copy
    n_jobs = options.get("n_jobs", -1)  # Default to all cores

    # Use all cores unless specified otherwise
    if n_jobs == -1:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free

    # Get the loss function wrapper
    loss_function, reset_loss_initial_guess = target_function_wrapper(
        guess, __ZCATS__, **options
    )

    # -------------------------------------------------
    # Scale optimization section
    # -------------------------------------------------
    if not options["_kFixScales"]:
        print("[INFO][python/helper_minimizer/scan_ll] scanning scales")

        # Collect weights and indices for diagonal categories
        diagonal_cats = [
            (cat.weight, cat.lead_index)
            for cat in __ZCATS__
            if cat.valid and cat.lead_index == cat.sublead_index
        ]

        # Sort by weight (we want to optimize high-weight categories first)
        diagonal_cats.sort(key=lambda x: x[0], reverse=True)

        # Track which scales we've already scanned
        scanned_scales = set()

        # Create a parameter grid for the scan
        scan_params = {
            "min": options["scan_min"],
            "max": options["scan_max"],
            "step": options["scan_step"],
        }
        scale_grid = np.arange(
            scan_params["min"], scan_params["max"], scan_params["step"]
        )

        for weight, scale_index in diagonal_cats:
            if scale_index in scanned_scales:
                continue

            # Add to scanned list so we don't repeat
            scanned_scales.add(scale_index)

            # Function to evaluate a specific scale value
            def evaluate_scale(scale_val):
                test_guess = guess.copy()
                test_guess[scale_index] = scale_val
                nll = loss_function(
                    test_guess,
                    __GUESS__,
                    __ZCATS__,
                    options["num_scales"],
                    options["num_smears"],
                )
                return scale_val, nll

            # Parallel evaluation of all scale values
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_scale)(val) for val in scale_grid
            )

            # Convert to arrays for efficient processing
            vals, nlls = zip(*results)
            vals = np.array(vals)
            nlls = np.array(nlls)

            # Filter invalid values (negative or infinite NLL)
            mask = (nlls > 0) & (nlls < 1e10)
            if np.any(mask):
                filtered_vals = vals[mask]
                filtered_nlls = nlls[mask]

                # Find the best scale value
                best_idx = np.argmin(filtered_nlls)
                best_val = filtered_vals[best_idx]

                # Update our guess
                guess[scale_index] = best_val
                print(
                    f"[INFO][python/nll] best guess for scale {scale_index} is {best_val:.6f}"
                )

    # -------------------------------------------------
    # Smearing optimization section
    # -------------------------------------------------
    if options["num_smears"] > 0:
        print("[INFO][python/helper_minimizer/scan_nll] scanning smearings:")

        # Similar process for smearings, but different scan range
        smear_diagonal_cats = [
            (cat.weight, cat.lead_smear_index)
            for cat in __ZCATS__
            if cat.valid and cat.lead_smear_index == cat.sublead_smear_index
        ]

        # Sort by weight
        smear_diagonal_cats.sort(key=lambda x: x[0], reverse=True)

        # Track which smearings we've already scanned
        scanned_smears = set()

        # Create a parameter grid for smearing scan - different range than scales
        low = options.get("smear_scan_min", 0.00025)
        high = options.get("smear_scan_max", 0.025)
        step = options.get("smear_scan_step", 0.00025)
        smear_grid = np.arange(low, high, step)

        # Process smearing parameters in parallel batches
        # This allows us to update the guess between batches for better convergence
        batch_size = 5  # Process 5 smearing parameters at a time

        for i in range(0, len(smear_diagonal_cats), batch_size):
            batch = smear_diagonal_cats[i : i + batch_size]

            # Process each smearing parameter in this batch
            for weight, smear_index in batch:
                if smear_index in scanned_smears:
                    continue

                # Add to scanned list
                scanned_smears.add(smear_index)

                # Function to evaluate a specific smearing value
                def evaluate_smear(smear_val):
                    test_guess = guess.copy()
                    test_guess[smear_index] = smear_val
                    nll = loss_function(
                        test_guess,
                        __GUESS__,
                        __ZCATS__,
                        options["num_scales"],
                        options["num_smears"],
                    )
                    return smear_val, nll

                # Parallel evaluation
                results = Parallel(n_jobs=n_jobs)(
                    delayed(evaluate_smear)(val) for val in smear_grid
                )

                # Process results
                vals, nlls = zip(*results)
                vals = np.array(vals)
                nlls = np.array(nlls)

                # Filter invalid values
                mask = (nlls > 0) & (nlls < 1e10)
                if np.any(mask):
                    filtered_vals = vals[mask]
                    filtered_nlls = nlls[mask]

                    # Find the best smearing value
                    best_idx = np.argmin(filtered_nlls)
                    best_val = filtered_vals[best_idx]

                    # Update our guess
                    guess[smear_index] = best_val
                    print(
                        f"[INFO][python/nll] best guess for smearing {smear_index} is {best_val:.6f}"
                    )

            # After each batch, reset the loss function's cache to use our improved guesses
            reset_loss_initial_guess(guess)

    print("[INFO][python/nll] scan complete")
    return guess


def adaptive_scan_nll(x, **options):
    """
    Advanced version of scan_nll with adaptive grid refinement for more accurate results.
    This uses a coarse-to-fine approach to efficiently find optimal parameter values.

    Args:
        x (iterable): iterable of floats, representing the scales and smearings
        **options: keyword arguments

    Returns:
        guess (numpy.ndarray): Optimized initial guess for scales and smearings
    """
    __ZCATS__ = options["zcats"]
    __GUESS__ = options["__GUESS__"]
    guess = np.array(x).copy()
    n_jobs = options.get("n_jobs", 1)  # default to 1 core to avoid memory issues

    # Create the loss function wrapper
    loss_function, reset_loss_initial_guess = target_function_wrapper(
        guess, __ZCATS__, **options
    )

    # -------------------------------------------------
    # Adaptive smearing optimization
    # -------------------------------------------------
    if options["num_smears"] > 0:
        print("[INFO][python/helper_minimizer/scan_nll] adaptively scanning smearings")

        # Similar approach for smearing parameters
        smear_diagonal_cats = [
            (cat.weight, cat.lead_smear_index)
            for cat in __ZCATS__
            if cat.valid and cat.lead_smear_index == cat.sublead_smear_index
        ]
        smear_diagonal_cats.sort(key=lambda x: x[0], reverse=True)

        scanned_smears = set()

        # Batch processing for smearings
        batch_size = 3  # Process fewer at a time for better convergence

        for i in range(0, len(smear_diagonal_cats), batch_size):
            batch = smear_diagonal_cats[i : i + batch_size]

            for weight, smear_index in batch:
                if smear_index in scanned_smears:
                    continue

                scanned_smears.add(smear_index)

                # Multi-stage adaptive search
                min_val = options.get("smear_scan_min", 0.00025)
                max_val = options.get("smear_scan_max", 0.025)

                # Stage 1: Logarithmic coarse scan (better for smearing parameters)
                n_points = 10
                # Use logarithmic spacing for smearing (often works better)
                if min_val <= 0:
                    min_val = 1e-6  # Avoid log(0)
                coarse_grid = np.logspace(
                    np.log10(min_val), np.log10(max_val), n_points
                )

                def evaluate_smear(val):
                    test_guess = guess.copy()
                    test_guess[smear_index] = val
                    nll = loss_function(
                        test_guess,
                        __GUESS__,
                        __ZCATS__,
                        options["num_scales"],
                        options["num_smears"],
                    )
                    return val, nll

                # Evaluate coarse grid
                results = Parallel(n_jobs=n_jobs)(
                    delayed(evaluate_smear)(val) for val in coarse_grid
                )

                vals, nlls = zip(*results)
                vals = np.array(vals)
                nlls = np.array(nlls)

                # Filter invalid values
                mask = (nlls > 0) & (nlls < 1e10)
                if not np.any(mask):
                    continue

                filtered_vals = vals[mask]
                filtered_nlls = nlls[mask]

                # Find best region
                best_idx = np.argmin(filtered_nlls)
                best_val = filtered_vals[best_idx]

                # Stage 2: Refine around best value
                # Use linear spacing for the refined grid
                window_factor = 2.0  # Wider window for smearing
                refined_min = best_val / window_factor
                refined_max = best_val * window_factor

                # Create refined grid (linear spacing for final precision)
                n_points = 15
                refined_grid = np.linspace(refined_min, refined_max, n_points)

                # Evaluate refined grid
                refined_results = Parallel(n_jobs=n_jobs)(
                    delayed(evaluate_smear)(val) for val in refined_grid
                )

                refined_vals, refined_nlls = zip(*refined_results)
                refined_vals = np.array(refined_vals)
                refined_nlls = np.array(refined_nlls)

                # Filter and find best value
                mask = (refined_nlls > 0) & (refined_nlls < 1e10)
                if np.any(mask):
                    filtered_vals = refined_vals[mask]
                    filtered_nlls = refined_nlls[mask]
                    best_idx = np.argmin(filtered_nlls)
                    best_val = filtered_vals[best_idx]

                    # Update guess
                    guess[smear_index] = best_val
                    print(
                        f"[INFO][python/nll] best guess for smearing {smear_index} is {best_val:.6f}"
                    )

            # Reset cache after each batch
            reset_loss_initial_guess(guess)

    # -------------------------------------------------
    # Adaptive scale optimization
    # -------------------------------------------------
    if not options["_kFixScales"]:
        print("[INFO][python/helper_minimizer/scan_ll] adaptively scanning scales")

        # Get diagonal categories for scales
        diagonal_cats = [
            (cat.weight, cat.lead_index)
            for cat in __ZCATS__
            if cat.valid and cat.lead_index == cat.sublead_index
        ]
        diagonal_cats.sort(key=lambda x: x[0], reverse=True)

        # Track scanned scales
        scanned_scales = set()

        for weight, scale_index in diagonal_cats:
            if scale_index in scanned_scales:
                continue

            scanned_scales.add(scale_index)

            # Multi-stage adaptive grid refinement
            # Start with a coarse grid
            min_val = options.get("scan_min", 0.8)
            max_val = options.get("scan_max", 1.2)

            # Stage 1: Coarse scan
            n_points = 11  # Use fewer points for initial scan
            coarse_grid = np.linspace(min_val, max_val, n_points)

            def evaluate_point(val):
                test_guess = guess.copy()
                test_guess[scale_index] = val
                nll = loss_function(
                    test_guess,
                    __GUESS__,
                    __ZCATS__,
                    options["num_scales"],
                    options["num_smears"],
                )
                return val, nll

            # Evaluate coarse grid in parallel
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_point)(val) for val in coarse_grid
            )

            # Find promising region
            vals, nlls = zip(*results)
            vals = np.array(vals)
            nlls = np.array(nlls)

            # Filter invalid values
            mask = (nlls > 0) & (nlls < 1e10)
            if not np.any(mask):
                continue  # Skip if all values are invalid

            filtered_vals = vals[mask]
            filtered_nlls = nlls[mask]

            # Find best value and narrow search region
            best_idx = np.argmin(filtered_nlls)
            best_val = filtered_vals[best_idx]

            # Stage 2: Refine around best value
            # Define a narrower region around the best value
            window = (max_val - min_val) / 10  # 10% of original range
            refined_min = max(min_val, best_val - window)
            refined_max = min(max_val, best_val + window)

            # Create a finer grid
            n_points = 15  # More points for refined scan
            refined_grid = np.linspace(refined_min, refined_max, n_points)

            # Evaluate refined grid
            refined_results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_point)(val) for val in refined_grid
            )

            refined_vals, refined_nlls = zip(*refined_results)
            refined_vals = np.array(refined_vals)
            refined_nlls = np.array(refined_nlls)

            # Filter and find best value
            mask = (refined_nlls > 0) & (refined_nlls < 1e10)
            if np.any(mask):
                filtered_vals = refined_vals[mask]
                filtered_nlls = refined_nlls[mask]
                best_idx = np.argmin(filtered_nlls)
                best_val = filtered_vals[best_idx]

                # Update guess
                guess[scale_index] = best_val
                print(
                    f"[INFO][python/nll] best guess for scale {scale_index} is {best_val:.6f}"
                )

    # -------------------------------------------------
    # Adaptive smearing optimization
    # -------------------------------------------------
    if options["num_smears"] > 0:
        print("[INFO][python/helper_minimizer/scan_nll] adaptively scanning smearings")

        # Similar approach for smearing parameters
        smear_diagonal_cats = [
            (cat.weight, cat.lead_smear_index)
            for cat in __ZCATS__
            if cat.valid and cat.lead_smear_index == cat.sublead_smear_index
        ]
        smear_diagonal_cats.sort(key=lambda x: x[0], reverse=True)

        scanned_smears = set()

        # Batch processing for smearings
        batch_size = 3  # Process fewer at a time for better convergence

        for i in range(0, len(smear_diagonal_cats), batch_size):
            batch = smear_diagonal_cats[i : i + batch_size]

            for weight, smear_index in batch:
                if smear_index in scanned_smears:
                    continue

                scanned_smears.add(smear_index)

                # Multi-stage adaptive search
                min_val = options.get("smear_scan_min", 0.00025)
                max_val = options.get("smear_scan_max", 0.025)

                # Stage 1: Logarithmic coarse scan (better for smearing parameters)
                n_points = 10
                # Use logarithmic spacing for smearing (often works better)
                if min_val <= 0:
                    min_val = 1e-6  # Avoid log(0)
                coarse_grid = np.logspace(
                    np.log10(min_val), np.log10(max_val), n_points
                )

                def evaluate_smear(val):
                    test_guess = guess.copy()
                    test_guess[smear_index] = val
                    nll = loss_function(
                        test_guess,
                        __GUESS__,
                        __ZCATS__,
                        options["num_scales"],
                        options["num_smears"],
                    )
                    return val, nll

                # Evaluate coarse grid
                results = Parallel(n_jobs=n_jobs)(
                    delayed(evaluate_smear)(val) for val in coarse_grid
                )

                vals, nlls = zip(*results)
                vals = np.array(vals)
                nlls = np.array(nlls)

                # Filter invalid values
                mask = (nlls > 0) & (nlls < 1e10)
                if not np.any(mask):
                    continue

                filtered_vals = vals[mask]
                filtered_nlls = nlls[mask]

                # Find best region
                best_idx = np.argmin(filtered_nlls)
                best_val = filtered_vals[best_idx]

                # Stage 2: Refine around best value
                # Use linear spacing for the refined grid
                window_factor = 2.0  # Wider window for smearing
                refined_min = best_val / window_factor
                refined_max = best_val * window_factor

                # Create refined grid (linear spacing for final precision)
                n_points = 15
                refined_grid = np.linspace(refined_min, refined_max, n_points)

                # Evaluate refined grid
                refined_results = Parallel(n_jobs=n_jobs)(
                    delayed(evaluate_smear)(val) for val in refined_grid
                )

                refined_vals, refined_nlls = zip(*refined_results)
                refined_vals = np.array(refined_vals)
                refined_nlls = np.array(refined_nlls)

                # Filter and find best value
                mask = (refined_nlls > 0) & (refined_nlls < 1e10)
                if np.any(mask):
                    filtered_vals = refined_vals[mask]
                    filtered_nlls = refined_nlls[mask]
                    best_idx = np.argmin(filtered_nlls)
                    best_val = filtered_vals[best_idx]

                    # Update guess
                    guess[smear_index] = best_val
                    print(
                        f"[INFO][python/nll] best guess for smearing {smear_index} is {best_val:.6f}"
                    )

            # Reset cache after each batch
            reset_loss_initial_guess(guess)

    print("[INFO][python/nll] adaptive scan complete")
    return guess

