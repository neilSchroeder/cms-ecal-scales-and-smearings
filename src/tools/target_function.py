import multiprocessing
from functools import lru_cache

import numpy as np
import scipy.optimize
from joblib import Parallel, delayed
from scipy.optimize import OptimizeResult


import numpy as np
import numba
from numba import njit, prange
import multiprocessing
from joblib import Parallel, delayed

# Import the gradient optimization utilities
from src.tools.gradients import fast_gradient_optimized

def enhanced_target_function_wrapper(initial_guess, ZCATS, num_scales, num_smears, **kwargs):
    """
    Enhanced wrapper for the target function with improved caching and gradient computation.
    
    Args:
        initial_guess: Initial parameter vector
        ZCATS: Category objects
        num_scales: Number of scales
        num_smears: Number of smears
        **kwargs: Additional keyword arguments
        
    Returns:
        A tuple containing:
        - The wrapped target function
        - A reset function
        - A function for gradient computation
    """
    previous_guess = [initial_guess.copy()]
    
    # Cache for the wrapped function
    function_cache = {}
    
    def wrapped_target_function(x, *args, **options):
        """Target function with caching and optimization"""
        # Try to use cache
        key = tuple(float(v) for v in x)
        if key in function_cache:
            return function_cache[key]
        
        # Extract whether verbose flag is set
        verbose = options.get('verbose', False)
        
        # Use the actual target function
        ret = target_function(
            x, previous_guess[0], ZCATS, num_scales, num_smears, 
            verbose=verbose, **options
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
    def compute_gradient(x, *args, h=1e-6, use_spsa=True, spsa_iterations=5, **options):
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
        return fast_gradient_optimized(
            wrapped_target_function, x, *args,
            h=h, 
            use_spsa=use_spsa, 
            spsa_iterations=spsa_iterations,
            **options
        )
    
    return wrapped_target_function, reset, compute_gradient


def target_function(x, previous, ZCATS, num_scales, num_smears, verbose=False, **options):
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

    # Cache the result
    if hasattr(x, "_cached_result"):
        x._cached_result = final_value

    return final_value


class OptimizedAdamWMinimizer:
    """
    Optimized AdamW implementation with improved computational efficiency
    and better convergence.
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

        # States - pre-allocate arrays
        self.m = None
        self.v = None
        self.t = 0
        
        # Cache for performance
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.one_minus_beta1 = 1 - betas[0]
        self.one_minus_beta2 = 1 - betas[1]
        self.weight_decay_factor = 1 - lr * weight_decay

    @numba.njit
    def _step_core(x, m, v, t, grad, beta1, beta2, one_minus_beta1, one_minus_beta2, 
                   weight_decay_factor, lr, eps):
        """Core step computation optimized with Numba"""
        # Update moments with better numerical stability
        m = beta1 * m + one_minus_beta1 * grad
        v = beta2 * v + one_minus_beta2 * (np.square(grad) + eps)

        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # AdamW update (combined operations)
        x_new = weight_decay_factor * x - lr * m_hat / (np.sqrt(v_hat) + eps)
        
        return x_new, m, v
    
    def _step(self, x, grad, func_val):
        """Optimized step function with cached computations"""
        # Initialize moments if first step
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)

        self.t += 1
        
        # Use numba-optimized core function
        x_new, self.m, self.v = self._step_core(
            x, self.m, self.v, self.t, grad, 
            self.beta1, self.beta2, 
            self.one_minus_beta1, self.one_minus_beta2,
            self.weight_decay_factor, self.lr, self.eps
        )
        
        return x_new

    def minimize(
        self, fun, x0, args=(), jac=None, bounds=None, callback=None, n_jobs=-1
    ):
        """Minimization with early stopping and learning rate scheduling."""
        # Convert to contiguous array for better memory access patterns
        x = np.ascontiguousarray(x0, dtype=np.float64)
        
        # Pre-compute bound arrays if needed
        if bounds is not None:
            lb, ub = np.asarray(list(zip(*bounds)))
        else:
            lb, ub = None, None

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
        
        # Update cached values
        self.weight_decay_factor = 1 - self.lr * self.weight_decay

        # Setup gradient function
        if jac is None:
            # Assuming fast_gradient is defined elsewhere
            grad_fn = lambda x_new: fast_gradient(x_new, *args, n_jobs=1)  # Force single thread
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

        # Pre-allocate arrays for intermediate results
        x_new = np.empty_like(x)
        g_new = np.empty_like(g)

        # Main optimization loop - avoid function calls in tight loop
        for i in range(self.max_iter):
            # Step with current parameters
            x_new = self._step(x, g, f)

            # Apply bounds
            if bounds is not None:
                np.clip(x_new, lb, ub, out=x_new)

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
                    # Update the cached weight decay factor
                    self.weight_decay_factor = 1 - self.lr * self.weight_decay
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
            x = x_new.copy()  # Copy to avoid aliasing issues
            f = f_new
            g = g_new.copy()  # Copy to avoid aliasing issues

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
    Uses a three-stage approach: smearings → scales → smearings again for better convergence.
    
    Args:
        x (iterable): iterable of floats, representing the scales and smearings
        **options: keyword arguments

    Returns:
        guess (numpy.ndarray): Optimized initial guess for scales and smearings
    """
    from joblib import parallel_backend, Parallel, delayed
    import multiprocessing
    import numpy as np
    
    __ZCATS__ = options["zcats"]
    __GUESS__ = options["__GUESS__"]
    guess = np.array(x).copy()
    
    # Configure parallel processing
    n_jobs = options.get("n_jobs", 1)
    if n_jobs == -1:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
    
    # Create the loss function wrapper
    loss_function, reset_loss_initial_guess = target_function_wrapper(
        guess, __ZCATS__, **options
    )
    
    # Define a helper function for parameter optimization that works for both scales and smearings
    def optimize_parameters(param_list, param_type, batch_size):
        """
        Helper function to optimize a set of parameters (either scales or smearings)
        
        Args:
            param_list: List of (weight, index) tuples for parameters
            param_type: "scale" or "smear" to determine search strategy
            batch_size: Number of parameters to process in each batch
        """
        scanned_params = set()
        
        # Process parameters in batches
        for i in range(0, len(param_list), batch_size):
            batch = param_list[i:i+batch_size]
            batch_params = []
            
            # Collect unprocessed parameters in this batch
            for weight, param_index in batch:
                if param_index not in scanned_params:
                    scanned_params.add(param_index)
                    batch_params.append(param_index)
            
            if not batch_params:
                continue
            
            # Configure grid search based on parameter type
            param_configs = []
            for param_index in batch_params:
                if param_type == "smear":
                    min_val = max(1e-6, options.get("smear_scan_min", 0.00025))
                    max_val = options.get("smear_scan_max", 0.025)
                    # Use logarithmic grid for smearings
                    n_points = 10
                    coarse_grid = np.logspace(np.log10(min_val), np.log10(max_val), n_points)
                else:  # scale
                    min_val = options.get("scan_min", 0.9)
                    max_val = options.get("scan_max", 1.1)
                    # Use linear grid for scales
                    n_points = 11
                    coarse_grid = np.linspace(min_val, max_val, n_points)
                
                param_configs.append((param_index, coarse_grid))
            
            # Evaluate parameters in this batch using batched workers
            _process_parameter_batch(param_configs, guess, param_type)
            
            # Reset cache after each parameter batch
            reset_loss_initial_guess(guess)
    
    def _process_parameter_batch(param_configs, current_guess, param_type):
        """
        Process a batch of parameters with coarse-to-fine grid search
        """
        # Function to evaluate all parameter values in one batch - more efficient
        def evaluate_batch(configs):
            results = []
            for param_idx, param_val in configs:
                test_guess = current_guess.copy()
                test_guess[param_idx] = param_val
                nll = loss_function(
                    test_guess,
                    __GUESS__,
                    __ZCATS__,
                    options["num_scales"],
                    options["num_smears"],
                )
                results.append((param_idx, param_val, nll))
            return results
        
        # Prepare batches for efficiency - each worker gets multiple evaluations
        max_evals_per_worker = 5  # Balance parallelism and overhead
        all_evals = [(p_idx, val) for p_idx, grid in param_configs for val in grid]
        eval_batches = [all_evals[j:j+max_evals_per_worker] 
                        for j in range(0, len(all_evals), max_evals_per_worker)]
        
        # Execute parallel evaluation with optimized backend
        with parallel_backend('loky', n_jobs=n_jobs):
            batch_results = Parallel(
                verbose=0,
                batch_size='auto',
                pre_dispatch='2*n_jobs',
                max_nbytes='100M',
                mmap_mode='r',
                temp_folder='/tmp'
            )(delayed(evaluate_batch)(batch) for batch in eval_batches)
        
        # Organize results by parameter for efficient processing
        param_results = {}
        for batch_result in batch_results:
            for param_idx, param_val, nll in batch_result:
                if param_idx not in param_results:
                    param_results[param_idx] = []
                param_results[param_idx].append((param_val, nll))
        
        # Process each parameter's results and do refinement
        for param_index, results in param_results.items():
            # Filter invalid values
            vals, nlls = zip(*results)
            vals = np.array(vals)
            nlls = np.array(nlls)
            
            mask = (nlls > 0) & (nlls < 1e10)
            if not np.any(mask):
                continue
                
            filtered_vals = vals[mask]
            filtered_nlls = nlls[mask]
            
            # Find best region from coarse scan
            best_idx = np.argmin(filtered_nlls)
            best_val = filtered_vals[best_idx]
            
            # Define refined grid around best value - different strategy for scales vs smearings
            if param_type == "smear":
                window_factor = 2.0
                refined_min = best_val / window_factor
                refined_max = best_val * window_factor
                n_points = 15
                refined_grid = np.linspace(refined_min, refined_max, n_points)
            else:  # scale
                window = (options.get("scan_max", 1.2) - options.get("scan_min", 0.8)) / 10
                refined_min = max(options.get("scan_min", 0.8), best_val - window)
                refined_max = min(options.get("scan_max", 1.2), best_val + window)
                n_points = 15
                refined_grid = np.linspace(refined_min, refined_max, n_points)
            
            # Prepare refinement evaluation in batches
            refined_evals = [(param_index, val) for val in refined_grid]
            refined_batches = [refined_evals[j:j+max_evals_per_worker] 
                              for j in range(0, len(refined_evals), max_evals_per_worker)]
            
            # Execute parallel refinement evaluation
            with parallel_backend('loky', n_jobs=n_jobs):
                refined_results_batches = Parallel(
                    verbose=0,
                    batch_size='auto',
                    max_nbytes='100M',
                    mmap_mode='r',
                    temp_folder='/tmp'
                )(delayed(evaluate_batch)(batch) for batch in refined_batches)
            
            # Collect results for this parameter
            param_refined_results = []
            for batch_result in refined_results_batches:
                for p_idx, param_val, nll in batch_result:
                    if p_idx == param_index:
                        param_refined_results.append((param_val, nll))
            
            if param_refined_results:
                # Find best value from refined search
                refined_vals, refined_nlls = zip(*param_refined_results)
                refined_vals = np.array(refined_vals)
                refined_nlls = np.array(refined_nlls)
                
                mask = (refined_nlls > 0) & (refined_nlls < 1e10)
                if np.any(mask):
                    filtered_vals = refined_vals[mask]
                    filtered_nlls = refined_nlls[mask]
                    best_idx = np.argmin(filtered_nlls)
                    best_val = filtered_vals[best_idx]
                    
                    # Update guess with best value
                    current_guess[param_index] = best_val
                    type_str = "smearing" if param_type == "smear" else "scale"
                    print(f"[INFO][python/nll] best guess for {type_str} {param_index} is {best_val:.6f}")
    
    # Collect categories for parameter optimization
    if options["num_smears"] > 0:
        smear_diagonal_cats = [
            (cat.weight, cat.lead_smear_index)
            for cat in __ZCATS__
            if cat.valid and cat.lead_smear_index == cat.sublead_smear_index
        ]
        smear_diagonal_cats.sort(key=lambda x: x[0], reverse=True)
    
    if not options["_kFixScales"]:
        scale_diagonal_cats = [
            (cat.weight, cat.lead_index)
            for cat in __ZCATS__
            if cat.valid and cat.lead_index == cat.sublead_index
        ]
        scale_diagonal_cats.sort(key=lambda x: x[0], reverse=True)
    
    # -------------------------------------------------
    # STAGE 1: First smearing optimization
    # -------------------------------------------------
    if options["num_smears"] > 0:
        print("[INFO][python/helper_minimizer/scan_nll] stage 1: adaptively scanning smearings")
        optimize_parameters(smear_diagonal_cats, "smear", batch_size=5)
    
    # -------------------------------------------------
    # STAGE 2: Scale optimization
    # -------------------------------------------------
    if not options["_kFixScales"]:
        print("[INFO][python/helper_minimizer/scan_ll] stage 2: adaptively scanning scales")
        optimize_parameters(scale_diagonal_cats, "scale", batch_size=10)
    
    # -------------------------------------------------
    # STAGE 3: Second smearing optimization (refinement)
    # -------------------------------------------------
    if options["num_smears"] > 0:
        print("[INFO][python/helper_minimizer/scan_nll] stage 3: refining smearings")
        # Use smaller batch size for refinement stage to focus more on individual parameters
        optimize_parameters(smear_diagonal_cats, "smear", batch_size=3)

    print("[INFO][python/nll] adaptive scan complete")
    return guess