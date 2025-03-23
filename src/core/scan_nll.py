import multiprocessing

import numpy as np
from joblib import Parallel, delayed, parallel_backend

from src.core.target_function import enhanced_target_function_wrapper


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
    loss_function, reset_loss_initial_guess, _ = enhanced_target_function_wrapper(
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
    """Advanced version of scan_nll with adaptive grid refinement for more accurate results."""

    __ZCATS__ = options["zcats"]
    __GUESS__ = options["__GUESS__"]

    # Ensure we're using float64 for all parameters
    guess = np.array(x, dtype=np.float64).copy()

    # Configure parallel processing
    n_jobs = options.get("n_jobs", 1)
    if n_jobs == -1:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free

    # Create the loss function wrapper
    loss_function, reset_loss_initial_guess, _ = enhanced_target_function_wrapper(
        guess, __ZCATS__, **options
    )

    # check loss function and reset_loss_initial_guess
    print(loss_function)
    print(reset_loss_initial_guess)

    # test loss function with two different guesses
    print(
        loss_function(
            guess, __GUESS__, __ZCATS__, options["num_scales"], options["num_smears"]
        )
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
            batch = param_list[i : i + batch_size]
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
                    coarse_grid = np.logspace(
                        np.log10(min_val), np.log10(max_val), n_points
                    )
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
        """Process a batch of parameters with coarse-to-fine grid search"""

        # Function to evaluate all parameter values in one batch - more efficient
        def evaluate_batch(configs):
            results = []
            for param_idx, param_val in configs:
                # Create a float copy of the guess to avoid integer conversion
                test_guess = current_guess.copy().astype(np.float64)
                # Explicitly cast param_val to float
                test_guess[param_idx] = float(param_val)

                # For debugging
                print(f"Testing param {param_idx} = {param_val}, guess: {test_guess}")

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
        eval_batches = [
            all_evals[j : j + max_evals_per_worker]
            for j in range(0, len(all_evals), max_evals_per_worker)
        ]

        # Execute parallel evaluation with optimized backend
        with parallel_backend("loky", n_jobs=n_jobs):
            batch_results = Parallel(
                verbose=0,
                batch_size="auto",
                pre_dispatch="2*n_jobs",
                max_nbytes="100M",
                mmap_mode="r",
                temp_folder="/tmp",
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
                window = (
                    options.get("scan_max", 1.2) - options.get("scan_min", 0.8)
                ) / 10
                refined_min = max(options.get("scan_min", 0.8), best_val - window)
                refined_max = min(options.get("scan_max", 1.2), best_val + window)
                n_points = 15
                refined_grid = np.linspace(refined_min, refined_max, n_points)

            # Prepare refinement evaluation in batches
            refined_evals = [(param_index, val) for val in refined_grid]
            refined_batches = [
                refined_evals[j : j + max_evals_per_worker]
                for j in range(0, len(refined_evals), max_evals_per_worker)
            ]

            # Execute parallel refinement evaluation
            with parallel_backend("loky", n_jobs=n_jobs):
                refined_results_batches = Parallel(
                    verbose=0,
                    batch_size="auto",
                    max_nbytes="100M",
                    mmap_mode="r",
                    temp_folder="/tmp",
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
                    print(
                        f"[INFO][python/nll] best guess for {type_str} {param_index} is {best_val:.6f}"
                    )

    # Collect categories for parameter optimization
    if options["num_smears"] > 0 and not options["_kClosure"]:
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
    if options["num_smears"] > 0 and not options["_kClosure"]:
        print(
            "[INFO][python/helper_minimizer/scan_nll] stage 1: adaptively scanning smearings"
        )
        optimize_parameters(smear_diagonal_cats, "smear", batch_size=5)

    # -------------------------------------------------
    # STAGE 2: Scale optimization
    # -------------------------------------------------
    if not options["_kFixScales"]:
        print(
            "[INFO][python/helper_minimizer/scan_ll] stage 2: adaptively scanning scales"
        )
        optimize_parameters(scale_diagonal_cats, "scale", batch_size=10)

    # -------------------------------------------------
    # STAGE 3: Second smearing optimization (refinement)
    # -------------------------------------------------
    if options["num_smears"] > 0 and not options["_kClosure"]:
        print("[INFO][python/helper_minimizer/scan_nll] stage 3: refining smearings")
        # Use smaller batch size for refinement stage to focus more on individual parameters
        optimize_parameters(smear_diagonal_cats, "smear", batch_size=3)

    print("[INFO][python/nll] adaptive scan complete")
    return guess
