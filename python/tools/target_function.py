import numpy as np
import scipy.optimize
from scipy.optimize import OptimizeResult

from python.classes.constant_classes import CategoryConstants as cc
from python.classes.constant_classes import DataConstants as dc


def target_function_wrapper(initial_guess, __ZCATS__, *args, **kwargs):
    """
    Wrapper for the target function. This is necessary to keep track of the previous guess, to eliminate redundant calculations.

    Args:
        __GUESS__ (iterable): iterable of floats, representing the initial guess for the scales and smearings
        __ZCATS__ (list): list of zcat objects, each representing a category
        **options: keyword arguments, which contain the following:
            num_scales (int): number of scales to be derived
            num_smears (int): number of smearings to be derived
    Returns:
        target_function (function): target function
    """

    previous_guess = [initial_guess]

    def wrapped_target_function(x, *args, **options):
        (previous, __ZCATS__, __num_scales__, __num_smears__) = args
        ret = target_function(
            x, previous_guess[0], __ZCATS__, __num_scales__, __num_smears__, **options
        )
        previous_guess[0] = x
        return ret

    def reset(x=None):
        previous_guess[0] = x if x is not None else initial_guess

    return wrapped_target_function, reset


def target_function(x, *args, verbose=False, **options):
    """
    This is the target function, which returns an event weighted -2*Delta NLL
    This function features a small verbose option for debugging purposes.
    target_function accepts an iterable of floats and uses them to evaluate the NLL in each category.
    Some 'smart' checks prevent the function from evaluating all N(N+1)/2 categories unless absolutely necessary.

    Args:
        x (iterable): iterable of floats, representing the scales and smearings chosen by the minimizer
        *args: a tuple of arguments, which contains the following:
            __GUESS__ (iterable): iterable of floats, representing the initial guess for the scales and smearings
            __ZCATS__ (list): list of zcat objects, each representing a category
            __num_scales__ (int): number of scales to be derived
            __num_smears__ (int): number of smearings to be derived
    """

    # unpack args
    (previous, __ZCATS__, __num_scales__, __num_smears__) = args

    # find where __GUESS__ and x differ
    # no use updating categories if they don't need to be updated
    updated_scales = [i for i in range(len(x)) if x[i] != previous[i]]

    # find all the categories that need to be updated
    mask = np.array(
        [
            cat.valid
            and (
                cat.lead_index in updated_scales
                or cat.sublead_index in updated_scales
                or cat.lead_smear_index in updated_scales
                or cat.sublead_smear_index in updated_scales
            )
            for cat in __ZCATS__
        ]
    )

    cats_to_update = np.array(__ZCATS__)[mask]

    active_indices = [
        (
            cat.lead_index,
            cat.sublead_index,
            cat.lead_smear_index,
            cat.sublead_smear_index,
        )
        for cat in __ZCATS__
        if cat.valid
    ]

    # count how many times each scale is used
    scale_counts = np.zeros(len(x))
    for ind in active_indices:
        scale_counts[ind[0]] += 1
        scale_counts[ind[1]] += 1

        if __num_smears__ > 0:
            scale_counts[ind[2]] += 1
            scale_counts[ind[3]] += 1

    # if any scale is not used at all, give the user a warning
    for i, count in enumerate(scale_counts):
        if count == 0:
            print(
                f"[WARNING][python/nll] scale {i} is not used in any category. This is likely a mistake."
            )

    # update the categories
    [
        (
            cat.update(x[cat.lead_index], x[cat.sublead_index])
            if __num_smears__ == 0
            else cat.update(
                x[cat.lead_index],
                x[cat.sublead_index],
                lead_smear=x[cat.lead_smear_index],
                sublead_smear=x[cat.sublead_smear_index],
            )
        )
        for cat in cats_to_update
    ]

    if verbose:
        print("------------- zcat info -------------")
        [cat.print() for cat in cats_to_update]
        print("-------------------------------------")
        print()

    tot = sum([cat.weight for cat in __ZCATS__ if cat.valid])
    ret = sum([cat.NLL * cat.weight for cat in __ZCATS__ if cat.valid])
    final_value = ret / tot if tot != 0 else 9e30

    if verbose:
        print("------------- total info -------------")
        # print("weighted nll:",ret/tot)
        print(
            "diagonal nll vals:",
            [
                cat.NLL * cat.weight / tot
                for cat in __ZCATS__
                if cat.lead_index == cat.sublead_index and cat.valid
            ],
        )
        print("using scales:", x)
        print("--------------------------------------")

    return final_value


def calculate_gradient(x, *args, h=1e-6, verbose=False, **options):
    """
    Numerically calculates the gradient of the target function using central difference method.

    Args:
        x (iterable): iterable of floats, representing the scales and smearings
        *args: a tuple of arguments to pass to target_function
        h (float): step size for numerical differentiation
        verbose (bool): whether to print debugging information
        **options: keyword arguments to pass to target_function

    Returns:
        gradient (numpy.ndarray): gradient vector of the target function at point x
    """
    gradient = np.zeros_like(x, dtype=float)
    base_loss = target_function(x, *args, verbose=False, **options)

    if verbose:
        print("Calculating gradient, base loss:", base_loss)

    for i in range(len(x)):
        # Create copies with one parameter slightly adjusted
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] += h
        x_minus[i] -= h

        # Calculate losses for the adjusted points
        loss_plus = target_function(x_plus, *args, verbose=False, **options)
        loss_minus = target_function(x_minus, *args, verbose=False, **options)

        # Central difference approximation of the derivative
        gradient[i] = (loss_plus - loss_minus) / (2 * h)

        if verbose:
            print(
                f"Parameter {i}: derivative = {gradient[i]:.6f} (f({x[i]+h:.6f})={loss_plus:.6f}, f({x[i]-h:.6f})={loss_minus:.6f})"
            )

    return gradient


def scan_nll(x, **options):
    """
    Performs the NLL scan to initialize the variables.

    Args:
        x (iterable): iterable of floats, representing the scales and smearings chosen by the minimizer
        **options: keyword arguments, which contain the following:
            __GUESS__ (iterable): iterable of floats, representing the initial guess for the scales and smearings
            __ZCATS__ (list): list of zcat objects, each representing a category
            _kFixScales (bool): whether or not to fix the scales
            num_scales (int): number of scales to be derived
            num_smears (int): number of smearings to be derived
            scan_min (float): minimum value for the scan
            scan_max (float): maximum value for the scan
            scan_step (float): step size for the scan
    Returns:
        guess (iterable): iterable of floats, representing the scales and smearings chosen by the minimizer
    """
    __ZCATS__ = options["zcats"]
    __GUESS__ = options["__GUESS__"]
    guess = x
    scanned = []

    # find most sensitive category and scan that first
    print("[INFO][python/helper_minimizer/scan_ll] scanning scales")
    weights = [
        (cat.weight, cat.lead_index)
        for cat in __ZCATS__
        if cat.valid and cat.lead_index == cat.sublead_index
    ]
    weights.sort(key=lambda x: x[0])
    loss_function, reset_loss_initial_guess = target_function_wrapper(
        guess, __ZCATS__, **options
    )

    if not options["_kFixScales"]:
        while weights:
            max_index = cc.empty
            tup = weights.pop(0)

            if tup[cc.i_eta_min] not in scanned:
                max_index = tup[cc.i_eta_min]
                scanned.append(tup[cc.i_eta_min])

            if max_index != cc.empty:
                x = np.arange(
                    options["scan_min"], options["scan_max"], options["scan_step"]
                )
                my_guesses = []

                # generate a few guesses
                for j, val in enumerate(x):
                    guess[max_index] = val
                    my_guesses.append(guess[:])

                # evaluate nll for each guess
                nll_vals = np.array(
                    [
                        loss_function(
                            g,
                            __GUESS__,
                            __ZCATS__,
                            options["num_scales"],
                            options["num_smears"],
                        )
                        for g in my_guesses
                    ]
                )
                mask = [
                    y > 0 for y in nll_vals
                ]  # addresses edge cases of scale being too large/small
                x = x[mask]
                nll_vals = nll_vals[mask]

                if len(nll_vals) > 0:
                    guess[max_index] = x[nll_vals.argmin()]
                    print(
                        "[INFO][python/nll] best guess for scale {} is {}".format(
                            max_index, guess[max_index]
                        )
                    )

    print("[INFO][python/helper_minimizer/scan_nll] scanning smearings:")
    scanned = []
    weights = [
        (cat.weight, cat.lead_smear_index)
        for cat in __ZCATS__
        if cat.valid and cat.lead_smear_index == cat.sublead_smear_index
    ]
    weights.sort(key=lambda x: x[0])

    low = 0.00025
    high = 0.025
    step = 0.00025
    x = np.arange(low, high, step)
    if options["num_smears"] > 0:
        while weights:
            max_index = cc.empty
            tup = weights.pop(0)

            if tup[cc.i_eta_min] not in scanned:
                max_index = tup[cc.i_eta_min]
                scanned.append(tup[cc.i_eta_min])

            # smearings are different, so use different values for low,high,step
            if max_index != cc.empty:
                my_guesses = []

                # generate a few guesses
                for j, val in enumerate(x):
                    guess[max_index] = val
                    my_guesses.append(guess[:])

                # evaluate nll for each guess
                nll_vals = np.array(
                    [
                        loss_function(
                            g,
                            __GUESS__,
                            __ZCATS__,
                            options["num_scales"],
                            options["num_smears"],
                        )
                        for g in my_guesses
                    ]
                )
                mask = [
                    y > 0 for y in nll_vals
                ]  # addresses edge cases of scale being too large/small
                x = x[mask]
                nll_vals = nll_vals[mask]
                if len(nll_vals) > 0:
                    guess[max_index] = x[nll_vals.argmin()]
                    print(
                        f"[INFO][python/nll] best guess for smearing {max_index} is {guess[max_index]}"
                    )

    print("[INFO][python/nll] scan complete")
    return guess


class AdamWMinimizer:
    """
    Implementation of AdamW optimizer compatible with scipy.optimize.minimize interface.

    AdamW is Adam with decoupled weight decay regularization, which can improve
    generalization performance in optimization problems.
    """

    def __init__(
        self,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        max_iter=1000,
        tol=1e-5,
        verbose=False,
    ):
        """
        Initialize AdamW optimizer.

        Args:
            lr: Learning rate
            betas: Coefficients for computing running averages of gradient and its square
            eps: Term added to denominator to improve numerical stability
            weight_decay: Weight decay coefficient
            max_iter: Maximum number of iterations
            tol: Tolerance for termination
            verbose: Whether to print progress
        """
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # States to be initialized when optimize is called
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.t = 0  # Timestep

    def _step(self, x, grad, func_val):
        """
        Perform one optimization step.

        Args:
            x: Current parameter values
            grad: Gradient of objective function at x
            func_val: Value of objective function at x

        Returns:
            new_x: Updated parameter values
        """
        # Initialize moment estimates on first call
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)

        self.t += 1

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad

        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad * grad)

        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # AdamW decoupled weight decay
        x_wd = x * (1 - self.lr * self.weight_decay)

        # Update parameters
        new_x = x_wd - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return new_x

    def minimize(self, fun, x0, args=(), jac=None, bounds=None, callback=None):
        """
        Minimize a function using AdamW.

        Args:
            fun: Objective function to minimize
            x0: Initial guess
            args: Extra arguments passed to the objective function
            jac: Method for computing the gradient vector
            bounds: Bounds for variables
            callback: Called after each iteration

        Returns:
            OptimizeResult describing the solution
        """
        x = np.asarray(x0).copy()
        if bounds is not None:
            x = np.clip(x, *zip(*bounds))

        # Initialize best solution tracker
        best_x = x.copy()
        best_fun = float("inf")

        # Initialize history for tracking convergence
        fun_history = []

        # Reset optimizer state
        self.m = None
        self.v = None
        self.t = 0

        # Function to get both function value and gradient
        if jac is None:
            # Use finite difference if no gradient provided
            def get_fun_and_grad(x_new):
                f = fun(x_new, *args)
                g = scipy.optimize._numdiff.approx_fprime(x_new, fun, 1e-8, args=args)
                return f, g

        else:

            def get_fun_and_grad(x_new):
                if callable(jac):
                    f = fun(x_new, *args)
                    g = jac(x_new, *args)
                    return f, g
                else:
                    # If jac == True, fun should return both value and gradient
                    f, g = fun(x_new, *args)
                    return f, g

        # Initial evaluation
        f, g = get_fun_and_grad(x)
        fun_history.append(f)

        # Update best solution
        if f < best_fun:
            best_fun = f
            best_x = x.copy()

        if self.verbose:
            print(f"Initial loss: {f:.6f}")

        # Main optimization loop
        for i in range(self.max_iter):
            # Perform a step
            x_new = self._step(x, g, f)

            # Apply bounds if provided
            if bounds is not None:
                x_new = np.clip(x_new, *zip(*bounds))

            # Evaluate function and gradient at new point
            f_new, g_new = get_fun_and_grad(x_new)
            fun_history.append(f_new)

            # Update best solution if improved
            if f_new < best_fun:
                best_fun = f_new
                best_x = x_new.copy()

            # Call user-provided callback if present
            if callback is not None:
                callback(x_new)

            # Check for convergence
            x_diff = np.linalg.norm(x_new - x)
            f_diff = abs(f_new - f)
            g_norm = np.linalg.norm(g_new)

            if self.verbose and (i % 20 == 0 or i == self.max_iter - 1):
                print(
                    f"Iter {i}: f={f_new:.6f}, |g|={g_norm:.6f}, |x_diff|={x_diff:.6f}"
                )

            # Update for next iteration
            x = x_new
            f = f_new
            g = g_new

            # Convergence criteria
            if x_diff < self.tol and f_diff < self.tol:
                if self.verbose:
                    print(f"Converged after {i+1} iterations.")
                break

        # Return in the same format as scipy.optimize.minimize
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


# Function to provide scipy.optimize.minimize compatible interface
def adamw_minimize(
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
    verbose=False,
    **kwargs,
):
    """
    Minimization of scalar function using AdamW algorithm.

    This creates a consistent interface with scipy.optimize.minimize.

    Args:
        fun: Objective function
        x0: Initial guess
        args: Extra arguments to pass to function
        jac: Jacobian (gradient) of objective function
        bounds: Bounds for variables
        callback: Called after each iteration
        lr: Learning rate
        betas: Coefficients for computing running averages
        eps: Term added to denominator for numerical stability
        weight_decay: Weight decay coefficient
        max_iter: Maximum number of iterations
        tol: Tolerance for termination
        verbose: Whether to print progress

    Returns:
        OptimizeResult
    """
    optimizer = AdamWMinimizer(
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
    )

    return optimizer.minimize(fun, x0, args, jac, bounds, callback)


def minimize(
    fun, x0, args=(), method="adamw", jac=None, bounds=None, callback=None, options=None
):
    """
    Wrapper to integrate AdamW with scipy.optimize.minimize interface.

    Args:
        fun: Objective function
        x0: Initial guess
        args: Extra arguments to pass to function
        method: When 'adamw', use AdamW optimizer, otherwise pass to scipy.optimize.minimize
        jac: Jacobian (gradient) of objective function
        bounds: Bounds for variables
        callback: Called after each iteration
        options: Dictionary with parameters for AdamW

    Returns:
        OptimizeResult
    """
    if method.lower() == "adamw":
        # Set default options
        default_options = {
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.01,
            "max_iter": 1000,
            "tol": 1e-5,
            "verbose": False,
        }

        # Update with user-provided options
        if options is not None:
            default_options.update(options)

        return adamw_minimize(fun, x0, args, jac, bounds, callback, **default_options)
    else:
        # Fall back to scipy's implementation
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
