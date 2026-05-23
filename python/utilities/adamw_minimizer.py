"""AdamW minimizer with a scipy.optimize.minimize-compatible API."""

import numpy as np
from scipy.optimize import OptimizeResult


def adamw_minimize(
    fun,
    x0,
    args=(),
    bounds=None,
    options=None,
    **ignored,
):
    """
    Minimize a scalar function using AdamW (Adam with decoupled weight decay).

    Drop-in replacement for ``scipy.optimize.minimize(method='L-BFGS-B')``.
    Gradients are estimated via central finite differences.

    Parameters
    ----------
    fun : callable
        Objective function ``fun(x, *args) -> float``.
    x0 : array_like
        Initial guess, shape ``(n,)``.
    args : tuple, optional
        Extra arguments passed to *fun*.
    bounds : sequence of (min, max) pairs, optional
        Bounds for each variable.  ``None`` entries mean unbounded.
    options : dict, optional
        Solver options:

        - **lr** (*float*) – learning rate (default ``1e-3``).
        - **eps** (*float*) – finite-difference step and Adam epsilon
          (default ``1e-5``).
        - **beta1** (*float*) – first moment decay (default ``0.9``).
        - **beta2** (*float*) – second moment decay (default ``0.999``).
        - **weight_decay** (*float*) – decoupled weight decay (default ``1e-4``).
        - **maxiter** (*int*) – maximum iterations (default ``2000``).
        - **ftol** (*float*) – convergence tolerance on function value
          change (default ``1e-9``).
        - **gtol** (*float*) – convergence tolerance on gradient norm
          (default ``1e-7``).
        - **patience** (*int*) – stop after this many iterations without
          improvement (default ``200``).
        - **disp** (*bool*) – print progress every 100 iterations
          (default ``False``).

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        With fields ``x``, ``fun``, ``nit``, ``nfev``, ``success``,
        ``message``, and ``jac``.
    """
    opts = options or {}
    lr = float(opts.get("lr", 1e-3))
    fd_eps = float(opts.get("eps", 1e-5))
    beta1 = float(opts.get("beta1", 0.9))
    beta2 = float(opts.get("beta2", 0.999))
    weight_decay = float(opts.get("weight_decay", 1e-4))
    maxiter = int(opts.get("maxiter", 2000))
    ftol = float(opts.get("ftol", 1e-9))
    gtol = float(opts.get("gtol", 1e-7))
    patience = int(opts.get("patience", 200))
    disp = bool(opts.get("disp", False))
    adam_eps = 1e-8  # numerical stability constant for Adam denominator

    x = np.array(x0, dtype=np.float64).copy()
    n = len(x)
    nfev = 0

    # parse bounds
    lower = np.full(n, -np.inf)
    upper = np.full(n, np.inf)
    if bounds is not None:
        for i, b in enumerate(bounds):
            if b is not None:
                if b[0] is not None:
                    lower[i] = b[0]
                if b[1] is not None:
                    upper[i] = b[1]

    # project onto feasible region
    x = np.clip(x, lower, upper)

    # Adam state
    m = np.zeros(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.float64)

    # evaluate initial objective
    f_val = fun(x, *args)
    nfev += 1
    best_f = f_val
    best_x = x.copy()
    no_improve = 0

    message = "maximum iterations reached"
    success = False

    for t in range(1, maxiter + 1):
        # central finite-difference gradient
        grad = np.zeros(n, dtype=np.float64)
        for i in range(n):
            step = fd_eps * max(1.0, abs(x[i]))
            x_fwd = x.copy()
            x_bwd = x.copy()
            x_fwd[i] = min(x[i] + step, upper[i])
            x_bwd[i] = max(x[i] - step, lower[i])
            actual_step = x_fwd[i] - x_bwd[i]
            if actual_step == 0:
                grad[i] = 0.0
            else:
                f_fwd = fun(x_fwd, *args)
                f_bwd = fun(x_bwd, *args)
                nfev += 2
                grad[i] = (f_fwd - f_bwd) / actual_step

        # gradient norm convergence check
        grad_norm = np.linalg.norm(grad, ord=np.inf)
        if grad_norm < gtol:
            message = f"gradient norm {grad_norm:.2e} below gtol={gtol:.2e}"
            success = True
            break

        # Adam moment updates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2

        # bias-corrected moments
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # AdamW update: decoupled weight decay + Adam step
        x = x - lr * (m_hat / (np.sqrt(v_hat) + adam_eps) + weight_decay * x)

        # project back onto bounds
        x = np.clip(x, lower, upper)

        # evaluate new objective
        f_prev = f_val
        f_val = fun(x, *args)
        nfev += 1

        # track best
        if f_val < best_f:
            best_f = f_val
            best_x = x.copy()
            no_improve = 0
        else:
            no_improve += 1

        # function-value convergence check
        if abs(f_val - f_prev) < ftol:
            message = (
                f"function value change {abs(f_val - f_prev):.2e} below ftol={ftol:.2e}"
            )
            success = True
            break

        # early stopping on patience
        if no_improve >= patience:
            message = f"no improvement for {patience} iterations (early stop)"
            success = True
            x = best_x.copy()
            f_val = best_f
            break

        if disp and t % 100 == 0:
            print(
                f"[AdamW] iter {t:5d} | f={f_val:.8e} | best={best_f:.8e} | "
                f"|grad|={grad_norm:.2e}"
            )

    # always return the best point found
    if f_val > best_f:
        x = best_x.copy()
        f_val = best_f

    return OptimizeResult(
        x=x,
        fun=f_val,
        nit=t if "t" in dir() else 0,
        nfev=nfev,
        success=success,
        message=message,
        jac=grad if "grad" in dir() else np.zeros(n),
    )
