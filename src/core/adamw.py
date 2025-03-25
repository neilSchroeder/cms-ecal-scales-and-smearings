import numba
import numpy as np
from scipy.optimize import OptimizeResult

from src.core.gradients import gradient_function


@numba.njit
def _step_core(
    x,
    m,
    v,
    t,
    grad,
    beta1,
    beta2,
    one_minus_beta1,
    one_minus_beta2,
    weight_decay_factor,
    lr,
    eps,
):
    """Core step computation optimized with Numba"""
    # Update moments with better numerical stability
    m = beta1 * m + one_minus_beta1 * grad
    v = beta2 * v + one_minus_beta2 * (np.square(grad) + eps)

    # Bias correction
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    # AdamW update (combined operations)
    x_new = weight_decay_factor * x - lr * m_hat / (np.sqrt(v_hat) + eps)

    return x_new, m, v


@numba.njit
def _step_core_inplace(
    x,
    m,
    v,
    t,
    grad,
    beta1,
    beta2,
    one_minus_beta1,
    one_minus_beta2,
    weight_decay_factor,
    lr,
    eps,
    m_out,
    v_out,
    x_out,
):
    """Core step computation optimized with Numba using in-place operations"""
    # Update moments with better numerical stability
    for i in range(len(x)):
        m_out[i] = beta1 * m[i] + one_minus_beta1 * grad[i]
        v_out[i] = beta2 * v[i] + one_minus_beta2 * (grad[i] * grad[i] + eps)

        # Bias correction
        m_hat = m_out[i] / (1 - beta1**t)
        v_hat = v_out[i] / (1 - beta2**t)

        # AdamW update
        x_out[i] = weight_decay_factor * x[i] - lr * m_hat / (np.sqrt(v_hat) + eps)


@numba.njit
def _vec_norm(x):
    """Calculate L2 norm using Numba"""
    norm_sq = 0.0
    for i in range(len(x)):
        norm_sq += x[i] * x[i]
    return np.sqrt(norm_sq)


def gradient_function(func, x, *args):
    """Calculate numerical gradient of func at x"""
    eps = np.sqrt(np.finfo(float).eps)
    grad = np.zeros_like(x)
    x_plus = x.copy()

    # Calculate gradients
    for i in range(len(x)):
        x_plus[i] = x[i] + eps
        f_plus = func(x_plus, *args)
        f_minus = func(x, *args)
        grad[i] = (f_plus - f_minus) / eps
        x_plus[i] = x[i]  # Reset

    return grad


class OptimizedAdamWMinimizer:
    """
    Optimized AdamW implementation with improved computational efficiency
    and better convergence.
    """

    def __init__(
        self,
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-6,
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

        # States - initialize to None but will be allocated properly in _step
        self.m = None
        self.v = None
        self.t = 0

        # Cache for performance
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.one_minus_beta1 = 1 - betas[0]
        self.one_minus_beta2 = 1 - betas[1]
        self.weight_decay_factor = 1 - lr * weight_decay

    def _step(self, x, grad, func_val):
        """Optimized step function with cached computations"""
        # Initialize moments if first step
        if self.m is None or self.v is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)

        self.t += 1

        # Better error messaging
        if x is None:
            raise ValueError("Input array 'x' is None")
        if grad is None:
            raise ValueError(
                "Gradient array 'grad' is None - check your gradient function"
            )
        if self.m is None:
            raise ValueError("Momentum array 'm' is None")
        if self.v is None:
            raise ValueError("Velocity array 'v' is None")

        # Make sure all scalar inputs are proper Python scalars, not None
        scalar_params = {
            "t": self.t,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "one_minus_beta1": self.one_minus_beta1,
            "one_minus_beta2": self.one_minus_beta2,
            "weight_decay_factor": self.weight_decay_factor,
            "lr": self.lr,
            "eps": self.eps,
        }

        for name, value in scalar_params.items():
            if value is None:
                raise ValueError(f"Scalar parameter '{name}' is None")

        # Use numba-optimized core function
        x_new, self.m, self.v = _step_core(
            x,
            self.m,
            self.v,
            self.t,
            grad,
            self.beta1,
            self.beta2,
            self.one_minus_beta1,
            self.one_minus_beta2,
            self.weight_decay_factor,
            self.lr,
            self.eps,
        )

        return x_new

    def _step_inplace(self, x, grad, func_val, x_out):
        """In-place optimized step function with cached computations"""
        # Initialize moments if first step
        if self.m is None or self.v is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)

        self.t += 1

        # Better error messaging (minimized for performance)
        if x is None or grad is None or self.m is None or self.v is None:
            raise ValueError("One of the required arrays is None")

        # Use numba-optimized in-place core function
        _step_core_inplace(
            x,
            self.m,
            self.v,
            self.t,
            grad,
            self.beta1,
            self.beta2,
            self.one_minus_beta1,
            self.one_minus_beta2,
            self.weight_decay_factor,
            self.lr,
            self.eps,
            self.m,  # Output for m (in-place)
            self.v,  # Output for v (in-place)
            x_out,  # Output for x
        )

        return x_out

    def minimize(self, fun, x0, args=(), jac=None, bounds=None, callback=None):
        """Minimization with early stopping and learning rate scheduling."""
        # Provide default bounds if none are given
        if bounds is None and len(x0) > 0:
            # Create bounds arrays directly instead of list
            lb = np.empty(len(x0), dtype=np.float64)
            ub = np.empty(len(x0), dtype=np.float64)

            # Set default bounds more efficiently
            scale_count = min(8, len(x0))
            lb[:scale_count] = 0.98
            ub[:scale_count] = 1.02

            if scale_count < len(x0):
                lb[scale_count:] = 0.005
                ub[scale_count:] = 0.05
        elif bounds is not None:
            # Convert bounds to arrays for faster access
            lb, ub = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
        else:
            lb, ub = None, None

        # Convert to contiguous array for better memory access patterns
        x = np.ascontiguousarray(x0, dtype=np.float64)

        # Pre-allocate arrays for better performance
        x_new = np.empty_like(x)
        best_x = x.copy()

        # Apply bounds to initial x0 if needed
        if bounds is not None:
            np.clip(x, lb, ub, out=x)

        # Setup gradient function with error handling
        if jac is None:

            def safe_grad_fn(x_new):
                try:
                    grad = gradient_function(fun, x_new, *args)
                    if grad is None:
                        print(
                            "Warning: Gradient function returned None, using default gradient"
                        )
                        return np.ones_like(x_new) * 1e-6
                    return grad
                except Exception as e:
                    print(f"Error in gradient calculation: {e}")
                    return np.ones_like(x_new) * 1e-6

            grad_fn = safe_grad_fn
        else:
            # Wrap custom jacobian with same error handling
            def safe_jac_fn(x_new):
                try:
                    grad = jac(x_new, *args)
                    if grad is None:
                        print(
                            "Warning: Custom gradient function returned None, using default gradient"
                        )
                        g = jac(x_new, *args, verbose=True)
                        print(f"gradient returned {g}")
                        return np.ones_like(x_new) * 1e-6
                    return grad
                except Exception as e:
                    print(f"Error in custom gradient: {e}")
                    return np.ones_like(x_new) * 1e-6

            grad_fn = safe_jac_fn

        # Setup tracking variables
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

        # Initial evaluation
        f = fun(x, *args)
        g = grad_fn(x)
        g_new = np.empty_like(g)  # Pre-allocate gradient array
        fun_history.append(f)

        if f < best_fun:
            best_fun = f
            np.copyto(best_x, x)  # Use copyto instead of copy

        if self.verbose:
            print(f"Initial loss: {f:.6f}")

        # Main optimization loop - avoid function calls in tight loop
        i = 0
        for i in range(self.max_iter):
            # Step with current parameters using in-place operations
            self._step_inplace(x, g, f, x_new)

            # Apply bounds in-place
            if bounds is not None:
                np.clip(x_new, lb, ub, out=x_new)

            # Evaluate at new point
            f_new = fun(x_new, *args)
            if self.verbose and i % 50 == 0:  # Reduced logging frequency
                print(f"Iter {i}: f={f_new:.6f}")

            g_new = grad_fn(x_new)  # Get new gradient
            fun_history.append(f_new)

            # Update best solution without copying
            if f_new < best_fun:
                improve_ratio = (best_fun - f_new) / (best_fun + self.eps)
                best_fun = f_new
                np.copyto(best_x, x_new)  # Use copyto instead of copy
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

            # Convergence checks (less frequently to improve performance)
            if i % 5 == 0:  # Only check every 5 iterations
                x_diff = _vec_norm(
                    x_new - x
                )  # Use numba function instead of np.linalg.norm
                f_diff = abs(f_new - f)
                g_norm = _vec_norm(g_new)  # Use numba function

                if self.verbose and (
                    i % 20 == 0 or i == self.max_iter - 1
                ):  # Reduced frequency
                    print(
                        f"Iter {i}: f={f_new:.6f}, |g|={g_norm:.6f}, |x_diff|={x_diff:.6f}, lr={self.lr:.8f}"
                    )

                # Strict convergence check
                if x_diff < self.tol and f_diff < self.tol and g_norm < self.tol:
                    if self.verbose:
                        print(f"Converged after {i+1} iterations.")
                    break

            # Prepare for next iteration - swap arrays instead of copying
            x, x_new = x_new, x
            f = f_new
            g, g_new = g_new, g

        # Return result
        result = OptimizeResult(
            x=best_x,
            fun=best_fun,
            jac=g,
            nit=i + 1,
            nfev=i + 1,
            success=(i < self.max_iter - 1) or (_vec_norm(g) < self.tol),
            message=(
                "Optimization terminated successfully."
                if ((i < self.max_iter - 1) or (_vec_norm(g) < self.tol))
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

    return optimizer.minimize(fun, x0, args, jac, bounds, callback)
