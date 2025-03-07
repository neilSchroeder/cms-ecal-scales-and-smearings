import numba
import numpy as np
from scipy.optimize import OptimizeResult

from src.core.gradients import gradient_function

@numba.njit
def _step_core(
    x, m, v, t, 
    grad, beta1, beta2, one_minus_beta1, 
    one_minus_beta2, weight_decay_factor, lr, eps
    ):
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
    
    def _step(self, x, grad, func_val):
        """Optimized step function with cached computations"""
        # Initialize moments if first step
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)

        self.t += 1
        
        # Use numba-optimized core function
        
        x_new, self.m, self.v = _step_core(
            x, self.m, self.v, self.t, 
            grad, self.beta1, self.beta2, self.one_minus_beta1, 
            self.one_minus_beta2, self.weight_decay_factor, self.lr, self.eps
        )
        
        return x_new

    def minimize(
        self, fun, x0, args=(), jac=None, bounds=None, callback=None
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
            grad_fn = lambda x_new: gradient_function(x_new, *args)  # Force single thread
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