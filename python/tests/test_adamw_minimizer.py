"""Tests for the AdamW minimizer."""

import numpy as np
import pytest

from python.utilities.adamw_minimizer import adamw_minimize


class TestAdamWMinimizer:
    """Test AdamW minimizer matches scipy.optimize.minimize API."""

    def test_rosenbrock_unbounded(self):
        """Minimise the Rosenbrock function (classic test) without bounds."""

        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        result = adamw_minimize(
            rosenbrock,
            x0=[0.0, 0.0],
            options={"lr": 1e-3, "maxiter": 10000, "ftol": 1e-12, "patience": 2000},
        )
        assert result.success
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=0.05)

    def test_quadratic_with_bounds(self):
        """Minimize a simple quadratic with bounds."""

        def quadratic(x):
            return (x[0] - 3.0) ** 2 + (x[1] - 5.0) ** 2

        result = adamw_minimize(
            quadratic,
            x0=[0.0, 0.0],
            bounds=[(0, 4), (0, 4)],  # x[1] bounded away from true min
            options={"lr": 1e-2, "maxiter": 5000},
        )
        assert result.success
        np.testing.assert_allclose(result.x[0], 3.0, atol=0.05)
        np.testing.assert_allclose(
            result.x[1], 4.0, atol=0.05
        )  # clamped to upper bound

    def test_extra_args_forwarded(self):
        """Extra args tuple is forwarded to the objective."""

        def shifted_quad(x, shift):
            return (x[0] - shift) ** 2

        result = adamw_minimize(
            shifted_quad,
            x0=[0.0],
            args=(7.0,),
            options={"lr": 5e-2, "maxiter": 5000},
        )
        assert result.success
        np.testing.assert_allclose(result.x[0], 7.0, atol=0.1)

    def test_result_fields(self):
        """Result has all standard OptimizeResult fields."""

        def f(x):
            return x[0] ** 2

        result = adamw_minimize(f, x0=[5.0], options={"maxiter": 100})
        assert hasattr(result, "x")
        assert hasattr(result, "fun")
        assert hasattr(result, "success")
        assert hasattr(result, "message")
        assert hasattr(result, "nit")
        assert hasattr(result, "nfev")
        assert hasattr(result, "jac")

    def test_convergence_on_gtol(self):
        """Should converge when gradient is small enough."""

        def f(x):
            return x[0] ** 2

        result = adamw_minimize(
            f, x0=[0.001], options={"gtol": 1e-3, "maxiter": 5000, "lr": 1e-3}
        )
        assert result.success
        assert "gtol" in result.message

    def test_nfev_positive(self):
        """Number of function evaluations should be tracked."""

        def f(x):
            return x[0] ** 2

        result = adamw_minimize(f, x0=[5.0], options={"maxiter": 10})
        assert result.nfev > 0

    def test_one_sided_bounds(self):
        """None entries in bounds mean unbounded on that side."""

        def f(x):
            return (x[0] + 10) ** 2

        result = adamw_minimize(
            f,
            x0=[5.0],
            bounds=[(0, None)],  # lower-bounded at 0, true min at -10
            options={"lr": 1e-2, "maxiter": 3000},
        )
        np.testing.assert_allclose(result.x[0], 0.0, atol=0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
