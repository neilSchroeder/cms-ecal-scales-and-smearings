import src.classes.breit_wigner as breit_wigner
import src.classes.crystal_ball as crystal_ball
import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy
import matplotlib.pyplot as plt

EPSILON = 1e-10

def NLL(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Negative Log-Likelihood (NLL) with penalty and chi-square scaling.
    This function calculates a modified negative log-likelihood value between two arrays,
    including a penalty term for deviations and scaling by a chi-square factor.
    Parameters
    ----------
    a : numpy.ndarray
        Observed values or weights (typically histogram bin contents).
    b : numpy.ndarray
        Expected or model values (must be same shape as 'a').
    Returns
    -------
    float
        The scaled negative log-likelihood value, computed as -2*(nll + penalty)*chi_sqr.
        Smaller values indicate better agreement between 'a' and 'b'.
    Notes
    -----
    - The function handles zero values in log calculations by setting -inf to 0.
    - The penalty term accounts for deviations in the complementary probability space.
    - The chi-square scaling provides additional sensitivity to larger deviations.
    """
    if len(a) != len(b):
        raise ValueError("Arrays 'a' and 'b' must have the same length.")
    
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Arrays 'a' and 'b' must have at least two elements.")
    
    # b must be normalized
    if sum(b) != 1:
        b = b / sum(b)

    # fill zeros with epsilon
    a[a == 0] = EPSILON
    b[b == 0] = EPSILON

    nll = xlogy(a, b)
    nll[nll == -np.inf] = 0
    nll = np.sum(nll) / len(nll)

    penalty = xlogy(np.sum(a) - a, 1 - b)
    penalty[penalty == -np.inf] = 0
    penalty = np.sum(penalty) / len(penalty)

    b = np.sum(a) * b / np.sum(b)
    chi_sqr = np.divide(np.multiply(a - b, a - b), a+EPSILON) / (len(a) - 1 + EPSILON)
    # remove inf
    chi_sqr = chi_sqr[np.isfinite(chi_sqr)]
    chi_sqr = np.sum(chi_sqr) / len(chi_sqr)

    return -2 * (nll + penalty) * chi_sqr


def target(guess, *args) -> float:
    """
    Calculates the negative log-likelihood (NLL) between a model and observed data.
    The model is a convolution of a Breit-Wigner function and a Crystal Ball function,
    where the Crystal Ball parameters are updated with the provided guess.
    Parameters
    ----------
    guess : array-like
        Parameters to update the Crystal Ball function
    *args : tuple
        Contains three objects:
        - thisBW : object
            Breit-Wigner function with a defined y attribute
        - thisCB : object
            Crystal Ball function that can be updated with parameters and provides a getY method
        - data : array-like
            Observed data to compare against the model
    Returns
    -------
    float
        Negative log-likelihood value representing how well the model fits the data
    """
    thisBW, thisCB, data = args

    thisCB.update(guess)
    y_vals = np.convolve(thisBW.y, thisCB.getY(), mode="same")
    y_vals = y_vals / np.sum(y_vals)

    return NLL(data, y_vals)


def fit_bw_cb(x: np.array, y: np.array, guess_cb: list) -> dict:
    """
    Fit a Breit-Wigner convoluted with a Crystal Ball to a set of data

    Args:
        x (np.array): x values
        y (np.array): y values
        guess_cb (list): guess for the crystal ball parameters [alpha, n, mu, sigma]
        Returns:
                dictionary: dictionary containing the fit parameters
    """
    # make bw distribution
    thisBW = breit_wigner.bw(x)
    thisCB = crystal_ball.cb(x, guess_cb)
    guess = guess_cb

    bounds = [(0.01, 100), (0.01, 100), (-5, 5), (0.1, 10)]

    print(guess)
    result = minimize(
        target,
        np.array(guess),
        args=(thisBW, thisCB, y),
        method="Nelder-Mead",
        bounds=bounds,
        options={"eps": 0.0001},
    )

    print(result)
    thisCB.update(result.x)
    y_vals = np.convolve(thisBW.getY(), thisCB.getY(), mode="same")
    y_vals = np.sum(y) * y_vals / (np.sum(y_vals)+EPSILON)
    chi_sqr = np.sum(np.divide(np.multiply(y - y_vals, y - y_vals), y + EPSILON)) / (len(y) - 1)
    print("minimization complete:")
    print("mu:", 91.188 + result.x[2])
    print("sigma:", result.x[3])
    print("reduced chi squared:", chi_sqr)
    print()

    return {
        "mu": 91.188 + result.x[2],
        "sigma": result.x[3],
        "chi_sqr": chi_sqr,
        "fit_hist": y_vals,
        "fit_params": result.x,
    }
