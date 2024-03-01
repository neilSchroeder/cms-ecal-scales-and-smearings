import python.classes.breit_wigner as breit_wigner
import python.classes.crystal_ball as crystal_ball
import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy
import matplotlib.pyplot as plt

def NLL(a,b):

    nll = xlogy(a,b)
    nll[nll==-np.inf] = 0
    nll = np.sum(nll)/len(nll)

    penalty = xlogy(np.sum(a) - a, 1 - b)
    penalty[penalty==-np.inf] = 0
    penalty = np.sum(penalty)/len(penalty)

    b = np.sum(a)*b/np.sum(b)
    chi_sqr = np.sum(np.divide(np.multiply(a-b,a-b),a))/(len(a)-1)
    
    return -2*(nll+penalty)*chi_sqr


def target(guess):
    global thisBW
    global thisCB
    global _DATA_

    thisCB.update(guess)
    y_vals = np.convolve(thisBW.y,thisCB.getY(),mode='same')
    y_vals = y_vals/np.sum(y_vals)

    return NLL(_DATA_,y_vals)


def fit_bw_cb(x: np.array, y: np.array, guess_cb: list) -> dict:
    """
    Fit a Breit-Wigner convoluted with a Crystal Ball to a set of data

    Args:
        x (np.array): x values
        y (np.array): y values
        guess_cb (list): guess for the crystal ball parameters
	Returns:
		dictionary: dictionary containing the fit parameters
    """
    global thisBW
    global thisCB
    global _DATA_
    
    #make bw distribution
    thisBW = breit_wigner.bw(x)
    thisCB = crystal_ball.cb(x, guess_cb)
    _DATA_ = y
    guess = guess_cb

    bounds = [(0.01,100), (0.01,100), (-5,5), (0.1,10)]
    
    print(guess)
    result = minimize(target, 
            np.array(guess), 
            method="L-BFGS-B",
            bounds=bounds,
            options={"eps":0.0001}
            )

    print(result)
    thisCB.update(result.x)
    y_vals = np.convolve(thisBW.getY(),thisCB.getY(),mode='same')
    y_vals = np.sum(y)*y_vals/np.sum(y_vals)
    chi_sqr = np.sum(np.divide(np.multiply(y-y_vals,y-y_vals),y))/(len(y)-1)
    print("minimization complete:")
    print("mu:", 91.188+result.x[2])
    print("sigma:", result.x[3])
    print("reduced chi squared:", chi_sqr)
    print()    
    
    return {"mu": 91.188+result.x[2], "sigma": result.x[3], "chi_sqr": chi_sqr, "fit_hist": y_vals, "fit_params": result.x}